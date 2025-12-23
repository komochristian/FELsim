import numpy as np
import os
from cosySimulator import COSYSimulator
from physicalConstants import PhysicalConstants
import warnings


class COSYParticleSimulator(COSYSimulator):
    """
    Particle tracking through COSY INFINITY beamline models.

    Handles coordinate transformations between FELsim and COSY conventions,
    particle generation, and statistical analysis at arbitrary beamline positions.

    Coordinate conventions:
        FELsim: [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF(10^-3), δW/W(10^-3)]
        COSY:   [x(m), a=px/p0, y(m), b=py/p0, l(m), δK=(K-K0)/K]

    Notes:
        - a, b are normalized transverse momenta (≈ angles for paraxial beams)
        - l is longitudinal coordinate: -(t-t0)*v0*γ/(1+γ)
        - δK is relative kinetic energy deviation
    """

    def __init__(self, excel_path, json_config_path=None, config_dict=None,
                 debug=None, use_enge_coeffs=True, use_mge_for_dipoles=False):
        super().__init__(excel_path, json_config_path, config_dict,
                         debug, use_enge_coeffs, use_mge_for_dipoles)

        # Particle tracking configuration
        self.particle_tracking_enabled = False
        self.particle_save_points = []
        self.n_particles = 0
        self.f = 2856e6  # RF frequency (Hz)

        # File I/O
        self.particle_input_file = 'particles_in.dat'
        self.particle_output_file = 'particles_out.dat'

        # Internal storage
        self.initial_particles_felsim = None
        self.initial_particles_cosy = None
        self.final_particles_cosy = None
        self.final_particles_felsim = None
        self.checkpoint_particles = {}

    # -------------------------------------------------------------------------
    # Particle generation
    # -------------------------------------------------------------------------

    def generate_6d_gaussian(self, mean=0, std_dev=None, num_particles=1000,
                             energy=None, epsilon_n=None, beam_size=None,
                             bunch_length=None, energy_spread=None,
                             energy_chirp=0):
        """
        Generate 6D Gaussian particle distribution.

        Two interfaces available:
        1. Direct: specify std_dev array
        2. Beam physics: specify epsilon_n, beam_size, etc.

        Parameters:
            mean: distribution center
            std_dev: [σx(mm), σx'(mrad), σy(mm), σy'(mrad), σΔt/T(10^-3), σδW/W(10^-3)]
            num_particles: particle count

            Beam physics interface (alternative to std_dev):
            energy: beam kinetic energy [MeV]
            epsilon_n: normalized emittance [π·mm·mrad], scalar or (εx, εy)
            beam_size: RMS beam size [mm], scalar or (σx, σy)
            bunch_length: RMS bunch length [ps]
            energy_spread: RMS energy spread [%]
            energy_chirp: linac chirp [1/s]

        Returns:
            Particle array (N, 6) in FELsim coordinates

        Example:
            # Beam physics specification
            particles = sim.generate_6d_gaussian(
                epsilon_n=8.0, beam_size=0.8, bunch_length=2.0,
                energy_spread=0.5, num_particles=10000
            )
        """
        if std_dev is None and epsilon_n is None:
            raise ValueError("Specify either std_dev or beam physics parameters")

        if std_dev is not None:
            std_dev = np.array(std_dev)
            if std_dev.shape != (6,):
                raise ValueError(f"std_dev must be (6,), got {std_dev.shape}")

            particles = np.random.normal(mean, std_dev, size=(num_particles, 6))

            if energy_chirp != 0:
                T_RF = 1.0 / self.f
                tof_dist = particles[:, 4] / self.f
                particles[:, 5] += energy_chirp * tof_dist

        else:
            # Beam physics calculation
            if energy is None:
                energy = self.KE

            E0 = self.E0
            gamma, beta = PhysicalConstants.relativistic_parameters(energy, E0)
            norm_factor = beta * gamma

            # Parse emittance (scalar or tuple)
            if np.isscalar(epsilon_n):
                epsilon_n_x = epsilon_n_y = epsilon_n
            else:
                epsilon_n_x, epsilon_n_y = epsilon_n

            # Parse beam size (scalar or tuple)
            if np.isscalar(beam_size):
                sigma_x = sigma_y = beam_size
            else:
                sigma_x, sigma_y = beam_size

            epsilon_x = epsilon_n_x / norm_factor
            epsilon_y = epsilon_n_y / norm_factor

            sigma_xp = epsilon_x / sigma_x
            sigma_yp = epsilon_y / sigma_y

            tof_std = bunch_length * 1e-12 * self.f * 1e3 if bunch_length else 0.0
            energy_std = energy_spread * 10 if energy_spread else 0.0

            std_dev = np.array([sigma_x, sigma_xp, sigma_y, sigma_yp, tof_std, energy_std])
            particles = np.random.normal(mean, std_dev, size=(num_particles, 6))

            if energy_chirp != 0:
                tof_dist = particles[:, 4] * 1e-3 * (1.0 / self.f)
                particles[:, 5] += energy_chirp * tof_dist * 1e3

        self.initial_particles_felsim = particles
        self.n_particles = num_particles

        if self.debug:
            self.logger.debug(f"Generated {num_particles} particles in FELsim coordinates")
            self.logger.debug(f"  RMS values: σx={np.std(particles[:, 0]):.3f} mm, "
                              f"σx'={np.std(particles[:, 1]):.3f} mrad")
            self.logger.debug(f"              σy={np.std(particles[:, 2]):.3f} mm, "
                              f"σy'={np.std(particles[:, 3]):.3f} mrad")
            self.logger.debug(f"              σΔt/T={np.std(particles[:, 4]):.3f}×10⁻³, "
                              f"σδW/W={np.std(particles[:, 5]):.3f}×10⁻³")

        return particles

    def generate_matched_beam(self, twiss_x, twiss_y, twiss_z=None,
                              num_particles=1000, energy=None):
        """Generate beam matched to Twiss parameters."""
        raise NotImplementedError("Matched beam generation not implemented")

    # -------------------------------------------------------------------------
    # Statistical analysis and Twiss calculations
    # -------------------------------------------------------------------------

    def calculate_beam_statistics(self, particles_felsim, energy=None):
        """
        Calculate statistical moments and emittances.

        Computes RMS values, geometric and normalized emittances, correlations.

        Parameters:
            particles_felsim: (N, 6) array in FELsim coordinates
            energy: beam kinetic energy [MeV]

        Returns:
            dict with keys: mean, rms, emittance_geometric, emittance_normalized,
                           centroid, correlation_xy, num_particles
        """
        if energy is None:
            energy = self.KE

        N = particles_felsim.shape[0]
        mean = np.mean(particles_felsim, axis=0)
        rms = np.std(particles_felsim, axis=0)

        x = particles_felsim[:, 0]
        xp = particles_felsim[:, 1]
        y = particles_felsim[:, 2]
        yp = particles_felsim[:, 3]

        x_c = x - mean[0]
        xp_c = xp - mean[1]
        y_c = y - mean[2]
        yp_c = yp - mean[3]

        x2 = np.mean(x_c ** 2)
        xp2 = np.mean(xp_c ** 2)
        xxp = np.mean(x_c * xp_c)

        y2 = np.mean(y_c ** 2)
        yp2 = np.mean(yp_c ** 2)
        yyp = np.mean(y_c * yp_c)

        # ε² = ⟨x²⟩⟨x'²⟩ - ⟨xx'⟩²
        emittance_x_sq = max(x2 * xp2 - xxp ** 2, 0.0)
        emittance_y_sq = max(y2 * yp2 - yyp ** 2, 0.0)

        epsilon_x = np.sqrt(emittance_x_sq)
        epsilon_y = np.sqrt(emittance_y_sq)

        gamma, beta = PhysicalConstants.relativistic_parameters(energy, self.E0)
        norm_factor = beta * gamma

        epsilon_nx = norm_factor * epsilon_x
        epsilon_ny = norm_factor * epsilon_y

        correlation_xy = np.corrcoef(x_c, y_c)[0, 1]

        results = {
            'mean': mean,
            'rms': rms,
            'emittance_geometric': {'x': epsilon_x, 'y': epsilon_y},
            'emittance_normalized': {'x': epsilon_nx, 'y': epsilon_ny},
            'centroid': {'x': mean[0], 'xp': mean[1], 'y': mean[2], 'yp': mean[3]},
            'correlation_xy': correlation_xy,
            'num_particles': N
        }

        if self.debug:
            self.logger.debug(f"\n{'=' * 70}")
            self.logger.debug(f"Beam Statistics")
            self.logger.debug(f"{'=' * 70}")
            self.logger.debug(f"Number of particles: {N}")
            self.logger.debug(f"\nCentroid (mean values):")
            self.logger.debug(f"  x  = {mean[0]:9.3f} mm")
            self.logger.debug(f"  x' = {mean[1]:9.3f} mrad")
            self.logger.debug(f"  y  = {mean[2]:9.3f} mm")
            self.logger.debug(f"  y' = {mean[3]:9.3f} mrad")
            self.logger.debug(f"\nRMS values:")
            self.logger.debug(f"  σx     = {rms[0]:9.3f} mm")
            self.logger.debug(f"  σx'    = {rms[1]:9.3f} mrad")
            self.logger.debug(f"  σy     = {rms[2]:9.3f} mm")
            self.logger.debug(f"  σy'    = {rms[3]:9.3f} mrad")
            self.logger.debug(f"  σΔt/T  = {rms[4]:9.3f} ×10⁻³")
            self.logger.debug(f"  σδW/W  = {rms[5]:9.3f} ×10⁻³")
            self.logger.debug(f"\nEmittances (geometric):")
            self.logger.debug(f"  εx = {epsilon_x:.3f} π·mm·mrad")
            self.logger.debug(f"  εy = {epsilon_y:.3f} π·mm·mrad")
            self.logger.debug(f"\nEmittances (normalized at {energy:.1f} MeV, βγ={norm_factor:.3f}):")
            self.logger.debug(f"  εnx = {epsilon_nx:.3f} π·mm·mrad")
            self.logger.debug(f"  εny = {epsilon_ny:.3f} π·mm·mrad")
            self.logger.debug(f"\nCorrelation coefficient (x-y): {correlation_xy:.6f}")
            self.logger.debug(f"{'=' * 70}\n")

        return results

    def calculate_twiss_from_particles(self, particles_felsim, energy=None, plane='both'):
        """
        Calculate Twiss parameters from second moments.

        Uses standard formulas:
            ε² = ⟨x²⟩⟨x'²⟩ - ⟨xx'⟩²
            β = ⟨x²⟩/ε,  α = -⟨xx'⟩/ε,  γ = ⟨x'²⟩/ε
        with βγ - α² = 1

        Parameters:
            particles_felsim: (N, 6) array
            energy: beam kinetic energy [MeV]
            plane: 'x', 'y', or 'both'

        Returns:
            dict with Twiss parameters: beta [m], alpha, gamma [1/m],
            emittance, emittance_normalized [π·mm·mrad]
        """
        if energy is None:
            energy = self.KE

        if plane not in ['x', 'y', 'both']:
            raise ValueError(f"plane must be 'x', 'y', or 'both'")

        mean = np.mean(particles_felsim, axis=0)
        gamma, beta_rel = PhysicalConstants.relativistic_parameters(energy, self.E0)
        norm_factor = beta_rel * gamma

        results = {}

        def calc_plane_twiss(particles, coord_idx, name):
            pos = particles[:, coord_idx]
            ang = particles[:, coord_idx + 1]

            pos_c = pos - mean[coord_idx]
            ang_c = ang - mean[coord_idx + 1]

            pos2 = np.mean(pos_c ** 2)
            ang2 = np.mean(ang_c ** 2)
            pos_ang = np.mean(pos_c * ang_c)

            emittance_sq = pos2 * ang2 - pos_ang ** 2

            if emittance_sq < 0:
                if emittance_sq < -1e-10:
                    warnings.warn(
                        f"Negative emittance² detected for {name}-plane: {emittance_sq:.3e}. "
                        f"This indicates numerical precision issues in the particle distribution. "
                        f"Setting to small positive value."
                    )
                emittance_sq = max(emittance_sq, 1e-15)

            epsilon = np.sqrt(emittance_sq)
            epsilon_n = norm_factor * epsilon

            # Convert units: mm → m, mrad → rad for Twiss parameters
            pos2_m = pos2 * 1e-6
            ang2_rad = ang2 * 1e-6
            pos_ang_m_rad = pos_ang * 1e-6
            epsilon_m_rad = epsilon * 1e-6

            beta = pos2_m / epsilon_m_rad
            gamma_twiss = ang2_rad / epsilon_m_rad
            alpha = -pos_ang_m_rad / epsilon_m_rad

            twiss_relation = beta * gamma_twiss - alpha ** 2

            plane_results = {
                'beta': beta,
                'alpha': alpha,
                'gamma': gamma_twiss,
                'emittance': epsilon,
                'emittance_normalized': epsilon_n,
                'twiss_relation_check': twiss_relation
            }

            if self.debug:
                self.logger.debug(f"\n{name.upper()}-plane Twiss parameters:")
                self.logger.debug(f"  β = {beta:.6f} m")
                self.logger.debug(f"  α = {alpha:.6f}")
                self.logger.debug(f"  γ = {gamma_twiss:.6f} m⁻¹")
                self.logger.debug(f"  ε = {epsilon:.6f} π·mm·mrad")
                self.logger.debug(f"  εn = {epsilon_n:.6f} π·mm·mrad")
                self.logger.debug(f"  Twiss relation check (βγ - α²): {twiss_relation:.9f} "
                                  f"(should be 1.0)")

                if abs(twiss_relation - 1.0) > 1e-6:
                    self.logger.warning(f"  Twiss relation deviates from 1.0 by "
                                        f"{abs(twiss_relation - 1.0):.3e}")

            return plane_results

        if plane in ['x', 'both']:
            results['x'] = calc_plane_twiss(particles_felsim, 0, 'x')

        if plane in ['y', 'both']:
            results['y'] = calc_plane_twiss(particles_felsim, 2, 'y')

        if plane != 'both':
            return results[plane]

        if self.debug and plane == 'both':
            self.logger.debug(f"\n{'=' * 70}")
            self.logger.debug(f"Statistical Twiss Parameters (at {energy:.1f} MeV)")
            self.logger.debug(f"{'=' * 70}\n")

        return results

    def calculate_twiss_evolution(self, checkpoint_indices=None, transform_to_felsim=True):
        """
        Calculate Twiss parameters at multiple beamline locations.

        Reads checkpoint files and computes statistical Twiss at each location.

        Parameters:
            checkpoint_indices: element indices for checkpoints (uses self.particle_save_points if None)
            transform_to_felsim: whether to transform from COSY before calculation

        Returns:
            dict mapping element_idx to Twiss parameters and metadata
        """
        if checkpoint_indices is None:
            if not self.particle_save_points:
                raise ValueError("No checkpoint indices specified and particle_save_points is empty")
            checkpoint_indices = self.particle_save_points

        if self.debug:
            self.logger.debug(f"\nCalculating Twiss evolution at {len(checkpoint_indices)} locations...")
            self.logger.debug(f"Checkpoint indices: {checkpoint_indices}")

        evolution = {}

        for elem_idx in checkpoint_indices:
            try:
                particles_cosy = self.read_particle_file(elem_idx, format='auto', output_dir='results')

                particles = self.transform_from_cosy_coordinates(
                    particles_cosy) if transform_to_felsim else particles_cosy

                twiss = self.calculate_twiss_from_particles(particles, plane='both')
                twiss['num_particles'] = particles.shape[0]
                twiss['s_position'] = None

                evolution[elem_idx] = twiss

                self.logger.debug(f"  Element {elem_idx}: βx={twiss['x']['beta']:.3f} m, "
                                  f"βy={twiss['y']['beta']:.3f} m")

            except FileNotFoundError:
                self.logger.warning(f"Checkpoint file for element {elem_idx} not found")
            except Exception as e:
                self.logger.error(f"Error processing element {elem_idx}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        self.logger.debug(f"\nSuccessfully calculated Twiss at {len(evolution)}/{len(checkpoint_indices)} locations")

        return evolution

    def compare_twiss_methods(self, element_idx=None):
        """Compare transfer-map vs statistical Twiss parameters."""
        raise NotImplementedError(
            "Transfer-map/statistical Twiss comparison not yet implemented"
        )

    def diagnose_particle_distribution(self, particles, coordinate_system='felsim'):
        """
        Check particle distribution for NaN, Inf, and unphysical values.

        Parameters:
            particles: (N, 6) array to diagnose
            coordinate_system: 'felsim' or 'cosy'

        Returns:
            dict with diagnostic information
        """
        N = particles.shape[0]

        has_nan = np.isnan(particles).any(axis=1)
        has_inf = np.isinf(particles).any(axis=1)

        n_nan = np.sum(has_nan)
        n_inf = np.sum(has_inf)
        n_valid = N - n_nan - n_inf

        print(f"\n{'=' * 70}")
        print(f"Particle Distribution Diagnostics ({coordinate_system.upper()} coordinates)")
        print(f"{'=' * 70}")
        print(f"Total particles: {N}")
        print(f"Valid particles: {n_valid} ({100 * n_valid / N:.1f}%)")
        print(f"Particles with NaN: {n_nan} ({100 * n_nan / N:.1f}%)")
        print(f"Particles with Inf: {n_inf} ({100 * n_inf / N:.1f}%)")

        if n_valid > 0:
            valid = particles[~(has_nan | has_inf)]

            coord_names = {
                'felsim': ['x(mm)', "x'(mrad)", 'y(mm)', "y'(mrad)", 'Δt/T(10^-3)', 'δW/W(10^-3)'],
                'cosy': ['x(m)', 'a', 'y(m)', 'b', 'l(m)', 'δK']
            }
            names = coord_names.get(coordinate_system, [f'coord_{i}' for i in range(6)])

            print(f"\nValid particle statistics:")
            print(f"{'Coordinate':<15} {'Min':>12} {'Max':>12} {'Mean':>12} {'RMS':>12}")
            print(f"{'-' * 70}")

            for i, name in enumerate(names):
                vals = valid[:, i]
                print(f"{name:<15} {np.min(vals):>12.6g} {np.max(vals):>12.6g} "
                      f"{np.mean(vals):>12.6g} {np.std(vals):>12.6g}")

            if coordinate_system == 'cosy':
                a = valid[:, 1]
                b = valid[:, 3]
                delta_K = valid[:, 5]

                a2_plus_b2 = a ** 2 + b ** 2
                n_large = np.sum(a2_plus_b2 > 0.1)
                n_very_large = np.sum(a2_plus_b2 > 1.0)
                n_neg_KE = np.sum(delta_K < -1.0)

                print(f"\nCOSY-specific checks:")
                print(f"  Particles with a² + b² > 0.1: {n_large} ({100 * n_large / n_valid:.1f}%)")
                print(f"  Particles with a² + b² > 1.0: {n_very_large} ({100 * n_very_large / n_valid:.1f}%)")
                print(f"  Particles with δK < -1 (negative KE): {n_neg_KE} "
                      f"({100 * n_neg_KE / n_valid:.1f}%)")

                if n_very_large > 0:
                    print(f"  WARNING: {n_very_large} particles have unphysical momentum ratios!")
                    print(f"           These may cause NaN in coordinate transformation.")
        else:
            print("\nWARNING: No valid particles found!")

        print(f"{'=' * 70}\n")

        return {
            'total': N,
            'valid': n_valid,
            'nan': n_nan,
            'inf': n_inf,
            'valid_fraction': n_valid / N if N > 0 else 0
        }

    def analyze_particle_distribution(self, particles_felsim,
                                      include_correlations=True,
                                      include_twiss=True,
                                      energy=None):
        """
        Full beam characterization: statistics, Twiss, correlations.

        Parameters:
            particles_felsim: (N, 6) array
            include_correlations: compute 6×6 correlation matrix
            include_twiss: compute Twiss parameters
            energy: beam kinetic energy [MeV]

        Returns:
            dict containing statistics, twiss (optional), correlation_matrix (optional)
        """
        if energy is None:
            energy = self.KE

        results = {
            'energy': energy,
            'num_particles': particles_felsim.shape[0]
        }

        stats = self.calculate_beam_statistics(particles_felsim, energy=energy)
        results['statistics'] = stats

        if include_twiss:
            twiss = self.calculate_twiss_from_particles(particles_felsim, energy=energy, plane='both')
            results['twiss'] = twiss

        if include_correlations:
            mean = np.mean(particles_felsim, axis=0)
            particles_c = particles_felsim - mean
            corr = np.corrcoef(particles_c.T)
            results['correlation_matrix'] = corr

            if self.debug:
                self.logger.debug(f"\n{'=' * 70}")
                self.logger.debug(f"Correlation Matrix")
                self.logger.debug(f"{'=' * 70}")
                self.logger.debug("        x      x'      y      y'    Δt/T   δW/W")
                coord_names = ['x   ', "x'  ", 'y   ', "y'  ", 'Δt/T', 'δW/W']
                for i, name in enumerate(coord_names):
                    row_str = f"{name}  " + "".join(f"{corr[i, j]:6.3f} " for j in range(6))
                    self.logger.debug(row_str)
                self.logger.debug(f"{'=' * 70}\n")

        if self.debug:
            self.logger.debug(f"\nParticle Distribution Analysis Complete")
            self.logger.debug(f"  Particles: {results['num_particles']}")
            self.logger.debug(f"  Energy: {energy:.3f} MeV")
            if include_twiss:
                self.logger.debug(f"  βx = {results['twiss']['x']['beta']:.3f} m")
                self.logger.debug(f"  βy = {results['twiss']['y']['beta']:.3f} m")
                self.logger.debug(f"  εx = {results['twiss']['x']['emittance']:.3f} π·mm·mrad")
                self.logger.debug(f"  εy = {results['twiss']['y']['emittance']:.3f} π·mm·mrad")

        return results

    def load_particles_from_file(self, filepath, format='felsim'):
        """Load particle distribution from file."""
        raise NotImplementedError("File loading not yet implemented")

    # -------------------------------------------------------------------------
    # Coordinate transformations
    # -------------------------------------------------------------------------

    def transform_to_cosy_coordinates(self, particles_felsim, energy=None):
        """
        Transform FELsim → COSY coordinates.

        Transformation details:
        - Position: mm → m (×1e-3)
        - Angles → momentum ratios: exact 3D momentum decomposition
          tan(θ) = p_perp/p_z, normalized by p0
        - Longitudinal: ΔToF/T_RF → l = -(t-t0)*v0*γ/(1+γ)
        - Energy: δW/W → δK = (KE-KE0)/KE0

        See Dragt et al., "Lie Methods for Nonlinear Dynamics with Applications
        to Accelerator Physics" for coordinate system details.

        Parameters:
            particles_felsim: (N, 6) array [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF, δW/W]
            energy: beam kinetic energy [MeV]

        Returns:
            (N, 6) array [x(m), a, y(m), b, l(m), δK] in COSY coordinates
        """
        if energy is None:
            energy = self.KE

        E0 = self.E0
        C = self.C

        KE0 = energy
        gamma = 1 + (KE0 / E0)
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)

        N = particles_felsim.shape[0]
        particles_cosy = np.zeros_like(particles_felsim)

        x_mm = particles_felsim[:, 0]
        xp_mrad = particles_felsim[:, 1]
        y_mm = particles_felsim[:, 2]
        yp_mrad = particles_felsim[:, 3]
        z_tof = particles_felsim[:, 4]
        zp_energy = particles_felsim[:, 5]

        # Transverse coordinates
        particles_cosy[:, 0] = x_mm * 1e-3
        particles_cosy[:, 2] = y_mm * 1e-3

        # Exact angle → momentum ratio transformation
        xp_rad = xp_mrad * 1e-3
        yp_rad = yp_mrad * 1e-3

        tan_xp = np.tan(xp_rad)
        tan_yp = np.tan(yp_rad)

        KE_particle = KE0 + zp_energy * (KE0 + E0) / 1000
        pc = np.sqrt(KE_particle ** 2 + 2 * KE_particle * E0)

        # 3D momentum: p_i = p * tan(θ_i) / sqrt(1 + tan²(θ_x) + tan²(θ_y))
        denom = np.sqrt(1 + tan_xp ** 2 + tan_yp ** 2)
        px = pc * tan_xp / denom
        py = pc * tan_yp / denom

        particles_cosy[:, 1] = px / p0c
        particles_cosy[:, 3] = py / p0c

        # Longitudinal coordinates
        T_RF = 1.0 / self.f
        DeltaToF = z_tof * 1e-3 * T_RF

        beta0 = p0c / (gamma * E0)
        v0 = beta0 * C

        particles_cosy[:, 4] = -DeltaToF * v0 * gamma / (1 + gamma)
        particles_cosy[:, 5] = (KE_particle - KE0) / KE0

        # Validation: a² + b² must be < k²
        k = pc / p0c
        k_sq = k ** 2
        a_sq_plus_b_sq = particles_cosy[:, 1] ** 2 + particles_cosy[:, 3] ** 2
        violation = a_sq_plus_b_sq - k_sq
        non_compliant_mask = violation > 1e-12
        num_non_compliant = np.sum(non_compliant_mask)

        if num_non_compliant > 0:
            self.logger.error(f"{num_non_compliant}/{N} particles violate a² + b² ≤ k²")
            self.logger.error(f"This should never happen - indicates a bug in the transformation!")

            if self.debug:
                # Detailed diagnostics for troubleshooting
                non_comp_idx = np.where(non_compliant_mask)[0]
                num_to_show = min(5, num_non_compliant)

                self.logger.debug(f"\nShowing first {num_to_show} non-compliant particles:")
                for idx in non_comp_idx[:num_to_show]:
                    self.logger.debug(f"\n  Particle {idx}:")
                    self.logger.debug(f"    FELsim coordinates:")
                    self.logger.debug(f"      x'  = {xp_mrad[idx]:12.6f} mrad")
                    self.logger.debug(f"      y'  = {yp_mrad[idx]:12.6f} mrad")
                    self.logger.debug(f"      δW/W = {zp_energy[idx]:12.6f} ×10⁻³")
                    self.logger.debug(f"    Intermediate calculations:")
                    self.logger.debug(f"      tan(x') = {tan_xp[idx]:12.6e}")
                    self.logger.debug(f"      tan(y') = {tan_yp[idx]:12.6e}")
                    self.logger.debug(f"      KE      = {KE_particle[idx]:12.6f} MeV")
                    self.logger.debug(f"      pc      = {pc[idx]:12.6f} MeV/c")
                    self.logger.debug(f"      k       = {k[idx]:12.6f}")
                    self.logger.debug(f"    COSY coordinates:")
                    self.logger.debug(f"      a       = {particles_cosy[idx, 1]:12.6e}")
                    self.logger.debug(f"      b       = {particles_cosy[idx, 3]:12.6e}")
                    self.logger.debug(f"    Validation check:")
                    self.logger.debug(f"      a² + b² = {a_sq_plus_b_sq[idx]:12.6e}")
                    self.logger.debug(f"      k²      = {k_sq[idx]:12.6e}")
                    self.logger.debug(f"      violation = {violation[idx]:12.6e}")

                if num_non_compliant > num_to_show:
                    self.logger.debug(f"\n  ... and {num_non_compliant - num_to_show} more")

                # Overall statistics for non-compliant particles
                self.logger.debug(f"\n  Statistics for all {num_non_compliant} non-compliant particles:")
                self.logger.debug(f"    KE range: [{np.min(KE_particle[non_compliant_mask]):.6f}, "
                                  f"{np.max(KE_particle[non_compliant_mask]):.6f}] MeV")
                self.logger.debug(f"    k range:  [{np.min(k[non_compliant_mask]):.6f}, "
                                  f"{np.max(k[non_compliant_mask]):.6f}]")
                self.logger.debug(f"    Max violation: {np.max(violation[non_compliant_mask]):.6e}")

                # Check for negative KE
                neg_KE_mask = KE_particle < 0
                if np.any(neg_KE_mask):
                    n_neg = np.sum(neg_KE_mask)
                    self.logger.warning(f"    {n_neg} particles have negative KE!")
                    self.logger.warning(f"    This occurs when δW/W < "
                                        f"{-KE0 * 1000 / (KE0 + E0):.1f}×10⁻³")

        if self.debug:
            self.logger.debug(f"\n{'=' * 70}")
            self.logger.debug(f"FELsim → COSY Coordinate Transformation")
            self.logger.debug(f"{'=' * 70}")
            self.logger.debug(f"Particles transformed: {N}")
            self.logger.debug(f"\nReference particle parameters:")
            self.logger.debug(f"  KE₀ = {KE0:.4f} MeV")
            self.logger.debug(f"  E₀ (rest energy) = {E0:.6f} MeV")
            self.logger.debug(f"  γ₀  = {gamma:.6f}")
            self.logger.debug(f"  β₀  = {beta0:.6f}")
            self.logger.debug(f"  p₀c = {p0c:.4f} MeV/c")
            self.logger.debug(f"  v₀  = {v0:.6e} m/s")
            self.logger.debug(f"  T_RF = {T_RF:.6e} s (f = {self.f:.3e} Hz)")
            self.logger.debug(f"\nTransformation method:")
            self.logger.debug(f"  Angles: EXACT (using tan(x'), tan(y'))")
            self.logger.debug(f"  Geometry: Full 3D momentum decomposition")
            self.logger.debug(f"  Longitudinal factor: γ/(1+γ) = {gamma / (1 + gamma):.6f}")
            self.logger.debug(f"\nOutput RMS values (COSY coordinates):")
            self.logger.debug(f"  x  = {np.std(particles_cosy[:, 0]):9.6e} m")
            self.logger.debug(f"  a  = {np.std(particles_cosy[:, 1]):9.6e}")
            self.logger.debug(f"  y  = {np.std(particles_cosy[:, 2]):9.6e} m")
            self.logger.debug(f"  b  = {np.std(particles_cosy[:, 3]):9.6e}")
            self.logger.debug(f"  l  = {np.std(particles_cosy[:, 4]):9.6e} m")
            self.logger.debug(f"  δK = {np.std(particles_cosy[:, 5]):9.6e}")
            self.logger.debug(f"{'=' * 70}\n")

        return particles_cosy

    def transform_from_cosy_coordinates(self, particles_cosy, energy=None,
                                        validate=True, filter_invalid=False):
        """
        Transform COSY → FELsim coordinates.

        Inverse of transform_to_cosy_coordinates().

        Momentum ratios → angles via:
            tan(θ) = a / sqrt(k² - a² - b²)
        where k = p/p0 is calculated from δK.

        Parameters:
            particles_cosy: (N, 6) array [x(m), a, y(m), b, l(m), δK]
            energy: beam kinetic energy [MeV]
            validate: check for physically invalid particles
            filter_invalid: remove invalid particles instead of raising error

        Returns:
            (N, 6) array in FELsim coordinates

        Raises:
            ValueError if validation fails and filter_invalid=False
        """
        if particles_cosy is None:
            raise ValueError("particles_cosy is None - COSY may have failed to track particles")

        if not isinstance(particles_cosy, np.ndarray):
            raise TypeError(f"particles_cosy must be numpy array, got {type(particles_cosy)}")

        if particles_cosy.size == 0:
            raise ValueError("particles_cosy is empty - no particles to transform")

        if energy is None:
            energy = self.KE

        E0 = self.E0
        C = self.C

        KE0 = energy
        gamma = 1 + (KE0 / E0)
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)

        N = particles_cosy.shape[0]
        particles_felsim = np.zeros_like(particles_cosy)

        with np.errstate(over='ignore', invalid='ignore'):
            x_m = particles_cosy[:, 0]
            a_norm = particles_cosy[:, 1]
            y_m = particles_cosy[:, 2]
            b_norm = particles_cosy[:, 3]
            l_m = particles_cosy[:, 4]
            delta_K = particles_cosy[:, 5]

            # Check for negative kinetic energy
            neg_KE_mask = delta_K < -1.0
            n_neg_KE = np.sum(neg_KE_mask)

            if n_neg_KE > 0 and validate:
                raise ValueError(
                    f"Invalid particles: {n_neg_KE} with negative KE (δK < -1)\n"
                    f"This indicates unphysical particle energies from COSY.\n"
                    f"Use validate=False, filter_invalid=True to remove these particles."
                )
            elif n_neg_KE > 0:
                self.logger.warning(f"{n_neg_KE} particles with negative KE will be filtered")

            # Transverse
            particles_felsim[:, 0] = x_m * 1e3
            particles_felsim[:, 2] = y_m * 1e3

            KE_particle = KE0 * (1 + delta_K)
            momentum_sq = KE_particle ** 2 + 2 * KE_particle * E0
            momentum_sq = np.where(momentum_sq >= 0, momentum_sq, np.nan)
            pc = np.sqrt(momentum_sq)
            k = pc / p0c

            # Inverse transformation: tan(θ) = a / sqrt(k² - a² - b²)
            k_sq = k ** 2
            a_sq = a_norm ** 2
            b_sq = b_norm ** 2
            discriminant = k_sq - a_sq - b_sq

            if validate and np.any(discriminant < 0):
                n_invalid = np.sum(discriminant < 0)
                max_viol = np.min(discriminant)

                invalid_mask = discriminant < 0

                # Enhanced error message with diagnostic information
                error_msg = (
                    f"Invalid momentum ratios detected: {n_invalid}/{N} particles have a² + b² > k²\n"
                    f"Maximum violation: k² - a² - b² = {max_viol:.6e}\n\n"
                    f"This indicates COSY has produced unphysical particle coordinates.\n"
                    f"Possible causes:\n"
                    f"  1. Particles lost to apertures (should have been removed by COSY)\n"
                    f"  2. Numerical errors in COSY tracking\n"
                    f"  3. Coordinate system mismatch between COSY and this code\n"
                    f"  4. Extremely large momentum deviations\n\n"
                    f"Statistics for invalid particles:\n"
                    f"  a:  mean={np.mean(a_norm[invalid_mask]):.6e}, "
                    f"std={np.std(a_norm[invalid_mask]):.6e}\n"
                    f"  b:  mean={np.mean(b_norm[invalid_mask]):.6e}, "
                    f"std={np.std(b_norm[invalid_mask]):.6e}\n"
                    f"  δK: mean={np.mean(delta_K[invalid_mask]):.6e}, "
                    f"std={np.std(delta_K[invalid_mask]):.6e}\n"
                    f"  k:  mean={np.mean(k[invalid_mask]):.6e}, "
                    f"std={np.std(k[invalid_mask]):.6e}\n\n"
                    f"Solutions:\n"
                    f"  - Use filter_invalid=True to automatically remove invalid particles\n"
                    f"  - Check COSY tracking configuration (step size, apertures)\n"
                    f"  - Verify element parameters in Excel file\n"
                    f"  - Enable COSY's particle loss tracking"
                )

                if self.debug:
                    # Additional detailed diagnostics
                    self.logger.debug(f"\nDetailed diagnostics for invalid particles:")
                    invalid_idx = np.where(invalid_mask)[0]
                    num_to_show = min(5, n_invalid)

                    for idx in invalid_idx[:num_to_show]:
                        self.logger.debug(f"\n  Particle {idx}:")
                        self.logger.debug(f"    COSY coordinates:")
                        self.logger.debug(f"      a       = {a_norm[idx]:12.6e}")
                        self.logger.debug(f"      b       = {b_norm[idx]:12.6e}")
                        self.logger.debug(f"      δK      = {delta_K[idx]:12.6e}")
                        self.logger.debug(f"    Derived quantities:")
                        self.logger.debug(f"      KE      = {KE_particle[idx]:12.6f} MeV")
                        self.logger.debug(f"      pc      = {pc[idx]:12.6f} MeV/c")
                        self.logger.debug(f"      k       = {k[idx]:12.6f}")
                        self.logger.debug(f"    Validation:")
                        self.logger.debug(f"      a² + b² = {(a_sq + b_sq)[idx]:12.6e}")
                        self.logger.debug(f"      k²      = {k_sq[idx]:12.6e}")
                        self.logger.debug(f"      k² - a² - b² = {discriminant[idx]:12.6e}")

                    if n_invalid > num_to_show:
                        self.logger.debug(f"\n  ... and {n_invalid - num_to_show} more")

                if not filter_invalid:
                    raise ValueError(error_msg)
                else:
                    self.logger.warning(f"\n{error_msg}\n\nProceeding with filter_invalid=True")

            denom = np.sqrt(discriminant)
            tan_xp = a_norm / denom
            tan_yp = b_norm / denom

            xp_rad = np.arctan(tan_xp)
            yp_rad = np.arctan(tan_yp)

            particles_felsim[:, 1] = xp_rad * 1e3
            particles_felsim[:, 3] = yp_rad * 1e3

            # Longitudinal
            T_RF = 1.0 / self.f
            beta0 = p0c / (gamma * E0)
            v0 = beta0 * C

            DeltaToF = -l_m * (1 + gamma) / (v0 * gamma)
            particles_felsim[:, 4] = (DeltaToF / T_RF) * 1e3
            particles_felsim[:, 5] = KE0 * delta_K * 1000 / (KE0 + E0)

        if self.debug:
            with np.errstate(over='ignore', invalid='ignore'):
                self.logger.debug(f"\n{'=' * 70}")
                self.logger.debug(f"COSY → FELsim Coordinate Transformation")
                self.logger.debug(f"{'=' * 70}")
                self.logger.debug(f"Particles transformed: {N}")
                self.logger.debug(f"\nReference particle parameters:")
                self.logger.debug(f"  KE₀ = {KE0:.4f} MeV")
                self.logger.debug(f"  γ₀  = {gamma:.6f}")
                self.logger.debug(f"  β₀  = {beta0:.6f}")
                self.logger.debug(f"  p₀c = {p0c:.4f} MeV/c")
                self.logger.debug(f"\nTransformation method:")
                self.logger.debug(f"  Angles: EXACT (using arctan)")
                self.logger.debug(f"\nOutput RMS values (FELsim coordinates):")
                self.logger.debug(f"  x     = {np.std(particles_felsim[:, 0]):9.3f} mm")
                self.logger.debug(f"  x'    = {np.std(particles_felsim[:, 1]):9.3f} mrad")
                self.logger.debug(f"  y     = {np.std(particles_felsim[:, 2]):9.3f} mm")
                self.logger.debug(f"  y'    = {np.std(particles_felsim[:, 3]):9.3f} mrad")
                self.logger.debug(f"  Δt/T  = {np.std(particles_felsim[:, 4]):9.3f} ×10⁻³")
                self.logger.debug(f"  δW/W  = {np.std(particles_felsim[:, 5]):9.3f} ×10⁻³")
                self.logger.debug(f"{'=' * 70}\n")

        if filter_invalid:
            valid_mask = np.isfinite(particles_felsim).all(axis=1)
            n_removed = N - np.sum(valid_mask)
            particles_felsim = particles_felsim[valid_mask]

            if n_removed > 0:
                self.logger.warning(f"Filtered out {n_removed} invalid particles, "
                                    f"{particles_felsim.shape[0]} remain")

        return particles_felsim

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def write_particle_file(self, particles_cosy, filename=None,
                            format='sray', output_dir='results'):
        """
        Write particles to file for COSY input.

        Formats:
            'ascii': Simple 6-column format (for FILE2VE)
            'ascii_simple': Particle count header + data
            'rray', 'sray': Native COSY format (8 coords: X, A, Y, B, T, D, G, Z)

        Parameters:
            particles_cosy: (N, 6) array [x, a, y, b, l, δK]
            filename: output filename
            format: file format
            output_dir: output directory

        Returns:
            Full path to written file
        """
        if filename is None:
            filename = self.particle_input_file

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        n_particles = particles_cosy.shape[0]

        if format == 'ascii':
            with open(filepath, 'w') as f:
                for p in particles_cosy:
                    f.write(" ".join(f"{x:.12e}" for x in p) + "\n")

        elif format == 'ascii_simple':
            with open(filepath, 'w') as f:
                f.write(f"{n_particles}\n")
                for p in particles_cosy:
                    f.write(" ".join(f"{x:.12e}" for x in p) + "\n")

        elif format in ['rray', 'sray']:
            # COSY native format: 8 coordinates (our 6 + G and Z set to zero)
            with open(filepath, 'w') as f:
                f.write(f"# number of rays: {n_particles:4d}    (including the reference)\n")

                coord_labels = ['X', 'A', 'Y', 'B', 'T', 'D', 'G', 'Z']

                for coord_idx, label in enumerate(coord_labels):
                    f.write(f"# {label}\n")

                    for p in particles_cosy:
                        value = p[coord_idx] if coord_idx < 6 else 0.0
                        f.write(f" {value:23.16E}\n")

                    f.write("\n")

        elif format == 'binary':
            raise NotImplementedError("Binary format not implemented")

        else:
            raise ValueError(f"Unknown format: {format}")

        if self.debug:
            self.logger.debug(f"Wrote {n_particles} particles to {filepath}")
            self.logger.debug(f"  Format: {format}")
            self.logger.debug(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
            if format in ['rray', 'sray']:
                self.logger.debug(f"  Coordinates: X, A, Y, B, T, D, G, Z (G and Z set to zero)")

        return filepath

    def read_particle_file(self, filename=None, format='auto', output_dir='results'):
        """
        Read particle distribution from COSY output.

        Parameters:
            filename: filename or element index (int N → 'fort.{base+N}')
            format: 'auto', 'ascii', 'ascii_simple', 'rray', 'sray'
            output_dir: directory containing file

        Returns:
            (N, 6) array in COSY coordinates [x, a, y, b, l, δK]

        Examples:
            particles = sim.read_particle_file(5)  # Element 5 checkpoint
            particles = sim.read_particle_file('fort.10001')
        """
        if isinstance(filename, int):
            element_idx = filename
            base_unit = getattr(self, 'particle_checkpoint_base_unit', 10000)
            filename = f'fort.{base_unit + element_idx}'
            self.logger.debug(f"Element index {element_idx} → filename '{filename}'")

        elif filename is None:
            filename = self.particle_output_file

        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Particle file not found: {filepath}")

        # Auto-detect format
        if format == 'auto':
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()

                if first_line.startswith('# number of rays'):
                    format = 'rray'
                    self.logger.debug("Auto-detected format: RRAY/SRAY")
                elif first_line.startswith('#'):
                    format = 'ascii'
                    self.logger.debug("Auto-detected format: ASCII (with comments)")
                else:
                    try:
                        int(first_line)
                        format = 'ascii_simple'
                        self.logger.debug("Auto-detected format: ASCII simple (with count)")
                    except ValueError:
                        try:
                            float(first_line.split()[0])
                            format = 'ascii'
                            self.logger.debug("Auto-detected format: ASCII (raw data)")
                        except:
                            raise ValueError(f"Cannot determine file format from: {first_line}")

        # Read file
        if format in ['rray', 'sray']:
            particles = self._read_rray_format(filepath)
        elif format == 'ascii_simple':
            particles = np.loadtxt(filepath, skiprows=1)
        elif format == 'ascii':
            particles = np.loadtxt(filepath, comments='#')
        elif format == 'binary':
            raise NotImplementedError("Binary format not implemented")
        else:
            raise ValueError(f"Unknown format: {format}")

        # Validate shape
        if particles.ndim == 1:
            if particles.shape[0] == 6:
                particles = particles.reshape(1, 6)
            else:
                raise ValueError(f"Expected 6 coordinates, got {particles.shape[0]}")
        elif particles.shape[1] != 6:
            raise ValueError(f"Expected 6 columns, got {particles.shape[1]}")

        if self.debug:
            with np.errstate(over='ignore', invalid='ignore'):
                self.logger.debug(f"Read {particles.shape[0]} particles from {filepath}")
                self.logger.debug(f"  Format: {format}")
                self.logger.debug(f"  RMS values: x={np.std(particles[:, 0]):.6e} m, "
                                  f"a={np.std(particles[:, 1]):.6e}")
                self.logger.debug(f"              y={np.std(particles[:, 2]):.6e} m, "
                                  f"b={np.std(particles[:, 3]):.6e}")
                self.logger.debug(f"              l={np.std(particles[:, 4]):.6e} m, "
                                  f"δK={np.std(particles[:, 5]):.6e}")

        return particles

    def read_checkpoints(self, element_indices, output_dir='results',
                         transform_to_felsim=True, validate=True, filter_invalid=False):
        """
        Read multiple checkpoint files at once.

        Parameters:
            element_indices: list of element indices
            output_dir: directory with checkpoint files
            transform_to_felsim: convert to FELsim coordinates
            validate: check for invalid particles during transformation
            filter_invalid: remove invalid particles instead of raising error

        Returns:
            dict mapping element_idx → particle array
        """
        checkpoints = {}

        for elem_idx in element_indices:
            try:
                particles_cosy = self.read_particle_file(elem_idx, format='auto', output_dir=output_dir)

                if transform_to_felsim:
                    particles = self.transform_from_cosy_coordinates(
                        particles_cosy, validate=validate, filter_invalid=filter_invalid
                    )
                else:
                    particles = particles_cosy

                checkpoints[elem_idx] = particles
                self.logger.debug(f"Successfully read checkpoint for element {elem_idx}")

            except FileNotFoundError:
                self.logger.warning(f"Checkpoint file for element {elem_idx} not found")
            except Exception as e:
                self.logger.error(f"Error reading element {elem_idx}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        self.logger.debug(f"Read {len(checkpoints)}/{len(element_indices)} checkpoints")

        return checkpoints

    def _read_rray_format(self, filepath):
        """
        Parse COSY RRAY/SRAY format.

        Format: header + 8 coordinate blocks (X, A, Y, B, T, D, G, Z)
        Extracts first 6 coordinates, discards G and Z.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        first_line = lines[0].strip()
        if not first_line.startswith('# number of rays'):
            raise ValueError(f"Invalid RRAY format: expected '# number of rays' header, "
                             f"got: {first_line}")

        try:
            n_rays_str = first_line.split(':')[1].split('(')[0].strip()
            n_rays = int(n_rays_str)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Cannot parse number of rays from: {first_line}") from e

        self.logger.debug(f"Reading RRAY format: {n_rays} rays")

        coord_data = []
        expected_labels = ['X', 'A', 'Y', 'B', 'T', 'D', 'G', 'Z']

        line_idx = 1
        for coord_idx, label in enumerate(expected_labels):
            if line_idx >= len(lines):
                raise ValueError(f"Unexpected end of file while reading coordinate {label}")

            label_line = lines[line_idx].strip()

            # Accept alternate labels: l for T, various delta forms for D
            if not (label_line.startswith(f'# {label}') or
                    (label == 'T' and '# l' in label_line) or
                    (label == 'D' and ('# D' in label_line or '# delta' in label_line))):
                raise ValueError(f"Expected '# {label}' header at line {line_idx + 1}, "
                                 f"got: {label_line}")

            line_idx += 1

            coord_values = []
            for ray_idx in range(n_rays):
                if line_idx >= len(lines):
                    raise ValueError(f"Unexpected end of file while reading {label}, "
                                     f"ray {ray_idx + 1}/{n_rays}")

                value_line = lines[line_idx].strip()
                try:
                    value = float(value_line)
                    coord_values.append(value)
                except ValueError as e:
                    raise ValueError(f"Cannot parse value at line {line_idx + 1}: "
                                     f"{value_line}") from e

                line_idx += 1

            coord_data.append(coord_values)
            line_idx += 1

        coord_array = np.array(coord_data)
        particles = coord_array[:6, :].T

        if self.debug:
            self.logger.debug(f"  Successfully parsed {n_rays} rays with 8 coordinates")
            self.logger.debug(f"  Extracted first 6 coordinates (X, A, Y, B, T, D)")
            self.logger.debug(f"  Discarded G and Z coordinates")

            if coord_array.shape[0] >= 8:
                g_nonzero = np.any(coord_array[6, :] != 0)
                z_nonzero = np.any(coord_array[7, :] != 0)
                if g_nonzero:
                    self.logger.warning(f"  G coordinate has non-zero values "
                                        f"(max: {np.max(np.abs(coord_array[6, :])):.3e})")
                if z_nonzero:
                    self.logger.warning(f"  Z coordinate has non-zero values "
                                        f"(max: {np.max(np.abs(coord_array[7, :])):.3e})")

        return particles

    def validate_coordinate_transformation(self, num_test_particles=1000, tolerance=1e-12):
        """
        Test round-trip transformation: FELsim → COSY → FELsim

        Returns:
            dict with validation results and max errors
        """
        test_particles = np.random.normal(
            0, [1.0, 0.1, 1.0, 0.1, 5.0, 1.0],
            size=(num_test_particles, 6)
        )

        cosy_coords = self.transform_to_cosy_coordinates(test_particles)
        felsim_recovered = self.transform_from_cosy_coordinates(cosy_coords)

        abs_errors = np.abs(test_particles - felsim_recovered)
        rel_errors = abs_errors / (np.abs(test_particles) + 1e-15)

        max_abs = np.max(abs_errors, axis=0)
        max_rel = np.max(rel_errors, axis=0)

        coord_names = ['x(mm)', "x'(mrad)", 'y(mm)', "y'(mrad)", 'Δt/T(10^-3)', 'δW/W(10^-3)']

        results = {
            'passed': np.all(max_abs < tolerance),
            'max_absolute_errors': dict(zip(coord_names, max_abs)),
            'max_relative_errors': dict(zip(coord_names, max_rel)),
            'tolerance': tolerance
        }

        if self.debug or not results['passed']:
            self.logger.debug(f"\n{'=' * 70}")
            self.logger.debug(f"Coordinate Transformation Validation")
            self.logger.debug(f"{'=' * 70}")
            self.logger.debug(f"Test particles: {num_test_particles}")
            self.logger.debug(f"Tolerance: {tolerance:.2e}")
            self.logger.debug(f"\nMaximum errors per coordinate:")
            for i, name in enumerate(coord_names):
                self.logger.debug(f"  {name:15s}: abs={max_abs[i]:.3e}, rel={max_rel[i]:.3e}")
            self.logger.debug(f"\nValidation: {'PASSED ✓' if results['passed'] else 'FAILED ✗'}")
            self.logger.debug(f"{'=' * 70}\n")

        return results