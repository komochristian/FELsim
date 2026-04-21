"""
Adapter for RF-Track particle accelerator simulation code.

RF-Track is a tracking code developed at CERN for the design and optimisation
of particle accelerators. It solves fully relativistic equations of motion
and supports space charge, RF cavities, wakefields, and electromagnetic field maps.

See: https://pypi.org/project/RF-Track/

Author: Eremey Valetov
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from simulatorBase import (
    SimulatorBase, SimulationResult, BeamlineElement,
    CoordinateSystem, SimulationMode
)
from beamEvolution import BeamEvolution, ElementInfo
from loggingConfig import get_logger_with_fallback
from physicalConstants import PhysicalConstants

# Attempt to import RF-Track
try:
    import RF_Track as rft
    _RFTRACK_AVAILABLE = True
except ImportError:
    _RFTRACK_AVAILABLE = False
    rft = None


class RFTrackAdapter(SimulatorBase):
    """
    Adapter providing unified interface to RF-Track simulator.

    Coordinate System (RF-Track Bunch6d):
        [x(mm), x'(mrad), y(mm), y'(mrad), t(mm/c), P(MeV/c)]

    This adapter transforms to/from FELsim coordinates:
        [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF×10³, ΔK/K₀×10³]
    """

    # Class-level capabilities for factory introspection
    CAPABILITIES = {
        'particle_tracking': True,
        'transfer_matrix': False,
        'space_charge': True,
        'rf_cavities': True,
        'field_maps': True,
        'wakefields': True,
        'synchrotron_radiation': True,
    }

    NATIVE_COORDINATES = CoordinateSystem.RFTRACK

    # Default aperture for elements (5 cm) - prevents tracking hangs
    DEFAULT_APERTURE = 0.05

    # Physical apertures for UH MkV FEL beamline [m]
    QUAD_HALF_APERTURE = 0.0135        # 27 mm bore / 2
    DIPOLE_HALF_GAP = 0.00724          # 14.48 mm gap / 2
    DIPOLE_HALF_WIDTH = 0.025          # 50 mm placeholder / 2
    BEAM_PIPE_RADIUS = 0.0127          # 1" beam pipe

    # Near-zero length for DPW thin-lens quadrupoles — true zero segfaults in RF-Track
    DPW_THIN_LENS_LENGTH = 1e-10

    # Angles below this threshold are treated as zero in sector-bend correction
    SECTOR_BEND_MIN_ANGLE = 1e-12

    def __init__(self,
                 lattice_path: Optional[str] = None,
                 excel_path: Optional[str] = None,
                 mode: str = 'particle_tracking',
                 space_charge: bool = False,
                 sc_mesh: Optional[tuple] = None,
                 beam_energy: float = 45.0,
                 particle_mass: Optional[float] = None,
                 particle_charge: float = -1.0,
                 aperture: float = 0.05,
                 G_quad: Optional[float] = None,
                 dipole_slices: int = 20,
                 rf_frequency: float = None,
                 debug: bool = None):
        """
        Initialise RF-Track adapter.

        Parameters
        ----------
        lattice_path : str, optional
            Path to lattice file (Excel, JSON, or YAML)
        excel_path : str, optional
            Backward-compatible alias for lattice_path
        mode : str
            Simulation mode ('particle_tracking' only for RF-Track)
        space_charge : bool
            Enable space charge calculation
        sc_mesh : tuple, optional
            Space charge mesh size (nx, ny, nz). Default: (32, 32, 64)
        beam_energy : float
            Beam kinetic energy in MeV
        particle_mass : float, optional
            Particle rest mass in MeV/c². Default: electron mass
        particle_charge : float
            Particle charge in elementary charges. Default: -1 (electron)
        aperture : float
            Default aperture for elements in metres. Default: 0.05 (5 cm)
        G_quad : float, optional
            Quadrupole gradient calibration constant in T/A/m.
            Converts current to field gradient: gradient = G_quad × current.
            Default: 2.694 T/A/m (UH FEL quadrupoles)
        dipole_slices : int
            Non-zero enables analytical sector-bend correction for dipoles.
            RF-Track v2.5.5 SBend is broken (P/δ confusion); dipoles are
            tracked as Drifts with post-tracking analytical correction
            (body focusing + dispersion + R₅₆). Set to 0 to disable.
            Default: 20
        rf_frequency : float, optional
            RF frequency in Hz for FELsim ↔ RF-Track coord5 conversion.
            Default: 2856 MHz (UH FEL S-band linac)
        debug : bool, optional
            Enable debug logging
        """
        if not _RFTRACK_AVAILABLE:
            raise ImportError(
                "RF-Track is not installed. Install with: pip install RF-Track"
            )

        super().__init__(
            name="RF-Track",
            native_coordinates=CoordinateSystem.RFTRACK,
            debug=debug
        )

        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

        # RF-Track only supports particle tracking
        if mode != 'particle_tracking':
            self.logger.warning(
                f"RF-Track only supports particle_tracking mode, ignoring '{mode}'"
            )
        self.simulation_mode = SimulationMode.PARTICLE_TRACKING

        # Beam parameters - use RF-Track's electron mass if not specified
        self.particle_mass = particle_mass if particle_mass is not None else rft.electronmass
        self.particle_charge = particle_charge
        self.beam_energy = beam_energy
        self._update_relativistic_params()

        # Default aperture for elements
        self.default_aperture = aperture

        # Quadrupole gradient calibration constant (T/A/m)
        # Default is for UH FEL quadrupoles (2.694 T/A/m)
        self.G_quad = G_quad if G_quad is not None else PhysicalConstants.G_quad_default

        # Dipole slicing (workaround for broken SBend in RF-Track v2.5.5)
        self.dipole_slices = dipole_slices

        # Space charge configuration
        self.space_charge_enabled = space_charge
        self.sc_mesh = sc_mesh or (32, 32, 64)
        self.sc_nsteps = 1  # SC kicks per element (Lattice mode)
        self._space_charge_effect = None

        # RF frequency for FELsim ↔ RF-Track coord5 conversion
        self._rf_frequency = rf_frequency if rf_frequency is not None else PhysicalConstants.f_RF_default
        self._t_scale = PhysicalConstants.C / self._rf_frequency  # c/f [m] ≈ 0.105

        # Physical aperture mode
        self._physical_apertures = False

        # RF-Track native objects
        self._lattice: Optional[rft.Lattice] = None
        self._bunch: Optional[rft.Bunch6d] = None
        self._native_elements: List[Any] = []

        # Element type mapping
        self._element_type_map = {
            'DRIFT': 'Drift',
            'QUAD_F': 'Quadrupole',
            'QPF': 'Quadrupole',
            'QUAD_D': 'Quadrupole',
            'QPD': 'Quadrupole',
            'DIPOLE': 'SBend',
            'DPH': 'SBend',
            'DIPOLE_WEDGE': 'SBend',
            'DPW': 'SBend',
            'SOLENOID': 'Solenoid',
            'RF_CAVITY': 'Cavity',
            'SEXTUPOLE': 'Sextupole',
        }

        # Load beamline if provided
        path = lattice_path or excel_path
        self.lattice_path = path
        self.excel_path = excel_path  # backward compat
        if path:
            self._load_lattice(path)

    def _update_relativistic_params(self):
        if self.beam_energy <= 0:
            raise ValueError(f"beam_energy must be positive, got {self.beam_energy}")
        self._gamma = 1 + self.beam_energy / self.particle_mass
        self._beta = np.sqrt(1 - 1/self._gamma**2)
        self._Pc = self._gamma * self._beta * self.particle_mass  # MeV/c

    def simulate(self,
                 particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        """
        Run RF-Track particle tracking simulation.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Initial distribution in FELsim coordinates:
            [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF×10³, ΔK/K₀×10³]
        mode : SimulationMode, optional
            Ignored (RF-Track only supports particle tracking)

        Returns
        -------
        SimulationResult
            Contains final particles, Twiss parameters, and metadata
        """
        if mode and mode != SimulationMode.PARTICLE_TRACKING:
            raise NotImplementedError(
                f"RF-Track only supports particle tracking, not {mode}"
            )

        if particles is None:
            raise ValueError("particles array required for simulation")

        self.validate_particles(particles)

        if self._lattice is None:
            raise ValueError(
                "Beamline not set. Call set_beamline() or provide excel_path"
            )

        particles_rftrack = self.transform_coordinates(
            particles, CoordinateSystem.FELSIM, CoordinateSystem.RFTRACK
        )

        self._bunch = rft.Bunch6d(
            self.particle_mass,
            self.particle_charge,
            self._Pc,
            particles_rftrack
        )

        self.logger.debug(
            f"Created bunch: {self._bunch.size()} particles, "
            f"Pc={self._Pc:.2f} MeV/c"
        )

        if self.space_charge_enabled and self._space_charge_effect is None:
            self._setup_space_charge()

        self.logger.info(
            f"Tracking {particles.shape[0]} particles through "
            f"{self._lattice.size()} elements (L={self._lattice.get_length():.3f} m)"
        )

        # Use segmented tracking if any dipoles need analytical correction
        has_analytical = any(
            e.parameters.get('_analytical_dipole', False) for e in self.beamline
        )

        if has_analytical:
            final_rftrack = self._track_segmented(particles_rftrack)
            n_good = final_rftrack.shape[0] if final_rftrack.ndim == 2 else 0
            n_lost = particles.shape[0] - n_good
        else:
            tracked_bunch = self._lattice.track(self._bunch)
            final_rftrack = tracked_bunch.get_phase_space()
            n_good = tracked_bunch.get_ngood()
            n_lost = tracked_bunch.get_nlost()

        if isinstance(final_rftrack, np.ndarray) and final_rftrack.ndim == 2 and final_rftrack.shape[0] > 0:
            final_particles = self.transform_coordinates(
                final_rftrack, CoordinateSystem.RFTRACK, CoordinateSystem.FELSIM
            )
        else:
            final_particles = np.empty((0, 6))
        twiss = self._calculate_twiss(final_particles) if final_particles.shape[0] > 0 else {}

        self.logger.info(
            f"Tracking complete: {n_good} good, {n_lost} lost particles"
        )

        return SimulationResult(
            simulator_name=self.name,
            success=True,
            final_particles=final_particles,
            twiss_parameters_statistical={'final': twiss},
            metadata={
                'num_particles': particles.shape[0],
                'num_good': n_good,
                'num_lost': n_lost,
                'beam_energy_mev': self.beam_energy,
                'momentum_mev_c': self._Pc,
                'space_charge': self.space_charge_enabled,
                'particle_mass_mev': self.particle_mass,
                'particle_charge': self.particle_charge,
                'lattice_length': self._lattice.get_length(),
            }
        )

    def optimize(self,
                 objectives: Dict,
                 variables: Dict,
                 initial_point: Dict,
                 method: Optional[str] = None,
                 **kwargs) -> SimulationResult:
        """
        Run optimization using RF-Track simulations.

        Uses scipy.optimize with RF-Track simulation as objective function.

        Parameters
        ----------
        objectives : dict
            Optimization objectives
        variables : dict
            Variable definitions
        initial_point : dict
            Initial values and bounds
        method : str, optional
            Optimization method (default: 'Nelder-Mead')
        **kwargs
            particles : ndarray (required)

        Returns
        -------
        SimulationResult
        """
        from scipy import optimize

        particles = kwargs.get('particles')
        if particles is None:
            raise ValueError("particles required for optimization")

        method = method or 'Nelder-Mead'

        def objective(x):
            for idx, (var_name, param_name, transform) in variables.items():
                value = transform(x[list(variables.keys()).index(idx)])
                self._modify_element(idx, **{param_name: value})
            self._build_lattice()
            result = self.simulate(particles)

            cost = 0.0
            twiss = result.twiss_parameters_statistical.get('final', {})
            for elem_idx, obj_list in objectives.items():
                if elem_idx == 'optimizer_settings':
                    continue
                for obj in obj_list:
                    axis, param = obj['measure']
                    goal = obj['goal']
                    weight = obj.get('weight', 1.0)
                    value = twiss.get(axis, {}).get(param, 0)
                    cost += weight * (value - goal)**2

            return cost

        x0 = [initial_point[v[0]]['start'] for v in variables.values()]
        bounds = [initial_point[v[0]].get('bounds') for v in variables.values()]

        result = optimize.minimize(
            objective, x0, method=method,
            bounds=bounds if any(bounds) else None
        )

        opt_vars = {
            var_name: transform(result.x[i])
            for i, (idx, (var_name, param_name, transform)) in enumerate(variables.items())
        }
        final = self.simulate(particles)

        return SimulationResult(
            simulator_name=self.name,
            success=result.success,
            twiss_parameters_statistical=final.twiss_parameters_statistical,
            final_particles=final.final_particles,
            optimization_variables=opt_vars,
            metadata={
                'method': method,
                'objective_value': result.fun,
                'num_iterations': getattr(result, 'nit', None),
                **final.metadata
            }
        )

    def _convert_element_to_native(self, element: BeamlineElement) -> Any:
        elem_type = element.element_type.upper()
        params = element.parameters
        length = element.length

        if elem_type == 'DRIFT':
            elem = rft.Drift(length)
            ap_x = ap_y = self.BEAM_PIPE_RADIUS if self._physical_apertures else self.default_aperture

        elif elem_type in ['QUAD_F', 'QPF']:
            elem = rft.Quadrupole()
            elem.set_length(length)
            k1 = self._current_to_k1(params.get('current', 0.0), length, focusing=True)
            elem.set_strength(k1 * length)  # integrated strength
            r = self.QUAD_HALF_APERTURE if self._physical_apertures else self.default_aperture
            ap_x = ap_y = r

        elif elem_type in ['QUAD_D', 'QPD']:
            elem = rft.Quadrupole()
            elem.set_length(length)
            k1 = self._current_to_k1(params.get('current', 0.0), length, focusing=False)
            elem.set_strength(k1 * length)
            r = self.QUAD_HALF_APERTURE if self._physical_apertures else self.default_aperture
            ap_x = ap_y = r

        elif elem_type in ['DIPOLE_WEDGE', 'DPW']:
            # FELsim models edge kicks as thin-lens matrices. RF-Track's SBend
            # set_E1/set_E2 is informational only, so we use a thin-lens
            # Quadrupole with K1L = -K0 * tan(wedge_angle) to reproduce the
            # horizontal edge kick exactly and the vertical kick approximately
            # (triangle-model fringe correction not included here).
            wedge_angle = np.radians(params.get('angle', 0.0))
            K0 = params.get('dipole_K0', 0.0)
            if wedge_angle != 0 and K0 != 0:
                K1L = -K0 * np.tan(wedge_angle)
                elem = rft.Quadrupole()
                elem.set_length(self.DPW_THIN_LENS_LENGTH)
                elem.set_strength(K1L)
            else:
                elem = rft.Drift(0)
            ap_x = ap_y = self.default_aperture

        elif elem_type in ['DIPOLE', 'DPH']:
            angle = params.get('angle', 0.0)
            if self._physical_apertures:
                ap_x = self.DIPOLE_HALF_WIDTH
                ap_y = self.DIPOLE_HALF_GAP
            else:
                ap_x = ap_y = self.default_aperture

            if angle != 0 and length > 0 and self.dipole_slices > 0:
                # RF-Track v2.5.5 SBend body tracking is broken (treats
                # absolute P as δ). Use a Drift for the lattice placeholder
                # (correct path length and y-plane) and mark for analytical
                # sector-bend correction in segmented tracking.
                elem = rft.Drift(length)
                params['_analytical_dipole'] = True
            else:
                elem = rft.Drift(length)

        elif elem_type == 'SOLENOID':
            elem = rft.Solenoid()
            elem.set_length(length)
            elem.set_Bz(params.get('field', 0.0))
            ap_x = ap_y = self.default_aperture

        elif elem_type == 'SEXTUPOLE':
            elem = rft.Sextupole()
            elem.set_length(length)
            elem.set_strength(params.get('strength', 0.0))
            ap_x = ap_y = self.default_aperture

        else:
            self.logger.warning(f"Unknown element type '{elem_type}', using drift")
            elem = rft.Drift(length)
            ap_x = ap_y = self.BEAM_PIPE_RADIUS if self._physical_apertures else self.default_aperture

        if hasattr(elem, 'set_aperture'):
            elem.set_aperture(ap_x, ap_y)

        if hasattr(elem, 'set_name') and 'name' in params:
            elem.set_name(params['name'])

        return elem

    def transform_coordinates(self,
                              particles: np.ndarray,
                              from_system: CoordinateSystem,
                              to_system: CoordinateSystem) -> np.ndarray:
        """
        Transform particle coordinates between systems.

        Coordinate systems:
        - FELSIM: [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF×10³, ΔK/K₀×10³]
        - RFTRACK: [x(mm), x'(mrad), y(mm), y'(mrad), ct(mm/c), P(MeV/c)]

        Coord5 conversion: ct_mm = coord5 × c/f_RF (the ×10³ and m→mm cancel).
        Note: RF-Track Bunch6d uses mm, mrad, mm/c, and MeV/c units.
        Column 5 is momentum P, not energy E.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Particle distribution
        from_system : CoordinateSystem
            Source coordinate system
        to_system : CoordinateSystem
            Target coordinate system

        Returns
        -------
        ndarray (N, 6)
            Transformed particle distribution
        """
        if from_system == to_system:
            return particles.copy()

        result = np.zeros_like(particles)

        if from_system == CoordinateSystem.FELSIM and to_system == CoordinateSystem.RFTRACK:
            # FELsim: [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T×10³, ΔK/K₀×10³]
            # RF-Track: [x(mm), x'(mrad), y(mm), y'(mrad), t(mm/c), P(MeV/c)]
            # Transverse coordinates: same units, no conversion needed
            result[:, 0:4] = particles[:, 0:4]
            # Longitudinal: FELsim ΔToF/T_RF×10³ → RF-Track ct [mm]
            # ct_mm = coord5 × c/f_RF (the ×10³ and m→mm cancel)
            result[:, 4] = particles[:, 4] * self._t_scale
            # FELsim coord6 = ΔK/K₀ × 10³ → RF-Track P [MeV/c] (exact)
            K = self.beam_energy * (1.0 + particles[:, 5] * 1e-3)
            E = K + self.particle_mass
            result[:, 5] = np.sqrt(E**2 - self.particle_mass**2)

        elif from_system == CoordinateSystem.RFTRACK and to_system == CoordinateSystem.FELSIM:
            # RF-Track: [x(mm), x'(mrad), y(mm), y'(mrad), t(mm/c), P(MeV/c)]
            # FELsim: [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T×10³, ΔK/K₀×10³]
            # Transverse coordinates: same units
            result[:, 0:4] = particles[:, 0:4]
            # Longitudinal: RF-Track ct [mm] → FELsim ΔToF/T_RF×10³
            result[:, 4] = particles[:, 4] / self._t_scale
            # RF-Track P [MeV/c] → FELsim coord6 = ΔK/K₀ × 10³ (exact)
            K = np.sqrt(particles[:, 5]**2 + self.particle_mass**2) - self.particle_mass
            result[:, 5] = (K / self.beam_energy - 1.0) * 1e3

        elif from_system == CoordinateSystem.COSY and to_system == CoordinateSystem.RFTRACK:
            # COSY uses m and rad; RF-Track Bunch6d uses mm and mrad
            result[:, 0:4] = particles[:, 0:4] * 1e3  # m/rad → mm/mrad
            result[:, 4] = particles[:, 4] / (self._beta * PhysicalConstants.C) * 1e3  # l(m) → t(mm/c)
            # COSY δ = ΔK/K₀ → RF-Track P [MeV/c] (exact)
            E0 = self.beam_energy + self.particle_mass  # total energy [MeV]
            E = E0 + self.beam_energy * particles[:, 5]  # E₀ + K₀δ
            result[:, 5] = np.sqrt(E**2 - self.particle_mass**2)  # P [MeV/c]

        elif from_system == CoordinateSystem.RFTRACK and to_system == CoordinateSystem.COSY:
            # RF-Track uses mm/mrad; COSY uses m/rad
            result[:, 0:4] = particles[:, 0:4] * 1e-3  # mm/mrad → m/rad
            result[:, 4] = particles[:, 4] * (self._beta * PhysicalConstants.C) * 1e-3  # t(mm/c) → l(m)
            # RF-Track P [MeV/c] → COSY δ = ΔK/K₀ (exact)
            K = np.sqrt(particles[:, 5]**2 + self.particle_mass**2) - self.particle_mass
            result[:, 5] = K / self.beam_energy - 1.0

        else:
            raise NotImplementedError(
                f"Transformation {from_system.value} → {to_system.value} "
                "not implemented. Transform via FELSIM as intermediate."
            )

        return result

    def set_beamline(self, elements: List[Union[BeamlineElement, Any]]):
        """
        Set beamline from generic or native elements.

        Parameters
        ----------
        elements : list
            List of BeamlineElement or native RF-Track elements
        """
        super().set_beamline(elements)
        self._build_lattice()

    def _annotate_dipole_edges(self):
        """Scan beamline for DPW-DPH-DPW triplets and annotate DPW elements.

        FELsim models dipole edge kicks as separate DIPOLE_WEDGE (DPW) elements
        flanking each DIPOLE (DPH). RF-Track's SBend set_E1/set_E2 is
        informational only (no effect on tracking), so we convert each DPW
        to a thin-lens quadrupole applying the equivalent edge kick:
            K1L = -K0 * tan(wedge_angle)
        where K0 = theta/L is the dipole curvature.

        This method writes 'dipole_K0' into each DPW's parameters so
        _convert_element_to_native can compute the thin-lens strength.
        """
        n_annotated = 0
        for i in range(len(self.beamline) - 2):
            e0, e1, e2 = self.beamline[i], self.beamline[i + 1], self.beamline[i + 2]
            if (e0.element_type.upper() in ('DIPOLE_WEDGE', 'DPW') and
                    e1.element_type.upper() in ('DIPOLE', 'DPH') and
                    e2.element_type.upper() in ('DIPOLE_WEDGE', 'DPW')):
                dph_angle = e1.parameters.get('angle', 0.0)
                dph_length = e1.length
                if dph_angle != 0 and dph_length > 0:
                    # Use |K0| to match FELsim convention (R = L/|θ|).
                    # Edge kick sign depends on pole face geometry, not bending direction.
                    K0 = abs(np.radians(dph_angle) / dph_length)
                    e0.parameters['dipole_K0'] = K0
                    e2.parameters['dipole_K0'] = K0
                    n_annotated += 1
        if n_annotated:
            self.logger.info(
                f"Annotated {n_annotated} DPW-DPH-DPW triplets for edge kick conversion"
            )
        else:
            self.logger.debug("No DPW-DPH-DPW triplets found")

    def _build_sliced_dipole(self, length, angle_rad, ap_x, ap_y):
        """Build a sector bend from N Corrector+Drift slices (split-operator).

        Workaround for RF-Track v2.5.5 SBend bug. Each slice:
            half-Drift(ds/2) → Corrector(Kx) → half-Drift(ds/2)

        The Corrector normalisation requires Kx = BdL/P₀ (same convention
        as Quadrupole set_strength taking k1*L, not P/q*k1*L).
        Since BdL = Bρ·dθ·1000 and Bρ = P₀/c, this simplifies to
        Kx = dθ·1000/c = dθ/0.299792458, independent of beam energy.

        Provides correct deflection and chromatic dispersion (1/P scaling).
        Body focusing (R11 = cos θ) is NOT reproduced — that requires
        a curved reference frame which only SBend provides.

        Note: This method is retained for validation and future use. The primary
        dipole tracking path uses ``_apply_sector_bend_correction()`` instead.
        """
        N = self.dipole_slices
        ds = length / N
        dtheta = angle_rad / N
        # Corrector Kx: normalised integrated field per slice
        c_mev = PhysicalConstants.C * 1e-6  # 299.792458 MeV/c per T·m
        dKx = dtheta * 1000 / c_mev  # sign follows angle sign

        elements = []
        for _ in range(N):
            d1 = rft.Drift(ds / 2)
            d1.set_aperture(ap_x, ap_y)
            elements.append(d1)

            c = rft.Corrector(0, dKx, 0)
            c.set_aperture(ap_x, ap_y)
            elements.append(c)

            d2 = rft.Drift(ds / 2)
            d2.set_aperture(ap_x, ap_y)
            elements.append(d2)

        return elements

    @staticmethod
    def _apply_sector_bend_correction(ps, length, angle_rad, Pc, particle_mass):
        """Apply analytical sector-bend body correction to phase space.

        RF-Track's SBend is broken (v2.5.5), so DPH elements are tracked as
        Drifts. This method corrects the tracked phase space from drift
        behaviour to sector-bend behaviour by applying:

            M_correction = M_sector × M_drift⁻¹

        to the horizontal plane (x, x'). The vertical plane (y, y') is
        already correct (both drift and sector bend give identity + L).

        Also applies:
        - Dispersion: R₁₆ = ρ(1-cos θ), R₂₆ = sin θ (via δ = ΔP/P₀)
        - Path length: R₅₆ = -(ρ sin θ - L) (geometric only). The 1/γ²
          velocity dispersion is already included by RF-Track's Drift tracking
          (verified numerically). The correction converts geometric path-length
          difference to time via 1/β₀.

        Parameters
        ----------
        ps : ndarray (N, 6)
            Phase space in RF-Track coordinates AFTER tracking through Drift.
            Columns: x[mm], x'[mrad], y[mm], y'[mrad], t[mm/c], P[MeV/c]
        length : float
            Dipole length [m]
        angle_rad : float
            Bend angle [rad]
        Pc : float
            Reference momentum [MeV/c]
        particle_mass : float
            Particle rest mass [MeV/c²]

        Limitations
        -----------
        - Fringe field multipoles (Enge sextupole, octupole) not included
        - Body focusing is linear only (no off-axis higher-order terms)
        - No synchrotron radiation or space charge inside dipole body
        """
        if ps.ndim != 2 or ps.shape[0] == 0:
            return ps
        if abs(angle_rad) < RFTrackAdapter.SECTOR_BEND_MIN_ANGLE:
            logger, _ = get_logger_with_fallback(__name__)
            logger.warning("_apply_sector_bend_correction: near-zero angle (%.2e), skipping", angle_rad)
            return ps

        theta = angle_rad
        L = length
        rho = L / theta
        C = np.cos(theta)
        S = np.sin(theta)

        # The drift that was applied: M_drift = [[1, L], [0, 1]]
        # We want: M_sector = [[C, ρS], [-S/ρ, C]]
        # Correction: M_corr = M_sector × M_drift⁻¹
        #   M_drift⁻¹ = [[1, -L], [0, 1]]
        #   M_corr = [[C, ρS - LC], [-S/ρ, LS/ρ + C]]
        # Units: R12 [m] = [mm/mrad], R21 [1/m] = [mrad/mm] — no scaling
        # needed because both x and x' differ from SI by the same factor (1000).
        R11 = C
        R12 = rho * S - L * C    # m (= mm/mrad)
        R21 = -S / rho           # 1/m (= mrad/mm)
        R22 = L * S / rho + C

        x = ps[:, 0].copy()    # mm
        xp = ps[:, 1].copy()   # mrad

        ps[:, 0] = R11 * x + R12 * xp
        ps[:, 1] = R21 * x + R22 * xp

        # Dispersion: Δx += ρ(1-C) × δ, Δx' += S × δ (in mrad)
        # RF-Track 6th col is P [MeV/c]; δ = (P - Pc) / Pc
        delta = (ps[:, 5] - Pc) / Pc
        # Use 2sin²(θ/2) instead of (1-cosθ) to avoid catastrophic cancellation for small θ
        ps[:, 0] += rho * 2 * np.sin(theta / 2)**2 * 1000 * delta   # mm
        ps[:, 1] += S * 1000 * delta                # mrad

        # Geometric path-length correction: Δs = -(ρ sin θ - L) × δ
        # The 1/γ² velocity dispersion is already handled by RF-Track's Drift.
        # Convert path → time: Δ(ct) = Δs/β₀
        gamma = np.sqrt(1 + (Pc / particle_mass)**2)
        beta = np.sqrt(1 - 1/gamma**2)
        R56_sector = -(rho * S - L)  # metres (negative for bunch compression)
        ps[:, 4] += R56_sector * 1000 * delta / beta  # mm/c

        return ps

    def _track_segmented(self, particles_rftrack):
        """Track through beamline segment-by-segment with analytical dipole corrections.

        Groups consecutive non-analytical-dipole elements into sub-lattices
        for efficient RF-Track tracking. At each analytical dipole, applies
        the correction from _apply_sector_bend_correction.
        """
        ps = particles_rftrack.copy()

        # Build segments: list of (sub_lattice, dipole_info_or_None)
        # Each segment is either a group of normal elements (tracked by RF-Track)
        # or a single analytical dipole (corrected after drift tracking)
        segments = []
        current_group = []

        for elem in self.beamline:
            params = elem.parameters
            is_analytical = params.get('_analytical_dipole', False)

            if is_analytical:
                # Flush accumulated group
                if current_group:
                    segments.append(('lattice', current_group))
                    current_group = []
                # Add dipole as its own segment
                segments.append(('dipole', elem))
            else:
                current_group.append(elem)

        if current_group:
            segments.append(('lattice', current_group))

        # Track through segments
        for seg_type, seg_data in segments:
            if ps.ndim != 2 or ps.shape[0] == 0:
                break

            if seg_type == 'lattice':
                lat = rft.Lattice()
                for elem in seg_data:
                    native = self._convert_element_to_native(elem)
                    if isinstance(native, list):
                        for ne in native:
                            lat.append(ne)
                    else:
                        lat.append(native)

                bunch = rft.Bunch6d(
                    self.particle_mass, self.particle_charge, self._Pc, ps
                )
                tracked = lat.track(bunch)
                ps = np.array(tracked.get_phase_space())

            else:  # 'dipole'
                elem = seg_data
                angle_rad = np.radians(elem.parameters.get('angle', 0.0))
                length = elem.length

                # Track through drift (for y-plane and path length)
                drift = rft.Drift(length)
                ap = self.DIPOLE_HALF_WIDTH if self._physical_apertures else self.default_aperture
                drift.set_aperture(ap, ap)
                lat = rft.Lattice()
                lat.append(drift)
                bunch = rft.Bunch6d(
                    self.particle_mass, self.particle_charge, self._Pc, ps
                )
                tracked = lat.track(bunch)
                ps = np.array(tracked.get_phase_space())

                # Apply analytical sector-bend correction
                if ps.ndim == 2 and ps.shape[0] > 0:
                    self._apply_sector_bend_correction(
                        ps, length, angle_rad, self._Pc, self.particle_mass
                    )

        return ps

    def track_elements(self, ps_rftrack, start_idx, end_idx):
        """Track through beamline[start_idx:end_idx] with analytical dipole corrections.

        Like _track_segmented() but for an arbitrary element range.
        Input/output are RF-Track coordinates (N, 6).
        """
        ps = ps_rftrack.copy()
        elements = self.beamline[start_idx:end_idx]

        segments = []
        current_group = []

        for elem in elements:
            if elem.parameters.get('_analytical_dipole', False):
                if current_group:
                    segments.append(('lattice', current_group))
                    current_group = []
                segments.append(('dipole', elem))
            else:
                current_group.append(elem)

        if current_group:
            segments.append(('lattice', current_group))

        for seg_type, seg_data in segments:
            if ps.ndim != 2 or ps.shape[0] == 0:
                break

            if seg_type == 'lattice':
                lat = rft.Lattice()
                for elem in seg_data:
                    native = self._convert_element_to_native(elem)
                    if isinstance(native, list):
                        for ne in native:
                            lat.append(ne)
                    else:
                        lat.append(native)
                lat.set_aperture(self.default_aperture, self.default_aperture)

                bunch = rft.Bunch6d(
                    self.particle_mass, self.particle_charge, self._Pc, ps
                )
                tracked = lat.track(bunch)
                ps = np.array(tracked.get_phase_space())

            else:  # 'dipole'
                elem = seg_data
                angle_rad = np.radians(elem.parameters.get('angle', 0.0))
                length = elem.length

                drift = rft.Drift(length)
                ap = self.DIPOLE_HALF_WIDTH if self._physical_apertures else self.default_aperture
                drift.set_aperture(ap, ap)
                lat = rft.Lattice()
                lat.append(drift)
                bunch = rft.Bunch6d(
                    self.particle_mass, self.particle_charge, self._Pc, ps
                )
                tracked = lat.track(bunch)
                ps = np.array(tracked.get_phase_space())

                if ps.ndim == 2 and ps.shape[0] > 0:
                    self._apply_sector_bend_correction(
                        ps, length, angle_rad, self._Pc, self.particle_mass
                    )

        return ps

    def _build_lattice(self):
        self._lattice = rft.Lattice()
        self._native_elements = []

        self._annotate_dipole_edges()

        for elem in self.beamline:
            native_elem = self._convert_element_to_native(elem)
            if isinstance(native_elem, list):
                for ne in native_elem:
                    self._native_elements.append(ne)
                    self._lattice.append(ne)
            else:
                self._native_elements.append(native_elem)
                self._lattice.append(native_elem)

        # Set lattice aperture to match element apertures
        self._lattice.set_aperture(self.default_aperture, self.default_aperture)

        self.logger.debug(
            f"Built RF-Track lattice: {self._lattice.size()} elements, "
            f"L={self._lattice.get_length():.3f} m"
        )

    def _modify_element(self, index: int, **kwargs):
        if 0 <= index < len(self.beamline):
            for key, value in kwargs.items():
                self.beamline[index].parameters[key] = value

    def set_space_charge(self, enabled: bool, mesh: Optional[tuple] = None,
                         method: str = 'PIC'):
        """
        Configure space charge calculation.

        Parameters
        ----------
        enabled : bool
            Enable/disable space charge
        mesh : tuple, optional
            Mesh size (nx, ny, nz) for 3D solver
        method : str
            Space charge method: 'PIC', 'P2P', 'FreeSpace'
        """
        self.space_charge_enabled = enabled
        if mesh:
            self.sc_mesh = mesh
        self._sc_method = method

        if enabled:
            self._setup_space_charge()
        else:
            self._space_charge_effect = None
            if self._lattice is not None:
                for i in range(self._lattice.size()):
                    self._lattice[i].set_sc_nsteps(0)

        self.logger.info(
            f"Space charge: {enabled}, method={method}, mesh={self.sc_mesh}"
        )

    def enable_physical_apertures(self, quad_half_aperture=None,
                                   dipole_half_gap=None, dipole_half_width=None,
                                   beam_pipe_radius=None):
        """Enable per-element physical apertures for particle tracking.

        Parameters
        ----------
        quad_half_aperture : float, optional
            Quadrupole half-aperture [m]. Default: 13.5 mm (27 mm bore/2).
        dipole_half_gap : float, optional
            Dipole vertical half-gap [m]. Default: 7.24 mm.
        dipole_half_width : float, optional
            Dipole horizontal half-width [m]. Default: 25 mm (placeholder).
        beam_pipe_radius : float, optional
            Default beam pipe radius [m]. Default: 12.7 mm (1" pipe).
        """
        self._physical_apertures = True
        if quad_half_aperture is not None:
            self.QUAD_HALF_APERTURE = quad_half_aperture
        if dipole_half_gap is not None:
            self.DIPOLE_HALF_GAP = dipole_half_gap
        if dipole_half_width is not None:
            self.DIPOLE_HALF_WIDTH = dipole_half_width
        if beam_pipe_radius is not None:
            self.BEAM_PIPE_RADIUS = beam_pipe_radius

        self.logger.info(
            f"Physical apertures enabled: quad={self.QUAD_HALF_APERTURE*1e3:.1f} mm, "
            f"dipole gap={self.DIPOLE_HALF_GAP*1e3:.2f} mm, "
            f"dipole width={self.DIPOLE_HALF_WIDTH*1e3:.1f} mm, "
            f"pipe={self.BEAM_PIPE_RADIUS*1e3:.1f} mm"
        )

        if self._lattice is not None and self.beamline:
            self._build_lattice()

    def disable_physical_apertures(self):
        """Disable per-element physical apertures, revert to default."""
        self._physical_apertures = False
        if self._lattice is not None and self.beamline:
            self._build_lattice()

    def _setup_space_charge(self):
        """Configure space charge via the global SC engine and per-element kicks.

        RF-Track 2.5.x handles space charge through:
        1. A global SC engine set via ``rft.cvar.SC_engine``
        2. Per-element activation via ``elem.set_sc_nsteps(N)``

        Note: SC_engine is global RF-Track state. Parallel instances will
        interfere. Each parallel worker must use a separate process.
        """
        nx, ny, nz = self.sc_mesh
        method = getattr(self, '_sc_method', 'PIC')

        if method == 'P2P':
            self._space_charge_effect = rft.SpaceCharge_P2P()
        else:
            self._space_charge_effect = rft.SpaceCharge_PIC_FreeSpace(nx, ny, nz)

        rft.cvar.SC_engine = self._space_charge_effect

        if self._lattice is not None:
            for i in range(self._lattice.size()):
                self._lattice[i].set_sc_nsteps(self.sc_nsteps)

    def collect_evolution(self,
                          particles: np.ndarray,
                          checkpoint_elements: Union[str, List[int]] = 'all') -> BeamEvolution:
        """
        Collect beam evolution data at element boundaries.

        Tracks element-by-element to capture phase space at each checkpoint.
        Uses RF-Track Lattice environment which tracks in curvilinear coordinates
        along the design orbit.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Initial distribution in FELsim coordinates
        checkpoint_elements : str or list
            'all' or list of element indices for checkpoints

        Returns
        -------
        BeamEvolution
        """
        self.validate_particles(particles)

        if not self.beamline:
            raise ValueError("Beamline not set")

        evolution = BeamEvolution(
            simulator_name=self.name,
            num_particles=particles.shape[0],
            beam_energy=self.beam_energy
        )

        n_elements = len(self.beamline)
        if checkpoint_elements == 'all':
            checkpoint_set = set(range(n_elements))
        else:
            checkpoint_set = set(checkpoint_elements)

        # Initial state
        evolution.add_sample(0.0, particles.copy(), self._calculate_twiss(particles))

        # Convert to RF-Track coordinates
        particles_rftrack = self.transform_coordinates(
            particles, CoordinateSystem.FELSIM, CoordinateSystem.RFTRACK
        )

        # Track element by element
        s = 0.0
        n_initial = particles.shape[0]
        for idx, elem in enumerate(self.beamline):
            n_before = particles_rftrack.shape[0] if particles_rftrack.ndim == 2 else 0

            # Build lattice for this element (may be multi-element for sliced dipoles)
            native = self._convert_element_to_native(elem)
            single_lat = rft.Lattice()
            if isinstance(native, list):
                for ne in native:
                    single_lat.append(ne)
            else:
                single_lat.append(native)

            # Create bunch for this segment
            bunch = rft.Bunch6d(
                self.particle_mass, self.particle_charge, self._Pc, particles_rftrack
            )

            # Track through element (returns tracked bunch)
            tracked_bunch = single_lat.track(bunch)

            # Update position
            s += elem.length

            # Get phase space from tracked bunch
            n_good = tracked_bunch.get_ngood()
            n_lost_elem = n_before - n_good
            particles_rftrack = np.array(tracked_bunch.get_phase_space())

            # Apply analytical sector-bend correction for flagged dipoles
            if (elem.parameters.get('_analytical_dipole', False)
                    and particles_rftrack.ndim == 2
                    and particles_rftrack.shape[0] > 0):
                angle_rad = np.radians(elem.parameters.get('angle', 0.0))
                self._apply_sector_bend_correction(
                    particles_rftrack, elem.length, angle_rad, self._Pc,
                    self.particle_mass
                )

            if n_lost_elem > 0:
                self.logger.info(
                    f"Element {idx} ({elem.element_type}, s={s:.4f} m): "
                    f"lost {n_lost_elem}, remaining {n_good}/{n_initial} "
                    f"({n_good/n_initial:.1%})"
                )

            # Record element info with transmission data
            elem_params = dict(elem.parameters)
            elem_params['n_good'] = n_good
            elem_params['n_lost'] = n_lost_elem
            elem_params['transmission'] = n_good / n_initial if n_initial > 0 else 0

            evolution.elements.append(ElementInfo(
                element_type=elem.element_type,
                s_start=s - elem.length,
                s_end=s,
                length=elem.length,
                color=self._get_element_color(elem.element_type),
                index=idx,
                parameters=elem_params
            ))

            # Checkpoint if requested
            if idx in checkpoint_set and particles_rftrack.size > 0:
                ps_felsim = self.transform_coordinates(
                    particles_rftrack, CoordinateSystem.RFTRACK, CoordinateSystem.FELSIM
                )
                evolution.add_sample(s, ps_felsim, self._calculate_twiss(ps_felsim))

        evolution.total_length = s
        return evolution

    def _load_lattice(self, lattice_path: str):
        """Load beamline from Excel, JSON, or YAML lattice file."""
        import latticeLoader

        try:
            native_elements = latticeLoader.create_beamline(lattice_path)

            self.beamline = []
            for elem in native_elements:
                self.beamline.append(self._convert_element_from_native(elem))

            self._build_lattice()
            self.logger.info(f"Loaded {len(self.beamline)} elements from {lattice_path}")

        except Exception as e:
            self.logger.error(f"Failed to load beamline from {lattice_path}: {type(e).__name__}: {e}")
            raise

    def _convert_element_from_native(self, native_elem: Any) -> BeamlineElement:
        """Convert FELsim native element to generic BeamlineElement."""
        cls_name = type(native_elem).__name__

        type_map = {
            'driftLattice': 'DRIFT',
            'qpfLattice': 'QUAD_F',
            'qpdLattice': 'QUAD_D',
            'dipole': 'DIPOLE',
            'dipole_wedge': 'DIPOLE_WEDGE',
        }

        elem_type = type_map.get(cls_name, cls_name.upper())

        params = {}
        if hasattr(native_elem, 'current'):
            params['current'] = native_elem.current
        if hasattr(native_elem, 'angle'):
            params['angle'] = native_elem.angle
        if hasattr(native_elem, 'pole_gap'):
            params['pole_gap'] = native_elem.pole_gap
        if hasattr(native_elem, 'fringeType'):
            params['fringe_type'] = native_elem.fringeType
        if hasattr(native_elem, 'name') and native_elem.name:
            params['name'] = native_elem.name

        return BeamlineElement(
            element_type=elem_type,
            length=native_elem.length,
            **params
        )

    def _calculate_twiss(self, particles: np.ndarray) -> Dict:
        """
        Calculate Twiss parameters from particle distribution (FELsim coords).

        FELsim coordinates: [x(mm), x'(mrad), y(mm), y'(mrad), ...]
        Returns beta in m, gamma in rad/m, emittance in π·mm·mrad.
        """
        if particles.shape[0] < 2:
            return {}

        twiss = {}
        for plane, (pos_idx, ang_idx) in [('x', (0, 1)), ('y', (2, 3))]:
            cov = np.cov(particles[:, pos_idx], particles[:, ang_idx], ddof=1)
            sig_x2, sig_xp2, sig_xxp = cov[0, 0], cov[1, 1], cov[0, 1]

            emit_sq = sig_x2 * sig_xp2 - sig_xxp**2
            emittance = np.sqrt(max(0, emit_sq))  # π·mm·mrad

            if emittance > 0:
                # beta = mm²/(mm·mrad) = mm/mrad = m
                beta = sig_x2 / emittance
                alpha = -sig_xxp / emittance
                # gamma = mrad²/(mm·mrad) = mrad/mm = rad/m
                gamma = sig_xp2 / emittance
                if beta > 1e6:
                    self.logger.warning(f"Unphysical beta_{plane} = {beta:.1e} m — beam may be mismatched")
            else:
                beta = alpha = gamma = 0.0
                self.logger.warning(f"Zero emittance in {plane}-plane — degenerate beam distribution")

            twiss[plane] = {'beta': beta, 'alpha': alpha, 'gamma': gamma, 'emittance': emittance}

        return twiss

    def _current_to_k1(self, current: float, length: float, focusing: bool = True) -> float:
        """
        Convert quadrupole current to normalized gradient k1.

        Uses k = |Q·G·I| / (M·C·β·γ), consistent with FELsim's beamline.py.
        """
        if length <= 0 or current == 0:
            return 0.0

        mass_kg = self.particle_mass * PhysicalConstants.MeV_to_J / PhysicalConstants.C**2
        k1 = abs(PhysicalConstants.Q * self.G_quad * current) / (
            mass_kg * PhysicalConstants.C * self._beta * self._gamma
        )
        return k1 if focusing else -k1

    def _get_element_color(self, elem_type: str) -> str:
        colors = {
            'DRIFT': 'white',
            'QUAD_F': 'cornflowerblue',
            'QPF': 'cornflowerblue',
            'QUAD_D': 'lightcoral',
            'QPD': 'lightcoral',
            'DIPOLE': 'forestgreen',
            'DPH': 'forestgreen',
            'DIPOLE_WEDGE': 'lightgreen',
            'DPW': 'lightgreen',
            'SOLENOID': 'purple',
            'RF_CAVITY': 'gold',
            'SEXTUPOLE': 'orange',
        }
        return colors.get(elem_type.upper(), 'gray')

    def generate_particles(self,
                           num_particles: int = 1000,
                           distribution_type: str = 'gaussian',
                           **parameters) -> np.ndarray:
        """
        Generate initial particle distribution in FELsim coordinates.

        Parameters
        ----------
        num_particles : int
            Number of particles
        distribution_type : str
            'gaussian', 'uniform', 'waterbag', 'kv', or 'twiss'
        **parameters
            std_dev : list of 6 RMS values [mm, mrad, mm, mrad, -, -]
            twiss_x, twiss_y : dict with beta, alpha, emittance for 'twiss' type

        Returns
        -------
        ndarray (N, 6) in FELsim coordinates
        """
        std_dev = parameters.get('std_dev', [1.0, 0.1, 1.0, 0.1, 1.0, 0.1])
        mean = parameters.get('mean', 0.0)

        if distribution_type == 'gaussian':
            particles = np.random.randn(num_particles, 6) * std_dev + mean

        elif distribution_type == 'uniform':
            half_width = np.array(std_dev) * np.sqrt(3)
            particles = np.random.uniform(-half_width, half_width, (num_particles, 6))

        elif distribution_type == 'twiss':
            twiss_x = parameters.get('twiss_x', {'beta': 10, 'alpha': 0, 'emittance': 1})
            twiss_y = parameters.get('twiss_y', {'beta': 10, 'alpha': 0, 'emittance': 1})

            particles = np.zeros((num_particles, 6))
            for plane, twiss, idx in [('x', twiss_x, 0), ('y', twiss_y, 2)]:
                beta, alpha, emit = twiss['beta'], twiss['alpha'], twiss['emittance']
                u1, u2 = np.random.randn(num_particles), np.random.randn(num_particles)

                sigma_x = np.sqrt(emit * beta)
                sigma_xp = np.sqrt(emit / beta) if beta > 0 else 0

                particles[:, idx] = sigma_x * u1
                particles[:, idx+1] = sigma_xp * (-alpha * u1 / np.sqrt(beta) + u2) if beta > 0 else 0

            particles[:, 4] = np.random.randn(num_particles) * std_dev[4]
            particles[:, 5] = np.random.randn(num_particles) * std_dev[5]

        else:
            self.logger.warning(f"Distribution '{distribution_type}' not implemented, using Gaussian")
            particles = np.random.randn(num_particles, 6) * std_dev + mean

        return particles

    def set_beam_energy(self, energy_mev: float):
        super().set_beam_energy(energy_mev)
        self.beam_energy = energy_mev
        self._update_relativistic_params()
        self.logger.debug(f"Energy: {energy_mev} MeV, γ={self._gamma:.2f}, Pc={self._Pc:.2f} MeV/c")

    def set_particle_type(self, mass_mev: float, charge: float):
        self.particle_mass = mass_mev
        self.particle_charge = charge
        self._update_relativistic_params()
        self.logger.info(f"Particle: m={mass_mev} MeV/c², q={charge}e")

    def set_quadrupole_gradient(self, G_quad: float):
        """Set quadrupole gradient calibration (T/A/m). Default: 2.694 for UH FEL."""
        self.G_quad = G_quad
        self.logger.info(f"Quadrupole gradient calibration: G = {G_quad:.4f} T/A/m")
        if self._lattice is not None and self.beamline:
            self._build_lattice()

    def supports_mode(self, mode: SimulationMode) -> bool:
        return mode == SimulationMode.PARTICLE_TRACKING

    def supports_optimization(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps.update({
            'space_charge': self.space_charge_enabled,
            'rf_cavities': True,
            'field_maps': True,
            'particle_mass_mev': self.particle_mass,
            'particle_charge': self.particle_charge,
            'momentum_mev_c': self._Pc,
        })
        return caps

    def get_lattice(self) -> Any:
        return self._lattice

    def get_bunch(self) -> Any:
        return self._bunch


# Convenience function
def create_rftrack_simulator(**kwargs) -> RFTrackAdapter:
    return RFTrackAdapter(**kwargs)
