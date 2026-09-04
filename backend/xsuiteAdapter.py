"""
Xsuite adapter for the unified simulator interface.

Wraps xsuite (xtrack/xpart/xfields) behind SimulatorBase so it can be chained
in MultiCodeSimulator alongside FELsim, COSY, and RF-Track. Like the other
adapters, simulate() accepts and returns particles in FELsim coordinates and
converts to/from xsuite's native frame internally.

Supported elements: drift, quadrupole (thick, current to k1 via the same
calibration as FELsim/RF-Track), sector bend, and dipole edges via
xt.DipoleEdge with fint from FELsim's triangle fringe. Optional space
charge via xfields (frozen-Gaussian or 3D PIC), inserted split-operator
between short sub-slices of each element.

RF cavities are modelled as a multi-cell autophasing TW chain (Cavity +
ReferenceEnergyIncrease) that ramps the reference energy on-crest; off-crest
phase and self-consistent longitudinal bunching are not modelled.

Author: Eremey Valetov
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from simulatorBase import (
    SimulatorBase, SimulationResult, BeamlineElement,
    CoordinateSystem, SimulationMode,
)
from physicalConstants import PhysicalConstants
from ebeam import beam as ebeam_class

logger = logging.getLogger(__name__)

try:
    import xtrack as xt
    import xpart as xp
    import xfields as xf
    _XSUITE_AVAILABLE = True
except ImportError:
    _XSUITE_AVAILABLE = False
    xt = xp = xf = None

# Horizontal-focusing sign for a QUAD_F in xsuite's k1 convention (electrons,
# q0=-1). Validated against FELsim on a FODO cell; flip if planes swap.
_FOCUS_SIGN = 1.0


class XsuiteAdapter(SimulatorBase):
    """Adapter exposing xsuite tracking through the SimulatorBase interface."""

    NATIVE_COORDINATES = CoordinateSystem.XSUITE

    def __init__(self,
                 lattice_path: Optional[str] = None,
                 excel_path: Optional[str] = None,
                 space_charge: bool = False,
                 sc_method: str = "frozen",
                 sc_mesh: tuple = (64, 64, 64),
                 n_slice_sc: int = 80,
                 bunch_charge_nc: float = 1.0,
                 beam_energy: float = 45.0,
                 particle_mass: Optional[float] = None,
                 particle_charge: float = -1.0,
                 debug: bool = None,
                 **kwargs):
        if not _XSUITE_AVAILABLE:
            raise ImportError(
                "xsuite not available. Install with `pip install xsuite`."
            )
        super().__init__(name="Xsuite", native_coordinates=CoordinateSystem.XSUITE)

        self.simulation_mode = SimulationMode.PARTICLE_TRACKING
        self.beam_energy = beam_energy
        self.particle_mass = particle_mass or PhysicalConstants.E0_electron
        self.particle_charge = particle_charge
        self.G_quad = PhysicalConstants.G_quad_default

        self.space_charge_enabled = space_charge
        self.sc_method = sc_method        # 'frozen' | 'pic3d'
        self.sc_mesh = tuple(sc_mesh)
        self.n_slice_sc = n_slice_sc
        self.bunch_charge_C = bunch_charge_nc * 1e-9

        self._ebeam = ebeam_class()
        self._update_energy()

        path = lattice_path or excel_path
        if path:
            import latticeLoader
            native = latticeLoader.create_beamline(path)
            for e in native:
                e.setE(self.beam_energy)
            self.set_beamline(native)

    # ------------------------------------------------------------------ energy
    def _update_energy(self):
        m = self.particle_mass
        self._gamma = 1.0 + self.beam_energy / m
        self._beta = np.sqrt(1.0 - 1.0 / self._gamma ** 2)
        self._betagamma = self._beta * self._gamma

    def set_beam_energy(self, energy_mev: float):
        super().set_beam_energy(energy_mev)
        self._update_energy()

    # --------------------------------------------------------------- beamline
    def set_beamline(self, elements: List[Union[BeamlineElement, Any]]):
        """Accept generic BeamlineElement objects or native FELsim elements."""
        if not elements:
            self.beamline = []
            return
        if isinstance(elements[0], BeamlineElement):
            self.beamline = list(elements)
        else:
            self.beamline = [self._convert_element_from_native(e) for e in elements]

    def _convert_element_from_native(self, elem) -> BeamlineElement:
        cls_name = type(elem).__name__
        type_map = {
            'driftLattice': 'DRIFT', 'qpfLattice': 'QUAD_F',
            'qpdLattice': 'QUAD_D', 'dipole': 'DIPOLE',
            'dipole_wedge': 'DIPOLE_WEDGE', 'rfCavityLattice': 'RF_CAVITY',
        }
        etype = type_map.get(cls_name, cls_name.upper())
        params = {}
        for attr in ('current', 'angle', 'dipole_angle', 'dipole_length',
                     'pole_gap', 'enge_fct'):
            if hasattr(elem, attr):
                params[attr] = getattr(elem, attr)
        # carry RF-cavity parameters so the TW model can be built downstream
        for attr in ('frequency_hz', 'gradient_mv_per_m', 'voltage_mv',
                     'structure_type', 'phase_advance_deg', 'n_cells', 'phase_deg'):
            if getattr(elem, attr, None) is not None:
                params[attr] = getattr(elem, attr)
        return BeamlineElement(etype, getattr(elem, 'length', 0.0), **params)

    def _convert_element_to_native(self, element: BeamlineElement) -> Any:
        # xsuite elements are built lazily in _build_line; nothing to do here.
        return element

    def _current_to_k1(self, current: float, focusing: bool,
                       betagamma: Optional[float] = None) -> float:
        """k1 = |Q·G·I| / (m·c·β·γ), matching FELsim / RF-Track. betagamma
        defaults to the adapter energy; pass a local value for elements after an
        accelerating section, since the same physical magnet is weaker at higher
        energy (k1 scales as 1/p)."""
        if current == 0:
            return 0.0
        mass_kg = self.particle_mass * PhysicalConstants.MeV_to_J / PhysicalConstants.C ** 2
        charge_C = abs(self.particle_charge) * PhysicalConstants.Q
        bg = betagamma if betagamma is not None else self._betagamma
        k1 = abs(charge_C * self.G_quad * current) / (
            mass_kg * PhysicalConstants.C * bg)
        return _FOCUS_SIGN * (k1 if focusing else -k1)

    def _betagamma_at(self, K_mev: float) -> float:
        gamma = 1.0 + K_mev / self.particle_mass
        return np.sqrt(gamma ** 2 - 1.0)

    def _tw_synchronous_profile(self, K_in, E0_vpm, freq, phi_adv, n_cells, L_cell):
        """Autophasing RK4 of the synchronous particle through a TW structure;
        kinetic energy [MeV] at each cell boundary (length n_cells+1), injected at
        the phase that maximises the gain. None if the particle is lost."""
        c = PhysicalConstants.C
        mc2 = self.particle_mass
        omega = 2.0 * np.pi * freq

        def run(phi0):
            K, psi, Ks = K_in, phi0, [K_in]
            for n in range(n_cells):
                zs = np.linspace(n * L_cell, (n + 1) * L_cell, 11)
                for i in range(len(zs) - 1):
                    h = zs[i + 1] - zs[i]

                    def d(K, psi):
                        g = 1.0 + K / mc2
                        if g <= 1.0:
                            return None
                        b = np.sqrt(1.0 - 1.0 / g ** 2)
                        return (E0_vpm * np.cos(psi)) / 1e6, (omega / c) * (1.0 / b - 1.0)
                    k1 = d(K, psi)
                    if k1 is None:
                        return None
                    k2 = d(K + .5 * h * k1[0], psi + .5 * h * k1[1])
                    k3 = d(K + .5 * h * k2[0], psi + .5 * h * k2[1]) if k2 else None
                    k4 = d(K + h * k3[0], psi + h * k3[1]) if k3 else None
                    if None in (k2, k3, k4):
                        return None
                    K += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                    psi += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                Ks.append(K)
            return np.array(Ks)

        best = None
        for phi in np.deg2rad(np.arange(0, 360, 2.0)):
            prof = run(phi)
            if prof is not None and (best is None or prof[-1] > best[-1]):
                best = prof
        return best

    def _build_tw_cavity(self, elem, K_in, sc=None):
        """xtrack multi-cell TW chain (Cavity + ReferenceEnergyIncrease) for one
        RF_CAVITY, injected at K_in [MeV]. Returns (elements, delta_K_MeV,
        n_cells). The per-cell energy gains come from the autophasing model; the
        reference ramps so the synchronous particle stays at delta~0 and
        downstream optics see the right energy."""
        p = elem.parameters
        L = elem.length
        phase_deg = float(p.get('phase_deg') or 0.0)
        if abs(phase_deg) > 1e-6:
            logger.warning("Xsuite RF_CAVITY phase_deg=%.1f off-crest is not "
                           "modelled; treating as on-crest (max energy gain)",
                           phase_deg)
        if p.get('frequency_hz') is None:
            logger.warning("Xsuite RF_CAVITY missing frequency_hz; drift")
            return [xt.Drift(length=L)], 0.0, 0
        freq = float(p['frequency_hz'])
        phi_adv = np.deg2rad(float(p.get('phase_advance_deg', 120.0)))
        if p.get('gradient_mv_per_m') is not None:
            E0_vpm = float(p['gradient_mv_per_m']) * 1e6
        elif p.get('voltage_mv') is not None and L > 0:
            E0_vpm = float(p['voltage_mv']) * 1e6 / L
        else:
            logger.warning("Xsuite RF_CAVITY missing gradient/voltage; drift")
            return [xt.Drift(length=L)], 0.0, 0
        l_sync = PhysicalConstants.C * phi_adv / (2.0 * np.pi * freq)
        n_cells = p.get('n_cells')
        n_cells = int(round(L / l_sync)) if not n_cells else int(round(float(n_cells)))
        n_cells = max(1, n_cells)
        L_cell = L / n_cells          # span elem.length exactly (no drift mismatch)
        Ks = self._tw_synchronous_profile(K_in, E0_vpm, freq, phi_adv, n_cells, L_cell)
        if Ks is None:
            logger.warning("Xsuite RF_CAVITY: synchronous particle lost; drift")
            return [xt.Drift(length=L)], 0.0, 0
        mc2 = self.particle_mass
        def mom(K):
            return np.sqrt((K + mc2) ** 2 - mc2 ** 2) * 1e6   # eV/c
        elems = []
        for n in range(len(Ks) - 1):
            dK = (Ks[n + 1] - Ks[n]) * 1e6                     # eV
            dp = mom(Ks[n + 1]) - mom(Ks[n])                   # eV/c
            elems += [xt.Drift(length=L_cell / 2),
                      xt.Cavity(frequency=freq, voltage=dK, phase=np.pi / 2),
                      xt.ReferenceEnergyIncrease(Delta_p0c=dp),
                      xt.Drift(length=L_cell / 2)]
            if sc is not None:
                # SC kick after the reference ramp: the frozen model reads the
                # local (post-cell) energy automatically; PIC gets the local
                # gamma explicitly. This is what makes SC strong at the 1 MeV
                # injection and ~1/(beta^2 gamma^3)-suppressed downstream (fixed
                # charge: the 1/(beta^3 gamma^3) perveance loses one power of
                # beta because the line current scales with beta).
                env = sc.get('sig_env')   # flat [(sx, sy), ...] over all SC cells
                if env is not None:
                    off = sc.get('env_offset', 0)   # this cavity's first cell
                    sx, sy = env[min(off + n, len(env) - 1)]
                    upd = False           # prescribed envelope, deterministic
                else:
                    sx, sy, upd = sc['sig_x'], sc['sig_y'], True   # self-consistent
                elems.append(self._make_sc_element(
                    L_cell, sx, sy, sc['long_profile'],
                    gamma=1.0 + Ks[n + 1] / mc2, update=upd))
        return elems, float(Ks[-1] - K_in), len(Ks) - 1

    # ----------------------------------------------------------- line builder
    def _xsuite_elements_for(self, elem: BeamlineElement, length: float,
                             e_run: Optional[float] = None):
        """Return the xsuite element(s) for one FELsim element of given length
        (length may be a sub-slice of the full element when space charge is on).
        e_run is the local reference kinetic energy [MeV] (for energy-aware k1
        after an accelerating section); defaults to the adapter energy."""
        etype = elem.element_type.upper()
        if etype == 'DRIFT' or length <= 0:
            return [xt.Drift(length=length)]
        if etype in ('QUAD_F', 'QPF', 'QUAD_D', 'QPD'):
            focusing = etype in ('QUAD_F', 'QPF')
            bg = self._betagamma_at(e_run) if e_run is not None else None
            k1 = self._current_to_k1(elem.parameters.get('current', 0.0),
                                     focusing, betagamma=bg)
            return [xt.Quadrupole(length=length, k1=k1)]
        if etype in ('DIPOLE', 'DPH'):
            ang = math.radians(float(elem.parameters.get('angle', 0.0)))
            full = elem.length or length
            h = ang / full if full else 0.0       # curvature 1/rho (sector body)
            ang_sub = ang * (length / full) if full else 0.0
            # Sector body only, in radians (FELsim stores degrees). Edge/fringe
            # focusing lives entirely in the flanking DPW elements, exactly as
            # in FELsim (whose dipole is a y-drift): deactivate Bend's own edges
            # so they are not double-counted.
            bend = xt.Bend(length=length, angle=ang_sub, k0=h)
            bend.edge_entry_active = 0
            bend.edge_exit_active = 0
            return [bend]
        if etype in ('DIPOLE_WEDGE', 'DPW'):
            return [self._dpw_edge(elem)]
        logger.warning("Xsuite: unknown element %s; treated as drift", etype)
        return [xt.Drift(length=length)]

    def _dpw_edge(self, elem: BeamlineElement):
        """Thin dipole edge reproducing FELsim's ``dipole_wedge`` matrix.

        FELsim's DPW is a thin lens (its 10 mm bookkeeping length adds no
        drift): ``x' += tan(eta)/R * x`` and ``y' -= tan(eta - phi)/R * y``
        with the triangle-model fringe ``phi = (le/6)(1/R)(1+sin^2 eta)/cos
        eta`` and R taken from the flanking dipole's |angle|/length (unsigned
        for both bend signs). xtrack's linear ``DipoleEdge`` implements the
        MAD fringe ``psi = 2 fint hgap k (1+sin^2 e1)/cos e1``, so
        ``fint = le/(12 hgap)`` with ``hgap = pole_gap/2`` reproduces phi
        exactly; verified against the FELsim matrix in
        test/test_xsuite_dipole_edges.py.
        """
        p = elem.parameters
        d_ang = math.radians(abs(float(p.get('dipole_angle', 0.0) or 0.0)))
        d_len = float(p.get('dipole_length', 0.0) or 0.0)
        if d_ang <= 0.0 or d_len <= 0.0:
            logger.warning("Xsuite: DPW without flanking-dipole parameters; "
                           "treated as marker")
            return xt.Drift(length=0.0)
        k = d_ang / d_len                       # +1/R, unsigned like FELsim
        eta = math.radians(float(p.get('angle', 0.0) or 0.0))
        le = float(elem.length or 0.0)
        hgap = 0.5 * float(p.get('pole_gap', 0.0) or 0.0)
        fint = le / (12.0 * hgap) if hgap > 0.0 else 0.0
        return xt.DipoleEdge(k=k, e1=eta, hgap=hgap, fint=fint, model='linear')

    def _make_sc_element(self, length, sig_x, sig_y, long_profile, gamma=None,
                         update=True):
        # gamma sets the relativistic factor for the PIC solver; the frozen
        # BiGaussian kick reads the line's (ramped) reference energy at track
        # time, so it needs no explicit gamma. Pass the local gamma inside an
        # accelerating section so the SC suppression (~1/(beta^2 gamma^3)) is right.
        # update=False uses the supplied sigma as a prescribed (matched-envelope)
        # value instead of recomputing it from the bunch each step.
        if self.sc_method == "pic3d":
            nx, ny, nz = self.sc_mesh
            return xf.SpaceCharge3D(
                length=length, update_on_track=update,
                x_range=(-8 * sig_x, 8 * sig_x),
                y_range=(-8 * sig_y, 8 * sig_y),
                z_range=(-8 * long_profile.sigma_z, 8 * long_profile.sigma_z),
                nx=nx, ny=ny, nz=nz,
                solver="FFTSolver2p5D", gamma0=(gamma or self._gamma))
        return xf.SpaceChargeBiGaussian(
            length=length, longitudinal_profile=long_profile,
            sigma_x=sig_x, sigma_y=sig_y, mean_x=0.0, mean_y=0.0,
            update_on_track=update)

    def _build_line(self, sc_on, sig_x=None, sig_y=None, sig_z=None, n_e=None,
                    sig_env=None):
        # sig_env: optional FLAT [(sigma_x, sigma_y), ...] matched envelope over
        # all SC cells (across every RF_CAVITY, in order). When given, each
        # per-cell SC uses its local sigma (prescribed, deterministic); otherwise
        # sigma is recomputed self-consistently from the bunch (update_on_track,
        # the default).
        elements = []
        e_run = self.beam_energy          # running reference kinetic energy [MeV]
        sc_off = 0                        # running SC-cell index into sig_env
        if not sc_on:
            for elem in self.beamline:
                if elem.element_type.upper() == 'RF_CAVITY':
                    tw, dK, _ = self._build_tw_cavity(elem, e_run)
                    elements += tw
                    e_run += dK
                else:
                    elements += self._xsuite_elements_for(elem, elem.length, e_run)
        else:
            total_L = sum(e.length for e in self.beamline)
            ds = total_L / max(self.n_slice_sc, 1)
            long_profile = xf.LongitudinalProfileQGaussian(
                number_of_particles=n_e, sigma_z=sig_z, z0=0.0, q_parameter=1.0)
            for elem in self.beamline:
                if elem.element_type.upper() == 'RF_CAVITY':
                    # per-cell SC interleaved inside the linac; each kick uses the
                    # local energy after that cell's reference ramp
                    tw, dK, n_cells = self._build_tw_cavity(elem, e_run, sc=dict(
                        sig_x=sig_x, sig_y=sig_y, long_profile=long_profile,
                        sig_env=sig_env, env_offset=sc_off))
                    elements += tw
                    e_run += dK
                    sc_off += n_cells
                    continue
                L = elem.length
                if L <= 0 or elem.element_type.upper() in ('DIPOLE_WEDGE', 'DPW'):
                    # zero-length elements and thin dipole edges are unsliceable
                    elements += self._xsuite_elements_for(elem, L, e_run)
                    continue
                n_sub = max(1, int(round(L / ds)))
                sub_L = L / n_sub
                for _ in range(n_sub):
                    elements += self._xsuite_elements_for(elem, sub_L, e_run)
                    if sig_env is not None:
                        sx, sy = sig_env[min(sc_off, len(sig_env) - 1)]
                        elements.append(self._make_sc_element(
                            sub_L, sx, sy, long_profile, update=False))
                    else:
                        elements.append(self._make_sc_element(
                            sub_L, sig_x, sig_y, long_profile))
                    sc_off += 1     # advance the flat sig_env index for every SC
                                    # cell, so interspersed (non-cavity) SC stations
                                    # don't desync a later cavity's envelope offset

        line = xt.Line(elements=elements)
        line.particle_ref = xp.Particles(
            mass0=self.particle_mass * 1e6, q0=self.particle_charge,
            kinetic_energy0=self.beam_energy * 1e6)
        line.build_tracker()
        return line

    def sc_envelope_prepass(self, sig_x, sig_y, sig_xp, sig_yp, sig_z, n_e,
                            n_part=2000, seed=0):
        """Matched-envelope pre-pass: track a Gaussian bunch through the
        SELF-CONSISTENT SC-on linac and return the per-cell (sigma_x, sigma_y)
        the SC actually sees. Feed the result back as `sig_env` to drive a
        deterministic, low-noise SC model that reproduces the self-consistent run.
        Tracking through the SC-on (update_on_track) line is what converges the
        envelope: an SC-off pre-pass underestimates sigma where SC grows the beam,
        and a naive analytic adiabatic law is wrong without transverse focusing."""
        line = self._build_line(sc_on=True, sig_x=sig_x, sig_y=sig_y,
                                sig_z=sig_z, n_e=n_e)
        et = list(line.get_table().element_type)
        sc_idx = [i for i, t in enumerate(et) if t.startswith('SpaceCharge')]
        rng = np.random.default_rng(seed)
        p = line.build_particles(
            x=rng.normal(0, sig_x, n_part), px=rng.normal(0, sig_xp, n_part),
            y=rng.normal(0, sig_y, n_part), py=rng.normal(0, sig_yp, n_part),
            zeta=rng.normal(0, sig_z, n_part), delta=np.zeros(n_part))
        env, prev = [], 0
        for si in sc_idx:                       # sigma arriving at each SC kick
            line.track(p, ele_start=prev, ele_stop=si)
            prev = si
            m = p.state == 1
            env.append((float(np.std(p.x[m])), float(np.std(p.y[m]))))
        return env

    # -------------------------------------------------------------- simulate
    def simulate(self, particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        if mode and mode != SimulationMode.PARTICLE_TRACKING:
            raise NotImplementedError("Xsuite adapter only supports particle tracking")
        if particles is None:
            raise ValueError("particles required")
        if not self.beamline:
            raise ValueError("Beamline not set")
        self.validate_particles(particles)

        xs = self.transform_coordinates(
            particles, CoordinateSystem.FELSIM, CoordinateSystem.XSUITE)
        n_p = xs.shape[0]
        n_e = self.bunch_charge_C / PhysicalConstants.Q

        p = xp.Particles(
            mass0=self.particle_mass * 1e6, q0=self.particle_charge,
            kinetic_energy0=self.beam_energy * 1e6,
            x=xs[:, 0], px=xs[:, 1], y=xs[:, 2], py=xs[:, 3],
            zeta=xs[:, 4], delta=xs[:, 5],
            weight=(n_e / n_p if self.space_charge_enabled else 1.0))

        if self.space_charge_enabled:
            sig_x = float(np.std(xs[:, 0])) or 1e-9
            sig_y = float(np.std(xs[:, 2])) or 1e-9
            sig_z = float(np.std(xs[:, 4])) or 1e-6
            line = self._build_line(True, sig_x, sig_y, sig_z, n_e)
        else:
            line = self._build_line(False)

        line.track(p)

        # A particle whose coordinates went non-finite is lost regardless of
        # its state flag: at extreme amplitudes the exact maps can overflow a
        # step before xtrack flags the particle, and one NaN row would poison
        # every downstream moment (same guard as CanonicalParticles.alive_mask).
        coords = np.column_stack([
            np.asarray(p.x), np.asarray(p.px), np.asarray(p.y),
            np.asarray(p.py), np.asarray(p.zeta), np.asarray(p.delta)])
        alive = (np.asarray(p.state) > 0) & np.isfinite(coords).all(axis=1)
        n_good = int(alive.sum())
        n_lost = n_p - n_good
        if n_good == 0:
            return SimulationResult(simulator_name=self.name, success=False,
                                    final_particles=np.empty((0, 6)),
                                    metadata={'num_lost': n_lost})

        xs_out = coords[alive]
        # the line may ramp the reference energy (linac); convert the output at the
        # EXIT reference energy, not the injection energy, so delta and the angle
        # terms are right
        exit_energy = (float(np.asarray(p.energy0)[alive][0]) -
                       self.particle_mass * 1e6) / 1e6        # MeV
        final = self.transform_coordinates(
            xs_out, CoordinateSystem.XSUITE, CoordinateSystem.FELSIM,
            energy=exit_energy)

        twiss = self._calc_twiss(final)
        return SimulationResult(
            simulator_name=self.name, success=True,
            final_particles=final,
            twiss_parameters_statistical={'final': twiss},
            metadata={
                'num_particles': n_p, 'num_good': n_good, 'num_lost': n_lost,
                'beam_energy_mev': self.beam_energy,
                'exit_energy_mev': exit_energy,
                'space_charge': self.space_charge_enabled,
                'sc_method': self.sc_method if self.space_charge_enabled else None,
                'bunch_charge_nc': self.bunch_charge_C * 1e9,
            })

    def _calc_twiss(self, felsim_particles: np.ndarray) -> dict:
        if felsim_particles.shape[0] < 2:
            return {}
        _, _, twiss_df = self._ebeam.cal_twiss(felsim_particles, ddof=1)
        return {axis: twiss_df.loc[axis].to_dict() for axis in twiss_df.index}

    # ----------------------------------------------------------- coordinates
    def transform_coordinates(self, particles, from_system, to_system,
                              energy=None):
        # energy [MeV] sets the reference for the delta / angle conversion;
        # defaults to the injection energy. Pass the exit energy for the output
        # of an accelerating (linac) line.
        from simulatorFactory import CoordinateTransformer
        return CoordinateTransformer.transform(
            particles, from_system, to_system,
            energy if energy is not None else self.beam_energy,
            particle_mass_mev=self.particle_mass)

    # ------------------------------------------------------------- SC config
    def set_space_charge(self, enabled: bool, mesh: tuple = None,
                         method: str = None, bunch_charge_nc: float = None):
        self.space_charge_enabled = enabled
        if mesh is not None:
            self.sc_mesh = tuple(mesh)
        if method is not None:
            self.sc_method = method
        if bunch_charge_nc is not None:
            self.bunch_charge_C = bunch_charge_nc * 1e-9

    def supports_mode(self, mode: SimulationMode) -> bool:
        return mode == SimulationMode.PARTICLE_TRACKING
