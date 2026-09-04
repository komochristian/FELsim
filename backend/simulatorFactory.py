"""
Factory and utilities for creating and managing beamline simulators.

Author: Eremey Valetov
"""

import logging
from typing import Dict, List, Union, Optional
from enum import Enum
import numpy as np

from simulatorBase import SimulatorBase, CoordinateSystem
from felsimAdapter import FELsimAdapter
from cosyAdapter import COSYAdapter
from beamEvolution import BeamEvolution

logger = logging.getLogger(__name__)

# RF-Track is optional - import if available
try:
    from rftrackAdapter import RFTrackAdapter
    _RFTRACK_AVAILABLE = True
except ImportError:
    _RFTRACK_AVAILABLE = False
    RFTrackAdapter = None

# xsuite is optional - import if available
try:
    from xsuiteAdapter import XsuiteAdapter
    _XSUITE_AVAILABLE = True
except ImportError:
    _XSUITE_AVAILABLE = False
    XsuiteAdapter = None


class SimulatorType(Enum):
    """Supported simulator backends."""
    FELSIM = "felsim"
    COSY = "cosy"
    RFTRACK = "rftrack"
    XSUITE = "xsuite"
    MULTICODE = "multicode"


class SimulatorFactory:
    """
    Factory for creating simulator instances.

    Usage:
        sim = SimulatorFactory.create('felsim')
        sim = SimulatorFactory.create('cosy', mode='particle_tracking', config=config)
    """

    _registry: Dict[str, type] = {
        SimulatorType.FELSIM.value: FELsimAdapter,
        SimulatorType.COSY.value: COSYAdapter,
    }

    # Register RF-Track if available
    if _RFTRACK_AVAILABLE:
        _registry[SimulatorType.RFTRACK.value] = RFTrackAdapter

    # Register xsuite if available
    if _XSUITE_AVAILABLE:
        _registry[SimulatorType.XSUITE.value] = XsuiteAdapter

    # Known configuration keys per simulator type
    _KNOWN_KEYS: Dict[str, frozenset] = {
        'felsim': frozenset({
            'lattice_path', 'excel_path', 'debug',
        }),
        'cosy': frozenset({
            'lattice_path', 'excel_path', 'mode', 'config', 'debug',
            'transfer_matrix_order', 'fringe_field_order',
            'use_mge_for_dipoles', 'quad_aperture', 'dipole_aperture',
        }),
        'rftrack': frozenset({
            'lattice_path', 'excel_path', 'mode', 'debug',
            'space_charge', 'sc_mesh', 'beam_energy',
            'particle_mass', 'particle_charge', 'aperture',
            'G_quad', 'dipole_slices', 'rf_frequency',
        }),
        'xsuite': frozenset({
            'lattice_path', 'excel_path', 'mode', 'debug',
            'space_charge', 'sc_mesh', 'sc_method', 'n_slice_sc',
            'bunch_charge_nc', 'beam_energy', 'particle_mass', 'particle_charge',
        }),
        'multicode': frozenset({
            'sections', 'lattice_path', 'beam_energy', 'debug',
        }),
    }

    # Features only supported by specific simulators
    _FEATURE_SUPPORT: Dict[str, frozenset] = {
        'space_charge': frozenset({'rftrack', 'xsuite'}),
        'sc_mesh': frozenset({'rftrack', 'xsuite'}),
        'sc_method': frozenset({'xsuite'}),
        'n_slice_sc': frozenset({'xsuite'}),
        'bunch_charge_nc': frozenset({'xsuite'}),
        'transfer_matrix_order': frozenset({'cosy'}),
        'fringe_field_order': frozenset({'cosy'}),
        'use_mge_for_dipoles': frozenset({'cosy'}),
        'dipole_slices': frozenset({'rftrack'}),
        'G_quad': frozenset({'rftrack'}),
        'rf_frequency': frozenset({'rftrack'}),
    }

    @classmethod
    def _validate_config(cls, sim_type: str, kwargs: dict):
        """Warn on unknown keys and feature/simulator mismatches."""
        known = cls._KNOWN_KEYS.get(sim_type)
        if known is None:
            return

        unknown = set(kwargs) - known
        if unknown:
            logger.warning(
                "%s: unknown config key(s) %s (known: %s)",
                sim_type, sorted(unknown), sorted(known))

        for key, supported_sims in cls._FEATURE_SUPPORT.items():
            if key in kwargs and sim_type not in supported_sims:
                logger.warning(
                    "%s: '%s' is not supported (only %s); setting will be ignored",
                    sim_type, key, ', '.join(sorted(supported_sims)))

    @classmethod
    def create(cls,
               simulator_type: Union[str, SimulatorType],
               **kwargs) -> SimulatorBase:
        """
        Create simulator instance.

        Parameters
        ----------
        simulator_type : str or SimulatorType
            Simulator type ('felsim', 'cosy')
        **kwargs : dict
            Simulator-specific parameters:
            - Common: lattice_path (Excel, JSON, or YAML)
            - FELsim: (no additional parameters)
            - COSY: mode, config, debug
            - RF-Track: space_charge, aperture, G_quad, debug
        """
        if isinstance(simulator_type, SimulatorType):
            sim_type = simulator_type.value
        else:
            sim_type = simulator_type.lower()

        # MultiCodeSimulator uses lazy import to avoid circular dependency
        if sim_type == SimulatorType.MULTICODE.value:
            cls._validate_config(sim_type, kwargs)
            from multiCodeSimulator import MultiCodeSimulator
            try:
                return MultiCodeSimulator(**kwargs)
            except Exception as e:
                raise ValueError(f"Failed to create multicode simulator: {e}") from e

        if sim_type not in cls._registry:
            available = ', '.join(list(cls._registry.keys()) + ['multicode'])
            raise ValueError(f"Unknown simulator '{sim_type}'. Available: {available}")

        cls._validate_config(sim_type, kwargs)

        try:
            return cls._registry[sim_type](**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create {sim_type} simulator: {e}") from e

    @classmethod
    def get_available_simulators(cls) -> List[str]:
        """Return list of available simulator types."""
        return list(cls._registry.keys()) + [SimulatorType.MULTICODE.value]

    @classmethod
    def register_simulator(cls, name: str, simulator_class: type):
        """Register new simulator type. Class must inherit from SimulatorBase."""
        if not issubclass(simulator_class, SimulatorBase):
            raise TypeError(f"{simulator_class} must inherit from SimulatorBase")
        cls._registry[name.lower()] = simulator_class

    @classmethod
    def plot_comparison(cls,
                        simulators: List[SimulatorBase],
                        particles: np.ndarray,
                        **kwargs) -> Dict[str, BeamEvolution]:
        """
        Run and plot simulations from multiple backends for comparison.

        Parameters
        ----------
        simulators : list of SimulatorBase
            Simulator instances to compare
        particles : ndarray
            Initial particles in FELsim coordinates
        **kwargs : dict
            interval (float): for FELsim (default 0.01)
            checkpoint_elements: for COSY (default 'all')

        Returns
        -------
        dict
            {simulator_name: BeamEvolution}
        """
        results = {}
        interval = kwargs.get('interval', 0.01)
        checkpoints = kwargs.get('checkpoint_elements', 'all')

        for sim in simulators:
            if sim.name == "Python":
                evolution = sim.collect_evolution(particles, interval)
            elif sim.name == "COSY":
                evolution = sim.collect_evolution(particles, checkpoints)
            elif sim.name == "RF-Track":
                evolution = sim.collect_evolution(particles, checkpoints)
            else:
                # Generic fallback using collect_evolution if available
                if hasattr(sim, 'collect_evolution'):
                    evolution = sim.collect_evolution(particles, checkpoints)
                else:
                    raise ValueError(f"Unknown simulator: {sim.name}")
            results[sim.name] = evolution

        cls._plot_evolution_comparison(results)
        return results

    @staticmethod
    def _plot_evolution_comparison(evolutions: Dict[str, BeamEvolution]):
        """Plot envelope evolution comparison."""
        import matplotlib.pyplot as plt

        fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Use a color cycle instead of hardcoding
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for idx, (name, evolution) in enumerate(evolutions.items()):
            df = evolution.get_twiss_evolution()
            color = colors[idx % len(colors)]

            ax_x.plot(df['s'], df['envelope_x'], color=color,
                      label=name, linewidth=1.5)
            ax_x.scatter(df['s'], df['envelope_x'], color=color,
                         s=10, alpha=0.5)

            ax_y.plot(df['s'], df['envelope_y'], color=color,
                      label=name, linewidth=1.5)
            ax_y.scatter(df['s'], df['envelope_y'], color=color,
                         s=10, alpha=0.5)

        ax_x.set_ylabel(r'$E_x$ (mm)')
        ax_x.legend()
        ax_x.grid(True, alpha=0.3)

        ax_y.set_ylabel(r'$E_y$ (mm)')
        ax_y.set_xlabel('s (m)')
        ax_y.legend()
        ax_y.grid(True, alpha=0.3)

        plt.suptitle('Simulator Comparison')
        plt.tight_layout()
        plt.show()

    @classmethod
    def get_simulator_info(cls, simulator_type: str) -> Dict:
        """Get static information about a simulator type without instantiation."""
        sim_type = simulator_type.lower()
        if sim_type not in cls._registry:
            raise ValueError(f"Unknown simulator type: {simulator_type}")

        simulator_class = cls._registry[sim_type]
        info = {
            'type': sim_type,
            'class': simulator_class.__name__,
        }

        if hasattr(simulator_class, 'CAPABILITIES'):
            info['capabilities'] = simulator_class.CAPABILITIES
        if hasattr(simulator_class, 'NATIVE_COORDINATES'):
            info['native_coordinates'] = simulator_class.NATIVE_COORDINATES.value

        return info


class CoordinateTransformer:
    """
    Coordinate transformations between simulator coordinate systems.

    Implements FELsim ↔ COSY transforms directly using PhysicalConstants,
    without requiring a simulator instance or lattice file.

    FELsim: [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T_RF×10³, ΔK/K₀×10³]
    COSY:   [x(m), a=px/p0, y(m), b=py/p0, l(m), δK=(K-K0)/K₀]

    Usage:
        particles_cosy = CoordinateTransformer.transform(
            particles_felsim,
            from_system=CoordinateSystem.FELSIM,
            to_system=CoordinateSystem.COSY,
            energy_mev=45.0
        )
    """

    @staticmethod
    def _felsim_to_cosy(particles_felsim: np.ndarray, energy_mev: float) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        E0 = PhysicalConstants.E0_electron
        C = float(PhysicalConstants.C)
        f_RF = PhysicalConstants.f_RF_default

        KE0 = energy_mev
        gamma = 1 + KE0 / E0
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)
        beta0 = p0c / (gamma * E0)
        v0 = beta0 * C
        T_RF = 1.0 / f_RF

        out = np.zeros_like(particles_felsim)

        # Transverse: mm → m
        out[:, 0] = particles_felsim[:, 0] * 1e-3
        out[:, 2] = particles_felsim[:, 2] * 1e-3

        # Angles → momentum ratios (exact 3D decomposition)
        xp_rad = particles_felsim[:, 1] * 1e-3
        yp_rad = particles_felsim[:, 3] * 1e-3
        tan_xp = np.tan(xp_rad)
        tan_yp = np.tan(yp_rad)

        KE_particle = KE0 * (1 + particles_felsim[:, 5] * 1e-3)
        pc = np.sqrt(KE_particle ** 2 + 2 * KE_particle * E0)
        denom = np.sqrt(1 + tan_xp ** 2 + tan_yp ** 2)

        out[:, 1] = pc * tan_xp / (denom * p0c)
        out[:, 3] = pc * tan_yp / (denom * p0c)

        # Longitudinal
        DeltaToF = particles_felsim[:, 4] * 1e-3 * T_RF
        out[:, 4] = -DeltaToF * v0 * gamma / (1 + gamma)
        out[:, 5] = (KE_particle - KE0) / KE0

        return out

    @staticmethod
    def _cosy_to_felsim(particles_cosy: np.ndarray, energy_mev: float) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        E0 = PhysicalConstants.E0_electron
        C = float(PhysicalConstants.C)
        f_RF = PhysicalConstants.f_RF_default

        KE0 = energy_mev
        gamma = 1 + KE0 / E0
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)
        beta0 = p0c / (gamma * E0)
        v0 = beta0 * C
        T_RF = 1.0 / f_RF

        out = np.zeros_like(particles_cosy)

        # Transverse: m → mm
        out[:, 0] = particles_cosy[:, 0] * 1e3
        out[:, 2] = particles_cosy[:, 2] * 1e3

        # Momentum ratios → angles
        a = particles_cosy[:, 1]
        b = particles_cosy[:, 3]
        delta_K = particles_cosy[:, 5]

        KE_particle = KE0 * (1 + delta_K)
        momentum_sq = np.maximum(KE_particle ** 2 + 2 * KE_particle * E0, 0)
        pc = np.sqrt(momentum_sq)
        k = pc / p0c

        discriminant = np.maximum(k ** 2 - a ** 2 - b ** 2, 0)
        pz_norm = np.sqrt(discriminant)

        with np.errstate(divide='ignore', invalid='ignore'):
            out[:, 1] = np.where(pz_norm > 0, np.arctan(a / pz_norm) * 1e3, 0)
            out[:, 3] = np.where(pz_norm > 0, np.arctan(b / pz_norm) * 1e3, 0)

        # Longitudinal
        out[:, 4] = -particles_cosy[:, 4] * (1 + gamma) / (v0 * gamma * T_RF) * 1e3
        out[:, 5] = delta_K * 1e3

        return out

    @staticmethod
    def _felsim_to_rftrack(particles: np.ndarray, energy_mev: float,
                           particle_mass_mev: float = None,
                           rf_frequency_hz: float = None) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        mass = particle_mass_mev or PhysicalConstants.E0_electron
        f_RF = rf_frequency_hz or PhysicalConstants.f_RF_default
        t_scale = PhysicalConstants.C / f_RF

        out = np.zeros_like(particles)
        out[:, 0:4] = particles[:, 0:4]
        out[:, 4] = particles[:, 4] * t_scale
        K = energy_mev * (1.0 + particles[:, 5] * 1e-3)
        E = K + mass
        out[:, 5] = np.sqrt(E**2 - mass**2)
        return out

    @staticmethod
    def _rftrack_to_felsim(particles: np.ndarray, energy_mev: float,
                           particle_mass_mev: float = None,
                           rf_frequency_hz: float = None) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        mass = particle_mass_mev or PhysicalConstants.E0_electron
        f_RF = rf_frequency_hz or PhysicalConstants.f_RF_default
        t_scale = PhysicalConstants.C / f_RF

        out = np.zeros_like(particles)
        out[:, 0:4] = particles[:, 0:4]
        out[:, 4] = particles[:, 4] / t_scale
        K = np.sqrt(particles[:, 5]**2 + mass**2) - mass
        out[:, 5] = (K / energy_mev - 1.0) * 1e3
        return out

    @staticmethod
    def _cosy_to_rftrack(particles: np.ndarray, energy_mev: float,
                         particle_mass_mev: float = None,
                         rf_frequency_hz: float = None) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        mass = particle_mass_mev or PhysicalConstants.E0_electron
        gamma = 1 + energy_mev / mass
        beta = np.sqrt(1 - 1/gamma**2)

        out = np.zeros_like(particles)
        out[:, 0:4] = particles[:, 0:4] * 1e3  # m/rad → mm/mrad
        out[:, 4] = -particles[:, 4] * (1 + gamma) / (beta * gamma) * 1e3
        E0 = energy_mev + mass
        E = E0 + energy_mev * particles[:, 5]
        out[:, 5] = np.sqrt(E**2 - mass**2)
        return out

    @staticmethod
    def _rftrack_to_cosy(particles: np.ndarray, energy_mev: float,
                         particle_mass_mev: float = None,
                         rf_frequency_hz: float = None) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        mass = particle_mass_mev or PhysicalConstants.E0_electron
        gamma = 1 + energy_mev / mass
        beta = np.sqrt(1 - 1/gamma**2)

        out = np.zeros_like(particles)
        out[:, 0:4] = particles[:, 0:4] * 1e-3  # mm/mrad → m/rad
        out[:, 4] = -particles[:, 4] * beta * gamma / ((1 + gamma) * 1e3)
        K = np.sqrt(particles[:, 5]**2 + mass**2) - mass
        out[:, 5] = K / energy_mev - 1.0
        return out

    @staticmethod
    def _felsim_to_xsuite(particles: np.ndarray, energy_mev: float,
                          particle_mass_mev: float = None) -> np.ndarray:
        """FELsim -> xsuite. xsuite: x[m], px=p_x/p0, y[m], py=p_y/p0,
        zeta[m]=-v0*Δt, delta=(p-p0)/p0. Transverse momenta use the same exact
        3D decomposition as the COSY transform; longitudinal is a plain
        path-length offset (no COSY γ/(1+γ) factor)."""
        from physicalConstants import PhysicalConstants
        E0 = particle_mass_mev or PhysicalConstants.E0_electron
        C = float(PhysicalConstants.C)
        T_RF = 1.0 / PhysicalConstants.f_RF_default

        KE0 = energy_mev
        gamma = 1 + KE0 / E0
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)
        v0 = (p0c / (gamma * E0)) * C

        out = np.zeros_like(particles)
        out[:, 0] = particles[:, 0] * 1e-3
        out[:, 2] = particles[:, 2] * 1e-3

        tan_xp = np.tan(particles[:, 1] * 1e-3)
        tan_yp = np.tan(particles[:, 3] * 1e-3)
        KE = KE0 * (1 + particles[:, 5] * 1e-3)
        pc = np.sqrt(KE ** 2 + 2 * KE * E0)
        denom = np.sqrt(1 + tan_xp ** 2 + tan_yp ** 2)
        out[:, 1] = pc * tan_xp / (denom * p0c)
        out[:, 3] = pc * tan_yp / (denom * p0c)
        out[:, 5] = pc / p0c - 1.0

        DeltaToF = particles[:, 4] * 1e-3 * T_RF
        out[:, 4] = -v0 * DeltaToF
        return out

    @staticmethod
    def _xsuite_to_felsim(particles: np.ndarray, energy_mev: float,
                          particle_mass_mev: float = None) -> np.ndarray:
        from physicalConstants import PhysicalConstants
        E0 = particle_mass_mev or PhysicalConstants.E0_electron
        C = float(PhysicalConstants.C)
        T_RF = 1.0 / PhysicalConstants.f_RF_default

        KE0 = energy_mev
        gamma = 1 + KE0 / E0
        p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)
        v0 = (p0c / (gamma * E0)) * C

        out = np.zeros_like(particles)
        out[:, 0] = particles[:, 0] * 1e3
        out[:, 2] = particles[:, 2] * 1e3

        pc = p0c * (1.0 + particles[:, 5])
        KE = np.sqrt(np.maximum(pc ** 2 + E0 ** 2, 0)) - E0
        out[:, 5] = (KE / KE0 - 1.0) * 1e3

        px = particles[:, 1] * p0c
        py = particles[:, 3] * p0c
        pz = np.sqrt(np.maximum(pc ** 2 - px ** 2 - py ** 2, 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            out[:, 1] = np.where(pz > 0, np.arctan(px / pz) * 1e3, 0)
            out[:, 3] = np.where(pz > 0, np.arctan(py / pz) * 1e3, 0)

        out[:, 4] = -particles[:, 4] / v0 / T_RF * 1e3
        return out

    @staticmethod
    def transform(particles: np.ndarray,
                  from_system: CoordinateSystem,
                  to_system: CoordinateSystem,
                  energy_mev: float = 45.0,
                  **kwargs) -> np.ndarray:
        """
        Transform particles between coordinate systems.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Particle distribution
        from_system, to_system : CoordinateSystem
            Source and target coordinate systems
        energy_mev : float
            Beam kinetic energy in MeV
        **kwargs
            particle_mass_mev : float (default: electron mass)
            rf_frequency_hz : float (default: 2856 MHz)
        """
        if from_system == to_system:
            return particles.copy()

        mass = kwargs.get('particle_mass_mev')
        f_RF = kwargs.get('rf_frequency_hz')
        key = (from_system, to_system)

        dispatch = {
            (CoordinateSystem.FELSIM, CoordinateSystem.COSY):
                lambda: CoordinateTransformer._felsim_to_cosy(particles, energy_mev),
            (CoordinateSystem.COSY, CoordinateSystem.FELSIM):
                lambda: CoordinateTransformer._cosy_to_felsim(particles, energy_mev),
            (CoordinateSystem.FELSIM, CoordinateSystem.RFTRACK):
                lambda: CoordinateTransformer._felsim_to_rftrack(particles, energy_mev, mass, f_RF),
            (CoordinateSystem.RFTRACK, CoordinateSystem.FELSIM):
                lambda: CoordinateTransformer._rftrack_to_felsim(particles, energy_mev, mass, f_RF),
            (CoordinateSystem.COSY, CoordinateSystem.RFTRACK):
                lambda: CoordinateTransformer._cosy_to_rftrack(particles, energy_mev, mass, f_RF),
            (CoordinateSystem.RFTRACK, CoordinateSystem.COSY):
                lambda: CoordinateTransformer._rftrack_to_cosy(particles, energy_mev, mass, f_RF),
            (CoordinateSystem.FELSIM, CoordinateSystem.XSUITE):
                lambda: CoordinateTransformer._felsim_to_xsuite(particles, energy_mev, mass),
            (CoordinateSystem.XSUITE, CoordinateSystem.FELSIM):
                lambda: CoordinateTransformer._xsuite_to_felsim(particles, energy_mev, mass),
        }

        if key not in dispatch:
            raise NotImplementedError(
                f"Transformation {from_system.value} → {to_system.value} not implemented"
            )
        return dispatch[key]()

    @staticmethod
    def transform_with_simulators(particles: np.ndarray,
                                  from_simulator: SimulatorBase,
                                  to_simulator: SimulatorBase) -> np.ndarray:
        """Transform particles using simulator instances."""
        from_system = from_simulator.get_native_coordinate_system()
        to_system = to_simulator.get_native_coordinate_system()

        return from_simulator.transform_coordinates(
            particles, from_system=from_system, to_system=to_system
        )

    @staticmethod
    def validate_transformation(num_particles: int = 1000,
                                from_system: CoordinateSystem = CoordinateSystem.FELSIM,
                                to_system: CoordinateSystem = CoordinateSystem.COSY,
                                energy_mev: float = 45.0,
                                tolerance: float = 1e-12) -> Dict:
        """
        Validate round-trip coordinate transformation.

        Returns dict with 'passed' (bool) and error statistics.
        """
        # Generate test particles
        if from_system == CoordinateSystem.FELSIM:
            test_particles = np.random.normal(
                0, [1.0, 0.1, 1.0, 0.1, 5.0, 1.0],
                size=(num_particles, 6)
            )
        elif from_system == CoordinateSystem.COSY:
            test_particles = np.random.normal(
                0, [1e-3, 1e-4, 1e-3, 1e-4, 5e-3, 1e-3],
                size=(num_particles, 6)
            )
        else:
            raise ValueError(f"Validation not implemented for {from_system.value}")

        # Round-trip transformation
        intermediate = CoordinateTransformer.transform(
            test_particles, from_system, to_system, energy_mev
        )
        recovered = CoordinateTransformer.transform(
            intermediate, to_system, from_system, energy_mev
        )

        # Calculate errors
        abs_errors = np.abs(test_particles - recovered)
        rel_errors = abs_errors / (np.abs(test_particles) + 1e-15)

        max_abs = np.max(abs_errors, axis=0)
        max_rel = np.max(rel_errors, axis=0)

        return {
            'passed': np.all(max_abs < tolerance),
            'tolerance': tolerance,
            'from_system': from_system.value,
            'to_system': to_system.value,
            'num_particles': num_particles,
            'max_absolute_errors': max_abs.tolist(),
            'max_relative_errors': max_rel.tolist()
        }


def compare_simulators(simulators: List[SimulatorBase],
                       particles: np.ndarray,
                       coordinate_system: CoordinateSystem,
                       energy_mev: float = 45.0) -> Dict:
    """
    Compare results from multiple simulators on identical initial conditions.

    Parameters
    ----------
    simulators : list of SimulatorBase
        Simulator instances with identical beamlines
    particles : ndarray (N, 6)
        Initial particle distribution
    coordinate_system : CoordinateSystem
        Coordinate system of input particles
    energy_mev : float
        Beam energy in MeV

    Returns
    -------
    dict
        Comparison results with Twiss parameters from each simulator
    """
    comparison = {
        'energy_mev': energy_mev,
        'num_particles': particles.shape[0],
        'coordinate_system': coordinate_system.value,
        'simulators': {}
    }

    for sim in simulators:
        native_system = sim.get_native_coordinate_system()

        # Transform to native coordinates if needed
        if native_system != coordinate_system:
            sim_particles = CoordinateTransformer.transform(
                particles, coordinate_system, native_system, energy_mev
            )
        else:
            sim_particles = particles.copy()

        # Run simulation
        sim.set_beam_energy(energy_mev)
        result = sim.simulate(sim_particles)

        comparison['simulators'][sim.name] = {
            'success': result.success,
            'twiss': result.get_twiss(),
            'metadata': result.metadata
        }

    return comparison


def create_simulator(simulator_type: str = 'felsim', **kwargs) -> SimulatorBase:
    """
    Convenience wrapper for SimulatorFactory.create().

    Parameters
    ----------
    simulator_type : str
        Simulator type ('felsim', 'cosy')
    **kwargs : dict
        Simulator-specific parameters
    """
    return SimulatorFactory.create(simulator_type, **kwargs)