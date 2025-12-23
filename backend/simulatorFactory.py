"""
Factory and utilities for creating and managing beamline simulators.

Author: Eremey Valetov
"""

from typing import Dict, List, Union, Optional
from enum import Enum
import numpy as np

from simulatorBase import SimulatorBase, CoordinateSystem
from felsimAdapter import FELsimAdapter
from cosyAdapter import COSYAdapter
from beamEvolution import BeamEvolution


class SimulatorType(Enum):
    """Supported simulator backends."""
    FELSIM = "felsim"
    COSY = "cosy"


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
            - FELsim: (no special parameters)
            - COSY: excel_path, mode, config, debug
        """
        if isinstance(simulator_type, SimulatorType):
            sim_type = simulator_type.value
        else:
            sim_type = simulator_type.lower()

        if sim_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Unknown simulator '{sim_type}'. Available: {available}")

        try:
            return cls._registry[sim_type](**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create {sim_type} simulator: {e}") from e

    @classmethod
    def get_available_simulators(cls) -> List[str]:
        """Return list of available simulator types."""
        return list(cls._registry.keys())

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
            if sim.name == "FELsim":
                evolution = sim.collect_evolution(particles, interval)
            elif sim.name == "COSY":
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

    Usage:
        particles_cosy = CoordinateTransformer.transform(
            particles_felsim,
            from_system=CoordinateSystem.FELSIM,
            to_system=CoordinateSystem.COSY,
            energy_mev=45.0
        )
    """

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
        """
        if from_system == to_system:
            return particles.copy()

        # Create appropriate simulator for transformation
        if from_system == CoordinateSystem.FELSIM and to_system == CoordinateSystem.COSY:
            cosy_sim = SimulatorFactory.create('cosy', mode='particle_tracking')
            cosy_sim.set_beam_energy(energy_mev)
            native_sim = cosy_sim.get_native_simulator()
            return native_sim.transform_to_cosy_coordinates(particles, energy=energy_mev)

        elif from_system == CoordinateSystem.COSY and to_system == CoordinateSystem.FELSIM:
            cosy_sim = SimulatorFactory.create('cosy', mode='particle_tracking')
            cosy_sim.set_beam_energy(energy_mev)
            native_sim = cosy_sim.get_native_simulator()
            return native_sim.transform_from_cosy_coordinates(particles, energy=energy_mev)

        else:
            raise NotImplementedError(
                f"Transformation {from_system.value} → {to_system.value} not implemented"
            )

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
            'twiss': result.twiss_parameters,
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