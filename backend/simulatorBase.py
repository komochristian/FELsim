"""
Abstract base class for beamline simulators.

Provides unified interface for different simulation codes (FELsim, COSY, etc.)
while handling coordinate system transformations and element representations.

Author: Eremey Valetov
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np


class CoordinateSystem(Enum):
    """Supported coordinate systems for particle distributions."""
    FELSIM = "felsim"  # [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T(10^-3), δW/W(10^-3)]
    COSY = "cosy"  # [x(m), a, y(m), b, l(m), δK]
    ELEGANT = "elegant"  # [x(m), x'(rad), y(m), y'(rad), t(s), δ]


class SimulationMode(Enum):
    """Types of simulation that can be performed."""
    TRANSFER_MATRIX = "transfer_matrix"
    PARTICLE_TRACKING = "particle_tracking"
    ENVELOPE = "envelope"


class BeamlineElement:
    """Generic beamline element representation convertible to code-specific formats."""

    def __init__(self, element_type: str, length: float, **parameters):
        """
        Initialise beamline element.

        Parameters
        ----------
        element_type : str
            Element type (DRIFT, QUAD_F, QUAD_D, DIPOLE, etc.)
        length : float
            Physical length in metres
        **parameters : dict
            Element-specific parameters (current, angle, etc.)
        """
        self.element_type = element_type
        self.length = length
        self.parameters = parameters

    def __repr__(self):
        try:
            params_str = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        except Exception:
            params_str = f"<{len(self.parameters)} parameters>"
        return f"{self.element_type}(L={self.length}, {params_str})"


class SimulationResult:
    """Container for simulation results with metadata."""

    def __init__(self,
                 simulator_name: str,
                 success: bool,
                 transfer_map: Optional[np.ndarray] = None,
                 json_results: dict = None,
                 twiss_parameters_transfer_map: Optional[Dict] = None,
                 twiss_parameters_statistical: Optional[Dict] = None,
                 final_particles: Optional[np.ndarray] = None,
                 checkpoint_particles: Optional[Dict[int, np.ndarray]] = None,
                 optimization_variables: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialise simulation result.

        Parameters
        ----------
        simulator_name : str
            Name of simulator that produced results
        success : bool
            Whether simulation completed successfully
        transfer_map : ndarray, optional
            6×6 transfer matrix
        json_results : dict, optional
            Raw JSON output from simulator
        twiss_parameters_transfer_map : dict, optional
            Twiss from transfer map analysis
        twiss_parameters_statistical : dict, optional
            Twiss from statistical particle analysis
        final_particles : ndarray, optional
            Final particle distribution (N, 6)
        checkpoint_particles : dict, optional
            Distributions at checkpoints {element_idx: particles}
        optimization_variables : dict, optional
            Optimised values {var_name: value}
        metadata : dict, optional
            Runtime, convergence, etc.
        """
        self.simulator_name = simulator_name
        self.success = success
        self.transfer_map = transfer_map
        self.json_results = json_results
        self.twiss_parameters_transfer_map = twiss_parameters_transfer_map
        self.twiss_parameters_statistical = twiss_parameters_statistical
        self.final_particles = final_particles
        self.checkpoint_particles = checkpoint_particles or {}
        self.optimization_variables = optimization_variables or {}
        self.metadata = metadata or {}

    def get_twiss(self, element_idx: Optional[int] = None, source: str = 'statistical') -> Dict:
        """
        Get Twiss parameters at specific element or final values.

        Parameters
        ----------
        element_idx : int, optional
            Element index for checkpoint data
        source : str
            'statistical' for particle-based Twiss, 'transfer_map' for matrix-based

        Returns
        -------
        dict
            Twiss parameters
        """
        if source == 'transfer_map':
            return self.twiss_parameters_transfer_map or {}

        twiss = self.twiss_parameters_statistical or {}

        if element_idx is not None and 'checkpoints' in twiss:
            return twiss['checkpoints'].get(element_idx, {})

        return twiss.get('final', twiss)

    def get_particles(self, element_idx: Optional[int] = None,
                      coordinate_system: Optional[CoordinateSystem] = None) -> np.ndarray:
        """Get particle distribution at checkpoint or final position."""
        if element_idx is not None:
            particles = self.checkpoint_particles.get(element_idx)
        else:
            particles = self.final_particles

        # Coordinate transformation not yet implemented
        if coordinate_system is not None:
            raise NotImplementedError("Coordinate system conversion not yet implemented")

        return particles


class SimulatorBase(ABC):
    """
    Abstract base class for beamline simulators.

    Defines unified interface for different simulation codes whilst handling
    their specific requirements and coordinate systems.
    """

    def __init__(self, name: str, native_coordinates: CoordinateSystem, debug: bool = None):
        self.name = name
        self.native_coordinates = native_coordinates
        self.beamline: List[BeamlineElement] = []
        self.beam_energy = 45.0  # MeV
        self.simulation_mode = SimulationMode.TRANSFER_MATRIX

    # Core abstract methods - must be implemented by subclasses

    @abstractmethod
    def simulate(self,
                 particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        """
        Run simulation.

        Parameters
        ----------
        particles : ndarray, optional
            Initial distribution (N, 6) in native coordinates. Required for PARTICLE_TRACKING.
        mode : SimulationMode, optional
            Override default simulation mode

        Returns
        -------
        SimulationResult
        """
        pass

    @abstractmethod
    def optimize(self,
                 objectives: Dict,
                 variables: Dict,
                 initial_point: Dict,
                 method: Optional[str] = None,
                 **kwargs) -> SimulationResult:
        """
        Run optimization.

        Parameters
        ----------
        objectives : dict
            {element_idx: [objective_dicts]}
        variables : dict
            {element_idx: {param: var_name}}
        initial_point : dict
            {var_name: {'start': value, 'bounds': (min, max)}}
        method : str, optional
            Optimizer (code-specific)
        **kwargs
            Additional optimizer parameters

        Returns
        -------
        SimulationResult
        """
        pass

    @abstractmethod
    def _convert_element_to_native(self, element: BeamlineElement) -> Any:
        """Convert generic BeamlineElement to code-specific format."""
        pass

    @abstractmethod
    def transform_coordinates(self,
                              particles: np.ndarray,
                              from_system: CoordinateSystem,
                              to_system: CoordinateSystem) -> np.ndarray:
        """Transform particle coordinates between systems."""
        pass

    @staticmethod
    def validate_particles(particles: np.ndarray) -> None:
        """Validate particle array shape and values."""
        if particles.ndim != 2:
            raise ValueError(f"Expected 2D array, got {particles.ndim}D")
        if particles.shape[1] != 6:
            raise ValueError(f"Expected 6 coordinates, got {particles.shape[1]}")
        if np.any(np.isnan(particles)):
            raise ValueError("Particle array contains NaN values")
        if np.any(np.isinf(particles)):
            raise ValueError("Particle array contains infinite values")

    # Common interface methods

    def set_beamline(self, elements: List[Union[BeamlineElement, Any]]):
        """Set beamline from generic or code-specific elements."""
        self.beamline = []
        for elem in elements:
            if isinstance(elem, BeamlineElement):
                self.beamline.append(elem)
            else:
                # Assume native element, convert to generic
                self.beamline.append(self._convert_element_from_native(elem))

    def _convert_element_from_native(self, native_element: Any) -> BeamlineElement:
        """Convert code-specific element to generic BeamlineElement."""
        try:
            return BeamlineElement(
                element_type=type(native_element).__name__,
                length=getattr(native_element, 'length', 0.0),
                **{k: v for k, v in vars(native_element).items()
                   if not k.startswith('_')}
            )
        except Exception as e:
            raise NotImplementedError(
                f"Must implement _convert_element_from_native for {self.name}"
            ) from e

    def set_beam_energy(self, energy_mev: float):
        """Set beam kinetic energy in MeV."""
        self.beam_energy = energy_mev

    def set_simulation_mode(self, mode: SimulationMode):
        """Set simulation mode."""
        self.simulation_mode = mode

    def generate_particles(self,
                           num_particles: int = 1000,
                           distribution_type: str = "gaussian",
                           **parameters) -> np.ndarray:
        """
        Generate initial particle distribution.

        Parameters
        ----------
        num_particles : int
            Number of particles
        distribution_type : str
            "gaussian" or "matched"
        **parameters : dict
            For Gaussian - choose one interface:

            Interface 1 (direct):
                std_dev: [σx(mm), σx'(mrad), σy(mm), σy'(mrad), σΔt/T, σδW/W]
                mean: scalar (default 0)

            Interface 2 (beam physics):
                epsilon_n: normalized emittance [π·mm·mrad] (scalar or tuple)
                beam_size: RMS size [mm] (scalar or tuple)
                bunch_length: RMS length [ps]
                energy_spread: RMS spread [%]
                energy_chirp: chirp [1/s] (default 0)

            For matched:
                twiss_x: dict with β, α, ε
                twiss_y: dict with β, α, ε

            Common:
                energy: beam energy [MeV] (defaults to self.beam_energy)

        Returns
        -------
        ndarray (N, 6)
            Particles in FELsim coordinates
        """
        if distribution_type == "gaussian":
            energy = parameters.pop('energy', None)

            # Use COSYParticleSimulator if available for sophisticated generation
            if hasattr(self, '_particle_sim') and self._particle_sim is not None:
                particles = self._particle_sim.generate_6d_gaussian(
                    num_particles=num_particles,
                    energy=energy,
                    **parameters
                )
            else:
                # Basic fallback
                std_dev = parameters.get('std_dev', [0.3, 0.03, 0.3, 0.03, 2.0, 0.3])
                mean = parameters.get('mean', 0.0)
                particles = np.random.normal(mean, std_dev, size=(num_particles, 6))

            return particles

        elif distribution_type == "matched":
            if not hasattr(self, '_particle_sim'):
                raise NotImplementedError("Matched beam requires COSYParticleSimulator")

            twiss_x = parameters.get('twiss_x')
            twiss_y = parameters.get('twiss_y')
            energy = parameters.get('energy')

            if twiss_x is None or twiss_y is None:
                raise ValueError("Matched beam requires 'twiss_x' and 'twiss_y' parameters")

            return self._particle_sim.generate_matched_beam(
                twiss_x=twiss_x,
                twiss_y=twiss_y,
                num_particles=num_particles,
                energy=energy
            )

        else:
            raise NotImplementedError(
                f"Distribution type '{distribution_type}' not implemented"
            )

    def get_native_coordinate_system(self) -> CoordinateSystem:
        """Get this simulator's native coordinate system."""
        return self.native_coordinates

    def supports_mode(self, mode: SimulationMode) -> bool:
        """Check if simulator supports given mode. Override in subclasses."""
        return True

    def supports_optimization(self) -> bool:
        """Check if simulator supports optimization. Override in subclasses."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get simulator capabilities and features."""
        return {
            'name': self.name,
            'native_coordinates': self.native_coordinates.value,
            'supports_transfer_matrix': self.supports_mode(SimulationMode.TRANSFER_MATRIX),
            'supports_particle_tracking': self.supports_mode(SimulationMode.PARTICLE_TRACKING),
            'supports_optimization': self.supports_optimization(),
            'beamline_length': len(self.beamline),
            'beam_energy': self.beam_energy
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"elements={len(self.beamline)}, "
                f"energy={self.beam_energy} MeV)")