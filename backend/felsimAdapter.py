"""
FELsim adapter for unified simulator interface.

Wraps existing FELsim beamline classes to provide SimulatorBase interface
whilst maintaining full backwards compatibility.

Author: Eremey Valetov
"""

import numpy as np
from typing import Dict, List, Optional, Any
from simulatorBase import (
    SimulatorBase, SimulationResult, BeamlineElement,
    CoordinateSystem, SimulationMode
)
from beamEvolution import BeamEvolution, ElementInfo
from evolutionPlotter import EvolutionPlotter
from beamPropagator import propagate
import latticeLoader
from beamline import driftLattice, qpfLattice, qpdLattice, dipole, dipole_wedge
from ebeam import beam as ebeam_class
from beamOptimizer import beamOptimizer


class FELsimAdapter(SimulatorBase):
    """
    Adapter for legacy FELsim simulator.

    Provides unified interface whilst maintaining backwards compatibility.
    Use get_native_beamline() for direct access to FELsim objects.
    """

    def __init__(self, lattice_path: Optional[str] = None,
                 excel_path: Optional[str] = None, **kwargs):
        super().__init__(name="Python", native_coordinates=CoordinateSystem.FELSIM)

        self.simulation_mode = SimulationMode.PARTICLE_TRACKING  # Only mode supported
        self._native_beamline: List[Any] = []
        self._ebeam = ebeam_class()
        self._optimizer: Optional[beamOptimizer] = None

        # Map generic types to FELsim classes
        self._elem_map = {
            'DRIFT': driftLattice,
            'QUAD_F': qpfLattice, 'QPF': qpfLattice,
            'QUAD_D': qpdLattice, 'QPD': qpdLattice,
            'DIPOLE': dipole, 'DPH': dipole,
            'DIPOLE_WEDGE': dipole_wedge, 'DPW': dipole_wedge
        }

        path = lattice_path or excel_path
        if path:
            self._load_lattice(path)

    # --- Core interface ---

    def simulate(self, particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        """
        Run particle tracking through beamline.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Initial distribution in FELsim coordinates:
            [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T(10^-3), ΔK/K₀(10^-3)]
        mode : SimulationMode, optional
            Only PARTICLE_TRACKING supported

        Returns
        -------
        SimulationResult
            Final particles and Twiss parameters
        """
        if mode and mode != SimulationMode.PARTICLE_TRACKING:
            raise NotImplementedError("Only particle tracking supported")

        if particles is None:
            raise ValueError("particles required")
        if not self._native_beamline:
            raise ValueError("Beamline not set")

        # Track through beamline
        current = particles.copy()
        for seg in self._native_beamline:
            current = np.array(seg.useMatrice(current))

        # Calculate final Twiss
        _, _, twiss_df = self._ebeam.cal_twiss(current, ddof=1)
        twiss_dict = {axis: twiss_df.loc[axis].to_dict() for axis in twiss_df.index}

        return SimulationResult(
            simulator_name=self.name,
            success=True,
            twiss_parameters_statistical={'final': twiss_dict},
            final_particles=current,
            metadata={
                'num_particles': particles.shape[0],
                'num_elements': len(self._native_beamline),
                'beam_energy_mev': self.beam_energy
            }
        )

    def optimize(self, objectives: Dict, variables: Dict, initial_point: Dict,
                 method: Optional[str] = None, **kwargs) -> SimulationResult:
        """
        Run beamline optimisation using FELsim's beamOptimizer.

        Parameters
        ----------
        objectives : dict
            {elem_idx: [{"measure": [axis, param], "goal": val, "weight": val}]}
        variables : dict
            {elem_idx: [var_name, param_name, transform_func]}
        initial_point : dict
            {var_name: {"start": val, "bounds": (min, max)}}
        method : str
            Optimisation method (default: 'Nelder-Mead')
        **kwargs
            particles : initial distribution (required)
            plot_progress, plot_beam, print_results : bool
        """
        particles = kwargs.get('particles')
        if particles is None:
            raise ValueError("particles required")

        method = method or 'Nelder-Mead'

        self._optimizer = beamOptimizer(self._native_beamline, particles)

        result = self._optimizer.calc(
            method=method,
            segmentVar=variables,
            startPoint=initial_point,
            objectives=objectives,
            plotProgress=kwargs.get('plot_progress', False),
            plotBeam=kwargs.get('plot_beam', False),
            printResults=kwargs.get('print_results', False)
        )

        # Extract optimised values
        opt_vars = {}
        for elem_idx, var_info in variables.items():
            var_name, param_name, transform = var_info
            var_idx = self._optimizer.variablesToOptimize.index(var_name)
            opt_vars[var_name] = transform(result.x[var_idx])

        # Get final state
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
                'num_evaluations': getattr(result, 'nfev', None),
                **final.metadata
            }
        )

    def _load_lattice(self, lattice_path: str):
        native = latticeLoader.create_beamline(lattice_path)

        for elem in native:
            elem.setE(self.beam_energy)

        self._native_beamline = native
        self.beamline = [self._convert_element_from_native(e) for e in native]

    def _convert_element_to_native(self, elem: BeamlineElement) -> Any:
        elem_type = elem.element_type.upper()

        if elem_type not in self._elem_map:
            raise ValueError(f"Unknown element type: {elem_type}")

        lattice_cls = self._elem_map[elem_type]
        params = elem.parameters

        if elem_type == 'DRIFT':
            return lattice_cls(elem.length)

        elif elem_type in ['QUAD_F', 'QPF', 'QUAD_D', 'QPD']:
            return lattice_cls(
                current=params.get('current', 0.0),
                length=elem.length,
                fringeType=params.get('fringe_type')
            )

        elif elem_type in ['DIPOLE', 'DPH']:
            return lattice_cls(
                length=elem.length,
                angle=params.get('angle', 0.0),
                fringeType=params.get('fringe_type')
            )

        elif elem_type in ['DIPOLE_WEDGE', 'DPW']:
            return dipole_wedge(
                length=elem.length,
                angle=params.get('wedge_angle', 0.0),
                dipole_length=params.get('dipole_length', 0.0889),
                dipole_angle=params.get('dipole_angle', 1.5),
                pole_gap=params.get('pole_gap', 0.014478),
                enge_fct=params.get('enge_fct', 0),
                fringeType=params.get('fringe_type', 'decay')
            )

        raise ValueError(f"Conversion not implemented for {elem_type}")

    def _convert_element_from_native(self, native_elem: Any) -> BeamlineElement:
        cls_name = type(native_elem).__name__

        type_map = {
            'driftLattice': 'DRIFT',
            'qpfLattice': 'QUAD_F',
            'qpdLattice': 'QUAD_D',
            'dipole': 'DIPOLE',
            'dipole_wedge': 'DIPOLE_WEDGE'
        }

        elem_type = type_map.get(cls_name, cls_name)

        params = {}
        if hasattr(native_elem, 'current'):
            params['current'] = native_elem.current
        if hasattr(native_elem, 'angle'):
            params['angle'] = native_elem.angle
        if hasattr(native_elem, 'fringeType'):
            params['fringe_type'] = native_elem.fringeType

        return BeamlineElement(
            element_type=elem_type,
            length=native_elem.length,
            **params
        )

    def transform_coordinates(self, particles: np.ndarray,
                              from_system: CoordinateSystem,
                              to_system: CoordinateSystem) -> np.ndarray:
        if from_system == to_system:
            return particles.copy()

        # COSY transformations not yet integrated - need to coordinate with
        # COSYParticleSimulator for consistent conversion
        raise NotImplementedError(
            f"Coordinate transformation {from_system.value} → {to_system.value}"
        )

    def collect_evolution(self, particles: np.ndarray,
                          interval: float = 0.01) -> BeamEvolution:
        """
        Collect beam evolution at specified interval.

        Uses beamPropagator for efficient stepping through beamline.
        """
        if not self._native_beamline:
            raise ValueError("Beamline not set")

        evolution = BeamEvolution(
            simulator_name=self.name,
            num_particles=particles.shape[0],
            beam_energy=self.beam_energy
        )

        for cp in propagate(self._native_beamline, particles, interval, rounding=2):
            evolution.s_positions.append(cp.s)
            evolution.particles[cp.s] = cp.particles
            evolution.twiss[cp.s] = self._calc_twiss(cp.particles)

            # Record element boundaries for plotting
            if cp.is_element_boundary and cp.element is not None:
                evolution.elements.append(ElementInfo(
                    element_type=type(cp.element).__name__,
                    s_start=cp.s - cp.element.length,
                    s_end=cp.s,
                    length=cp.element.length,
                    color=getattr(cp.element, 'color', 'gray'),
                    index=cp.element_index,
                    parameters={'current': getattr(cp.element, 'current', None)}
                ))

        evolution.total_length = evolution.s_positions[-1] if evolution.s_positions else 0.0
        return evolution

    def _calc_twiss(self, particles: np.ndarray) -> dict:
        _, _, twiss_df = self._ebeam.cal_twiss(particles, ddof=1)

        # Extract both planes
        twiss = {}
        for plane in ['x', 'y']:
            twiss[plane] = {
                'beta': twiss_df.loc[plane, r'$\beta$ (m)'],
                'alpha': twiss_df.loc[plane, r'$\alpha$'],
                'gamma': twiss_df.loc[plane, r'$\gamma$ (rad/m)'],
                'emittance': twiss_df.loc[plane, r'$\epsilon$ ($\pi$.mm.mrad)'],
                'dispersion': twiss_df.loc[plane, r'$D$ (m)'],
                'dispersion_prime': twiss_df.loc[plane, r"$D^{\prime}$"]
            }
        return twiss

    def plot_transport(self, particles: np.ndarray, interval: float = 0.01,
                       **kwargs) -> BeamEvolution:
        evolution = self.collect_evolution(particles, interval)
        plotter = EvolutionPlotter()
        plotter.plot(evolution, **kwargs)
        return evolution

    # FELsim-specific interface

    def set_beamline(self, elements: List):
        if not elements:
            self.beamline = []
            self._native_beamline = []
            return

        # Check if already FELsim objects
        if hasattr(elements[0], 'useMatrice'):
            self._native_beamline = list(elements)
            self.beamline = [self._convert_element_from_native(e) for e in elements]
            for elem in self._native_beamline:
                elem.setE(self.beam_energy)
        else:
            super().set_beamline(elements)
            self._native_beamline = [self._convert_element_to_native(e) for e in self.beamline]
            for elem in self._native_beamline:
                elem.setE(self.beam_energy)

    def get_native_beamline(self) -> List:
        """Get native FELsim beamline for legacy code compatibility."""
        return self._native_beamline

    def set_beam_energy(self, energy_mev: float):
        super().set_beam_energy(energy_mev)
        for elem in self._native_beamline:
            elem.setE(energy_mev)

    def change_beam_type(self, particle_type: str, kinetic_energy: float):
        """
        Change particle type using FELsim's native method.

        Parameters
        ----------
        particle_type : str
            'electron', 'proton', or isotope format 'A,Z' (e.g. '12,6' for C12)
        kinetic_energy : float
            Kinetic energy in MeV
        """
        if self._native_beamline:
            self._native_beamline[0].changeBeamType(
                particle_type, kinetic_energy, self._native_beamline
            )
        self.beam_energy = kinetic_energy

    def generate_particles(self, num_particles: int = 1000,
                           distribution_type: str = "gaussian",
                           **params) -> np.ndarray:
        if distribution_type != "gaussian":
            raise NotImplementedError("Only Gaussian distributions supported")

        mean = params.get('mean', 0)
        std_dev = params.get('std_dev', [1.0, 0.1, 1.0, 0.1, 1.0, 0.1])

        return self._ebeam.gen_6d_gaussian(mean, std_dev, num_particles)

    def supports_mode(self, mode: SimulationMode) -> bool:
        return mode == SimulationMode.PARTICLE_TRACKING

    def get_ebeam_instance(self) -> ebeam_class:
        """Get underlying ebeam instance for direct Twiss calculations."""
        return self._ebeam

    def get_optimizer_instance(self) -> Optional[beamOptimizer]:
        """Get beamOptimizer (available after optimize() is called)."""
        return self._optimizer