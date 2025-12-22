"""
Adapter for COSY INFINITY simulator.

Author: Eremey Valetov
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from simulatorBase import (
    SimulatorBase, SimulationResult, BeamlineElement,
    CoordinateSystem, SimulationMode
)
from beamEvolution import BeamEvolution, ElementInfo
from evolutionPlotter import EvolutionPlotter
from loggingConfig import get_logger_with_fallback

try:
    from cosySimulator import COSYSimulator
    from cosyParticleSimulator import COSYParticleSimulator
    _COSY_AVAILABLE = True
except ImportError:
    _COSY_AVAILABLE = False
    COSYSimulator = None
    COSYParticleSimulator = None


class COSYAdapter(SimulatorBase):
    """Adapter providing unified interface to COSY INFINITY simulator."""

    def __init__(self,
                 excel_path: Optional[str] = None,
                 mode: str = 'transfer_matrix',
                 config: Optional[Dict] = None,
                 debug: bool = None):

        if not _COSY_AVAILABLE:
            raise ImportError("COSY components not available")

        super().__init__(name="COSY", native_coordinates=CoordinateSystem.COSY, debug=None)

        self.logger, self.debug = get_logger_with_fallback(__name__, debug)
        self._config = config or {}
        self._beamline_parsed = False

        if mode == 'transfer_matrix':
            self.simulation_mode = SimulationMode.TRANSFER_MATRIX
            self._native_sim = COSYSimulator(
                excel_path=excel_path or 'dummy.xlsx',
                config_dict=self._config,
                debug=debug
            )
            self._particle_sim = None

        elif mode == 'particle_tracking':
            self.simulation_mode = SimulationMode.PARTICLE_TRACKING
            self._particle_sim = COSYParticleSimulator(
                excel_path=excel_path or 'dummy.xlsx',
                config_dict=self._config,
                debug=debug
            )
            self._native_sim = self._particle_sim

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'transfer_matrix' or 'particle_tracking'")

        self._element_type_map = {
            'DRIFT': 'DRIFT',
            'QUAD_F': 'QPF',
            'QPF': 'QPF',
            'QUAD_D': 'QPD',
            'QPD': 'QPD',
            'DIPOLE': 'DPH',
            'DPH': 'DPH',
            'DIPOLE_WEDGE': 'DPW',
            'DPW': 'DPW'
        }

        self.excel_path = excel_path

        if excel_path and excel_path != 'dummy.xlsx':
            try:
                parsed_elements = self._native_sim.parse_beamline()
                self.logger.debug(f"Parsed {len(parsed_elements)} beamline elements from {excel_path}")
            except FileNotFoundError:
                self.logger.debug(f"Excel file not found: {excel_path}")
            except Exception as e:
                self.logger.warning(f"Could not parse beamline from {excel_path}: {e}")

    def _ensure_beamline_parsed(self):
        """Parse beamline from Excel if not already done."""
        if self._beamline_parsed:
            return

        if hasattr(self._native_sim, 'beamline') and len(self._native_sim.beamline) > 0:
            self._beamline_parsed = True
            return

        if self.excel_path and self.excel_path != 'dummy.xlsx':
            try:
                parsed = self._native_sim.parse_beamline()
                self.logger.debug(f"Auto-parsed {len(parsed)} elements before simulation")
                self._beamline_parsed = True
                return
            except Exception as e:
                raise ValueError(f"Cannot parse beamline from {self.excel_path}: {e}") from e

        raise ValueError("No beamline available. Provide excel_path or call set_beamline()")

    def collect_evolution(self,
                          particles: np.ndarray,
                          checkpoint_elements: Union[str, List[int]] = 'all'
                          ) -> BeamEvolution:
        """Collect beam evolution data at element boundaries."""
        self._ensure_beamline_parsed()
        native_sim = self.get_native_simulator()

        evolution = BeamEvolution(
            simulator_name=self.name,
            num_particles=particles.shape[0],
            beam_energy=self._native_sim.KE
        )

        evolution.s_positions.append(0.0)
        evolution.particles[0.0] = particles.copy()
        evolution.twiss[0.0] = self._calculate_twiss(particles)

        n_elements = len(native_sim.beamline)
        if checkpoint_elements == 'all':
            checkpoint_list = list(range(1, n_elements + 1))
        else:
            checkpoint_list = checkpoint_elements

        particles_cosy = native_sim.transform_to_cosy_coordinates(particles)
        native_sim.enable_particle_tracking(checkpoint_elements=checkpoint_list)
        native_sim.write_particle_file(particles_cosy, format='rray', output_dir='results')

        result = native_sim.run_simulation()
        if result.get('status') != 'success':
            raise RuntimeError(f"COSY simulation failed: {result}")

        s = 0.0
        s_positions_map = {}

        for idx, elem in enumerate(native_sim.beamline):
            elem_length = elem.get('length', 0)
            s_start = s
            s += elem_length
            s_positions_map[idx + 1] = s

            evolution.elements.append(ElementInfo(
                element_type=elem.get('type', 'UNKNOWN'),
                s_start=s_start,
                s_end=s,
                length=elem_length,
                color=self._get_element_color(elem.get('type')),
                index=idx,
                parameters={
                    'current': elem.get('current'),
                    'angle': elem.get('angle')
                }
            ))

        evolution.total_length = s

        checkpoints = native_sim.read_checkpoints(
            checkpoint_list,
            transform_to_felsim=True,
            validate=False,
            filter_invalid=True
        )

        for elem_idx, particles_at_elem in checkpoints.items():
            s_pos = s_positions_map.get(elem_idx, 0)

            if particles_at_elem.shape[0] > 0:
                evolution.s_positions.append(s_pos)
                evolution.particles[s_pos] = particles_at_elem
                evolution.twiss[s_pos] = self._calculate_twiss(particles_at_elem)

        evolution.s_positions = sorted(evolution.s_positions)

        return evolution

    def _calculate_twiss(self, particles_felsim: np.ndarray) -> dict:
        """Calculate Twiss parameters from FELsim-coordinate particles."""
        if self._particle_sim is None:
            from ebeam import beam
            ebeam = beam()
            _, _, twiss_df = ebeam.cal_twiss(particles_felsim, ddof=1)
            return {
                'x': {
                    'beta': twiss_df.loc['x', r'$\beta$ (m)'],
                    'alpha': twiss_df.loc['x', r'$\alpha$'],
                    'gamma': twiss_df.loc['x', r'$\gamma$ (rad/m)'],
                    'emittance': twiss_df.loc['x', r'$\epsilon$ ($\pi$.mm.mrad)']
                },
                'y': {
                    'beta': twiss_df.loc['y', r'$\beta$ (m)'],
                    'alpha': twiss_df.loc['y', r'$\alpha$'],
                    'gamma': twiss_df.loc['y', r'$\gamma$ (rad/m)'],
                    'emittance': twiss_df.loc['y', r'$\epsilon$ ($\pi$.mm.mrad)']
                }
            }

        return self._particle_sim.calculate_twiss_from_particles(particles_felsim)

    def _get_element_color(self, elem_type: str) -> str:
        """Map element type to display color."""
        colors = {
            'DRIFT': 'white',
            'QPF': 'cornflowerblue',
            'QPD': 'lightcoral',
            'DPH': 'forestgreen',
            'DPW': 'lightgreen',
            'DIPOLE_CONSOLIDATED': 'forestgreen'
        }
        return colors.get(elem_type, 'gray')

    def plot_transport(self,
                       particles: np.ndarray,
                       checkpoint_elements: Union[str, List[int]] = 'all',
                       **kwargs) -> BeamEvolution:
        """Simulate and plot beam transport."""
        evolution = self.collect_evolution(particles, checkpoint_elements)
        plotter = EvolutionPlotter()
        plotter.plot(evolution, **kwargs)
        return evolution

    def parse_beamline(self):
        """Parse beamline from Excel file."""
        return self._native_sim.parse_beamline()

    def find_elements(self, element_type=None, **criteria):
        """Find beamline elements matching criteria."""
        self._ensure_beamline_parsed()
        return self._native_sim.find_elements(element_type, **criteria)

    def print_beamline(self):
        """Print beamline elements as formatted table."""
        self._ensure_beamline_parsed()
        return self._native_sim.print_beamline()

    def get_beamline(self):
        """Get parsed beamline elements."""
        self._ensure_beamline_parsed()
        return self._native_sim.beamline

    def apply_variable_mapping(self, xVar, validation=True):
        """Apply variable mappings to beamline elements."""
        self._ensure_beamline_parsed()
        return self._native_sim.apply_variable_mapping(xVar, validation)

    def modify_element(self, index, **kwargs):
        """Modify beamline element parameters."""
        self._ensure_beamline_parsed()
        return self._native_sim.modify_element(index, **kwargs)

    def simulate(self,
                 particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        """Run COSY simulation."""
        self._ensure_beamline_parsed()

        mode = mode or self.simulation_mode

        if mode == SimulationMode.TRANSFER_MATRIX:
            if self._native_sim.particle_tracking_mode:
                old_state = self._native_sim.particle_tracking_mode
                self._native_sim.disable_particle_tracking()

            result_dict = self._native_sim.run_simulation()

            if not result_dict.get('status') == 'success':
                return SimulationResult(
                    simulator_name=self.name,
                    success=False,
                    metadata={'error': result_dict}
                )

            reader = self._native_sim.analyze_results()
            twiss = reader.get_twiss_from_transfer_map()

            return SimulationResult(
                simulator_name=self.name,
                success=True,
                twiss_parameters_transfer_map=twiss,
                optimization_variables=reader.get_variables(),
                metadata={
                    'beam_energy_mev': self._native_sim.KE,
                    'optimization_enabled': reader.optimization_enabled
                },
                transfer_map=reader.read_linear_transfer_map(),
                json_results=reader.read_json_results()
            )

        elif mode == SimulationMode.PARTICLE_TRACKING:
            if self._particle_sim is None:
                raise ValueError("Particle tracking not available. Create COSYAdapter with mode='particle_tracking'")

            if particles is None:
                raise ValueError("particles required for PARTICLE_TRACKING mode")

            if particles.shape[1] != 6:
                raise ValueError(f"Expected 6 coordinates, got {particles.shape[1]}")

            self._particle_sim.write_particle_file(
                particles,
                format='rray',
                output_dir='results'
            )

            result_dict = self._particle_sim.run_simulation()

            if result_dict.get('status') != 'success':
                return SimulationResult(
                    simulator_name=self.name,
                    success=False,
                    metadata={'error': result_dict}
                )

            checkpoint_config = self._particle_sim.get_particle_tracking_config()

            final_particles_cosy = None
            checkpoint_particles = {}

            if checkpoint_config['checkpoint_files']:
                try:
                    last_file = checkpoint_config['checkpoint_files'][-1]
                    final_particles_cosy = self._particle_sim.read_particle_file(
                        filename=last_file.split('/')[-1],
                        format='auto'
                    )
                    self.logger.debug(f"Read final particles from {last_file}")
                except FileNotFoundError as e:
                    self.logger.warning(f"Could not read final particles: {e}")
                except Exception as e:
                    self.logger.error(f"Error reading final particles: {e}")

                if checkpoint_config.get('checkpoint_elements'):
                    try:
                        checkpoint_particles = self._particle_sim.read_checkpoints(
                            checkpoint_config['checkpoint_elements'],
                            transform_to_felsim=False
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not read checkpoint particles: {e}")
            else:
                self.logger.warning("No checkpoint files generated")
                self.logger.warning(f"Checkpoint count: {checkpoint_config['checkpoints_written']}")
                self.logger.warning(f"Checkpoint elements: {checkpoint_config['checkpoint_elements']}")

            reader = self._particle_sim.analyze_results()
            twiss = reader.get_twiss_from_transfer_map()

            if self.debug:
                self._particle_sim.diagnose_particle_distribution(final_particles_cosy, 'cosy')

            self.logger.debug(final_particles_cosy)
            self.logger.debug("\n=== Attempting filtered transformation ===")

            filtered_particles = self._particle_sim.transform_from_cosy_coordinates(
                final_particles_cosy,
                validate=False,
                filter_invalid=True
            )

            if filtered_particles.shape[0] > 0:
                self.logger.debug(f"\nFiltered to {filtered_particles.shape[0]} valid particles")
                twiss_filtered = self._particle_sim.calculate_twiss_from_particles(filtered_particles)
                self.logger.debug("Twiss from filtered particles:")
                self.logger.debug(f"  βx = {twiss_filtered['x']['beta']:.6f} m")
                self.logger.debug(f"  βy = {twiss_filtered['y']['beta']:.6f} m")

            return SimulationResult(
                simulator_name=self.name,
                success=True,
                twiss_parameters_transfer_map=twiss,
                final_particles=final_particles_cosy,
                checkpoint_particles=checkpoint_particles,
                metadata={
                    'num_particles': particles.shape[0],
                    'beam_energy_mev': self._particle_sim.KE,
                    'checkpoint_config': checkpoint_config
                },
                transfer_map=reader.read_linear_transfer_map(),
                json_results=reader.read_json_results(),
                twiss_parameters_statistical={
                    'final': self._particle_sim.calculate_twiss_from_particles(final_particles_cosy)
                },
            )

        else:
            raise NotImplementedError(f"Simulation mode {mode} not implemented for COSY")

    def optimize(self,
                 objectives: Dict,
                 variables: Dict,
                 initial_point: Dict,
                 method: Optional[str] = None,
                 **kwargs) -> SimulationResult:
        """Run COSY optimization using internal FIT command."""
        self._native_sim.set_optimization_initial_point(initial_point, reset=True)
        self._native_sim.apply_variable_mapping(variables)

        if 'optimizer_settings' not in objectives:
            objectives['optimizer_settings'] = {}

        if method is not None:
            algorithm_map = {
                'nelder-mead': 3,
                'powell': 4,
                'conjugate-gradient': 5
            }
            if isinstance(method, str):
                objectives['optimizer_settings']['Nalgorithm'] = algorithm_map.get(
                    method.lower(), 3
                )
            else:
                objectives['optimizer_settings']['Nalgorithm'] = method

        if 'eps' in kwargs:
            objectives['optimizer_settings']['eps'] = kwargs['eps']
        if 'Nmax' in kwargs:
            objectives['optimizer_settings']['Nmax'] = kwargs['Nmax']

        self._native_sim.set_optimization_objectives(objectives, reset=True)

        result_dict = self._native_sim.run_simulation()

        if result_dict.get('status') != 'success':
            return SimulationResult(
                simulator_name=self.name,
                success=False,
                metadata={'error': result_dict}
            )

        reader = self._native_sim.analyze_results()
        twiss = reader.get_twiss_from_transfer_map()
        opt_vars = reader.get_variables()

        return SimulationResult(
            simulator_name=self.name,
            success=True,
            twiss_parameters_transfer_map=twiss,
            optimization_variables=opt_vars,
            metadata={
                'beam_energy_mev': self._native_sim.KE,
                'optimization_enabled': reader.optimization_enabled,
                'objectives': objectives
            }
        )

    def _convert_element_to_native(self, element: BeamlineElement) -> Dict:
        """Convert BeamlineElement to COSY Excel format."""
        elem_type = self._element_type_map.get(
            element.element_type.upper(),
            element.element_type
        )

        elem_dict = {
            'Element': elem_type,
            'z_end': element.length,
            'z_start': 0.0
        }

        if elem_type in ['QPF', 'QPD']:
            elem_dict['Current (A)'] = element.parameters.get('current', 0.0)

        elif elem_type == 'DPH':
            elem_dict['Dipole Angle (deg)'] = element.parameters.get('angle', 0.0)
            elem_dict['Dipole length (m)'] = element.length

        elif elem_type == 'DPW':
            elem_dict['Dipole wedge (deg)'] = element.parameters.get('wedge_angle', 0.0)
            elem_dict['Gap wedge (m)'] = element.length
            elem_dict['Pole gap (m)'] = element.parameters.get('pole_gap', 0.014478)

        return elem_dict

    def transform_coordinates(self,
                              particles: np.ndarray,
                              from_system: CoordinateSystem,
                              to_system: CoordinateSystem) -> np.ndarray:
        """Transform particle coordinates between systems."""
        if from_system == to_system:
            return particles.copy()

        if self._particle_sim is None:
            raise ValueError("Particle simulator required for coordinate transformations")

        if from_system == CoordinateSystem.FELSIM and to_system == CoordinateSystem.COSY:
            return self._particle_sim.transform_to_cosy_coordinates(particles)

        elif from_system == CoordinateSystem.COSY and to_system == CoordinateSystem.FELSIM:
            return self._particle_sim.transform_from_cosy_coordinates(particles)

        else:
            raise NotImplementedError(
                f"Transformation {from_system.value} → {to_system.value} not implemented"
            )

    def get_native_simulator(self) -> COSYSimulator:
        """Get underlying COSYSimulator for direct access."""
        return self._native_sim

    def generate_particles(self,
                           num_particles: int = 1000,
                           distribution_type: str = "gaussian",
                           **parameters) -> np.ndarray:
        """
        Generate particles in COSY coordinates.

        For gaussian: specify either std_dev (COSY coords) or physical parameters
        (energy, epsilon_n, beam_size, bunch_length, energy_spread, energy_chirp).
        For matched: specify target Twiss parameters.
        """
        if self._particle_sim is None:
            raise ValueError("Particle generation requires particle_tracking mode")

        if distribution_type == "gaussian":
            return self._particle_sim.generate_6d_gaussian(
                num_particles=num_particles,
                **parameters
            )
        elif distribution_type == "matched":
            return self._particle_sim.generate_matched_beam(
                num_particles=num_particles,
                **parameters
            )
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

    def enable_particle_checkpoints(self,
                                    checkpoint_elements: Optional[List[int]] = None):
        """Enable particle distribution checkpoints."""
        if self._particle_sim is None:
            raise ValueError("Particle tracking mode required for checkpoints")

        self._particle_sim.enable_particle_tracking(
            checkpoint_elements=checkpoint_elements
        )

    def set_beam_energy(self, energy_mev: float):
        """Set beam energy and update COSY configuration."""
        super().set_beam_energy(energy_mev)
        self._native_sim.update_simulation_config(KE=energy_mev)

    def supports_mode(self, mode: SimulationMode) -> bool:
        """Check if simulation mode is supported."""
        return mode in [SimulationMode.TRANSFER_MATRIX, SimulationMode.PARTICLE_TRACKING]

    def validate_coordinate_transformation(self, **kwargs):
        """Validate round-trip coordinate transformation."""
        if self._particle_sim is None:
            raise ValueError("Particle simulator required for validation")

        return self._particle_sim.validate_coordinate_transformation(**kwargs)