"""
Multi-code beamline simulator orchestrator.

Runs different simulation backends on different beamline sections,
chaining them in sequence. All adapters accept and return FELsim
coordinates, so no inter-section coordinate transforms are needed:
each adapter handles its own internal transforms.

Author: Eremey Valetov
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from simulatorBase import (
    SimulatorBase, SimulationResult, BeamlineElement,
    CoordinateSystem, SimulationMode
)

logger = logging.getLogger(__name__)

# Config keys applied per-call via setter methods (not at construction time)
_RUNTIME_CONFIG_KEYS = frozenset({
    'space_charge', 'sc_mesh', 'sc_method',
    'physical_apertures', 'dipole_slices', 'aperture',
})

# All known per-section config keys (runtime + creation-time)
_ALL_SECTION_CONFIG_KEYS = _RUNTIME_CONFIG_KEYS | frozenset({
    'G_quad', 'particle_mass', 'particle_charge', 'rf_frequency',
    'transfer_matrix_order', 'fringe_field_order', 'use_mge_for_dipoles',
    'quad_aperture', 'dipole_aperture', 'mode', 'beam_energy',
})

# FELsim class name → generic element type
_FELSIM_TYPE_MAP = {
    'driftLattice': 'DRIFT',
    'qpfLattice': 'QUAD_F',
    'qpdLattice': 'QUAD_D',
    'dipole': 'DIPOLE',
    'dipole_wedge': 'DIPOLE_WEDGE',
    'alphaMagnetLattice': 'ALPHA_MAGNET',
    'rfCavityLattice': 'RF_CAVITY',
}

# Attributes to copy for each element type
_PARAM_KEYS = {
    'DRIFT': (),
    'QUAD_F': ('current',),
    'QUAD_D': ('current',),
    'DIPOLE': ('angle',),
    'DIPOLE_WEDGE': ('angle', 'dipole_length', 'dipole_angle', 'pole_gap'),
    'ALPHA_MAGNET': ('current', 'gradient_per_amp'),
    'RF_CAVITY': ('frequency_hz', 'phase_deg', 'voltage_mv',
                  'gradient_mv_per_m', 'structure_type',
                  'phase_advance_deg', 'n_cells'),
}


def _felsim_to_generic(elem) -> BeamlineElement:
    """Convert a FELsim native element to a generic BeamlineElement."""
    cls_name = type(elem).__name__
    elem_type = _FELSIM_TYPE_MAP.get(cls_name, cls_name.upper())
    keys = _PARAM_KEYS.get(elem_type, ())
    params = {k: getattr(elem, k) for k in keys if hasattr(elem, k)}
    if hasattr(elem, 'fringeType'):
        params['fringe_type'] = elem.fringeType
    return BeamlineElement(element_type=elem_type, length=elem.length, **params)


@dataclass
class SimSection:
    """One section of a multi-code simulation."""
    name: str
    simulator_key: str
    element_range: tuple  # (start_idx, end_idx), half-open
    config: dict = field(default_factory=dict)


class MultiCodeSimulator(SimulatorBase):
    """
    Orchestrator that chains multiple SimulatorBase instances on
    contiguous beamline sections.

    All adapters (FELsim, COSY, RF-Track) accept and return particles
    in FELsim coordinates; each handles its own internal coordinate
    transforms. The orchestrator simply passes FELsim-format particles
    from one section's output to the next section's input.

    Usage::

        mc = MultiCodeSimulator(sections=[
            SimSection('prefix', 'felsim', (0, 87)),
            SimSection('suffix', 'rftrack', (87, 118)),
        ], lattice_path='var/UH_FEL_beamline.json')
        result = mc.simulate(particles)
    """

    def __init__(self,
                 sections: List[SimSection],
                 lattice_path: Optional[str] = None,
                 beam_energy: float = 45.0,
                 debug: bool = None,
                 **kwargs):
        super().__init__(
            name="MultiCode",
            native_coordinates=CoordinateSystem.FELSIM,
            debug=debug,
        )
        self.beam_energy = beam_energy
        self.sections = sections
        self._simulators: Dict[str, SimulatorBase] = {}
        self._master_beamline: List[Any] = []
        self._lattice_path = lattice_path
        self._extra_kwargs = kwargs

        if lattice_path:
            self._load_master_beamline(lattice_path)
            self._init_simulators()

    def _load_master_beamline(self, lattice_path: str):
        import latticeLoader
        self._master_beamline = latticeLoader.create_beamline(lattice_path)
        for elem in self._master_beamline:
            elem.setE(self.beam_energy)

    def _init_simulators(self):
        from simulatorFactory import SimulatorFactory

        for section in self.sections:
            cfg = section.config or {}

            # Warn on unknown config keys
            unknown = set(cfg) - _ALL_SECTION_CONFIG_KEYS
            if unknown:
                logger.warning(
                    "Section '%s' (%s): unknown config key(s) %s",
                    section.name, section.simulator_key, sorted(unknown))

            creation_cfg = {k: v for k, v in cfg.items()
                           if k not in _RUNTIME_CONFIG_KEYS}
            cache_key = (section.simulator_key,
                         tuple(sorted(creation_cfg.items())))

            if cache_key not in self._simulators:
                sim = SimulatorFactory.create(
                    section.simulator_key, **creation_cfg)
                sim.set_beam_energy(self.beam_energy)
                self._simulators[cache_key] = sim

            section._sim_cache_key = cache_key

    def _prepare_beamline_slice(self, sim: SimulatorBase,
                                start: int, end: int) -> List:
        """Get beamline slice in the format the target simulator expects."""
        raw_slice = self._master_beamline[start:end]
        if not raw_slice:
            return raw_slice

        is_felsim_native = hasattr(raw_slice[0], 'useMatrice')
        sim_is_felsim = sim.native_coordinates == CoordinateSystem.FELSIM

        if is_felsim_native and sim_is_felsim:
            return raw_slice
        elif is_felsim_native:
            # Non-FELsim adapter needs generic BeamlineElement objects
            return [_felsim_to_generic(e) for e in raw_slice]
        else:
            return raw_slice

    @staticmethod
    def _apply_section_config(sim: SimulatorBase, config: dict):
        """Apply per-section runtime config to a simulator before tracking."""
        if not config:
            return

        if 'space_charge' in config and hasattr(sim, 'set_space_charge'):
            sc_kwargs = {}
            if 'sc_mesh' in config:
                sc_kwargs['mesh'] = config['sc_mesh']
            if 'sc_method' in config:
                sc_kwargs['method'] = config['sc_method']
            sim.set_space_charge(config['space_charge'], **sc_kwargs)
        elif 'sc_method' in config and hasattr(sim, '_sc_method'):
            sim._sc_method = config['sc_method']

        if 'dipole_slices' in config and hasattr(sim, 'dipole_slices'):
            sim.dipole_slices = config['dipole_slices']

        if 'physical_apertures' in config:
            if config['physical_apertures']:
                if hasattr(sim, 'enable_physical_apertures'):
                    sim.enable_physical_apertures()
            else:
                if hasattr(sim, 'disable_physical_apertures'):
                    sim.disable_physical_apertures()

        if 'aperture' in config and hasattr(sim, 'default_aperture'):
            sim.default_aperture = config['aperture']

    def simulate(self,
                 particles: Optional[np.ndarray] = None,
                 mode: Optional[SimulationMode] = None) -> SimulationResult:
        """
        Run multi-code simulation: track particles through each section
        in order.

        All adapters accept and return FELsim coordinates, so particles
        are passed directly between sections without coordinate transforms.

        Parameters
        ----------
        particles : ndarray (N, 6)
            Initial distribution in FELsim coordinates.
        mode : SimulationMode, optional
            Ignored (each section uses its own mode).
        """
        if particles is None:
            raise ValueError("particles required")
        if not self.sections:
            raise ValueError("No sections configured")

        self.validate_particles(particles)

        current_particles = particles.copy()
        all_checkpoints = {}
        section_metadata = []

        for i, section in enumerate(self.sections):
            sim = self._simulators[section._sim_cache_key]

            # Apply per-section runtime config before tracking
            self._apply_section_config(sim, section.config)

            # Set beamline slice for this section
            start, end = section.element_range
            bl_slice = self._prepare_beamline_slice(sim, start, end)
            if bl_slice and hasattr(sim, 'set_beamline'):
                sim.set_beamline(bl_slice)

            result = sim.simulate(particles=current_particles)

            if not result.success:
                return SimulationResult(
                    simulator_name=self.name,
                    success=False,
                    metadata={'failed_section': section.name,
                              'section_index': i}
                )

            # Collect checkpoint particles with global element indices
            for local_idx, cp_particles in result.checkpoint_particles.items():
                all_checkpoints[start + local_idx] = cp_particles

            section_metadata.append({
                'name': section.name,
                'simulator': section.simulator_key,
                'elements': list(section.element_range),
                'num_particles_in': current_particles.shape[0],
                'num_particles_out': (result.final_particles.shape[0]
                                      if result.final_particles is not None
                                      else 0),
            })

            current_particles = result.final_particles
            if current_particles is None:
                return SimulationResult(
                    simulator_name=self.name,
                    success=False,
                    metadata={'failed_section': section.name,
                              'reason': 'no output particles'}
                )

        return SimulationResult(
            simulator_name=self.name,
            success=True,
            final_particles=current_particles,
            checkpoint_particles=all_checkpoints,
            metadata={
                'sections': section_metadata,
                'num_sections': len(self.sections),
                'beam_energy_mev': self.beam_energy,
            }
        )

    def _convert_element_to_native(self, element):
        raise NotImplementedError("MultiCodeSimulator delegates to child simulators")

    def transform_coordinates(self, particles, from_system, to_system):
        from simulatorFactory import CoordinateTransformer
        return CoordinateTransformer.transform(
            particles, from_system, to_system, self.beam_energy
        )

    def set_beam_energy(self, energy_mev: float):
        super().set_beam_energy(energy_mev)
        for elem in self._master_beamline:
            elem.setE(energy_mev)
        for sim in self._simulators.values():
            sim.set_beam_energy(energy_mev)

    @classmethod
    def from_config(cls, config: dict, **kwargs):
        """
        Create from a configuration dict.

        Config format::

            {
                "lattice_path": "var/UH_FEL_beamline.json",
                "beam_energy_mev": 40.0,
                "sections": [
                    {"name": "prefix", "simulator": "felsim", "elements": [0, 87]},
                    {"name": "suffix", "simulator": "rftrack", "elements": [87, 118]},
                ]
            }
        """
        sections = []
        for s in config['sections']:
            sections.append(SimSection(
                name=s.get('name', f"section_{len(sections)}"),
                simulator_key=s['simulator'],
                element_range=tuple(s['elements']),
                config=s.get('config', {}),
            ))
        return cls(
            sections=sections,
            lattice_path=config.get('lattice_path'),
            beam_energy=config.get('beam_energy_mev', 45.0),
            **kwargs
        )
