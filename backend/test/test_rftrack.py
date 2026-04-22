"""
Basic tests for RF-Track adapter.

Author: Eremey Valetov
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from pathlib import Path

pytest.importorskip("RF_Track", reason="RF-Track not installed (pip install RF-Track)")

from rftrackAdapter import RFTrackAdapter, _RFTRACK_AVAILABLE
from simulatorBase import BeamlineElement, CoordinateSystem, SimulationMode
from simulatorFactory import SimulatorFactory
from physicalConstants import PhysicalConstants

BEAM_ENERGY = 45.0


def test_adapter_creation():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    assert sim.name == "RF-Track"
    assert sim.native_coordinates == CoordinateSystem.RFTRACK
    assert sim.simulation_mode == SimulationMode.PARTICLE_TRACKING
    assert sim.G_quad == PhysicalConstants.G_quad_default

    sim_custom = RFTrackAdapter(beam_energy=30.0, G_quad=3.0)
    assert sim_custom.G_quad == 3.0
    assert sim_custom.beam_energy == 30.0


def test_factory_registration():
    available = SimulatorFactory.get_available_simulators()
    assert 'rftrack' in available

    sim = SimulatorFactory.create('rftrack', beam_energy=BEAM_ENERGY)
    assert sim.name == "RF-Track"

    info = SimulatorFactory.get_simulator_info('rftrack')
    assert info['class'] == 'RFTrackAdapter'


def test_relativistic_params():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    gamma_expected, beta_expected = PhysicalConstants.relativistic_parameters(
        BEAM_ENERGY, PhysicalConstants.E0_electron
    )

    assert abs(sim._gamma - gamma_expected) < 1e-6
    assert abs(sim._beta - beta_expected) < 1e-6

    sim.set_beam_energy(100.0)
    gamma_100, _ = PhysicalConstants.relativistic_parameters(100.0, PhysicalConstants.E0_electron)
    assert abs(sim._gamma - gamma_100) < 1e-6


def test_current_to_k1():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    k1_focus = sim._current_to_k1(1.0, 0.1, focusing=True)
    k1_defocus = sim._current_to_k1(1.0, 0.1, focusing=False)

    assert k1_focus > 0
    assert k1_defocus < 0
    assert abs(k1_focus) == abs(k1_defocus)

    # Verify against manual calculation
    mass_kg = sim.particle_mass * PhysicalConstants.MeV_to_J / PhysicalConstants.C**2
    k1_manual = abs(PhysicalConstants.Q * sim.G_quad * 1.0) / (
        mass_kg * PhysicalConstants.C * sim._beta * sim._gamma
    )
    assert abs(k1_focus - k1_manual) < 1e-10

    assert sim._current_to_k1(0.0, 0.1, focusing=True) == 0.0
    assert sim._current_to_k1(1.0, 0.0, focusing=True) == 0.0


def test_coordinate_transforms():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)
    particles = np.array([
        [1.0, 0.1, 2.0, 0.2, 0.5, 0.3],
        [-0.5, -0.05, 1.0, -0.1, -0.2, 0.1],
    ])

    # FELsim -> RF-Track -> FELsim
    p_rft = sim.transform_coordinates(particles, CoordinateSystem.FELSIM, CoordinateSystem.RFTRACK)
    p_back = sim.transform_coordinates(p_rft, CoordinateSystem.RFTRACK, CoordinateSystem.FELSIM)
    assert np.max(np.abs(particles - p_back)) < 1e-10

    # COSY -> RF-Track -> COSY
    particles_cosy = np.array([[1e-3, 1e-4, 2e-3, 2e-4, 1e-3, 1e-4]])
    p_rft2 = sim.transform_coordinates(particles_cosy, CoordinateSystem.COSY, CoordinateSystem.RFTRACK)
    p_back2 = sim.transform_coordinates(p_rft2, CoordinateSystem.RFTRACK, CoordinateSystem.COSY)
    assert np.max(np.abs(particles_cosy - p_back2)) < 1e-10

    # Identity transform
    p_same = sim.transform_coordinates(particles, CoordinateSystem.FELSIM, CoordinateSystem.FELSIM)
    assert np.allclose(particles, p_same)


def test_beamline_setup():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    elements = [
        BeamlineElement('DRIFT', 0.5),
        BeamlineElement('QUAD_F', 0.1, current=1.5),
        BeamlineElement('DRIFT', 0.3),
        BeamlineElement('QUAD_D', 0.1, current=1.5),
        BeamlineElement('DRIFT', 0.5),
    ]

    sim.set_beamline(elements)

    assert len(sim.beamline) == 5
    assert sim._lattice is not None
    assert sim._lattice.size() == 5

    total_length = sim._lattice.get_length()
    expected_length = sum(e.length for e in elements)
    assert abs(total_length - expected_length) < 1e-6


def test_particle_generation():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    # Gaussian
    p_gauss = sim.generate_particles(1000, distribution_type='gaussian',
                                      std_dev=[1.0, 0.1, 1.0, 0.1, 0.5, 0.2])
    assert p_gauss.shape == (1000, 6)

    # Uniform
    p_uniform = sim.generate_particles(500, distribution_type='uniform',
                                        std_dev=[2.0, 0.2, 2.0, 0.2, 1.0, 0.5])
    assert p_uniform.shape == (500, 6)

    # Twiss-matched
    p_twiss = sim.generate_particles(500, distribution_type='twiss',
                                      twiss_x={'beta': 5.0, 'alpha': -1.0, 'emittance': 2.0},
                                      twiss_y={'beta': 8.0, 'alpha': 0.5, 'emittance': 2.0})
    assert p_twiss.shape == (500, 6)


def test_simulation():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    elements = [
        BeamlineElement('DRIFT', 0.3),
        BeamlineElement('QUAD_F', 0.08, current=2.0),
        BeamlineElement('DRIFT', 0.2),
        BeamlineElement('QUAD_D', 0.08, current=2.0),
        BeamlineElement('DRIFT', 0.3),
    ]
    sim.set_beamline(elements)

    particles = sim.generate_particles(200, std_dev=[0.5, 0.05, 0.5, 0.05, 0.1, 0.1])
    result = sim.simulate(particles)

    assert result.success
    assert result.simulator_name == "RF-Track"
    assert result.final_particles is not None
    assert result.final_particles.shape[0] > 0

    twiss = result.twiss_parameters_statistical['final']
    assert 'x' in twiss and 'y' in twiss
    assert all(k in twiss['x'] for k in ['beta', 'alpha', 'gamma', 'emittance'])


def test_twiss_calculation():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)

    particles = sim.generate_particles(2000, std_dev=[1.0, 0.1, 1.0, 0.1, 0.5, 0.2])
    twiss = sim._calculate_twiss(particles)

    assert twiss['x']['beta'] > 0
    assert twiss['y']['beta'] > 0
    assert twiss['x']['emittance'] > 0
    assert twiss['y']['emittance'] > 0
    assert all(k in twiss['x'] for k in ['beta', 'alpha', 'gamma', 'emittance'])


def test_g_quad_modification():
    sim = RFTrackAdapter(beam_energy=BEAM_ENERGY)
    elements = [
        BeamlineElement('DRIFT', 0.2),
        BeamlineElement('QUAD_F', 0.1, current=1.0),
        BeamlineElement('DRIFT', 0.2),
    ]
    sim.set_beamline(elements)

    k1_default = sim._current_to_k1(1.0, 0.1, focusing=True)

    sim.set_quadrupole_gradient(5.0)
    assert sim.G_quad == 5.0

    k1_new = sim._current_to_k1(1.0, 0.1, focusing=True)
    ratio = k1_new / k1_default
    expected_ratio = 5.0 / PhysicalConstants.G_quad_default

    assert abs(ratio - expected_ratio) < 1e-6
