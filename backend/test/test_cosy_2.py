# test_cosy_2.py
"""
Usage examples for simulator adapters.

Demonstrates unified simulator interface for FELsim and COSY backends.

Author: Eremey Valetov
"""
#import sys
#import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from simulatorFactory import SimulatorFactory, create_simulator, compare_simulators
from simulatorBase import CoordinateSystem, SimulationMode, BeamlineElement

EXCEL_PATH = '../../beam_excel/Beamline_elements.xlsx'

def example_felsim_basic():
    """Basic FELsim simulation using adapter interface"""
    print("=== FELsim Basic Simulation ===\n")

    sim = create_simulator('felsim')
    sim.set_beam_energy(45.0)

    elements = [
        BeamlineElement('DRIFT', length=0.5),
        BeamlineElement('QUAD_F', length=0.0889, current=2.0),
        BeamlineElement('DRIFT', length=0.25),
        BeamlineElement('QUAD_D', length=0.0889, current=2.0),
        BeamlineElement('DRIFT', length=0.5)
    ]
    sim.set_beamline(elements)

    particles = sim.generate_particles(
        num_particles=1000,
        distribution_type='gaussian',
        std_dev=[0.8, 0.1, 0.8, 0.1, 2.0, 0.5]
    )

    result = sim.simulate(particles)

    print(f"Success: {result.success}")
    print(f"Tracked {result.metadata['num_particles']} particles")

    twiss = result.twiss_parameters_statistical.get('final', {}).get('x', {})
    if twiss:
        print("Final Twiss (x):", ", ".join(f"{k}={v:.3f}" for k, v in twiss.items()))

    return sim, result


def example_felsim_optimization():
    """FELsim optimization with beamOptimizer"""
    print("\n=== FELsim Optimization ===\n")

    sim = create_simulator('felsim')
    sim.set_beam_energy(45.0)

    elements = [
        BeamlineElement('DRIFT', length=0.5),
        BeamlineElement('QUAD_F', length=0.0889, current=2.0),
        BeamlineElement('DRIFT', length=0.25),
        BeamlineElement('QUAD_D', length=0.0889, current=2.0),
        BeamlineElement('DRIFT', length=0.5)
    ]
    sim.set_beamline(elements)

    particles = sim.generate_particles(
        num_particles=1000,
        std_dev=[0.8, 0.1, 0.8, 0.1, 2.0, 0.5]
    )

    objectives = {
        4: [
            {"measure": ["x", "alpha"], "goal": 0.0, "weight": 1.0},
            {"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}
        ]
    }

    variables = {
        1: ["I1", "current", lambda x: x],
        3: ["I2", "current", lambda x: x]
    }

    initial_point = {
        "I1": {"start": 2.0, "bounds": (0, 10)},
        "I2": {"start": 2.0, "bounds": (0, 10)}
    }

    result = sim.optimize(
        objectives=objectives,
        variables=variables,
        initial_point=initial_point,
        method='Nelder-Mead',
        particles=particles,
        print_results=True
    )

    if result.success:
        print(f"\nOptimised currents: I1={result.optimization_variables['I1']:.4f}, "
              f"I2={result.optimization_variables['I2']:.4f}")
        twiss = result.twiss_parameters_statistical.get('final', {}).get('x', {})
        if twiss:
            print("Final Twiss (x):", ", ".join(f"{k}={v:.3f}" for k, v in twiss.items()))

    return sim, result


def example_cosy_transfer_matrix():
    """COSY transfer matrix calculation"""
    print("\n=== COSY Transfer Matrix ===\n")

    sim = create_simulator(
        'cosy',
        excel_path=EXCEL_PATH,
        mode='transfer_matrix',
        debug=False
    )
    sim.set_beam_energy(45.0)

    result = sim.simulate()

    print(f"Success: {result.success}")
    print("Transfer map:\n", result.transfer_map)
    print("\nTwiss from transfer map:", result.twiss_parameters_transfer_map)

    return sim, result


def example_cosy_particle_tracking():
    """
    COSY particle tracking with checkpoints.

    Note: Checkpoint elements must exist in the beamline for particles to be saved.
    """
    print("\n=== COSY Particle Tracking ===\n")

    sim = create_simulator(
        'cosy',
        excel_path=EXCEL_PATH,
        mode='particle_tracking',
        debug=False
    )
    sim.set_beam_energy(45.0)
    sim.enable_particle_checkpoints(checkpoint_elements=[10])

    # COSY uses much smaller divergences than FELsim
    particles = sim.generate_particles(
        num_particles=10000,
        distribution_type='gaussian',
        std_dev=[0.08, 0.008, 0.08, 0.008, 0.5, 0.1],
        mean=0.0
    )

    result = sim.simulate(particles)

    print(f"Success: {result.success}")

    if result.success:
        print("Transfer map:\n", result.transfer_map)
        print("\nTwiss (transfer map):", result.twiss_parameters_transfer_map)
        print("Twiss (statistical):", result.twiss_parameters_statistical)

        if result.final_particles is not None:
            print(f"\nFinal particles: {result.final_particles.shape}")
        else:
            print("\nWarning: No final particles available")

        if result.checkpoint_particles:
            print(f"Checkpoints saved at elements: {list(result.checkpoint_particles.keys())}")
        else:
            print("Warning: No checkpoint particles available")
            checkpoint_config = sim.get_native_simulator().get_particle_tracking_config()
            print(f"Checkpoint config: mode={checkpoint_config['checkpoint_mode']}, "
                  f"elements={checkpoint_config['checkpoint_elements']}, "
                  f"written={checkpoint_config['checkpoints_written']}")

    return sim, result


def example_cosy_optimization():
    """COSY internal FIT optimization"""
    print("\n=== COSY Optimization (FIT) ===\n")

    sim = create_simulator(
        'cosy',
        excel_path=EXCEL_PATH,
        mode='transfer_matrix'
    )
    sim.set_beam_energy(45.0)

    objectives = {
        86: [{"measure": ["y", "alpha"], "goal": 0.0, "weight": 1.0}],
        "optimizer_settings": {
            "eps": 1e-8,
            "Nmax": 1000,
            "Nalgorithm": 3
        }
    }

    variables = {
        1: {"current": "I1"},
        9: {"current": "I2"}
    }

    initial_point = {
        "I1": {"start": 1.0, "bounds": (0, 10)},
        "I2": {"start": 1.0, "bounds": (0, 10)}
    }

    result = sim.optimize(
        objectives=objectives,
        variables=variables,
        initial_point=initial_point
    )

    print(f"Success: {result.success}")
    if result.success:
        print("Optimised variables:", result.optimization_variables)

    return sim, result


def example_cross_simulator_coordinates():
    """Coordinate transformation between FELsim and COSY"""
    print("\n=== Cross-Simulator Coordinate Transform ===\n")

    from simulatorFactory import CoordinateTransformer

    felsim = create_simulator('felsim', excel_path='Beamline_elements.xlsx')
    felsim.set_beam_energy(45.0)

    particles_felsim = felsim.generate_particles(
        num_particles=1000,
        std_dev=[0.8, 0.1, 0.8, 0.1, 2.0, 0.5]
    )

    print(f"FELsim particles shape: {particles_felsim.shape}")
    print(f"FELsim coordinates: [x(mm), x'(mrad), y(mm), y'(mrad), Δt/T(10^-3), δW/W(10^-3)]")
    print(f"Sample particle (FELsim): {particles_felsim[0]}")

    particles_cosy = CoordinateTransformer.transform(
        particles_felsim,
        from_system=CoordinateSystem.FELSIM,
        to_system=CoordinateSystem.COSY,
        energy_mev=45.0
    )

    print(f"\nCOSY particles shape: {particles_cosy.shape}")
    print(f"COSY coordinates: [x(m), a, y(m), b, l(m), δK]")
    print(f"Sample particle (COSY): {particles_cosy[0]}")

    particles_recovered = CoordinateTransformer.transform(
        particles_cosy,
        from_system=CoordinateSystem.COSY,
        to_system=CoordinateSystem.FELSIM,
        energy_mev=45.0
    )

    max_error = np.max(np.abs(particles_felsim - particles_recovered))
    print(f"\nRound-trip maximum error: {max_error:.3e}")

    return particles_felsim, particles_cosy


def example_simulator_comparison():
    """Compare FELsim and COSY on identical beamline"""
    print("\n=== Simulator Comparison ===\n")

    # Note: Requires identical beamline setup in both simulators
    felsim = create_simulator('felsim')
    cosy = create_simulator('cosy', excel_path='Beamline_elements.xlsx', mode='particle_tracking')

    energy = 45.0
    felsim.set_beam_energy(energy)
    cosy.set_beam_energy(energy)

    # Generate identical initial particles in FELsim coordinates
    particles_felsim = np.random.normal(
        0,
        [0.8, 0.1, 0.8, 0.1, 2.0, 0.5],
        size=(1000, 6)
    )

    results = compare_simulators(
        simulators=[felsim, cosy],
        particles=particles_felsim,
        coordinate_system=CoordinateSystem.FELSIM,
        energy_mev=energy
    )

    print("Comparison results:")
    for sim_name, sim_results in results['simulators'].items():
        print(f"\n{sim_name}:")
        print(f"  Success: {sim_results['success']}")
        if 'twiss' in sim_results and 'x' in sim_results['twiss']:
            beta_x = sim_results['twiss']['x'].get('beta', 'N/A')
            print(f"  Beta_x: {beta_x}")

    return results


def example_legacy_felsim_integration():
    """Using adapter with legacy FELsim objects"""
    print("\n=== Legacy FELsim Integration ===\n")

    from beamline import qpfLattice, driftLattice

    sim = create_simulator('felsim')

    legacy_elements = [
        driftLattice(0.5),
        qpfLattice(current=2.0, length=0.0889),
        driftLattice(0.5)
    ]
    sim.set_beamline(legacy_elements)

    native_beamline = sim.get_native_beamline()
    print(f"Native beamline type: {type(native_beamline)}")
    print(f"First element: {native_beamline[0]}")
    print("\nCan use with legacy code:")
    print("  - schematic.draw_beamline()")
    print("  - beamOptimizer")
    print("  - Direct beamline manipulation")

    return sim, native_beamline


def run_all_examples():
    """Run all examples with basic error handling"""

    print("Simulator Adapter Usage Examples")
    print("=" * 70)
    print("\nAvailable simulators:")
    for sim_type in SimulatorFactory.get_available_simulators():
        info = SimulatorFactory.get_simulator_info(sim_type)
        print(f"  {sim_type}: {info.get('class', 'Unknown')}")
    print()

    examples = [
        ("FELsim Basic", example_felsim_basic),
        ("FELsim Optimization", example_felsim_optimization),
        ("COSY Transfer Matrix", example_cosy_transfer_matrix),
        ("COSY Particle Tracking", example_cosy_particle_tracking),
        ("COSY Optimization", example_cosy_optimization),
        ("Coordinate Transform", example_cross_simulator_coordinates),
        ("Simulator Comparison", example_simulator_comparison),
        ("Legacy Integration", example_legacy_felsim_integration),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n{name} failed: {e}\n")

    print("\n" + "=" * 70)
    print("Examples completed")


if __name__ == "__main__":
    run_all_examples()
