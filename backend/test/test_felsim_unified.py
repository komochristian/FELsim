# test_felsim_unified.py
"""
Test unified plotting with FELsim adapter using the real beamline from Excel.
Limited to first N elements to avoid numerical instabilities.
"""
import sys
#import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np

EXCEL_PATH = Path("../../beam_excel/Beamline_elements.xlsx")
MAX_ELEMENTS = 50
BEAM_ENERGY_MEV = 45.0

from felsimAdapter import FELsimAdapter
from evolutionPlotter import EvolutionPlotter


def load_real_beamline_limited():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    print(f"Loading beamline from {EXCEL_PATH.name}")

    temp_sim = FELsimAdapter(excel_path=str(EXCEL_PATH))
    temp_sim.set_beam_energy(BEAM_ENERGY_MEV)

    full_beamline = temp_sim.get_native_beamline()
    limited = full_beamline[:MAX_ELEMENTS]

    cumulative = 0.0
    for i, el in enumerate(limited):
        l = getattr(el, 'length', 0)
        cumulative += l
        print(f"{i:3d}: {type(el).__name__:15s}  L={l:.6f} m  cumulative={cumulative:.4f} m")

    total_l = sum(el.length for el in limited)
    print(f"Using {MAX_ELEMENTS}/{len(full_beamline)} elements, total length: {total_l:.4f} m")

    for el in limited:
        el.setE(BEAM_ENERGY_MEV)

    return limited


def create_test_particles(n=1000):
    from ebeam import beam
    ebeam = beam()
    std_dev = np.array([1.0, 0.1, 1.0, 0.1, 1.0, 0.5])
    return ebeam.gen_6d_gaussian(0, std_dev, n)


def test_adapter_collect_evolution():
    print("\n--- Adapter Evolution Collection ---")
    beamline = load_real_beamline_limited()
    particles = create_test_particles(500)

    sim = FELsimAdapter()
    sim.set_beam_energy(BEAM_ENERGY_MEV)
    sim._native_beamline = beamline

    evolution = sim.collect_evolution(particles, interval=0.05)

    print(f"Collected {len(evolution.s_positions)} samples over {evolution.total_length:.4f} m")
    print(f"Snapshots: {len(evolution.particles)}, Twiss calculations: {len(evolution.twiss)}")

    assert len(evolution.s_positions) == len(evolution.particles) == len(evolution.twiss)

    twiss_x = evolution.twiss[evolution.s_positions[0]]['x']
    required_fields = ['beta', 'alpha', 'gamma', 'emittance', 'dispersion']
    assert all(f in twiss_x for f in required_fields)

    return evolution


def test_twiss_dataframe(evolution):
    print("\n--- DataFrame Export ---")
    df = evolution.get_twiss_evolution()
    print(f"Shape: {df.shape}")
    print(f"s ∈ [{df['s'].min():.4f}, {df['s'].max():.4f}] m")
    print(f"βx ∈ [{df['beta_x'].min():.3f}, {df['beta_x'].max():.3f}] m")
    print(f"Envelope_x ∈ [{df['envelope_x'].min():.3f}, {df['envelope_x'].max():.3f}] mm")

    assert 'dispersion_x' in df.columns
    return df


def test_plotter(evolution):
    print("\n--- Plotting ---")
    plotter = EvolutionPlotter(axis_mode='local')
    plotter.plot(
        evolution,
        show_phase_space=True,
        show_envelope=True,
        show_schematic=True,
        interactive=True,
        scatter=False
    )


def test_axis_modes():
    print("\n--- Axis Mode Comparison ---")
    beamline = load_real_beamline_limited()
    particles = create_test_particles(1000)

    sim = FELsimAdapter()
    sim._native_beamline = beamline
    sim.beam_energy = BEAM_ENERGY_MEV

    ev = sim.collect_evolution(particles, interval=0.03)

    print("Local axis mode (per-slice adaptation)...")
    EvolutionPlotter(axis_mode='local').plot(ev)

    print("Global axis mode (fixed scale)...")
    EvolutionPlotter(axis_mode='global').plot(ev)


def run_all_tests():
    evolution = test_adapter_collect_evolution()
    test_twiss_dataframe(evolution)
    test_plotter(evolution)
    print("\nTests completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help="Use full beamline (may be numerically unstable)")
    parser.add_argument('--max', type=int, default=20,
                        help="Maximum number of elements")
    args = parser.parse_args()

    if args.full:
        MAX_ELEMENTS = 999
    else:
        MAX_ELEMENTS = args.max

    if args.full or '--axis-modes' in sys.argv:
        test_axis_modes()
    else:
        run_all_tests()