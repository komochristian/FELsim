# test_cosy_unified.py
"""
Test unified plotting with COSY adapter.
"""
import sys
#import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warnings
import traceback
from pathlib import Path


# Custom warning handler for debugging
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Show full traceback for warnings during debugging."""
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.filterwarnings('error', category=RuntimeWarning)

# Check COSY availability
try:
    from cosyAdapter import COSYAdapter, _COSY_AVAILABLE
    if not _COSY_AVAILABLE:
        print("COSY components not available (cosySimulator/cosyParticleSimulator not found)")
        sys.exit(1)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Import unified components
try:
    from beamEvolution import BeamEvolution
    from evolutionPlotter import EvolutionPlotter
    from simulatorBase import CoordinateSystem, SimulationMode
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Import FELsim for comparison (optional)
try:
    from felsimAdapter import FELsimAdapter
    from ebeam import beam

    FELSIM_AVAILABLE = True
except ImportError:
    FELSIM_AVAILABLE = False

# Configuration
EXCEL_PATH = Path("../../beam_excel/Beamline_elements_3.xlsx")
BEAM_ENERGY = 45.0  # MeV


def create_test_particles_felsim(n=1000):
    """Generate test distribution in FELsim coordinates."""
    ebeam = beam()
    std_dev = [1.0, 0.1, 1.0, 0.1, 1.0, 0.5]
    return ebeam.gen_6d_gaussian(0, std_dev, n)


def test_cosy_adapter_creation():
    """Test COSYAdapter instantiation."""
    print("\n=== COSYAdapter Creation ===")

    if not EXCEL_PATH.exists():
        print(f"Excel file not found: {EXCEL_PATH}")
        print("Creating adapter without beamline")
        sim = COSYAdapter(mode='particle_tracking', debug=True)
    else:
        print(f"Loading beamline from: {EXCEL_PATH}")
        sim = COSYAdapter(
            excel_path=str(EXCEL_PATH),
            mode='particle_tracking',
            debug=True
        )

    print(f"Simulator: {sim.name}")
    print(f"Native coordinates: {sim.native_coordinates.value}")
    print(f"Mode: {sim.simulation_mode}")

    try:
        beamline = sim.get_beamline()
        n_elements = len(beamline)
        print(f"Beamline: {n_elements} elements")

        for i, elem in enumerate(beamline[:5]):
            elem_type = elem.get('type', 'UNKNOWN')
            length = elem.get('length', 0)
            print(f"  [{i}] {elem_type}: L={length:.4f} m")
        if n_elements > 5:
            print(f"  ... and {n_elements - 5} more elements")
    except ValueError as e:
        print(f"No beamline parsed: {e}")

    print("COSYAdapter created successfully")
    return sim


def test_particle_generation(sim):
    """Test particle generation and coordinate transformation."""
    print("\n=== Particle Generation and Coordinate Transformation ===")

    particles_felsim = create_test_particles_felsim(500)
    print(f"FELsim particles: {particles_felsim.shape}")
    print(f"  x range: [{particles_felsim[:, 0].min():.4f}, {particles_felsim[:, 0].max():.4f}] mm")

    native_sim = sim.get_native_simulator()
    particles_cosy = native_sim.transform_to_cosy_coordinates(particles_felsim, energy=BEAM_ENERGY)
    print(f"COSY particles: {particles_cosy.shape}")
    print(f"  x range: [{particles_cosy[:, 0].min():.6f}, {particles_cosy[:, 0].max():.6f}] m")

    particles_back = native_sim.transform_from_cosy_coordinates(particles_cosy, energy=BEAM_ENERGY)
    max_diff = np.max(np.abs(particles_felsim - particles_back))
    print(f"Round-trip max difference: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("Coordinate transformation validated")
    else:
        print(f"Warning: coordinate transformation error: {max_diff}")

    return particles_felsim


def test_collect_evolution(sim, particles_felsim):
    """Test COSYAdapter.collect_evolution()."""
    print("\n=== Evolution Collection ===")

    try:
        print("Running COSY simulation with checkpoints...")
        evolution = sim.collect_evolution(
            particles_felsim,
            checkpoint_elements=list(range(1, 21))
        )

        print(f"Evolution data:")
        print(f"  {len(evolution.s_positions)} s-positions")
        print(f"  {len(evolution.particles)} particle snapshots")
        print(f"  {len(evolution.twiss)} Twiss calculations")
        print(f"  {len(evolution.elements)} elements")
        print(f"  Total length: {evolution.total_length:.4f} m")

        # Verify data consistency
        assert len(evolution.s_positions) == len(evolution.particles), \
            f"Mismatch: {len(evolution.s_positions)} positions vs {len(evolution.particles)} particles"
        assert len(evolution.s_positions) == len(evolution.twiss), \
            f"Mismatch: {len(evolution.s_positions)} positions vs {len(evolution.twiss)} twiss"
        print("Data structures consistent")

        # Check Twiss fields
        if evolution.s_positions:
            s0 = evolution.s_positions[0]
            if s0 in evolution.twiss:
                twiss_x = evolution.twiss[s0].get('x', {})
                required_fields = ['beta', 'alpha', 'gamma', 'emittance']
                missing = [f for f in required_fields if f not in twiss_x]
                if missing:
                    print(f"Warning: missing Twiss fields: {missing}")
                else:
                    print("Twiss contains required fields")
                    print(f"  βx = {twiss_x['beta']:.4f} m")
                    print(f"  αx = {twiss_x['alpha']:.4f}")

        # Show s-position statistics
        if len(evolution.s_positions) > 1:
            spacings = np.diff(sorted(evolution.s_positions))
            print(f"  s-spacing: min={spacings.min():.4f}, max={spacings.max():.4f}, mean={spacings.mean():.4f} m")

        print("collect_evolution() completed successfully")
        return evolution

    except Exception as e:
        print(f"collect_evolution() failed: {e}")
        traceback.print_exc()
        return None


def test_twiss_dataframe(evolution):
    """Test BeamEvolution DataFrame export."""
    print("\n=== Twiss DataFrame Export ===")

    if evolution is None:
        print("Skipped: no evolution data")
        return None

    df = evolution.get_twiss_evolution()

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"s range: [{df['s'].min():.4f}, {df['s'].max():.4f}] m")

    if 'beta_x' in df.columns and not df['beta_x'].isna().all():
        print(f"βx range: [{df['beta_x'].min():.4f}, {df['beta_x'].max():.4f}] m")
        print(f"Envelope_x range: [{df['envelope_x'].min():.4f}, {df['envelope_x'].max():.4f}] mm")

    if 'dispersion_x' in df.columns:
        print("Dispersion columns present")
    else:
        print("Warning: dispersion columns missing")

    return df


def test_plotter(evolution):
    """Test EvolutionPlotter with COSY data."""
    print("\n=== Evolution Plotting ===")

    if evolution is None:
        print("Skipped: no evolution data")
        return False

    if len(evolution.particles) == 0:
        print("Skipped: no particle data in evolution")
        return False

    plotter = EvolutionPlotter(axis_mode='local')

    print("Plotting COSY evolution (close window to continue)...")
    try:
        plotter.plot(
            evolution,
            show_phase_space=True,
            show_envelope=True,
            show_schematic=True,
            interactive=True,
            scatter=False
        )
        print("Plot displayed successfully")
        return True
    except Exception as e:
        print(f"Plotting failed: {e}")
        traceback.print_exc()
        return False


def test_comparison_with_felsim(sim_cosy, particles_felsim):
    """Compare COSY evolution with FELsim on same initial conditions."""
    print("\n=== Comparison with FELsim ===")

    if not FELSIM_AVAILABLE:
        print("Skipped: FELsim not available")
        return None

    try:
        cosy_beamline = sim_cosy.get_beamline()
        n_elements = len(cosy_beamline)
        print(f"COSY beamline: {n_elements} elements")

        print("Running COSY simulation...")
        evolution_cosy = sim_cosy.collect_evolution(
            particles_felsim.copy(),
            checkpoint_elements='all'
        )

        if evolution_cosy is None or len(evolution_cosy.twiss) == 0:
            print("COSY evolution returned no data")
            return None

        final_s_cosy = max(evolution_cosy.s_positions)
        twiss_cosy = evolution_cosy.twiss.get(final_s_cosy, {})

        if 'x' in twiss_cosy:
            print(f"\nCOSY final Twiss (s={final_s_cosy:.4f} m):")
            print(f"  βx = {twiss_cosy['x'].get('beta', 'N/A'):.4f} m")
            print(f"  βy = {twiss_cosy['y'].get('beta', 'N/A'):.4f} m")
            print(f"  εx = {twiss_cosy['x'].get('emittance', 'N/A'):.6f} π·mm·mrad")

        print("\nCOSY simulation completed")
        print("(Full FELsim comparison requires equivalent beamline translation)")

        return evolution_cosy

    except Exception as e:
        print(f"Comparison failed: {e}")
        traceback.print_exc()
        return None


def test_selective_checkpoints(sim, particles_felsim):
    """Test with selective checkpoint elements."""
    print("\n=== Selective Checkpoints ===")

    try:
        beamline = sim.get_beamline()
        n_elements = len(beamline)

        if n_elements < 5:
            print("Beamline too short for selective checkpoint test")
            return

        checkpoint_list = list(range(1, 21))
        print(f"Checkpointing at elements: {checkpoint_list[:10]}{'...' if len(checkpoint_list) > 10 else ''}")

        evolution = sim.collect_evolution(
            particles_felsim,
            checkpoint_elements=list(range(1, 21))
        )

        print(f"Requested checkpoints (including at s=0): {len(checkpoint_list) + 1}")
        print(f"Obtained checkpoints: {len(evolution.s_positions)}")

        expected = len(checkpoint_list) + 1
        if len(evolution.s_positions) == expected:
            print("Correct number of checkpoints")
        else:
            print(f"Expected {expected}, got {len(evolution.s_positions)}")

    except Exception as e:
        print(f"Selective checkpoint test failed: {e}")
        traceback.print_exc()


def run_all_tests():
    """Run complete COSY test suite."""
    print("=" * 60)
    print("COSY Unified Plotting Test Suite")
    print("=" * 60)

    try:
        sim = test_cosy_adapter_creation()
        particles_felsim = test_particle_generation(sim)
        evolution = test_collect_evolution(sim, particles_felsim)
        test_twiss_dataframe(evolution)
        test_plotter(evolution)
        test_comparison_with_felsim(sim, particles_felsim)
        test_selective_checkpoints(sim, particles_felsim)

        print("\n" + "=" * 60)
        print("COSY test suite completed")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nTest suite failed: {e}")
        traceback.print_exc()
        return False


def visual_comparison():
    """Side-by-side comparison of FELsim and COSY plots."""
    print("\n=== Visual Comparison: FELsim vs COSY ===")

    if not EXCEL_PATH.exists():
        print(f"Excel file required: {EXCEL_PATH}")
        return

    particles = create_test_particles_felsim(1000)

    print("Displaying COSY plot...")
    sim_cosy = COSYAdapter(
        excel_path=str(EXCEL_PATH),
        mode='particle_tracking',
        debug=False
    )

    evolution_cosy = sim_cosy.collect_evolution(particles.copy(), checkpoint_elements='all')

    plotter = EvolutionPlotter(axis_mode='local')
    plotter.plot(evolution_cosy)

    print("Visual comparison complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test COSY unified plotting")
    parser.add_argument('--visual', action='store_true', help='Run visual comparison only')
    parser.add_argument('--excel', type=str, default=str(EXCEL_PATH), help='Path to beamline Excel file')

    args = parser.parse_args()
    EXCEL_PATH = Path(args.excel)

    if args.visual:
        visual_comparison()
    else:
        run_all_tests()