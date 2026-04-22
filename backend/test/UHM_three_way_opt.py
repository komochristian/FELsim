"""Three-code beamline optimization and cross-validation.

Optimizes the UH MkV FEL transport line with FELsim Python (fast matrix-based),
then validates the result with RF-Track (particle tracking) and COSY INFINITY
(transfer maps). This demonstrates the typical workflow: optimize quickly with
the linear code, then cross-check with higher-fidelity simulations.

Usage:
    cd backend && python test/UHM_three_way_opt.py
    cd backend && python test/UHM_three_way_opt.py --max-stage 4
    cd backend && python test/UHM_three_way_opt.py --n-particles 2000

Author: Eremey Valetov
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from UHM_beamline_opt_cosy import (
    build_stages, compute_targets, Energy, epsilon_n, x_std,
    FELSIM_S1_CURRENTS
)
from felsimAdapter import FELsimAdapter

try:
    from cosyAdapter import COSYAdapter
    from cosyOptHelper import parse_beamline_felsim_indexed
    _COSY_AVAILABLE = True
except ImportError:
    _COSY_AVAILABLE = False

try:
    import RF_Track as rft
    from rftrackAdapter import RFTrackAdapter
    _RFTRACK_AVAILABLE = True
except ImportError:
    _RFTRACK_AVAILABLE = False

# Gun to undulator entrance: 118 elements (indices 0-117)
N_ELEMENTS = 118


def stage_to_felsim_vars(stage):
    """Convert stage variable dict to beamOptimizer format.

    Includes mirror pairs so symmetric elements share the same variable.
    """
    identity = lambda x: x
    variables = {}
    for idx, var_name in stage['variables'].items():
        variables[idx] = [var_name, 'current', identity]
    for target_idx, source_idx in stage.get('mirror', {}).items():
        variables[target_idx] = [stage['variables'][source_idx], 'current', identity]
    return variables


def generate_particles(n, seed=42):
    """Generate 6D Gaussian distribution in FELsim coordinates."""
    from beamline import lattice as lat_class

    rng = np.random.default_rng(seed)
    rel = lat_class(1, fringeType=None)
    rel.setE(E=Energy)
    epsilon = epsilon_n / (rel.gamma * rel.beta)

    xp_std = epsilon / x_std
    tof_std = 0.5e-9 * 2.856e9   # 0.5 ps bunch length
    dk_std = 0.5 * 10             # 0.5% energy spread

    particles = rng.standard_normal((n, 6))
    for i, std in enumerate([x_std, xp_std, x_std, xp_std, tof_std, dk_std]):
        particles[:, i] *= std
    return particles


def extract_twiss(tw_dict):
    """Extract Twiss from statistical result, handling LaTeX/plain key formats."""
    key_map = {
        'beta': [r'$\beta$ (m)', 'beta'],
        'alpha': [r'$\alpha$', 'alpha'],
    }

    def _get(plane, param):
        d = tw_dict[plane]
        for key in key_map.get(param, [param]):
            if key in d:
                return d[key]
        return float('nan')

    return {
        'beta_x': _get('x', 'beta'), 'alpha_x': _get('x', 'alpha'),
        'beta_y': _get('y', 'beta'), 'alpha_y': _get('y', 'alpha'),
    }


# --- Step 1: FELsim Python optimization ---

def optimize_felsim(file_path, stages, particles, max_stage=None):
    """Run sequential stage optimization with FELsim Python.

    Returns (FELsimAdapter, dict of optimized currents).
    """
    if max_stage is not None:
        stages = stages[:max_stage]

    sim = FELsimAdapter(lattice_path=str(file_path), beam_energy=Energy)
    sim._native_beamline = sim._native_beamline[:N_ELEMENTS]

    for i, stage in enumerate(stages):
        variables = stage_to_felsim_vars(stage)
        result = sim.optimize(
            objectives=stage['objectives'],
            variables=variables,
            initial_point=stage['start_point'],
            method='Nelder-Mead',
            particles=particles.copy()
        )
        fval = result.metadata.get('objective_value', float('nan'))
        nfev = result.metadata.get('num_evaluations', '?')
        print(f"  Stage {i+1:2d}/{len(stages)}: {stage['name']:35s}  "
              f"f={fval:.4e}  nfev={nfev}")

    # Extract optimized currents from beamline
    currents = {}
    all_indices = set()
    for stage in stages:
        all_indices.update(stage['variables'].keys())
        all_indices.update(stage.get('mirror', {}).keys())

    for idx in all_indices:
        elem = sim._native_beamline[idx]
        if hasattr(elem, 'current'):
            currents[idx] = elem.current

    return sim, currents


# --- Step 2: RF-Track validation ---

def validate_rftrack(file_path, optimized_currents, particles):
    """Validate optimized optics with RF-Track particle tracking."""
    if not _RFTRACK_AVAILABLE:
        print("  RF-Track not available, skipping")
        return None

    rft_sim = RFTrackAdapter(lattice_path=str(file_path), beam_energy=Energy)
    rft_sim.beamline = rft_sim.beamline[:N_ELEMENTS]

    for idx, current in optimized_currents.items():
        rft_sim._modify_element(idx, current=current)
    rft_sim._build_lattice()

    result = rft_sim.simulate(particles.copy())
    n_in = particles.shape[0]
    n_out = result.final_particles.shape[0]
    print(f"  {n_out}/{n_in} particles survived")

    return extract_twiss(result.twiss_parameters_statistical['final'])


# --- Step 3: COSY INFINITY validation ---

def validate_cosy(file_path, optimized_currents, targets):
    """Validate optimized optics with COSY INFINITY transfer maps."""
    if not _COSY_AVAILABLE:
        print("  COSY not available, skipping")
        return None

    config = {'simulation': {'KE': Energy, 'order': 3, 'dimensions': 3}}
    adapter = COSYAdapter(
        lattice_path=str(file_path), mode='transfer_matrix',
        config=config, fringe_field_order=0, debug=False
    )
    sim = adapter.get_native_simulator()
    sim.beamline = parse_beamline_felsim_indexed(str(file_path))[:N_ELEMENTS]

    for idx, current in optimized_currents.items():
        if idx < len(sim.beamline) and 'current' in sim.beamline[idx]:
            sim.beamline[idx]['current'] = current

    beta_0 = targets['beta_0']
    sim.set_geometric_emittance(targets['epsilon'])
    sim.set_initial_twiss(beta_x=beta_0, alpha_x=0.0, beta_y=beta_0, alpha_y=0.0)

    result = sim.run_simulation()
    if result.get('status') != 'success':
        print(f"  COSY simulation failed: {result.get('error', 'unknown')}")
        return None

    reader = sim.analyze_results()
    twiss = reader.get_twiss_from_transfer_map(
        initial_twiss_x={'beta': beta_0, 'alpha': 0.0},
        initial_twiss_y={'beta': beta_0, 'alpha': 0.0},
    )
    print(f"  Transfer map computed (order 3)")
    return twiss


# --- Comparison ---

def print_comparison(twiss_results, targets, optimized_currents):
    """Print cross-code Twiss comparison and optimized currents."""
    print("\n" + "=" * 72)
    print("Cross-Code Twiss Comparison at Undulator Entrance")
    print("=" * 72)

    codes = [(name, tw) for name, tw in twiss_results.items() if tw is not None]

    header = f"{'Parameter':>12s}  {'Target':>8s}"
    for name, _ in codes:
        header += f"  {name:>10s}"
    print(header)
    print("-" * len(header))

    for param, target_key in [
        ('beta_x', 'beta_xm'), ('alpha_x', 'alpha_xm'),
        ('beta_y', 'beta_ym'), ('alpha_y', 'alpha_ym'),
    ]:
        row = f"{param:>12s}  {targets[target_key]:8.4f}"
        for _, tw in codes:
            row += f"  {tw.get(param, float('nan')):10.4f}"
        print(row)

    print(f"\n{'Elem':>6s}  {'Optimized':>10s}  {'Reference':>10s}  {'Delta':>8s}")
    print("-" * 40)
    for idx in sorted(optimized_currents):
        opt = optimized_currents[idx]
        ref = FELSIM_S1_CURRENTS.get(idx, float('nan'))
        delta = opt - ref if np.isfinite(ref) else float('nan')
        print(f"{idx:6d}  {opt:10.4f}  {ref:10.4f}  {delta:+8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-way beamline optimization: FELsim Python -> RF-Track -> COSY")
    parser.add_argument('--max-stage', type=int, default=None,
                        help='Run only stages 1..N')
    parser.add_argument('--n-particles', type=int, default=1000,
                        help='Number of test particles (default: 1000)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    file_path = (Path(__file__).resolve().parent.parent.parent
                 / 'beam_excel' / 'Beamline_elements.xlsx')

    targets = compute_targets()
    stages = build_stages(targets)
    particles = generate_particles(args.n_particles)

    n_stages = args.max_stage or len(stages)
    print(f"UH MkV FEL -- 3-Way Optimization")
    print(f"Stages: {n_stages}, Particles: {args.n_particles}")
    print(f"Geometric emittance: {targets['epsilon']:.4f} pi.mm.mrad")

    # Step 1: Optimize with FELsim Python (fast, matrix-based)
    print(f"\n--- FELsim Python optimization ---")
    sim, currents = optimize_felsim(
        file_path, stages, particles, max_stage=args.max_stage)
    felsim_result = sim.simulate(particles.copy())
    felsim_twiss = extract_twiss(felsim_result.twiss_parameters_statistical['final'])

    # Step 2: Validate with RF-Track (particle tracking)
    print(f"\n--- RF-Track validation ---")
    rftrack_twiss = validate_rftrack(file_path, currents, particles)

    # Step 3: Validate with COSY INFINITY (transfer maps)
    print(f"\n--- COSY INFINITY validation ---")
    cosy_twiss = validate_cosy(file_path, currents, targets)

    # Step 4: Cross-code comparison
    twiss_results = {'FELsim': felsim_twiss}
    if rftrack_twiss is not None:
        twiss_results['RF-Track'] = rftrack_twiss
    if cosy_twiss is not None:
        twiss_results['COSY'] = cosy_twiss

    print_comparison(twiss_results, targets, currents)

    if args.save:
        data = {
            'config': '3-way optimization',
            'energy_MeV': Energy,
            'n_particles': args.n_particles,
            'n_stages': n_stages,
            'targets': {k: float(v) for k, v in targets.items()
                        if isinstance(v, (int, float))},
            'optimized_currents': {str(k): float(v)
                                   for k, v in sorted(currents.items())},
            'twiss': {name: {k: float(v) for k, v in tw.items()}
                      for name, tw in twiss_results.items()},
        }
        with open(args.save, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.save}")
