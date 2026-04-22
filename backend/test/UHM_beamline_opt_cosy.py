"""UH MkV FEL Beamline Optimization via COSY INFINITY FIT.

Translates the 11-stage FELsim optimization into COSY's internal FIT command.
Supports S1 (2 ps) and S3 (0.5 ps) configs, multistart for Stage 5, and full
11-stage optimization in a single COSY run.

Author: Eremey Valetov
"""

import sys
import json
import copy
import argparse
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from cosyAdapter import COSYAdapter
from cosyOptHelper import add_stages, get_optimized_currents, parse_beamline_felsim_indexed

# --- Beam parameters ---
Energy = 40  # MeV
epsilon_n = 8  # pi.mm.mrad (normalized)
x_std = 0.8  # mm — initial beam size (same for y)

# Stage 5 multistart grid (I, I2, I3)
STAGE5_STARTS = [
    (0.28, 2.65, 2.69),  # FELsim S1 solution (I→37, I2→35, I3→33)
    (2.5, 0.6, 1.3),     # Close to FELsim (swapped)
    (2, 2, 2),            # Original default
    (1, 3, 1),            # Alternate
]


def compute_targets():
    """Compute undulator Twiss targets, geometric emittance, and initial Twiss."""
    from beamline import lattice
    relat = lattice(1, fringeType=None)
    relat.setE(E=Energy)
    norm = relat.gamma * relat.beta
    epsilon = epsilon_n / norm  # pi.mm.mrad (geometric)

    K = 1.2
    lambda_u = 2.3e-2  # m
    beta_ym = relat.gamma * lambda_u / (2 * np.pi * K)
    alpha_ym = 0.0
    beta_xm = 1.4
    alpha_xm = 0.47

    beta_0 = x_std**2 / epsilon  # Initial β from beam distribution

    return {
        'beta_xm': beta_xm, 'alpha_xm': alpha_xm,
        'beta_ym': beta_ym, 'alpha_ym': alpha_ym,
        'epsilon': epsilon, 'beta_0': beta_0,
    }


def build_stages(targets):
    """Define the 11 optimization stages.

    Variable indices are FELsim 0-based beamline indices.
    """
    axm = targets['alpha_xm']
    aym = targets['alpha_ym']
    bxm = targets['beta_xm']
    bym = targets['beta_ym']

    return [
        {   # Stage 1: First Quadrupole Doublet
            'name': 'First Quadrupole Doublet',
            'variables': {1: 'I', 3: 'I2'},
            'start_point': {
                'I': {'start': 1, 'bounds': (0, 10)},
                'I2': {'start': 1, 'bounds': (0, 10)},
            },
            'objectives': {
                8: [{'measure': ['x', 'alpha'], 'goal': 0, 'weight': 1},
                    {'measure': ['x', 'beta'], 'goal': 0.1, 'weight': 0.0}],
                9: [{'measure': ['y', 'alpha'], 'goal': 0, 'weight': 1},
                    {'measure': ['y', 'beta'], 'goal': 0.1, 'weight': 0.5}],
            },
        },
        {   # Stage 2: First Chromaticity Quad
            'name': 'First Chromaticity Quad',
            'variables': {10: 'I'},
            'start_point': {'I': {'start': 1, 'bounds': (0, 10)}},
            'objectives': {
                15: [{'measure': ['x', 'dispersion'], 'goal': 0, 'weight': 1}],
            },
        },
        {   # Stage 3: Quadrupole Triplet 1
            'name': 'Quadrupole Triplet 1',
            'variables': {16: 'I', 18: 'I2', 20: 'I3'},
            'start_point': {
                'I': {'start': 2, 'bounds': (0, 10)},
                'I2': {'start': 5, 'bounds': (0, 10)},
                'I3': {'start': 3, 'bounds': (0, 10)},
            },
            'objectives': {
                25: [{'measure': ['x', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['x', 'beta'], 'goal': 0.1, 'weight': 0.5}],
                26: [{'measure': ['y', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['y', 'beta'], 'goal': 0.1, 'weight': 0.5}],
            },
        },
        {   # Stage 4: Second Chromaticity Quad
            'name': 'Second Chromaticity Quad',
            'variables': {27: 'I'},
            'start_point': {'I': {'start': 1, 'bounds': (0, 10)}},
            'objectives': {
                32: [{'measure': ['x', 'dispersion'], 'goal': 0, 'weight': 1}],
            },
        },
        {   # Stage 5: Double Quadrupole Triplet (mirror symmetry)
            'name': 'Double Quadrupole Triplet',
            'variables': {37: 'I', 35: 'I2', 33: 'I3'},
            'start_point': {
                'I': {'start': 0.28, 'bounds': (0, 10)},
                'I2': {'start': 2.65, 'bounds': (0, 10)},
                'I3': {'start': 2.69, 'bounds': (0, 10)},
            },
            'objectives': {
                37: [{'measure': ['x', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['y', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['x', 'envelope'], 'goal': 2.0, 'weight': 1},
                     {'measure': ['y', 'envelope'], 'goal': 2.0, 'weight': 1}],
            },
            'mirror': {43: 33, 41: 35, 39: 37},
        },
        {   # Stage 6: Third Chromaticity Quad
            'name': 'Third Chromaticity Quad',
            'variables': {50: 'I'},
            'start_point': {'I': {'start': 1, 'bounds': (0, 10)}},
            'objectives': {
                55: [{'measure': ['x', 'dispersion'], 'goal': 0, 'weight': 1}],
            },
        },
        {   # Stage 7: IP Doublet
            'name': 'IP Doublet',
            'variables': {56: 'I', 58: 'I2'},
            'start_point': {
                'I': {'start': 2, 'bounds': (0, 10)},
                'I2': {'start': 2, 'bounds': (0, 10)},
            },
            'objectives': {
                59: [{'measure': ['x', 'envelope'], 'goal': 0.0, 'weight': 1},
                     {'measure': ['y', 'envelope'], 'goal': 0.0, 'weight': 1}],
            },
        },
        {   # Stage 8: Post-IP Doublet
            'name': 'Post-IP Doublet',
            'variables': {61: 'I', 63: 'I2'},
            'start_point': {
                'I': {'start': 2, 'bounds': (0, 10)},
                'I2': {'start': 2, 'bounds': (0, 10)},
            },
            'objectives': {
                68: [{'measure': ['x', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['x', 'beta'], 'goal': 0.1, 'weight': 0.5}],
                69: [{'measure': ['y', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['y', 'beta'], 'goal': 0.1, 'weight': 0.5}],
            },
        },
        {   # Stage 9: Fourth Chromaticity Quad
            'name': 'Fourth Chromaticity Quad',
            'variables': {70: 'I'},
            'start_point': {'I': {'start': 1, 'bounds': (0, 10)}},
            'objectives': {
                75: [{'measure': ['x', 'dispersion'], 'goal': 0, 'weight': 1}],
            },
        },
        {   # Stage 10: Quadrupole Triplet 2
            'name': 'Quadrupole Triplet 2',
            'variables': {76: 'I', 78: 'I2', 80: 'I3'},
            'start_point': {
                'I': {'start': 2, 'bounds': (0, 10)},
                'I2': {'start': 2, 'bounds': (0, 10)},
                'I3': {'start': 2, 'bounds': (0, 10)},
            },
            'objectives': {
                85: [{'measure': ['x', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['x', 'beta'], 'goal': 0.1, 'weight': 0.5}],
                86: [{'measure': ['y', 'alpha'], 'goal': 0, 'weight': 1},
                     {'measure': ['y', 'beta'], 'goal': 0.1, 'weight': 0.5}],
            },
        },
        {   # Stage 11: Chromaticity 5 + Final Triplet → Undulator Matching
            'name': 'Chromaticity 5 + Final Triplet',
            'variables': {87: 'Ic', 93: 'I', 95: 'I2', 97: 'I3'},
            'start_point': {
                'Ic': {'start': 4, 'bounds': (0, 10)},
                'I': {'start': 2, 'bounds': (0, 10)},
                'I2': {'start': 2, 'bounds': (0, 10)},
                'I3': {'start': 2, 'bounds': (0, 10)},
            },
            'objectives': {
                92: [{'measure': ['x', 'dispersion'], 'goal': 0, 'weight': 0.5}],
                117: [{'measure': ['x', 'alpha'], 'goal': axm, 'weight': 1},
                      {'measure': ['y', 'alpha'], 'goal': aym, 'weight': 1},
                      {'measure': ['x', 'beta'], 'goal': bxm, 'weight': 1},
                      {'measure': ['y', 'beta'], 'goal': bym, 'weight': 1}],
            },
        },
    ]


def apply_warm_start(stages, warm_currents):
    """Override stage start_point values with currents from a previous run."""
    for stage in stages:
        for elem_idx, var_name in stage['variables'].items():
            key = str(elem_idx)
            if key in warm_currents:
                val = warm_currents[key]
                stage['start_point'][var_name]['start'] = val
                lo, hi = stage['start_point'][var_name]['bounds']
                if val < lo:
                    stage['start_point'][var_name]['bounds'] = (val - 1, hi)
                if val > hi:
                    stage['start_point'][var_name]['bounds'] = (lo, val + 1)


def run_cosy_optimization(file_path, stages, targets, max_stage=None,
                          nmax=1000, nalg=1, generate_only=False,
                          fringe_field_order=0, order=3,
                          transfer_matrix_order=None, use_mge=False):
    """Run COSY FIT optimization for specified stages."""
    if max_stage is not None:
        stages = stages[:max_stage]

    config = {'simulation': {'KE': Energy, 'order': order, 'dimensions': 3}}
    if transfer_matrix_order is not None:
        config['simulation']['transfer_matrix_order'] = transfer_matrix_order

    adapter = COSYAdapter(
        lattice_path=str(file_path), mode='transfer_matrix',
        config=config,
        fringe_field_order=fringe_field_order,
        use_mge_for_dipoles=use_mge,
        debug=False
    )
    sim = adapter.get_native_simulator()
    # First 118 elements: gun to undulator entrance (index 117), excluding the undulator itself
    sim.beamline = parse_beamline_felsim_indexed(str(file_path))[:118]

    sim.set_geometric_emittance(targets['epsilon'])
    beta_0 = targets['beta_0']
    sim.set_initial_twiss(beta_x=beta_0, alpha_x=0.0, beta_y=beta_0, alpha_y=0.0)

    sim.fit_nmax = nmax
    sim.fit_eps = 1e-8
    sim.fit_nalgorithm = nalg
    sim.fit_combined_mse = True

    result = add_stages(sim, stages)
    index_map = result['index_map']

    if generate_only:
        fox_path = sim.generate_input(output_dir='results')
        print(f"FOX file generated: {fox_path}")
        return {'fox_path': fox_path, 'index_map': index_map}

    print(f"\nRunning COSY FIT with {len(stages)} stage(s), Nmax={nmax}, alg={nalg}...")
    sim_result = sim.run_simulation()

    if sim_result.get('status') != 'success':
        print(f"COSY FAILED: {sim_result}")
        return {'success': False, 'error': sim_result}

    # Print post-FIT diagnostic lines
    for line in sim_result.get('log', '').splitlines():
        if 'POST-FIT' in line:
            print(f"  {line.strip()}")

    reader = sim.analyze_results()
    twiss = reader.get_twiss_from_transfer_map(
        initial_twiss_x={'beta': beta_0, 'alpha': 0.0},
        initial_twiss_y={'beta': beta_0, 'alpha': 0.0},
    )
    opt_currents = get_optimized_currents(reader, stages)

    return {
        'success': True,
        'twiss': twiss,
        'currents': opt_currents,
        'index_map': index_map,
        'reader': reader,
        'stages': stages,
    }


def compute_mse(twiss, targets):
    return (
        (twiss['beta_x'] - targets['beta_xm'])**2 +
        (twiss['beta_y'] - targets['beta_ym'])**2 +
        (twiss['alpha_x'] - targets['alpha_xm'])**2 +
        (twiss['alpha_y'] - targets['alpha_ym'])**2
    ) / 4


def run_multistart(file_path, stages, targets, nmax=1000, nalg=1,
                    fringe_field_order=0, order=3, use_mge=False):
    """Run with multiple Stage 5 starting points, keep best result."""
    best_result = None
    best_mse = float('inf')

    for i, (s5_i, s5_i2, s5_i3) in enumerate(STAGE5_STARTS):
        trial_stages = copy.deepcopy(stages)
        trial_stages[4]['start_point']['I']['start'] = s5_i
        trial_stages[4]['start_point']['I2']['start'] = s5_i2
        trial_stages[4]['start_point']['I3']['start'] = s5_i3

        print(f"\n{'='*60}")
        print(f"Multistart trial {i+1}/{len(STAGE5_STARTS)}: "
              f"Stage 5 start = ({s5_i}, {s5_i2}, {s5_i3})")
        print(f"{'='*60}")

        result = run_cosy_optimization(
            file_path, trial_stages, targets, nmax=nmax, nalg=nalg,
            fringe_field_order=fringe_field_order, order=order,
            use_mge=use_mge)

        if not result.get('success'):
            print("  -> FAILED")
            continue

        mse = compute_mse(result['twiss'], targets)
        has_neg = any(v < 0 for v in result['currents'].values())
        print(f"  -> MSE = {mse:.6e}"
              f"{'  (has negative currents)' if has_neg else ''}")

        if mse < best_mse:
            best_mse = mse
            best_result = result

    return best_result


def print_results(results, targets, felsim_ref=None):
    """Print optimization results with optional FELsim comparison."""
    if not results.get('success'):
        print("Optimization FAILED")
        return

    twiss = results['twiss']
    currents = results['currents']

    print("\n" + "=" * 70)
    print("COSY FIT Optimization Results")
    print("=" * 70)

    if felsim_ref:
        print(f"\n{'Elem':>6s}  {'COSY':>8s}  {'FELsim':>8s}  {'Delta':>8s}")
        print("-" * 38)
        for idx in sorted(set(currents) | set(felsim_ref)):
            c = currents.get(idx, float('nan'))
            f = felsim_ref.get(idx, float('nan'))
            d = c - f
            print(f"{idx:6d}  {c:8.4f}  {f:8.4f}  {d:+8.4f}")
    else:
        print("\nOptimized quad currents:")
        for idx in sorted(currents):
            print(f"  [{idx:3d}] {currents[idx]:8.4f} A")

    print("\nFinal Twiss at undulator entrance:")
    print(f"  beta_x  = {twiss['beta_x']:8.4f} m   (target: {targets['beta_xm']:.4f})")
    print(f"  beta_y  = {twiss['beta_y']:8.4f} m   (target: {targets['beta_ym']:.4f})")
    print(f"  alpha_x = {twiss['alpha_x']:8.4f}     (target: {targets['alpha_xm']:.4f})")
    print(f"  alpha_y = {twiss['alpha_y']:8.4f}     (target: {targets['alpha_ym']:.4f})")
    if 'eta_x' in twiss:
        print(f"  D_x     = {twiss['eta_x']*1000:8.4f} mm  (target: 0)")

    mse = compute_mse(twiss, targets)
    print(f"\n  MSE (final Twiss) = {mse:.6e}")
    print("=" * 70)
    return mse


def save_results(results, targets, output_path, config_name, fringe_field_order=0):
    """Save results to JSON."""
    twiss = results['twiss']
    data = {
        'config': config_name,
        'energy_MeV': Energy,
        'epsilon_n': epsilon_n,
        'fringe_field_order': fringe_field_order,
        'targets': {k: v for k, v in targets.items() if isinstance(v, (int, float))},
        'mse': compute_mse(twiss, targets),
        'twiss_undulator': {k: float(v) for k, v in twiss.items()},
        'currents': {str(k): float(v) for k, v in sorted(results['currents'].items())},
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


# --- FELsim reference currents ---
FELSIM_S1_CURRENTS = {
    1: 0.8218, 3: 1.0430, 10: 3.8834,
    16: 2.2396, 18: 4.9532, 20: 3.4258, 27: 4.6657,
    33: 2.6942, 35: 2.6523, 37: 0.2768,
    39: 0.2768, 41: 2.6523, 43: 2.6942, 50: 4.6739,
    56: 3.1219, 58: 3.3129, 61: 5.1775, 63: 4.0434, 70: 4.6818,
    76: 3.9336, 78: 4.0787, 80: 0.0139,
    87: 1.3624, 93: 0.9452, 95: 2.8851, 97: 2.1921,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COSY FIT beamline optimization")
    parser.add_argument('--max-stage', type=int, default=None,
                        help='Run only stages 1..N (for incremental testing)')
    parser.add_argument('--nmax', type=int, default=1000,
                        help='Max FIT iterations per block')
    parser.add_argument('--nalg', type=int, default=1,
                        help='FIT algorithm (1=gradient/DA, 3=Nelder-Mead)')
    parser.add_argument('--multistart', action='store_true',
                        help='Try multiple Stage 5 starting points')
    parser.add_argument('--generate-only', action='store_true',
                        help='Generate FOX file without running COSY')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--fr', type=int, default=0, choices=[0, 1, 2, 3],
                        help='COSY fringe field order (0=none, 3=full)')
    parser.add_argument('--order', type=int, default=3, choices=[1, 2, 3],
                        help='COSY DA computation order (1=linear, 3=default)')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='JSON file with previous results to use as starting point')
    parser.add_argument('--mge', action='store_true',
                        help='Use MGE fieldmap for dipoles (requires chicane_dipole_fieldmap.dat)')
    args = parser.parse_args()

    file_path = (Path(__file__).resolve().parent.parent.parent
                 / 'beam_excel' / 'Beamline_elements.xlsx')

    targets = compute_targets()
    stages = build_stages(targets)

    if args.warm_start:
        with open(args.warm_start) as f:
            warm_data = json.load(f)
        apply_warm_start(stages, warm_data['currents'])
        print(f"Warm-started from {args.warm_start} "
              f"(FR {warm_data.get('fringe_field_order', '?')}, "
              f"MSE {warm_data.get('mse', '?'):.2e})")

    print(f"UH MkV FEL — COSY FIT Optimization (S1, 2 ps)")
    print(f"Stages: {args.max_stage or 'all'}, Nmax: {args.nmax}, Alg: {args.nalg}, "
          f"FR: {args.fr}, Order: {args.order}, MGE: {args.mge}")
    print(f"Geometric emittance: {targets['epsilon']:.4f} pi.mm.mrad")
    print(f"Initial beta_0: {targets['beta_0']:.4f} m")

    if args.multistart and not args.generate_only:
        results = run_multistart(
            file_path, stages, targets, nmax=args.nmax, nalg=args.nalg,
            fringe_field_order=args.fr, order=args.order, use_mge=args.mge)
    else:
        results = run_cosy_optimization(
            file_path, stages, targets,
            max_stage=args.max_stage, nmax=args.nmax, nalg=args.nalg,
            generate_only=args.generate_only,
            fringe_field_order=args.fr, order=args.order, use_mge=args.mge)

    if not args.generate_only and results:
        mse = print_results(results, targets, felsim_ref=FELSIM_S1_CURRENTS)
        if args.save:
            save_results(results, targets, args.save, 'S1_2ps',
                         fringe_field_order=args.fr)
