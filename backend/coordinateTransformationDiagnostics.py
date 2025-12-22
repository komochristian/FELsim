"""
Diagnostic utilities for COSY particle coordinate transformations.

Analyzes common failure modes in COSY -> MAD-X coordinate conversion.
"""

import numpy as np


class CoordinateTransformationDiagnostics:
    """
    Error analysis for coordinate transformations.

    Failure modes:
    - δK < -1: negative kinetic energy (sqrt of negative in momentum calc)
    - a² + b² > k²: unphysical transverse momentum
    - |δK| >> 1: extreme energy deviations suggesting tracking failure
    """

    def __init__(self, KE0=40.0, E0=0.511):
        """
        Parameters:
        -----------
        KE0 : float
            Reference kinetic energy [MeV]
        E0 : float
            Rest energy [MeV]
        """
        self.KE0 = KE0
        self.E0 = E0
        self.p0c = np.sqrt(KE0 ** 2 + 2 * KE0 * E0)

    def analyze_particle_errors(self, particles_cosy):
        """
        Analyze transformation errors in COSY coordinates.

        Parameters:
        -----------
        particles_cosy : np.ndarray (N, 6)
            [x(m), a, y(m), b, l(m), δK]

        Returns:
        --------
        dict : Error statistics by category
        """
        N = particles_cosy.shape[0]
        x, a, y, b, l, delta_K = particles_cosy.T

        KE_particle = self.KE0 * (1 + delta_K)

        # Error categorization
        negative_KE_mask = delta_K < -1.0
        near_zero_KE_mask = (delta_K >= -1.0) & (delta_K < -0.99)
        extreme_positive_mask = delta_K > 10.0
        extreme_negative_mask = (delta_K < -0.9) & (delta_K >= -1.0)

        with np.errstate(invalid='ignore'):
            momentum_squared = KE_particle ** 2 + 2 * KE_particle * self.E0
            pc = np.sqrt(np.maximum(momentum_squared, 0))
            k = pc / self.p0c

        discriminant = k ** 2 - a ** 2 - b ** 2
        invalid_momentum_mask = (discriminant < 0) | ~np.isfinite(discriminant)
        nan_mask = ~np.isfinite(particles_cosy).all(axis=1)

        results = {
            'total_particles': N,
            'categories': {
                'negative_KE': {
                    'count': int(np.sum(negative_KE_mask)),
                    'fraction': float(np.sum(negative_KE_mask) / N),
                    'min_delta_K': float(np.min(delta_K[negative_KE_mask])) if np.any(negative_KE_mask) else None,
                },
                'near_zero_KE': {
                    'count': int(np.sum(near_zero_KE_mask)),
                    'fraction': float(np.sum(near_zero_KE_mask) / N),
                },
                'extreme_positive': {
                    'count': int(np.sum(extreme_positive_mask)),
                    'fraction': float(np.sum(extreme_positive_mask) / N),
                    'max_delta_K': float(np.max(delta_K[extreme_positive_mask])) if np.any(
                        extreme_positive_mask) else None,
                },
                'extreme_negative': {
                    'count': int(np.sum(extreme_negative_mask)),
                    'fraction': float(np.sum(extreme_negative_mask) / N),
                },
                'invalid_momentum': {
                    'count': int(np.sum(invalid_momentum_mask)),
                    'fraction': float(np.sum(invalid_momentum_mask) / N),
                },
                'nan_or_inf': {
                    'count': int(np.sum(nan_mask)),
                    'fraction': float(np.sum(nan_mask) / N),
                }
            },
            'statistics': {
                'delta_K': {
                    'mean': float(np.nanmean(delta_K)),
                    'std': float(np.nanstd(delta_K)),
                    'min': float(np.nanmin(delta_K)),
                    'max': float(np.nanmax(delta_K)),
                    'percentiles': {
                        1: float(np.nanpercentile(delta_K, 1)),
                        5: float(np.nanpercentile(delta_K, 5)),
                        95: float(np.nanpercentile(delta_K, 95)),
                        99: float(np.nanpercentile(delta_K, 99)),
                    }
                },
                'transverse': {
                    'a_rms': float(np.nanstd(a)),
                    'b_rms': float(np.nanstd(b)),
                    'a_max': float(np.nanmax(np.abs(a))),
                    'b_max': float(np.nanmax(np.abs(b))),
                },
                'momentum_ratio': {
                    'k_mean': float(np.nanmean(k)),
                    'k_std': float(np.nanstd(k)),
                    'k_min': float(np.nanmin(k)),
                    'k_max': float(np.nanmax(k)),
                }
            }
        }

        any_problem = (negative_KE_mask | near_zero_KE_mask |
                       extreme_positive_mask | extreme_negative_mask |
                       invalid_momentum_mask | nan_mask)
        results['total_problematic'] = int(np.sum(any_problem))
        results['fraction_problematic'] = float(np.sum(any_problem) / N)

        return results

    def print_summary(self, results):
        """Print concise diagnostic summary."""
        N = results['total_particles']
        n_bad = results['total_problematic']

        print(f"\nDiagnostics: {N:,} particles, {n_bad:,} problematic ({100 * n_bad / N:.1f}%)")

        cats = results['categories']

        if cats['negative_KE']['count'] > 0:
            print(f"  Negative KE (δK < -1): {cats['negative_KE']['count']:,} "
                  f"(min δK = {cats['negative_KE']['min_delta_K']:.3f})")

        if cats['invalid_momentum']['count'] > 0:
            print(f"  Invalid momentum ratio: {cats['invalid_momentum']['count']:,}")

        if cats['near_zero_KE']['count'] > 0:
            print(f"  Near-zero KE: {cats['near_zero_KE']['count']:,}")

        if cats['extreme_positive']['count'] > 0:
            print(f"  Extreme energy gain: {cats['extreme_positive']['count']:,} "
                  f"(max δK = {cats['extreme_positive']['max_delta_K']:.1f})")

        if cats['nan_or_inf']['count'] > 0:
            print(f"  NaN/Inf: {cats['nan_or_inf']['count']:,}")

        stats = results['statistics']
        print(f"\nδK: μ={stats['delta_K']['mean']:.4f}, σ={stats['delta_K']['std']:.4f}, "
              f"range=[{stats['delta_K']['min']:.4f}, {stats['delta_K']['max']:.4f}]")
        print(f"Transverse: a_rms={stats['transverse']['a_rms']:.2e}, "
              f"b_rms={stats['transverse']['b_rms']:.2e}")

    def get_valid_mask(self, particles_cosy):
        """
        Return boolean mask for particles that can be safely transformed.

        Valid particles satisfy:
        - δK > -1 (positive kinetic energy)
        - a² + b² < k² (physical momentum ratio)
        - No NaN/Inf values
        """
        x, a, y, b, l, delta_K = particles_cosy.T

        # Basic validity checks
        finite_mask = np.isfinite(particles_cosy).all(axis=1)
        positive_KE_mask = delta_K > -1.0

        # Momentum ratio check
        KE_particle = self.KE0 * (1 + delta_K)
        with np.errstate(invalid='ignore'):
            momentum_squared = KE_particle ** 2 + 2 * KE_particle * self.E0
            pc = np.sqrt(np.maximum(momentum_squared, 0))
            k = pc / self.p0c

        discriminant = k ** 2 - a ** 2 - b ** 2
        valid_momentum_mask = (discriminant >= 0) & np.isfinite(discriminant)

        return finite_mask & positive_KE_mask & valid_momentum_mask


def test_error_cases():
    """Test diagnostics with known error cases."""
    diag = CoordinateTransformationDiagnostics(KE0=40.0, E0=0.511)

    # Generate test cases
    test_cases = {
        'negative_KE': np.array([[0.001, 0.0001, 0.001, 0.0001, 0.0, -1.5]]),
        'invalid_momentum': np.array([[0.001, 0.9, 0.001, 0.9, 0.0, 0.0]]),
        'extreme_energy': np.array([[0.001, 0.0001, 0.001, 0.0001, 0.0, 50.0]]),
        'valid': np.array([[0.001, 0.0001, 0.001, 0.0001, 0.0, 0.0]]),
    }

    for name, particles in test_cases.items():
        print(f"\nTest case: {name}")
        print(f"Input: {particles[0]}")
        results = diag.analyze_particle_errors(particles)
        valid_mask = diag.get_valid_mask(particles)
        print(f"Valid: {valid_mask[0]}")
        if results['total_problematic'] > 0:
            for cat, data in results['categories'].items():
                if data['count'] > 0:
                    print(f"  {cat}: {data['count']}")


if __name__ == "__main__":
    test_error_cases()

    # Example: analyze realistic distribution with some errors
    print("\n" + "=" * 60)
    print("Example: realistic beam with tracking errors")
    print("=" * 60)

    np.random.seed(42)
    N = 10000

    # Normal distribution
    particles_cosy = np.random.normal(0, [1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 0.001], (N, 6))

    # Inject 2% negative KE (particle loss)
    n_lost = int(0.02 * N)
    particles_cosy[:n_lost, 5] = np.random.uniform(-1.5, -1.01, n_lost)

    # Inject 0.5% invalid momentum (tracking error)
    n_bad = int(0.005 * N)
    particles_cosy[n_lost:n_lost + n_bad, 1] = 0.8
    particles_cosy[n_lost:n_lost + n_bad, 3] = 0.8

    diag = CoordinateTransformationDiagnostics(KE0=40.0, E0=0.511)
    results = diag.analyze_particle_errors(particles_cosy)
    diag.print_summary(results)

    # Show how to filter
    valid_mask = diag.get_valid_mask(particles_cosy)
    print(f"\nFiltering: {np.sum(valid_mask):,}/{N:,} particles are valid")
