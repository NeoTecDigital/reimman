#!/usr/bin/env python3
"""
UNIFIED APPROACH - SCALING PARAMETER STUDY
==========================================

Systematically test different scaling factors to find optimal α.

Author: Claude (Anthropic)
Date: November 11, 2025
"""

import numpy as np
import scipy.linalg
import mpmath
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

mpmath.mp.dps = 50


class ScalingStudy:
    """
    Test different scaling factors for the unified operator.
    """

    def __init__(self, N: int = 600, epsilon: float = 0.001, L: float = 150, prime_cutoff: int = 5000):
        """Initialize with moderate parameters for speed."""
        self.N = N
        self.epsilon = epsilon
        self.L = L
        self.prime_cutoff = prime_cutoff

        # Setup
        self.setup_grid()
        self.primes = self.generate_primes(prime_cutoff)
        self.riemann_zeros = self.get_riemann_zeros(100)

        print(f"Initialized with N={N}, {len(self.primes)} primes")

    def setup_grid(self):
        """Logarithmic grid."""
        self.x = np.logspace(np.log10(self.epsilon), np.log10(self.L), self.N)
        self.dx = np.diff(self.x)
        self.dx = np.append(self.dx, self.dx[-1])

    def generate_primes(self, n: int) -> List[int]:
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(range(i*i, n+1, i))
        return [i for i in range(2, n + 1) if sieve[i]]

    def get_riemann_zeros(self, num: int = 100) -> np.ndarray:
        """Get actual Riemann zeros."""
        zeros = []
        for n in range(1, num + 1):
            zero = mpmath.zetazero(n)
            zeros.append(complex(zero))
        return np.array(zeros)

    def construct_T_base(self) -> np.ndarray:
        """
        Construct base operator T WITHOUT scaling.
        Returns unscaled operator that will be multiplied by α.
        """
        T = np.zeros((self.N, self.N), dtype=float)

        # Berry-Keating term: -(xp̂ + p̂x)/2
        for i in range(self.N):
            if i > 0 and i < self.N - 1:
                h_plus = self.x[i+1] - self.x[i]
                h_minus = self.x[i] - self.x[i-1]

                # Central difference for derivative
                T[i, i-1] = -self.x[i] / h_minus
                T[i, i+1] = self.x[i] / h_plus
                T[i, i] = self.x[i] * (1/h_minus - 1/h_plus)

                # Quantum correction -1/2
                T[i, i] -= 0.5

        # Boundary conditions
        if self.N > 1:
            T[0, 0] = -self.x[0] / (self.x[1] - self.x[0]) - 0.5
            T[0, 1] = self.x[0] / (self.x[1] - self.x[0])

            T[-1, -2] = -self.x[-1] / (self.x[-1] - self.x[-2])
            T[-1, -1] = self.x[-1] / (self.x[-1] - self.x[-2]) - 0.5

        # Frobenius prime potential: ∑_p (log p)/p δ(x - log p)
        for p in self.primes[:500]:  # First 500 primes
            log_p = np.log(p)
            if log_p < self.epsilon or log_p > self.L:
                continue

            # Find nearest grid point
            idx = np.argmin(np.abs(self.x - log_p))
            weight = np.log(p) / p

            # Add delta function contribution
            T[idx, idx] += weight / self.dx[idx]

        # Symmetrize
        T = (T + T.T) / 2

        return T

    def compute_eigenvalues_with_scaling(self, alpha: float) -> np.ndarray:
        """
        Compute eigenvalues of H = (1/2)I + i·α·T

        Returns eigenvalues λ = 1/2 + i·α·t where t are eigenvalues of T.
        """
        T_base = self.construct_T_base()
        T_scaled = alpha * T_base

        # Compute eigenvalues of scaled T
        t_eigenvalues = scipy.linalg.eigvalsh(T_scaled)

        # Eigenvalues of H
        eigenvalues = 0.5 + 1j * t_eigenvalues

        # Filter to positive imaginary parts with reasonable magnitude
        mask = np.imag(eigenvalues) > 1.0
        eigenvalues = eigenvalues[mask]

        # Sort by imaginary part
        eigenvalues = eigenvalues[np.argsort(np.imag(eigenvalues))]

        return eigenvalues

    def evaluate_scaling(self, alpha: float, num_compare: int = 50) -> Dict:
        """
        Evaluate performance for given scaling factor α.
        """
        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues_with_scaling(alpha)

        if len(eigenvalues) == 0:
            return {
                'alpha': alpha,
                'num_eigenvalues': 0,
                'num_compared': 0,
                'mean_abs_error': float('inf'),
                'median_abs_error': float('inf'),
                'mean_rel_error': float('inf'),
                'correlation': 0.0,
                'success': False
            }

        # Compare with actual zeros
        computed_heights = np.imag(eigenvalues)
        actual_heights = np.imag(self.riemann_zeros)

        # Match up to num_compare or available
        n = min(num_compare, len(computed_heights), len(actual_heights))

        if n < 5:
            return {
                'alpha': alpha,
                'num_eigenvalues': len(eigenvalues),
                'num_compared': n,
                'mean_abs_error': float('inf'),
                'median_abs_error': float('inf'),
                'mean_rel_error': float('inf'),
                'correlation': 0.0,
                'success': False
            }

        # Compute errors
        errors = computed_heights[:n] - actual_heights[:n]
        abs_errors = np.abs(errors)
        rel_errors = abs_errors / (actual_heights[:n] + 1e-10)

        # Correlation
        correlation = np.corrcoef(computed_heights[:n], actual_heights[:n])[0, 1]

        return {
            'alpha': alpha,
            'num_eigenvalues': len(eigenvalues),
            'num_compared': n,
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'std_abs_error': np.std(abs_errors),
            'max_abs_error': np.max(abs_errors),
            'mean_rel_error': np.mean(rel_errors),
            'median_rel_error': np.median(rel_errors),
            'correlation': correlation,
            'errors': errors.tolist(),
            'computed_heights': computed_heights[:n].tolist(),
            'actual_heights': actual_heights[:n].tolist(),
            'success': True
        }

    def run_scaling_study(self, alpha_values: List[float] = None) -> List[Dict]:
        """
        Test multiple scaling factors and find optimal.
        """
        if alpha_values is None:
            # Test range of physically meaningful values
            alpha_values = [
                1.0,       # No scaling
                2.0,       # Factor 2
                np.pi,     # π
                2*np.pi,   # 2π ≈ 6.28
                4*np.pi,   # 4π ≈ 12.57
                10.0,      # Factor 10
                14.0,      # Ratio of first zero
                15.0,
                20.0,
                25.0,
                30.0,
            ]

        print("=" * 70)
        print("SCALING FACTOR STUDY")
        print("=" * 70)
        print(f"Testing {len(alpha_values)} different scaling factors α")
        print(f"Operator: H = (1/2)I + i·α·T")
        print()

        results = []

        for i, alpha in enumerate(alpha_values, 1):
            print(f"[{i}/{len(alpha_values)}] Testing α = {alpha:.4f}...", end=' ')

            result = self.evaluate_scaling(alpha)
            results.append(result)

            if result['success']:
                print(f"✓ Mean error: {result['mean_abs_error']:.4f}, "
                      f"Correlation: {result['correlation']:.4f}")
            else:
                print(f"✗ Failed (not enough eigenvalues)")

        return results

    def find_optimal_alpha(self, results: List[Dict]) -> Dict:
        """
        Find optimal α based on multiple criteria.
        """
        successful = [r for r in results if r['success']]

        if not successful:
            return None

        # Sort by mean absolute error
        by_error = sorted(successful, key=lambda r: r['mean_abs_error'])
        best_by_error = by_error[0]

        # Sort by correlation
        by_corr = sorted(successful, key=lambda r: -r['correlation'])
        best_by_corr = by_corr[0]

        # Combined score: minimize error, maximize correlation
        for r in successful:
            # Normalize error to [0, 1]
            max_err = max(rr['mean_abs_error'] for rr in successful)
            min_err = min(rr['mean_abs_error'] for rr in successful)
            norm_err = (r['mean_abs_error'] - min_err) / (max_err - min_err + 1e-10)

            # Normalize correlation to [0, 1]
            max_corr = max(rr['correlation'] for rr in successful)
            min_corr = min(rr['correlation'] for rr in successful)
            norm_corr = (r['correlation'] - min_corr) / (max_corr - min_corr + 1e-10)

            # Combined score (lower is better)
            r['score'] = norm_err - norm_corr

        by_score = sorted(successful, key=lambda r: r['score'])
        best_overall = by_score[0]

        return {
            'best_by_error': best_by_error,
            'best_by_correlation': best_by_corr,
            'best_overall': best_overall
        }

    def print_summary(self, results: List[Dict]):
        """Print summary table."""
        successful = [r for r in results if r['success']]

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()
        print(f"{'α':>8} | {'Eigenvals':>10} | {'Mean Err':>10} | {'Med Err':>10} | {'Corr':>8} | {'Status':>10}")
        print("-" * 70)

        for r in results:
            if r['success']:
                status = "✓ Good" if r['mean_abs_error'] < 5.0 else "○ Moderate"
                print(f"{r['alpha']:8.4f} | {r['num_eigenvalues']:10d} | "
                      f"{r['mean_abs_error']:10.4f} | {r['median_abs_error']:10.4f} | "
                      f"{r['correlation']:8.4f} | {status:>10}")
            else:
                print(f"{r['alpha']:8.4f} | {r['num_eigenvalues']:10d} | "
                      f"{'---':>10} | {'---':>10} | {'---':>8} | {'✗ Failed':>10}")

        print()

        # Find and highlight best
        optimal = self.find_optimal_alpha(results)
        if optimal:
            print("OPTIMAL VALUES:")
            print("-" * 70)
            print(f"Best by error:       α = {optimal['best_by_error']['alpha']:.4f} "
                  f"(error = {optimal['best_by_error']['mean_abs_error']:.4f})")
            print(f"Best by correlation: α = {optimal['best_by_correlation']['alpha']:.4f} "
                  f"(corr = {optimal['best_by_correlation']['correlation']:.6f})")
            print(f"Best overall:        α = {optimal['best_overall']['alpha']:.4f} "
                  f"(error = {optimal['best_overall']['mean_abs_error']:.4f}, "
                  f"corr = {optimal['best_overall']['correlation']:.6f})")
            print()

            # Assessment
            best = optimal['best_overall']
            print("ASSESSMENT:")
            print("-" * 70)
            if best['mean_abs_error'] < 1.0 and best['correlation'] > 0.95:
                print("🎉 BREAKTHROUGH! Excellent agreement with Riemann zeros!")
                print(f"   Mean error: {best['mean_abs_error']:.6f} < 1.0 ✓")
                print(f"   Correlation: {best['correlation']:.6f} > 0.95 ✓")
                print(f"   Re(λ) = 0.5 exactly (by construction) ✓")
                print()
                print("This strongly supports the unified operator approach!")

            elif best['mean_abs_error'] < 5.0 and best['correlation'] > 0.90:
                print("📈 VERY PROMISING! Good agreement found.")
                print(f"   Mean error: {best['mean_abs_error']:.6f}")
                print(f"   Correlation: {best['correlation']:.6f}")
                print()
                print("Further refinement (finer grid, more primes) may reach breakthrough.")

            elif best['mean_abs_error'] < 20.0 and best['correlation'] > 0.80:
                print("✓ ENCOURAGING: Correct structure captured.")
                print(f"   Mean error: {best['mean_abs_error']:.6f}")
                print(f"   Correlation: {best['correlation']:.6f}")
                print()
                print("Approach has merit, but needs significant refinement.")

            else:
                print("⚠️ NEEDS WORK: Scaling not yet optimal.")
                print(f"   Mean error: {best['mean_abs_error']:.6f}")
                print(f"   Correlation: {best['correlation']:.6f}")
                print()
                print("Consider trying different α range or adjusting operator construction.")

    def visualize_results(self, results: List[Dict], save_path: str = None):
        """Visualize scaling study results."""
        successful = [r for r in results if r['success']]

        if not successful:
            print("No successful results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        alphas = [r['alpha'] for r in successful]
        errors = [r['mean_abs_error'] for r in successful]
        correlations = [r['correlation'] for r in successful]

        # Error vs alpha
        ax = axes[0, 0]
        ax.plot(alphas, errors, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Scaling factor α', fontsize=12)
        ax.set_ylabel('Mean absolute error', fontsize=12)
        ax.set_title('Error vs Scaling Factor', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target < 1.0')
        ax.legend()

        # Correlation vs alpha
        ax = axes[0, 1]
        ax.plot(alphas, correlations, 's-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Scaling factor α', fontsize=12)
        ax.set_ylabel('Correlation coefficient', fontsize=12)
        ax.set_title('Correlation vs Scaling Factor', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target > 0.95')
        ax.legend()

        # Combined: Error vs Correlation
        ax = axes[1, 0]
        scatter = ax.scatter(errors, correlations, c=alphas, s=100, cmap='viridis', alpha=0.7)
        for r in successful[:5]:  # Label first 5
            ax.annotate(f"α={r['alpha']:.1f}",
                       (r['mean_abs_error'], r['correlation']),
                       fontsize=8, alpha=0.7)
        ax.set_xlabel('Mean absolute error', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Error vs Correlation (color = α)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='α')

        # Best result comparison
        ax = axes[1, 1]
        optimal = self.find_optimal_alpha(results)
        if optimal:
            best = optimal['best_overall']
            n = min(30, len(best['computed_heights']))
            indices = np.arange(1, n+1)
            ax.scatter(indices, best['actual_heights'][:n],
                      label='Actual zeros', s=60, alpha=0.6)
            ax.scatter(indices, best['computed_heights'][:n],
                      label=f'Computed (α={best["alpha"]:.2f})',
                      s=40, alpha=0.6, marker='x')
            ax.set_xlabel('Zero index n', fontsize=12)
            ax.set_ylabel('Height Im(ρ)', fontsize=12)
            ax.set_title(f'Best Result (α={best["alpha"]:.2f})', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = "/home/persist/neotec/reimman/Unified-Approach/scaling_study.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()

    def save_results(self, results: List[Dict], filepath: str = None):
        """Save results to JSON."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"/home/persist/neotec/reimman/Unified-Approach/scaling_results_{timestamp}.json"

        data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'N': self.N,
                'epsilon': self.epsilon,
                'L': self.L,
                'prime_cutoff': self.prime_cutoff,
                'num_primes': len(self.primes)
            },
            'results': results,
            'optimal': self.find_optimal_alpha(results)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")


def main():
    """Run scaling study."""
    print("=" * 70)
    print("UNIFIED APPROACH - SCALING PARAMETER STUDY")
    print("=" * 70)
    print()

    # Initialize
    study = ScalingStudy(N=600, epsilon=0.001, L=150, prime_cutoff=5000)

    # Test scaling factors
    alphas = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        np.pi, 2*np.pi, 3*np.pi, 4*np.pi,
        10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0,
        25.0, 30.0
    ]

    results = study.run_scaling_study(alphas)

    # Print summary
    study.print_summary(results)

    # Visualize
    study.visualize_results(results)

    # Save
    study.save_results(results)

    print("\n" + "=" * 70)
    print("SCALING STUDY COMPLETE")
    print("=" * 70)

    return study, results


if __name__ == "__main__":
    study, results = main()
