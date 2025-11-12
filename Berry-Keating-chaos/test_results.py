#!/usr/bin/env python3
"""
Test and Validation Script for Berry-Keating Implementation
Comprehensive testing against known Riemann zeros
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy import stats
from berry_keating_implementation import (
    BerryKeatingHamiltonian,
    RiemannZeroValidator,
    QuantizationAnalyzer,
    Visualizer
)
import json
from datetime import datetime
import os

# Set high precision for mpmath
mp.mp.dps = 100

class ComprehensiveValidator:
    """
    Comprehensive validation of Berry-Keating approach
    """

    def __init__(self):
        """Initialize validator"""
        self.validator = RiemannZeroValidator()
        # Load more known zeros for thorough testing
        self.extended_zeros = self._load_extended_zeros(200)

    def _load_extended_zeros(self, n: int) -> np.ndarray:
        """Load extended list of known zeros"""
        zeros = []
        for k in range(1, n + 1):
            zero = mp.zetazero(k)
            zeros.append(float(zero.imag))
        return np.array(zeros)

    def test_convergence_with_N(self) -> dict:
        """
        Test how accuracy improves with grid size N
        """
        print("\n" + "=" * 50)
        print("Testing convergence with grid size N")
        print("=" * 50)

        N_values = [50, 100, 200, 300, 400, 500, 600]
        errors = []
        correlations = []

        for N in N_values:
            print(f"Testing N = {N}...")
            ham = BerryKeatingHamiltonian(N=N, L=50.0, epsilon=0.01)
            H = ham.build_hamiltonian('weyl')
            eigenvalues = ham.compute_eigenvalues(H, n_eigs=50)
            zeros = ham.extract_zeros(eigenvalues)
            metrics = self.validator.compare_zeros(zeros, n_compare=30)

            errors.append(metrics['mean_absolute_error'])
            correlations.append(metrics['correlation'])

        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(N_values, errors, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Grid Size N')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Error Convergence with Grid Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        ax2.plot(N_values, correlations, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Grid Size N')
        ax2.set_ylabel('Correlation with Known Zeros')
        ax2.set_title('Correlation Improvement')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('visualizations/convergence_test.png', dpi=150)
        plt.show()

        return {
            'N_values': N_values,
            'errors': errors,
            'correlations': correlations
        }

    def test_critical_line_property(self) -> dict:
        """
        Test if computed zeros lie on critical line Re(s) = 1/2
        """
        print("\n" + "=" * 50)
        print("Testing critical line property")
        print("=" * 50)

        ham = BerryKeatingHamiltonian(N=500, L=75.0, epsilon=0.01)

        results = {}
        for ordering in ['standard', 'anti', 'weyl']:
            print(f"\nTesting {ordering} ordering...")
            H = ham.build_hamiltonian(ordering)
            eigenvalues = ham.compute_eigenvalues(H, n_eigs=100)
            zeros = ham.extract_zeros(eigenvalues)

            # Check real parts
            real_parts = zeros.real
            deviations = np.abs(real_parts - 0.5)

            results[ordering] = {
                'mean_real_part': np.mean(real_parts),
                'std_real_part': np.std(real_parts),
                'max_deviation': np.max(deviations),
                'mean_deviation': np.mean(deviations),
                'on_critical_line': np.all(deviations < 0.01)  # Tolerance
            }

            print(f"  Mean Re(ρ): {results[ordering]['mean_real_part']:.6f}")
            print(f"  Std Re(ρ): {results[ordering]['std_real_part']:.6f}")
            print(f"  Max deviation from 1/2: {results[ordering]['max_deviation']:.6f}")
            print(f"  On critical line: {results[ordering]['on_critical_line']}")

        return results

    def test_gue_statistics(self) -> dict:
        """
        Test if zero spacings follow GUE (Gaussian Unitary Ensemble) statistics
        """
        print("\n" + "=" * 50)
        print("Testing GUE statistics")
        print("=" * 50)

        ham = BerryKeatingHamiltonian(N=600, L=100.0, epsilon=0.001)
        H = ham.build_hamiltonian('weyl')
        eigenvalues = ham.compute_eigenvalues(H, n_eigs=200)

        # Get imaginary parts (heights of zeros)
        heights = np.sort(np.abs(eigenvalues.imag))
        heights = heights[heights > 0]  # Remove any zero eigenvalue

        # Compute normalized spacings
        spacings = np.diff(heights)
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing

        # GUE prediction for nearest neighbor spacing
        # P(s) ≈ (32/π²)s²exp(-4s²/π) for small s
        s_theory = np.linspace(0, 4, 100)
        p_gue = (32/np.pi**2) * s_theory**2 * np.exp(-4*s_theory**2/np.pi)

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of spacings
        ax1.hist(normalized_spacings, bins=30, density=True, alpha=0.7, label='Computed')
        ax1.plot(s_theory, p_gue, 'r-', linewidth=2, label='GUE prediction')
        ax1.set_xlabel('Normalized spacing s')
        ax1.set_ylabel('Probability density')
        ax1.set_title('Nearest Neighbor Spacing Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative distribution
        ax2.hist(normalized_spacings, bins=30, density=True, cumulative=True,
                alpha=0.7, label='Computed')
        ax2.set_xlabel('Normalized spacing s')
        ax2.set_ylabel('Cumulative probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/gue_statistics.png', dpi=150)
        plt.show()

        # Compute Kolmogorov-Smirnov test
        # Note: This is simplified - proper test would need exact GUE distribution
        ks_statistic = np.max(np.abs(np.sort(normalized_spacings) -
                                     np.linspace(0, 1, len(normalized_spacings))))

        return {
            'mean_spacing': mean_spacing,
            'normalized_spacings': normalized_spacings.tolist()[:50],  # First 50 for report
            'ks_statistic': ks_statistic,
            'follows_gue': ks_statistic < 0.1  # Rough criterion
        }

    def test_different_regularizations(self) -> dict:
        """
        Test different regularization schemes
        """
        print("\n" + "=" * 50)
        print("Testing different regularization parameters")
        print("=" * 50)

        epsilon_values = [0.001, 0.01, 0.1, 1.0]
        L_values = [10, 25, 50, 100, 200]

        results = []

        for epsilon in epsilon_values:
            for L in L_values:
                print(f"Testing epsilon={epsilon}, L={L}")
                try:
                    ham = BerryKeatingHamiltonian(N=300, L=L, epsilon=epsilon)
                    H = ham.build_hamiltonian('weyl')
                    eigenvalues = ham.compute_eigenvalues(H, n_eigs=50)
                    zeros = ham.extract_zeros(eigenvalues)
                    metrics = self.validator.compare_zeros(zeros, n_compare=20)

                    results.append({
                        'epsilon': epsilon,
                        'L': L,
                        'error': metrics['mean_absolute_error'],
                        'correlation': metrics['correlation']
                    })
                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    results.append({
                        'epsilon': epsilon,
                        'L': L,
                        'error': np.inf,
                        'correlation': 0
                    })

        # Find best combination
        best = min(results, key=lambda x: x['error'])
        print(f"\nBest regularization: epsilon={best['epsilon']}, L={best['L']}")
        print(f"Error: {best['error']:.6f}")

        return {
            'results': results,
            'best': best
        }

    def test_first_100_zeros(self) -> dict:
        """
        Comprehensive test against first 100 known zeros
        """
        print("\n" + "=" * 50)
        print("Testing against first 100 known zeros")
        print("=" * 50)

        # Use optimized parameters
        ham = BerryKeatingHamiltonian(N=800, L=150.0, epsilon=0.001)
        H = ham.build_hamiltonian('weyl')

        print("Computing eigenvalues (this may take a moment)...")
        eigenvalues = ham.compute_eigenvalues(H, n_eigs=150)
        zeros = ham.extract_zeros(eigenvalues)

        # Compare with known zeros
        computed_heights = np.sort(np.abs(zeros.imag))[:100]
        known_heights = self.extended_zeros[:100]

        # Compute various metrics
        absolute_errors = np.abs(computed_heights - known_heights)
        relative_errors = absolute_errors / known_heights

        # Statistical tests
        correlation = np.corrcoef(computed_heights, known_heights)[0, 1]
        rmse = np.sqrt(np.mean(absolute_errors**2))

        # Success criteria
        success_rate = np.sum(relative_errors < 0.01) / len(relative_errors)  # Within 1%

        results = {
            'n_zeros_tested': 100,
            'mean_absolute_error': np.mean(absolute_errors),
            'median_absolute_error': np.median(absolute_errors),
            'max_absolute_error': np.max(absolute_errors),
            'mean_relative_error': np.mean(relative_errors),
            'median_relative_error': np.median(relative_errors),
            'max_relative_error': np.max(relative_errors),
            'rmse': rmse,
            'correlation': correlation,
            'success_rate': success_rate,
            'first_10_computed': computed_heights[:10].tolist(),
            'first_10_known': known_heights[:10].tolist(),
            'first_10_errors': absolute_errors[:10].tolist()
        }

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Direct comparison
        ax = axes[0, 0]
        ax.scatter(known_heights, computed_heights, alpha=0.5, s=10)
        ax.plot(known_heights, known_heights, 'r--', alpha=0.5, label='Perfect match')
        ax.set_xlabel('Known zeros')
        ax.set_ylabel('Computed zeros')
        ax.set_title(f'First 100 Zeros Comparison\nCorrelation: {correlation:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[0, 1]
        ax.hist(absolute_errors, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Absolute error')
        ax.set_ylabel('Count')
        ax.set_title(f'Error Distribution\nMean: {np.mean(absolute_errors):.4f}')
        ax.grid(True, alpha=0.3)

        # Error vs zero index
        ax = axes[1, 0]
        ax.plot(range(1, 101), absolute_errors, 'o-', markersize=3)
        ax.set_xlabel('Zero index')
        ax.set_ylabel('Absolute error')
        ax.set_title('Error vs Zero Index')
        ax.grid(True, alpha=0.3)

        # Relative error (log scale)
        ax = axes[1, 1]
        ax.semilogy(range(1, 101), relative_errors, 'o-', markersize=3)
        ax.set_xlabel('Zero index')
        ax.set_ylabel('Relative error')
        ax.set_title('Relative Error (log scale)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/first_100_zeros_test.png', dpi=150)
        plt.show()

        print(f"\nResults for first 100 zeros:")
        print(f"  Mean absolute error: {results['mean_absolute_error']:.6f}")
        print(f"  Median absolute error: {results['median_absolute_error']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        print(f"  Correlation: {results['correlation']:.6f}")
        print(f"  Success rate (< 1% error): {results['success_rate']*100:.1f}%")

        return results


def run_all_tests():
    """
    Run comprehensive test suite
    """
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION OF BERRY-KEATING APPROACH")
    print("=" * 60)

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    validator = ComprehensiveValidator()

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Run all tests
    print("\n1. CONVERGENCE TEST")
    all_results['tests']['convergence'] = validator.test_convergence_with_N()

    print("\n2. CRITICAL LINE TEST")
    all_results['tests']['critical_line'] = validator.test_critical_line_property()

    print("\n3. GUE STATISTICS TEST")
    all_results['tests']['gue_statistics'] = validator.test_gue_statistics()

    print("\n4. REGULARIZATION TEST")
    all_results['tests']['regularization'] = validator.test_different_regularizations()

    print("\n5. FIRST 100 ZEROS TEST")
    all_results['tests']['first_100_zeros'] = validator.test_first_100_zeros()

    # Save results to JSON
    with open('test_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        json.dump(convert_numpy(all_results), f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print("\n✓ Convergence Test:")
    print(f"  - Tested grid sizes: {all_results['tests']['convergence']['N_values']}")
    print(f"  - Best error achieved: {min(all_results['tests']['convergence']['errors']):.6f}")

    print("\n✓ Critical Line Test:")
    for ordering, data in all_results['tests']['critical_line'].items():
        print(f"  - {ordering}: On critical line = {data['on_critical_line']}")

    print("\n✓ GUE Statistics Test:")
    print(f"  - Follows GUE: {all_results['tests']['gue_statistics']['follows_gue']}")
    print(f"  - KS statistic: {all_results['tests']['gue_statistics']['ks_statistic']:.6f}")

    print("\n✓ Regularization Test:")
    best = all_results['tests']['regularization']['best']
    print(f"  - Best params: ε={best['epsilon']}, L={best['L']}")
    print(f"  - Best error: {best['error']:.6f}")

    print("\n✓ First 100 Zeros Test:")
    test_100 = all_results['tests']['first_100_zeros']
    print(f"  - Mean absolute error: {test_100['mean_absolute_error']:.6f}")
    print(f"  - Correlation: {test_100['correlation']:.6f}")
    print(f"  - Success rate: {test_100['success_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    # Determine overall success
    success_criteria = [
        test_100['correlation'] > 0.9,
        test_100['mean_absolute_error'] < 5.0,
        all_results['tests']['critical_line']['weyl']['on_critical_line'],
        all_results['tests']['gue_statistics']['follows_gue']
    ]

    if all(success_criteria):
        print("✅ APPROACH SHOWS PROMISE")
        print("The Berry-Keating approach demonstrates correlation with Riemann zeros")
    elif sum(success_criteria) >= 2:
        print("⚠️ PARTIAL SUCCESS")
        print("Some aspects work but accuracy needs improvement")
    else:
        print("❌ APPROACH NEEDS REFINEMENT")
        print("Significant deviations from expected behavior")

    print("\nResults saved to:")
    print("  - test_results.json (numerical data)")
    print("  - visualizations/ (plots)")

    return all_results


if __name__ == "__main__":
    results = run_all_tests()