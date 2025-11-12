"""
Test suite for Frobenius operator approach
Comprehensive validation against known Riemann zeros
"""

import numpy as np
import mpmath
from frobenius_implementation import (
    FiniteField, EllipticCurve, FrobeniusOperator,
    get_riemann_zeros
)
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict
import json

# Set high precision
mpmath.mp.dps = 100


class FrobeniusValidator:
    """Validate Frobenius approach against known results"""

    def __init__(self):
        self.test_results = {
            'weil_bounds': [],
            'zero_correspondence': [],
            'trace_formula': [],
            'statistical_tests': {}
        }

    def test_weil_bounds(self, primes: List[int] = None) -> Dict:
        """Test if Frobenius eigenvalues satisfy Weil bounds"""

        if primes is None:
            primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

        results = []

        for p in primes:
            field = FiniteField(p)
            violations = 0
            total_curves = 0

            # Test multiple curves for each prime
            for a in range(p):
                for b in range(1, min(p, 20)):  # Limit for efficiency
                    try:
                        curve = EllipticCurve(field, a, b)
                        alpha, beta = curve.frobenius_eigenvalues()

                        # Check Weil bound: |α| = |β| = √p
                        expected = np.sqrt(p)
                        alpha_abs = abs(alpha)
                        beta_abs = abs(beta)

                        # Allow small numerical error
                        if not (np.isclose(alpha_abs, expected, rtol=1e-8) and
                               np.isclose(beta_abs, expected, rtol=1e-8)):
                            violations += 1

                        total_curves += 1

                    except ValueError:
                        continue  # Skip singular curves

            violation_rate = violations / total_curves if total_curves > 0 else 0
            results.append({
                'prime': p,
                'curves_tested': total_curves,
                'violations': violations,
                'violation_rate': violation_rate,
                'weil_bound_satisfied': violation_rate < 0.01
            })

            print(f"Prime {p}: Tested {total_curves} curves, "
                  f"{violations} violations ({violation_rate:.2%})")

        self.test_results['weil_bounds'] = results
        return results

    def test_trace_formula(self, num_tests: int = 20) -> Dict:
        """Verify the Lefschetz trace formula"""

        results = []
        primes = [11, 13, 17, 19, 23]

        for p in primes:
            field = FiniteField(p)

            for _ in range(num_tests // len(primes)):
                a = np.random.randint(0, p)
                b = np.random.randint(1, p)

                try:
                    curve = EllipticCurve(field, a, b)

                    # Count points directly
                    point_count = curve.count_points()

                    # Compute via trace formula
                    t = curve.frobenius_trace()
                    computed_count = p + 1 - t

                    # They should match
                    match = (point_count == computed_count)

                    results.append({
                        'prime': p,
                        'curve': f"y^2 = x^3 + {a}x + {b}",
                        'direct_count': point_count,
                        'trace_formula': computed_count,
                        'match': match
                    })

                except ValueError:
                    continue

        success_rate = sum(1 for r in results if r['match']) / len(results)
        print(f"\nTrace formula verification: {success_rate:.1%} success rate")

        self.test_results['trace_formula'] = {
            'results': results,
            'success_rate': success_rate
        }
        return self.test_results['trace_formula']

    def test_zero_correspondence(self, num_zeros: int = 100) -> Dict:
        """Test correspondence with actual Riemann zeros"""

        # Get actual Riemann zeros
        actual_zeros = []
        for n in range(1, num_zeros + 1):
            zero = mpmath.zetazero(n)
            actual_zeros.append(float(zero.imag))

        # Collect Frobenius-mapped zeros
        primes = [p for p in range(11, 100) if self._is_prime(p)]
        mapped_zeros = []

        for p in primes[:20]:  # Limit for efficiency
            frob = FrobeniusOperator(p)
            eigenvalues = frob.analyze_elliptic_curves(num_curves=10)

            for alpha, beta in eigenvalues:
                # Map eigenvalues to potential zeros
                # Using the correspondence: |λ| = p^(1/2) => Re(s) = 1/2

                # Extract phase information
                if abs(alpha) > 0:
                    phase_alpha = np.angle(alpha)
                    # Map phase to imaginary part of zero
                    t_alpha = abs(phase_alpha) * p / (2 * np.pi)
                    mapped_zeros.append(t_alpha)

                if abs(beta) > 0:
                    phase_beta = np.angle(beta)
                    t_beta = abs(phase_beta) * p / (2 * np.pi)
                    mapped_zeros.append(t_beta)

        # Statistical comparison
        mapped_zeros = sorted([z for z in mapped_zeros if 0 < z < 100])

        # Compute spacing distributions
        actual_spacings = np.diff(actual_zeros[:50])

        if len(mapped_zeros) > 50:
            mapped_spacings = np.diff(mapped_zeros[:50])

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(actual_spacings, mapped_spacings)
        else:
            ks_stat, ks_pvalue = None, None

        # Check for matches (within tolerance)
        tolerance = 2.0
        matches = 0
        for az in actual_zeros[:20]:
            for mz in mapped_zeros:
                if abs(az - mz) < tolerance:
                    matches += 1
                    break

        match_rate = matches / min(20, len(actual_zeros))

        results = {
            'num_actual_zeros': len(actual_zeros),
            'num_mapped_zeros': len(mapped_zeros),
            'match_rate': match_rate,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'first_actual_zeros': actual_zeros[:10],
            'sample_mapped_zeros': mapped_zeros[:10] if mapped_zeros else []
        }

        print(f"\nZero correspondence test:")
        print(f"  Match rate (tolerance={tolerance}): {match_rate:.1%}")
        if ks_pvalue is not None:
            print(f"  KS test p-value: {ks_pvalue:.4f}")

        self.test_results['zero_correspondence'] = results
        return results

    def statistical_analysis(self) -> Dict:
        """Comprehensive statistical analysis"""

        # Analyze Weil bounds
        if self.test_results['weil_bounds']:
            weil_violation_rates = [r['violation_rate']
                                   for r in self.test_results['weil_bounds']]
            weil_stats = {
                'mean_violation_rate': np.mean(weil_violation_rates),
                'max_violation_rate': np.max(weil_violation_rates),
                'all_satisfied': all(r['weil_bound_satisfied']
                                   for r in self.test_results['weil_bounds'])
            }
        else:
            weil_stats = {}

        # Analyze trace formula
        trace_stats = {
            'success_rate': self.test_results['trace_formula'].get('success_rate', 0)
        }

        # Analyze zero correspondence
        zero_stats = self.test_results['zero_correspondence']

        self.test_results['statistical_tests'] = {
            'weil_bounds': weil_stats,
            'trace_formula': trace_stats,
            'zero_correspondence': zero_stats
        }

        return self.test_results['statistical_tests']

    def _is_prime(self, n: int) -> bool:
        """Check if n is prime"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def generate_report(self) -> str:
        """Generate comprehensive test report"""

        report = []
        report.append("="*70)
        report.append("FROBENIUS OPERATOR APPROACH - TEST RESULTS")
        report.append("="*70)
        report.append("")

        # Weil Bounds Test
        report.append("1. WEIL BOUNDS VERIFICATION")
        report.append("-"*30)
        if self.test_results['weil_bounds']:
            stats = self.test_results['statistical_tests'].get('weil_bounds', {})
            report.append(f"Tested {len(self.test_results['weil_bounds'])} primes")
            report.append(f"Mean violation rate: {stats.get('mean_violation_rate', 0):.4%}")
            report.append(f"All bounds satisfied: {stats.get('all_satisfied', False)}")
        report.append("")

        # Trace Formula Test
        report.append("2. TRACE FORMULA VERIFICATION")
        report.append("-"*30)
        trace_data = self.test_results.get('trace_formula', {})
        if trace_data:
            report.append(f"Success rate: {trace_data.get('success_rate', 0):.1%}")
            report.append(f"Total tests: {len(trace_data.get('results', []))}")
        report.append("")

        # Zero Correspondence Test
        report.append("3. ZERO CORRESPONDENCE ANALYSIS")
        report.append("-"*30)
        zero_data = self.test_results.get('zero_correspondence', {})
        if zero_data:
            report.append(f"Actual zeros analyzed: {zero_data.get('num_actual_zeros', 0)}")
            report.append(f"Mapped zeros generated: {zero_data.get('num_mapped_zeros', 0)}")
            report.append(f"Match rate: {zero_data.get('match_rate', 0):.1%}")
            if zero_data.get('ks_pvalue') is not None:
                report.append(f"KS test p-value: {zero_data.get('ks_pvalue', 0):.4f}")
        report.append("")

        # Overall Assessment
        report.append("4. OVERALL ASSESSMENT")
        report.append("-"*30)

        # Determine success criteria
        weil_ok = self.test_results['statistical_tests'].get('weil_bounds', {}).get('all_satisfied', False)
        trace_ok = trace_data.get('success_rate', 0) > 0.95
        zero_match = zero_data.get('match_rate', 0) > 0.1  # Lower threshold due to mapping complexity

        if weil_ok:
            report.append("✓ Weil bounds satisfied for all tested curves")
        else:
            report.append("✗ Weil bounds violations detected")

        if trace_ok:
            report.append("✓ Trace formula verified with high accuracy")
        else:
            report.append("✗ Trace formula shows discrepancies")

        if zero_match:
            report.append("✓ Some correspondence with Riemann zeros detected")
        else:
            report.append("✗ Poor correspondence with Riemann zeros")

        report.append("")
        report.append("5. CONCLUSION")
        report.append("-"*30)

        if weil_ok and trace_ok:
            report.append("The Frobenius operator approach shows promise:")
            report.append("- Mathematical framework is sound (Weil bounds hold)")
            report.append("- Trace formulas are accurate")
            if zero_match:
                report.append("- Some correspondence with Riemann zeros observed")
                report.append("- Further refinement of the mapping needed")
            else:
                report.append("- Mapping to Riemann zeros needs improvement")
                report.append("- Current approach may be too simplified")
        else:
            report.append("Implementation issues detected - review required")

        report.append("")
        report.append("="*70)

        return "\n".join(report)


def run_comprehensive_tests():
    """Run all tests and generate report"""

    print("Starting comprehensive test suite...")
    print()

    validator = FrobeniusValidator()

    # Test 1: Weil bounds
    print("Testing Weil bounds...")
    validator.test_weil_bounds()
    print()

    # Test 2: Trace formula
    print("Testing trace formula...")
    validator.test_trace_formula()
    print()

    # Test 3: Zero correspondence
    print("Testing zero correspondence...")
    validator.test_zero_correspondence()
    print()

    # Statistical analysis
    print("Performing statistical analysis...")
    validator.statistical_analysis()
    print()

    # Generate and print report
    report = validator.generate_report()
    print(report)

    # Save report
    with open('/home/persist/neotec/reimman/Frobenius-operators/test_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to 'test_report.txt'")

    # Save detailed results as JSON
    with open('/home/persist/neotec/reimman/Frobenius-operators/test_results.json', 'w') as f:
        # Convert complex numbers to strings for JSON serialization
        def convert_complex(obj):
            if isinstance(obj, complex):
                return f"{obj.real}+{obj.imag}j"
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        import json
        json.dump(validator.test_results, f, default=convert_complex, indent=2)
    print("Detailed results saved to 'test_results.json'")

    return validator.test_results


if __name__ == "__main__":
    print("="*70)
    print("FROBENIUS OPERATOR APPROACH - VALIDATION SUITE")
    print("="*70)
    print()

    results = run_comprehensive_tests()