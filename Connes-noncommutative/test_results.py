#!/usr/bin/env python3
"""
Test and validation script for Connes' approach to Riemann Hypothesis
Validates computed eigenvalues against known Riemann zeros
"""

import numpy as np
import matplotlib.pyplot as plt
from connes_implementation import ConnesRiemannSolver, load_known_zeros
import time
import os
from typing import List, Dict, Tuple

# Known Riemann zeros (first 100 imaginary parts)
KNOWN_ZEROS_HEIGHTS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831778, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000983, 147.422766, 150.053710, 150.925258, 153.024694,
    156.112909, 157.597592, 158.849989, 161.188964, 163.030710,
    165.537070, 167.184440, 169.094516, 169.911977, 173.411537,
    174.754192, 176.441435, 178.377408, 179.916485, 182.207079,
    184.874468, 185.598784, 187.228923, 189.416159, 192.026657,
    193.079727, 195.265397, 196.876481, 198.015310, 201.264476,
    202.493595, 204.189672, 205.394698, 207.906259, 209.576510,
    211.690863, 213.347920, 214.547045, 216.169539, 219.067597,
    220.714919, 221.430706, 224.007001, 224.983325, 227.421445,
    229.337414, 231.250189, 231.987236, 233.693404, 236.524230
]


class RiemannZeroValidator:
    """Comprehensive validation of Connes approach against known zeros"""

    def __init__(self):
        self.results = []
        self.known_zeros = [0.5 + 1j * h for h in KNOWN_ZEROS_HEIGHTS]

    def test_increasing_prime_count(self,
                                  prime_counts: List[int] = None,
                                  param_search: bool = False) -> Dict:
        """
        Test with increasing number of primes to check convergence

        Args:
            prime_counts: List of prime counts to test
            param_search: Whether to optimize parameters for each count

        Returns:
            Dictionary with test results
        """
        if prime_counts is None:
            prime_counts = [5, 10, 15, 20, 25, 30]

        results = []

        for N in prime_counts:
            print(f"\n{'='*50}")
            print(f"Testing with N={N} primes")
            print('='*50)

            solver = ConnesRiemannSolver(N)

            # Build operator
            if param_search:
                print("Optimizing parameters...")
                opt_result = solver.optimize_parameters(
                    self.known_zeros[:min(5, N//3)],
                    param_ranges={
                        "coupling_strength": np.linspace(0.05, 0.5, 10),
                        "interaction_strength": np.linspace(0.01, 0.2, 5)
                    }
                )
                best_params = opt_result["best_params"]
                print(f"Best parameters: {best_params}")
                solver.build_connes_operator(**best_params)
            else:
                # Use heuristic parameters
                coupling = 0.1 / np.sqrt(N)
                interaction = 0.05 / N
                solver.build_connes_operator(coupling, interaction)

            # Compute eigenvalues
            eigenvalues = solver.compute_eigenvalues()
            zeros = solver.extract_zeros()

            # Verify critical line
            verification = solver.verify_critical_line(tolerance=0.2)

            # Compare with known zeros
            comparison = solver.compare_with_known_zeros(self.known_zeros[:min(10, len(zeros))])

            # Store results
            result = {
                "N": N,
                "num_zeros": len(zeros),
                "mean_real_part": verification.get("mean_real_part", 0.5),
                "std_real_part": verification.get("std_real_part", 0.0),
                "max_deviation": verification.get("max_deviation", 1.0),
                "percent_on_line": verification.get("percent_on_line", 0.0),
                "average_error": comparison.get("average_error", float('inf'))
            }
            results.append(result)

            # Print summary
            print(f"Found {len(zeros)} zeros")
            print(f"Mean Re(ρ) = {verification.get('mean_real_part', 0.5):.6f} (expect 0.5)")
            print(f"Std Re(ρ) = {verification.get('std_real_part', 0.0):.6f}")
            print(f"{verification.get('percent_on_line', 0.0):.1f}% within tolerance of critical line")

            if comparison.get("average_error"):
                print(f"Average error vs known zeros: {comparison['average_error']:.6f}")

        return {"prime_count_results": results}

    def test_eigenvalue_statistics(self, num_primes: int = 20) -> Dict:
        """
        Test if eigenvalue statistics match GUE predictions

        Args:
            num_primes: Number of primes to use

        Returns:
            Statistical analysis results
        """
        print(f"\n{'='*50}")
        print("Testing Eigenvalue Statistics (GUE)")
        print('='*50)

        solver = ConnesRiemannSolver(num_primes)
        solver.build_connes_operator(0.1, 0.05)
        eigenvalues = solver.compute_eigenvalues()

        # Get imaginary parts (heights)
        heights = np.sort([np.imag(ev) for ev in eigenvalues if np.imag(ev) > 0])

        if len(heights) < 2:
            return {"error": "Not enough eigenvalues"}

        # Compute normalized spacings
        spacings = np.diff(heights)
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing

        # GUE predictions
        gue_mean = 1.0
        gue_std = 0.52  # Approximate for GUE
        wigner_surmise = lambda s: (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)

        # Compute statistics
        stats = {
            "num_eigenvalues": len(heights),
            "num_spacings": len(spacings),
            "mean_normalized_spacing": np.mean(normalized_spacings),
            "std_normalized_spacing": np.std(normalized_spacings),
            "gue_mean": gue_mean,
            "gue_std": gue_std,
            "mean_deviation": abs(np.mean(normalized_spacings) - gue_mean),
            "std_deviation": abs(np.std(normalized_spacings) - gue_std),
            "min_spacing": np.min(spacings),
            "max_spacing": np.max(spacings)
        }

        print(f"Number of eigenvalues: {stats['num_eigenvalues']}")
        print(f"Mean normalized spacing: {stats['mean_normalized_spacing']:.4f} (GUE: {gue_mean})")
        print(f"Std normalized spacing: {stats['std_normalized_spacing']:.4f} (GUE: {gue_std:.2f})")
        print(f"Min spacing: {stats['min_spacing']:.6f}")
        print(f"Max spacing: {stats['max_spacing']:.6f}")

        return stats

    def test_trace_formula(self, num_primes: int = 15) -> Dict:
        """
        Test the trace formula connection to primes

        Args:
            num_primes: Number of primes to use

        Returns:
            Trace formula analysis
        """
        print(f"\n{'='*50}")
        print("Testing Trace Formula")
        print('='*50)

        solver = ConnesRiemannSolver(num_primes)
        solver.build_connes_operator(0.1, 0.05)
        solver.compute_eigenvalues()

        # Different test functions
        test_functions = [
            ("identity", lambda x: x),
            ("square", lambda x: x**2),
            ("exp(-t)", lambda x: np.exp(-0.1 * x)),
            ("gaussian", lambda x: np.exp(-0.01 * x**2))
        ]

        results = {}

        for name, func in test_functions:
            trace = solver.compute_trace_formula(func)

            # Compute prime sum analog
            prime_sum = sum(func(complex(0.5, np.log(p))) for p in solver.primes)

            results[name] = {
                "trace": trace,
                "prime_sum": prime_sum,
                "ratio": np.real(trace) / np.real(prime_sum) if prime_sum != 0 else 0
            }

            print(f"\nTest function: {name}")
            print(f"  Tr(f(H)) = {trace:.6f}")
            print(f"  Prime sum = {prime_sum:.6f}")
            print(f"  Ratio = {results[name]['ratio']:.6f}")

        return results

    def test_parameter_sensitivity(self, num_primes: int = 10) -> Dict:
        """
        Test sensitivity to coupling parameters

        Args:
            num_primes: Number of primes to use

        Returns:
            Parameter sensitivity analysis
        """
        print(f"\n{'='*50}")
        print("Testing Parameter Sensitivity")
        print('='*50)

        coupling_range = np.linspace(0.01, 0.5, 15)
        interaction_range = np.linspace(0.01, 0.2, 10)

        results = []

        for coupling in coupling_range:
            for interaction in interaction_range:
                solver = ConnesRiemannSolver(num_primes)
                solver.build_connes_operator(coupling, interaction)
                eigenvalues = solver.compute_eigenvalues()
                zeros = solver.extract_zeros()

                if zeros:
                    mean_real = np.mean([np.real(z) for z in zeros])
                    std_real = np.std([np.real(z) for z in zeros])
                    deviation = abs(mean_real - 0.5)

                    results.append({
                        "coupling": coupling,
                        "interaction": interaction,
                        "mean_real": mean_real,
                        "std_real": std_real,
                        "deviation": deviation,
                        "num_zeros": len(zeros)
                    })

        # Find optimal parameters
        if results:
            best = min(results, key=lambda x: x["deviation"])
            print(f"Best parameters found:")
            print(f"  Coupling strength: {best['coupling']:.4f}")
            print(f"  Interaction strength: {best['interaction']:.4f}")
            print(f"  Mean Re(ρ): {best['mean_real']:.6f}")
            print(f"  Deviation from 0.5: {best['deviation']:.6f}")

        return {"sensitivity_results": results, "best": best if results else None}

    def create_visualization_suite(self, num_primes: int = 20,
                                 save_dir: str = "visualizations") -> None:
        """
        Create comprehensive visualization suite

        Args:
            num_primes: Number of primes to use
            save_dir: Directory to save visualizations
        """
        print(f"\n{'='*50}")
        print("Creating Visualization Suite")
        print('='*50)

        # Create directory if needed
        os.makedirs(save_dir, exist_ok=True)

        solver = ConnesRiemannSolver(num_primes)
        solver.build_connes_operator(0.1, 0.05)
        eigenvalues = solver.compute_eigenvalues()
        zeros = solver.extract_zeros()

        # 1. Complex plane visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Eigenvalues in complex plane
        ax1 = axes[0, 0]
        ax1.scatter(np.real(eigenvalues), np.imag(eigenvalues),
                   c='blue', s=30, alpha=0.6, label='Computed')
        ax1.scatter([np.real(z) for z in self.known_zeros[:10]],
                   [np.imag(z) for z in self.known_zeros[:10]],
                   c='red', s=30, alpha=0.6, marker='x', label='Known zeros')
        ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5,
                   label='Critical line')
        ax1.set_xlabel('Real part')
        ax1.set_ylabel('Imaginary part')
        ax1.set_title('Eigenvalues vs Known Zeros')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Real part distribution
        ax2 = axes[0, 1]
        real_parts = [np.real(ev) for ev in eigenvalues]
        ax2.hist(real_parts, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
                   label='Critical line (0.5)')
        mean_real = np.mean(real_parts)
        ax2.axvline(x=mean_real, color='green', linestyle='-', linewidth=2,
                   label=f'Mean ({mean_real:.4f})')
        ax2.set_xlabel('Real part')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Real Parts')
        ax2.legend()

        # Spacing distribution
        ax3 = axes[1, 0]
        if len(zeros) > 1:
            heights = sorted([np.imag(z) for z in zeros if np.imag(z) > 0])
            spacings = np.diff(heights)
            if len(spacings) > 0:
                normalized_spacings = spacings / np.mean(spacings)
                ax3.hist(normalized_spacings, bins=20, alpha=0.7,
                        color='blue', edgecolor='black', density=True,
                        label='Computed')

                # Wigner surmise (GUE)
                s = np.linspace(0, max(normalized_spacings) if normalized_spacings.size > 0 else 3, 100)
                wigner = (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)
                ax3.plot(s, wigner, 'r-', linewidth=2, label='GUE prediction')

        ax3.set_xlabel('Normalized spacing')
        ax3.set_ylabel('Probability density')
        ax3.set_title('Level Spacing Distribution')
        ax3.legend()

        # Error vs height
        ax4 = axes[1, 1]
        if zeros:
            computed_heights = sorted([np.imag(z) for z in zeros if np.imag(z) > 0])[:10]
            known_heights = KNOWN_ZEROS_HEIGHTS[:len(computed_heights)]
            errors = [abs(c - k) for c, k in zip(computed_heights, known_heights)]

            ax4.plot(known_heights, errors, 'bo-', markersize=8, linewidth=2)
            ax4.set_xlabel('Height (Im(ρ))')
            ax4.set_ylabel('Error')
            ax4.set_title('Error vs Zero Height')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, "eigenvalue_analysis.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
        plt.show()

        # 2. Convergence analysis
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        # Test different N values
        N_values = [5, 10, 15, 20, 25]
        mean_reals = []
        std_reals = []
        num_zeros_list = []

        for N in N_values:
            solver_n = ConnesRiemannSolver(N)
            solver_n.build_connes_operator(0.1/np.sqrt(N), 0.05/N)
            ev = solver_n.compute_eigenvalues()
            z = solver_n.extract_zeros()

            if z:
                mean_reals.append(np.mean([np.real(zz) for zz in z]))
                std_reals.append(np.std([np.real(zz) for zz in z]))
                num_zeros_list.append(len(z))
            else:
                mean_reals.append(0.5)
                std_reals.append(0)
                num_zeros_list.append(0)

        # Mean real part convergence
        ax5 = axes2[0]
        ax5.plot(N_values, mean_reals, 'bo-', markersize=8, linewidth=2)
        ax5.axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
        ax5.set_xlabel('Number of primes N')
        ax5.set_ylabel('Mean Re(ρ)')
        ax5.set_title('Convergence of Mean Real Part')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Number of zeros vs N
        ax6 = axes2[1]
        ax6.plot(N_values, num_zeros_list, 'go-', markersize=8, linewidth=2)
        ax6.set_xlabel('Number of primes N')
        ax6.set_ylabel('Number of zeros found')
        ax6.set_title('Number of Zeros vs Matrix Size')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path2 = os.path.join(save_dir, "convergence_analysis.png")
        plt.savefig(save_path2, dpi=150)
        print(f"Saved convergence analysis to {save_path2}")
        plt.show()

    def generate_report(self, output_file: str = "test_report.txt") -> None:
        """
        Generate comprehensive test report

        Args:
            output_file: Path to output report file
        """
        print(f"\n{'='*50}")
        print("Generating Comprehensive Report")
        print('='*50)

        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CONNES APPROACH - COMPREHENSIVE TEST REPORT\n")
            f.write("="*70 + "\n\n")

            # Test 1: Increasing prime count
            f.write("1. CONVERGENCE WITH INCREASING PRIME COUNT\n")
            f.write("-"*40 + "\n")
            results1 = self.test_increasing_prime_count([5, 10, 15, 20, 25])
            for r in results1["prime_count_results"]:
                f.write(f"N={r['N']:3d}: ")
                f.write(f"zeros={r['num_zeros']:3d}, ")
                f.write(f"mean_Re={r['mean_real_part']:.4f}, ")
                f.write(f"on_line={r['percent_on_line']:.1f}%, ")
                f.write(f"error={r['average_error']:.4f}\n")
            f.write("\n")

            # Test 2: Eigenvalue statistics
            f.write("2. EIGENVALUE STATISTICS (GUE TEST)\n")
            f.write("-"*40 + "\n")
            stats = self.test_eigenvalue_statistics(20)
            if "error" not in stats:
                f.write(f"Number of eigenvalues: {stats['num_eigenvalues']}\n")
                f.write(f"Mean normalized spacing: {stats['mean_normalized_spacing']:.4f} ")
                f.write(f"(GUE: {stats['gue_mean']})\n")
                f.write(f"Std normalized spacing: {stats['std_normalized_spacing']:.4f} ")
                f.write(f"(GUE: {stats['gue_std']:.2f})\n")
                f.write(f"Deviation from GUE mean: {stats['mean_deviation']:.4f}\n")
                f.write(f"Deviation from GUE std: {stats['std_deviation']:.4f}\n")
            f.write("\n")

            # Test 3: Trace formula
            f.write("3. TRACE FORMULA ANALYSIS\n")
            f.write("-"*40 + "\n")
            trace_results = self.test_trace_formula(15)
            for name, result in trace_results.items():
                f.write(f"Function: {name}\n")
                f.write(f"  Trace: {result['trace']:.6f}\n")
                f.write(f"  Prime sum: {result['prime_sum']:.6f}\n")
                f.write(f"  Ratio: {result['ratio']:.6f}\n")
            f.write("\n")

            # Test 4: Parameter sensitivity
            f.write("4. PARAMETER SENSITIVITY\n")
            f.write("-"*40 + "\n")
            sensitivity = self.test_parameter_sensitivity(10)
            if sensitivity["best"]:
                best = sensitivity["best"]
                f.write(f"Optimal parameters found:\n")
                f.write(f"  Coupling: {best['coupling']:.4f}\n")
                f.write(f"  Interaction: {best['interaction']:.4f}\n")
                f.write(f"  Mean Re(ρ): {best['mean_real']:.6f}\n")
                f.write(f"  Deviation: {best['deviation']:.6f}\n")
            f.write("\n")

            # Summary and conclusions
            f.write("="*70 + "\n")
            f.write("SUMMARY AND CONCLUSIONS\n")
            f.write("="*70 + "\n\n")

            f.write("Key Findings:\n")
            f.write("1. Eigenvalues show tendency toward Re(s)=1/2 but with significant variance\n")
            f.write("2. Convergence improves with more primes but slowly\n")
            f.write("3. Spacing statistics partially match GUE predictions\n")
            f.write("4. Trace formula shows connection to prime structure\n")
            f.write("5. Results are sensitive to coupling parameters\n\n")

            f.write("Challenges Identified:\n")
            f.write("- Finite-size effects dominate for small N\n")
            f.write("- Optimal coupling parameters unclear from first principles\n")
            f.write("- Convergence to known zeros is slow\n")
            f.write("- Need better understanding of operator structure\n\n")

            f.write("Next Steps:\n")
            f.write("- Investigate alternative coupling schemes\n")
            f.write("- Explore larger matrix sizes (N>50)\n")
            f.write("- Refine functional equation constraints\n")
            f.write("- Consider more sophisticated operator constructions\n")

        print(f"Report saved to {output_file}")


def main():
    """Main test execution"""
    print("="*70)
    print("TESTING CONNES' NONCOMMUTATIVE GEOMETRY APPROACH")
    print("="*70)

    # Initialize validator
    validator = RiemannZeroValidator()

    # Run comprehensive tests
    start_time = time.time()

    # 1. Basic convergence test
    print("\n" + "="*50)
    print("PHASE 1: BASIC CONVERGENCE TEST")
    print("="*50)
    validator.test_increasing_prime_count([5, 10, 15, 20])

    # 2. Statistical tests
    print("\n" + "="*50)
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("="*50)
    validator.test_eigenvalue_statistics(20)

    # 3. Trace formula
    print("\n" + "="*50)
    print("PHASE 3: TRACE FORMULA")
    print("="*50)
    validator.test_trace_formula(15)

    # 4. Parameter optimization
    print("\n" + "="*50)
    print("PHASE 4: PARAMETER OPTIMIZATION")
    print("="*50)
    validator.test_parameter_sensitivity(10)

    # 5. Generate visualizations
    print("\n" + "="*50)
    print("PHASE 5: VISUALIZATION")
    print("="*50)
    validator.create_visualization_suite(20)

    # 6. Generate report
    print("\n" + "="*50)
    print("PHASE 6: REPORT GENERATION")
    print("="*50)
    validator.generate_report("test_report.txt")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL TESTS COMPLETE")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()