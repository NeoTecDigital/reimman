#!/usr/bin/env python3
"""
Quick test of Berry-Keating implementation
"""

import numpy as np
import mpmath as mp
from berry_keating_implementation import BerryKeatingHamiltonian, RiemannZeroValidator

# Set precision
mp.mp.dps = 50

def quick_test():
    """Run a quick test with small parameters"""
    print("=" * 60)
    print("QUICK TEST OF BERRY-KEATING IMPLEMENTATION")
    print("=" * 60)

    # Create Hamiltonian with modest parameters
    print("\nInitializing quantum system...")
    ham = BerryKeatingHamiltonian(N=200, L=50.0, epsilon=0.1)

    # Test different orderings
    orderings = ['standard', 'anti', 'weyl']
    validator = RiemannZeroValidator()

    results = {}

    for ordering in orderings:
        print(f"\n{'-'*40}")
        print(f"Testing {ordering} ordering...")
        print(f"{'-'*40}")

        # Build Hamiltonian
        H = ham.build_hamiltonian(ordering)

        # Check properties
        is_hermitian = np.allclose(H, H.conj().T)
        print(f"  Hermitian: {is_hermitian}")
        print(f"  Matrix norm: {np.linalg.norm(H):.2f}")

        # Compute eigenvalues
        print("  Computing eigenvalues...")
        eigenvalues = ham.compute_eigenvalues(H, n_eigs=30)

        # Extract zeros
        zeros = ham.extract_zeros(eigenvalues)
        print(f"  Number of zeros extracted: {len(zeros)}")

        # Check if on critical line
        if len(zeros) > 0:
            real_parts = zeros.real
            print(f"  Mean Re(ρ): {np.mean(real_parts):.6f}")
            print(f"  Std Re(ρ): {np.std(real_parts):.6f}")

            # Compare with known zeros
            metrics = validator.compare_zeros(zeros, n_compare=min(10, len(zeros)))

            print(f"\n  Comparison with known zeros (first 10):")
            print(f"  Mean absolute error: {metrics['mean_absolute_error']:.4f}")
            print(f"  Mean relative error: {metrics['mean_relative_error']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")

            # Show first few zeros
            print(f"\n  First 5 computed (Im part): {metrics['computed'][:5]}")
            print(f"  First 5 known (Im part):    {metrics['known'][:5]}")

            results[ordering] = {
                'is_hermitian': is_hermitian,
                'mean_error': metrics['mean_absolute_error'],
                'correlation': metrics['correlation']
            }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_ordering = min(results.keys(), key=lambda k: results[k]['mean_error'])
    print(f"\nBest ordering: {best_ordering}")
    print(f"  Mean error: {results[best_ordering]['mean_error']:.4f}")
    print(f"  Correlation: {results[best_ordering]['correlation']:.4f}")

    # Assessment
    print("\n" + "-" * 40)
    print("ASSESSMENT:")

    if results[best_ordering]['correlation'] > 0.8:
        print("✅ Strong correlation with Riemann zeros!")
        print("   The approach shows promise.")
    elif results[best_ordering]['correlation'] > 0.5:
        print("⚠️ Moderate correlation with Riemann zeros.")
        print("   The approach has potential but needs refinement.")
    else:
        print("❌ Weak correlation with Riemann zeros.")
        print("   The approach needs significant improvement.")

    # Check critical line property
    if 'weyl' in results and results['weyl']['is_hermitian']:
        print("\n✅ Weyl ordering gives Hermitian operator")
        print("   This ensures eigenvalues are real → zeros on critical line!")
    else:
        print("\n⚠️ No Hermitian ordering found")
        print("   Critical line property not guaranteed.")

    return results

if __name__ == "__main__":
    results = quick_test()