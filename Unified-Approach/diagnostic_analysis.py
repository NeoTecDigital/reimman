#!/usr/bin/env python3
"""
DIAGNOSTIC ANALYSIS - Understanding the Scaling Problem
========================================================

Deep dive into why scaling doesn't fix the imaginary part mismatch.

Author: Claude (Anthropic)
Date: November 11, 2025
"""

import numpy as np
import scipy.linalg
import mpmath
import matplotlib.pyplot as plt

mpmath.mp.dps = 50


class DiagnosticAnalysis:
    """Analyze the fundamental scaling issue."""

    def __init__(self):
        # Simple test case
        self.N = 400
        self.epsilon = 0.01
        self.L = 100
        self.setup_grid()
        self.primes = self.generate_primes(1000)
        self.riemann_zeros = self.get_riemann_zeros(50)

    def setup_grid(self):
        """Simple logarithmic grid."""
        self.x = np.logspace(np.log10(self.epsilon), np.log10(self.L), self.N)
        self.dx = np.diff(self.x)
        self.dx = np.append(self.dx, self.dx[-1])

    def generate_primes(self, n: int):
        """Quick sieve."""
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(range(i*i, n+1, i))
        return [i for i in range(2, n + 1) if sieve[i]]

    def get_riemann_zeros(self, num: int):
        """Get zeros."""
        zeros = []
        for n in range(1, num + 1):
            zeros.append(complex(mpmath.zetazero(n)))
        return np.array(zeros)

    def build_operator_components(self):
        """Build and analyze separate components."""
        print("=" * 70)
        print("COMPONENT ANALYSIS")
        print("=" * 70)

        # Component 1: Berry-Keating xp̂ term only
        print("\n1. BERRY-KEATING TERM ONLY: -(xp̂ + p̂x)/2")
        T_BK = self.build_berry_keating()
        eigs_BK = np.sort(scipy.linalg.eigvalsh(T_BK))
        print(f"   Eigenvalue range: [{eigs_BK[0]:.4f}, {eigs_BK[-1]:.4f}]")
        print(f"   First 10 positive: {eigs_BK[eigs_BK > 0][:10]}")

        # Component 2: Prime potential only
        print("\n2. FROBENIUS PRIME POTENTIAL ONLY: ∑_p (log p)/p δ(x-log p)")
        T_F = self.build_frobenius()
        eigs_F = np.sort(scipy.linalg.eigvalsh(T_F))
        print(f"   Eigenvalue range: [{eigs_F[0]:.4f}, {eigs_F[-1]:.4f}]")
        print(f"   Non-zero count: {np.sum(np.abs(eigs_F) > 1e-6)}")

        # Component 3: Combined
        print("\n3. COMBINED: T_BK + T_F")
        T_combined = T_BK + T_F
        eigs_combined = np.sort(scipy.linalg.eigvalsh(T_combined))
        print(f"   Eigenvalue range: [{eigs_combined[0]:.4f}, {eigs_combined[-1]:.4f}]")
        print(f"   First 10 positive: {eigs_combined[eigs_combined > 0][:10]}")

        # Riemann zeros for comparison
        print("\n4. RIEMANN ZEROS (for reference)")
        print(f"   First 10 heights: {np.imag(self.riemann_zeros[:10])}")

        # Analysis
        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        ratio = np.imag(self.riemann_zeros[0]) / eigs_combined[eigs_combined > 0][0]
        print(f"\nRatio (first zero / first eigenvalue): {ratio:.4f}")
        print(f"This suggests we need α ≈ {ratio:.1f}")

        # But test it
        print("\nTesting with α = {:.1f}:".format(ratio))
        scaled_eigs = ratio * eigs_combined[eigs_combined > 0][:10]
        actual_zeros = np.imag(self.riemann_zeros[:10])
        errors = scaled_eigs - actual_zeros
        print(f"   Scaled eigenvalues: {scaled_eigs}")
        print(f"   Actual zeros:       {actual_zeros}")
        print(f"   Errors:             {errors}")
        print(f"   Mean abs error:     {np.mean(np.abs(errors)):.4f}")

        return T_BK, T_F, T_combined

    def build_berry_keating(self):
        """Pure xp̂ operator."""
        T = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            if i > 0 and i < self.N - 1:
                h_plus = self.x[i+1] - self.x[i]
                h_minus = self.x[i] - self.x[i-1]

                # -x d/dx with Weyl correction
                T[i, i-1] = -self.x[i] / h_minus
                T[i, i+1] = self.x[i] / h_plus
                T[i, i] = self.x[i] * (1/h_minus - 1/h_plus) - 0.5

        # Symmetrize
        return (T + T.T) / 2

    def build_frobenius(self):
        """Pure prime potential."""
        T = np.zeros((self.N, self.N), dtype=float)

        for p in self.primes[:200]:
            log_p = np.log(p)
            if log_p < self.epsilon or log_p > self.L:
                continue

            idx = np.argmin(np.abs(self.x - log_p))
            weight = np.log(p) / p
            T[idx, idx] += weight / self.dx[idx]

        return T

    def test_different_constructions(self):
        """Test alternative operator constructions."""
        print("\n" + "=" * 70)
        print("ALTERNATIVE CONSTRUCTIONS")
        print("=" * 70)

        zeros_target = np.imag(self.riemann_zeros[:10])

        # Method 1: Current approach
        print("\n1. Current: T = -xp̂ + ∑_p (log p)/p δ")
        T1 = self.build_berry_keating() + self.build_frobenius()
        eigs1 = np.sort(scipy.linalg.eigvalsh(T1))
        eigs1_pos = eigs1[eigs1 > 0][:10]
        alpha1 = zeros_target[0] / eigs1_pos[0]
        scaled1 = alpha1 * eigs1_pos
        error1 = np.mean(np.abs(scaled1 - zeros_target))
        print(f"   Optimal α: {alpha1:.2f}")
        print(f"   Mean error: {error1:.4f}")

        # Method 2: Remove Weyl correction
        print("\n2. No Weyl correction: T = -xp̂ (no -1/2)")
        T2 = self.build_berry_keating_no_weyl() + self.build_frobenius()
        eigs2 = np.sort(scipy.linalg.eigvalsh(T2))
        eigs2_pos = eigs2[eigs2 > 0][:10]
        if len(eigs2_pos) > 0:
            alpha2 = zeros_target[0] / eigs2_pos[0]
            scaled2 = alpha2 * eigs2_pos
            error2 = np.mean(np.abs(scaled2 - zeros_target))
            print(f"   Optimal α: {alpha2:.2f}")
            print(f"   Mean error: {error2:.4f}")

        # Method 3: Different prime weighting
        print("\n3. Prime weight (log p) instead of (log p)/p")
        T3 = self.build_berry_keating() + self.build_frobenius_alt()
        eigs3 = np.sort(scipy.linalg.eigvalsh(T3))
        eigs3_pos = eigs3[eigs3 > 0][:10]
        alpha3 = zeros_target[0] / eigs3_pos[0]
        scaled3 = alpha3 * eigs3_pos
        error3 = np.mean(np.abs(scaled3 - zeros_target))
        print(f"   Optimal α: {alpha3:.2f}")
        print(f"   Mean error: {error3:.4f}")

        # Method 4: Berry-Keating only (no primes)
        print("\n4. Berry-Keating only (no primes)")
        T4 = self.build_berry_keating()
        eigs4 = np.sort(scipy.linalg.eigvalsh(T4))
        eigs4_pos = eigs4[eigs4 > 0][:10]
        alpha4 = zeros_target[0] / eigs4_pos[0]
        scaled4 = alpha4 * eigs4_pos
        error4 = np.mean(np.abs(scaled4 - zeros_target))
        print(f"   Optimal α: {alpha4:.2f}")
        print(f"   Mean error: {error4:.4f}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: Best construction")
        print("=" * 70)
        errors_all = [error1, error2, error3, error4]
        methods = ["Current", "No Weyl", "Alt prime weight", "BK only"]
        best_idx = np.argmin(errors_all)
        print(f"\nBest method: {methods[best_idx]} (error = {errors_all[best_idx]:.4f})")

    def build_berry_keating_no_weyl(self):
        """xp̂ without -1/2 correction."""
        T = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            if i > 0 and i < self.N - 1:
                h_plus = self.x[i+1] - self.x[i]
                h_minus = self.x[i] - self.x[i-1]

                T[i, i-1] = -self.x[i] / h_minus
                T[i, i+1] = self.x[i] / h_plus
                T[i, i] = self.x[i] * (1/h_minus - 1/h_plus)

        return (T + T.T) / 2

    def build_frobenius_alt(self):
        """Prime potential with (log p) weight."""
        T = np.zeros((self.N, self.N), dtype=float)

        for p in self.primes[:200]:
            log_p = np.log(p)
            if log_p < self.epsilon or log_p > self.L:
                continue

            idx = np.argmin(np.abs(self.x - log_p))
            weight = np.log(p)  # No /p
            T[idx, idx] += weight / self.dx[idx]

        return T

    def investigate_spectral_density(self):
        """Compare spectral density with zero density."""
        print("\n" + "=" * 70)
        print("SPECTRAL DENSITY ANALYSIS")
        print("=" * 70)

        T = self.build_berry_keating() + self.build_frobenius()
        eigs = np.sort(scipy.linalg.eigvalsh(T))
        eigs_pos = eigs[eigs > 1.0][:50]

        zeros = np.imag(self.riemann_zeros[:50])

        # Compute spacing
        if len(eigs_pos) > 1:
            eig_spacing = np.diff(eigs_pos)
            zero_spacing = np.diff(zeros)

            print(f"\nEigenvalue spacing:")
            print(f"   Mean: {np.mean(eig_spacing):.4f}")
            print(f"   Std:  {np.std(eig_spacing):.4f}")

            print(f"\nZero spacing:")
            print(f"   Mean: {np.mean(zero_spacing):.4f}")
            print(f"   Std:  {np.std(zero_spacing):.4f}")

            print(f"\nRatio (zero spacing / eig spacing): {np.mean(zero_spacing) / np.mean(eig_spacing):.4f}")

    def visualize_comparison(self):
        """Visualize the problem."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        T = self.build_berry_keating() + self.build_frobenius()
        eigs = np.sort(scipy.linalg.eigvalsh(T))
        eigs_pos = eigs[eigs > 0][:30]

        zeros = np.imag(self.riemann_zeros[:30])

        # Raw comparison
        ax = axes[0, 0]
        n = min(len(eigs_pos), len(zeros))
        indices = np.arange(1, n+1)
        ax.plot(indices, zeros[:n], 'o-', label='Riemann zeros', markersize=8)
        ax.plot(indices, eigs_pos[:n], 's-', label='Eigenvalues (unscaled)', markersize=6, alpha=0.7)
        ax.set_xlabel('Index n')
        ax.set_ylabel('Height')
        ax.set_title('Raw Comparison (Before Scaling)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Scaled comparison
        ax = axes[0, 1]
        alpha_opt = zeros[0] / eigs_pos[0]
        scaled_eigs = alpha_opt * eigs_pos[:n]
        ax.plot(indices, zeros[:n], 'o-', label='Riemann zeros', markersize=8)
        ax.plot(indices, scaled_eigs, 's-', label=f'Scaled (α={alpha_opt:.1f})', markersize=6, alpha=0.7)
        ax.set_xlabel('Index n')
        ax.set_ylabel('Height')
        ax.set_title(f'After Optimal Scaling (α={alpha_opt:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error after scaling
        ax = axes[1, 0]
        errors = scaled_eigs - zeros[:n]
        ax.bar(indices, errors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_xlabel('Index n')
        ax.set_ylabel('Error (scaled - actual)')
        ax.set_title(f'Errors After Scaling (mean = {np.mean(np.abs(errors)):.2f})')
        ax.grid(True, alpha=0.3, axis='y')

        # Spacing comparison
        ax = axes[1, 1]
        zero_spacing = np.diff(zeros[:n])
        eig_spacing = np.diff(scaled_eigs)
        ax.plot(zero_spacing, 'o-', label='Zero spacing', markersize=6)
        ax.plot(eig_spacing, 's-', label='Eigenvalue spacing', markersize=6, alpha=0.7)
        ax.set_xlabel('Index n')
        ax.set_ylabel('Spacing')
        ax.set_title('Spacing Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/persist/neotec/reimman/Unified-Approach/diagnostic_analysis.png', dpi=150)
        print("\nVisualization saved to: diagnostic_analysis.png")
        plt.close()


def main():
    """Run diagnostic analysis."""
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS - Understanding the Scaling Problem")
    print("=" * 70)

    diag = DiagnosticAnalysis()

    # 1. Component analysis
    diag.build_operator_components()

    # 2. Test alternative constructions
    diag.test_different_constructions()

    # 3. Spectral density
    diag.investigate_spectral_density()

    # 4. Visualize
    diag.visualize_comparison()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    return diag


if __name__ == "__main__":
    diag = main()
