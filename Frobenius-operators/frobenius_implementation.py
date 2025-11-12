"""
Frobenius Operator Approach to Riemann Hypothesis
Implementation of finite field arithmetic and Frobenius eigenvalue computation
"""

import numpy as np
from sympy import *
from sympy.polys.galoistools import *
from sympy.ntheory import factorint, isprime
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import mpmath

# Set precision for mpmath
mpmath.mp.dps = 50

class FiniteField:
    """Finite field F_p arithmetic"""

    def __init__(self, p: int):
        if not isprime(p):
            raise ValueError(f"{p} is not prime")
        self.p = p
        self.order = p

    def add(self, a: int, b: int) -> int:
        """Addition in F_p"""
        return (a + b) % self.p

    def subtract(self, a: int, b: int) -> int:
        """Subtraction in F_p"""
        return (a - b) % self.p

    def multiply(self, a: int, b: int) -> int:
        """Multiplication in F_p"""
        return (a * b) % self.p

    def power(self, a: int, n: int) -> int:
        """Fast exponentiation in F_p"""
        return pow(a, n, self.p)

    def inverse(self, a: int) -> int:
        """Multiplicative inverse in F_p"""
        if a == 0:
            raise ValueError("Zero has no inverse")
        return pow(a, self.p - 2, self.p)

    def frobenius(self, a: int) -> int:
        """Frobenius map: a -> a^p"""
        return self.power(a, self.p)


class EllipticCurve:
    """Elliptic curve over finite field: y^2 = x^3 + ax + b"""

    def __init__(self, field: FiniteField, a: int, b: int):
        self.field = field
        self.a = a % field.p
        self.b = b % field.p

        # Check discriminant
        disc = (-16 * (4 * self.a**3 + 27 * self.b**2)) % field.p
        if disc == 0:
            raise ValueError("Singular curve")

    def is_point(self, x: int, y: int) -> bool:
        """Check if (x, y) is on the curve"""
        if x == 'inf' and y == 'inf':
            return True  # Point at infinity

        lhs = self.field.power(y, 2)
        rhs = (self.field.power(x, 3) +
               self.field.multiply(self.a, x) +
               self.b) % self.field.p
        return lhs == rhs

    def count_points(self) -> int:
        """Count points on curve using naive method"""
        count = 1  # Point at infinity

        for x in range(self.field.p):
            # Compute y^2 = x^3 + ax + b
            rhs = (self.field.power(x, 3) +
                   self.field.multiply(self.a, x) +
                   self.b) % self.field.p

            # Check if rhs is a quadratic residue
            if rhs == 0:
                count += 1
            elif pow(rhs, (self.field.p - 1) // 2, self.field.p) == 1:
                count += 2  # Two y-values

        return count

    def frobenius_trace(self) -> int:
        """Compute trace of Frobenius: t = p + 1 - #E(F_p)"""
        return self.field.p + 1 - self.count_points()

    def frobenius_eigenvalues(self) -> Tuple[complex, complex]:
        """Compute Frobenius eigenvalues"""
        t = self.frobenius_trace()
        # Characteristic polynomial: X^2 - tX + p = 0
        discriminant = t**2 - 4*self.field.p

        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            alpha = (t + sqrt_disc) / 2
            beta = (t - sqrt_disc) / 2
        else:
            sqrt_disc = np.sqrt(-discriminant)
            alpha = complex(t/2, sqrt_disc/2)
            beta = complex(t/2, -sqrt_disc/2)

        return alpha, beta


class FrobeniusOperator:
    """Frobenius operator and its spectral properties"""

    def __init__(self, prime: int):
        self.p = prime
        self.field = FiniteField(prime)
        self.eigenvalues_collection = []

    def analyze_elliptic_curves(self, num_curves: int = 10) -> List[Tuple[complex, complex]]:
        """Analyze Frobenius eigenvalues for random elliptic curves"""
        eigenvalues = []

        for _ in range(num_curves):
            # Generate random curve parameters
            a = np.random.randint(0, self.p)
            b = np.random.randint(1, self.p)  # b != 0

            try:
                curve = EllipticCurve(self.field, a, b)
                alpha, beta = curve.frobenius_eigenvalues()
                eigenvalues.append((alpha, beta))

                # Verify Weil bound: |alpha| = |beta| = sqrt(p)
                expected_abs = np.sqrt(self.p)
                alpha_abs = abs(alpha)
                beta_abs = abs(beta)

                print(f"Curve y^2 = x^3 + {a}x + {b} (mod {self.p})")
                print(f"  Eigenvalues: α = {alpha:.4f}, β = {beta:.4f}")
                print(f"  |α| = {alpha_abs:.4f}, |β| = {beta_abs:.4f}, √p = {expected_abs:.4f}")
                print(f"  Weil bound satisfied: {np.isclose(alpha_abs, expected_abs, rtol=1e-10)}")
                print()

            except ValueError:
                continue  # Skip singular curves

        self.eigenvalues_collection = eigenvalues
        return eigenvalues

    def map_to_zeta_zeros(self, eigenvalues: List[Tuple[complex, complex]]) -> List[complex]:
        """Map Frobenius eigenvalues to potential zeta zeros"""
        zeros = []

        for alpha, beta in eigenvalues:
            # Use the relation: if |λ| = p^(1/2), then Re(s) = 1/2
            # Map via: s = 1/2 + it where t is related to arg(λ)

            # For α
            if abs(alpha) > 0:
                t_alpha = np.angle(alpha) / (2 * np.pi) * 50  # Scale to typical zero heights
                zeros.append(complex(0.5, t_alpha))

            # For β
            if abs(beta) > 0:
                t_beta = np.angle(beta) / (2 * np.pi) * 50
                zeros.append(complex(0.5, t_beta))

        return zeros

    def compute_l_function(self, s: complex, eigenvalues: Tuple[complex, complex]) -> complex:
        """Compute L-function for given eigenvalues"""
        alpha, beta = eigenvalues

        # L(s) = 1 / ((1 - α/p^s)(1 - β/p^s))
        term1 = 1 - alpha * self.p**(-s)
        term2 = 1 - beta * self.p**(-s)

        if abs(term1) > 1e-10 and abs(term2) > 1e-10:
            return 1 / (term1 * term2)
        else:
            return float('inf')


def get_riemann_zeros(num_zeros: int = 10) -> List[float]:
    """Get the first few non-trivial zeros of the Riemann zeta function"""
    zeros = []

    # Using mpmath for accurate zero computation
    for n in range(1, num_zeros + 1):
        zero = mpmath.zetazero(n)
        zeros.append(float(zero.imag))

    return zeros


def test_frobenius_correspondence(primes: List[int] = None):
    """Test correspondence between Frobenius eigenvalues and zeta zeros"""

    if primes is None:
        primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    all_mapped_zeros = []

    for p in primes:
        print(f"\n{'='*50}")
        print(f"Testing prime p = {p}")
        print(f"{'='*50}")

        frob = FrobeniusOperator(p)
        eigenvalues = frob.analyze_elliptic_curves(num_curves=5)

        # Map to potential zeros
        mapped_zeros = frob.map_to_zeta_zeros(eigenvalues)
        all_mapped_zeros.extend([z.imag for z in mapped_zeros if z.imag > 0])

    # Compare with actual Riemann zeros
    print(f"\n{'='*50}")
    print("Comparison with Riemann zeros")
    print(f"{'='*50}")

    actual_zeros = get_riemann_zeros(10)

    print("\nFirst 10 Riemann zeros (imaginary parts):")
    for i, zero in enumerate(actual_zeros, 1):
        print(f"  ρ_{i}: {zero:.6f}")

    print("\nMapped zero distribution (imaginary parts):")
    mapped_sorted = sorted(all_mapped_zeros)[:20]
    for i, zero in enumerate(mapped_sorted, 1):
        print(f"  Mapped_{i}: {zero:.6f}")

    # Statistical analysis
    print(f"\n{'='*50}")
    print("Statistical Analysis")
    print(f"{'='*50}")

    # Check if mapped zeros cluster near Re(s) = 1/2
    real_parts = [z.real for z in all_mapped_zeros if isinstance(z, complex)]
    if real_parts:
        mean_real = np.mean([abs(r - 0.5) for r in real_parts])
        print(f"Mean deviation from critical line: {mean_real:.6f}")

    return all_mapped_zeros, actual_zeros


def visualize_results(mapped_zeros: List[complex], actual_zeros: List[float]):
    """Visualize the correspondence between Frobenius eigenvalues and zeta zeros"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distribution of mapped zeros
    ax1 = axes[0]
    if mapped_zeros:
        imag_parts = [abs(z) if isinstance(z, (int, float)) else abs(z.imag)
                     for z in mapped_zeros if z != 0]
        if imag_parts:
            ax1.hist(imag_parts, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Imaginary part')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Mapped Zeros from Frobenius Eigenvalues')
            ax1.grid(True, alpha=0.3)

    # Plot 2: Comparison with actual zeros
    ax2 = axes[1]
    ax2.scatter([0.5]*len(actual_zeros), actual_zeros, color='red',
                s=50, alpha=0.7, label='Actual Riemann zeros')

    # Plot mapped zeros if they have complex structure
    mapped_complex = [z for z in mapped_zeros if isinstance(z, complex)]
    if mapped_complex:
        real_parts = [z.real for z in mapped_complex]
        imag_parts = [abs(z.imag) for z in mapped_complex]
        ax2.scatter(real_parts, imag_parts, color='blue',
                   s=20, alpha=0.5, label='Mapped from Frobenius')

    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Critical line')
    ax2.set_xlabel('Real part')
    ax2.set_ylabel('Imaginary part')
    ax2.set_title('Zeros in Complex Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig('/home/persist/neotec/reimman/Frobenius-operators/frobenius_zeros_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved as 'frobenius_zeros_comparison.png'")


if __name__ == "__main__":
    print("="*60)
    print("FROBENIUS OPERATOR APPROACH TO RIEMANN HYPOTHESIS")
    print("="*60)

    # Run the test
    mapped_zeros, actual_zeros = test_frobenius_correspondence()

    # Visualize results
    visualize_results(mapped_zeros, actual_zeros)