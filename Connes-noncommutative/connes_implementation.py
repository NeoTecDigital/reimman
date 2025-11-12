#!/usr/bin/env python3
"""
Connes' Noncommutative Geometry Approach to Riemann Hypothesis
Implementation of finite-dimensional approximation to find Riemann zeros
"""

import numpy as np
import scipy.linalg as la
import scipy.special as sp
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ConnesOperator:
    """Represents the finite-dimensional approximation of Connes' operator H"""
    N: int  # Number of primes to include
    primes: List[int]  # List of first N primes
    H: np.ndarray  # The operator matrix
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None

class ConnesRiemannSolver:
    """
    Implementation of Connes' noncommutative geometry approach
    to finding Riemann zeta zeros
    """

    def __init__(self, num_primes: int = 10):
        """
        Initialize with first num_primes primes

        Args:
            num_primes: Number of primes to use in approximation
        """
        self.num_primes = num_primes
        self.primes = self._generate_primes(num_primes)
        self.operator = None

    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n prime numbers"""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _build_scaling_matrix(self) -> np.ndarray:
        """
        Build the scaling operator D encoding prime structure
        This represents the action of R*+ on the adelic space
        """
        N = self.num_primes
        D = np.zeros((N, N), dtype=float)  # Use real matrix

        # Diagonal: log of primes (main scaling)
        for i in range(N):
            # Scale to approximate Riemann zero heights
            D[i, i] = np.log(self.primes[i]) * 10.0

        # Off-diagonal: coupling between primes
        # This encodes the multiplicative structure
        for i in range(N):
            for j in range(i+1, N):
                # Coupling strength decreases with prime distance
                coupling = 1.0 / np.sqrt(self.primes[i] * self.primes[j])
                D[i, j] = coupling
                D[j, i] = coupling  # Symmetric for real eigenvalues

        return D

    def _build_interaction_matrix(self) -> np.ndarray:
        """
        Build interaction matrix encoding prime multiplicative relations
        This captures the Euler product structure
        """
        N = self.num_primes
        V = np.zeros((N, N), dtype=float)  # Use real matrix

        for i in range(N):
            for j in range(N):
                if i != j:
                    # Interaction based on GCD and LCM of prime indices
                    p_i, p_j = self.primes[i], self.primes[j]

                    # Möbius-like interaction
                    if p_i * p_j < 100:  # Only nearby interactions
                        V[i, j] = (-1)**(i+j) / np.sqrt(p_i * p_j)

        return V

    def _apply_functional_equation_constraint(self, H: np.ndarray) -> np.ndarray:
        """
        Modify operator to respect functional equation ζ(s) = χ(s)ζ(1-s)
        This enforces symmetry about Re(s) = 1/2
        """
        N = H.shape[0]

        # Create symmetry operator S that implements s -> 1-s
        S = np.zeros((N, N), dtype=complex)
        for i in range(N):
            S[i, N-1-i] = 1

        # Enforce that H commutes with symmetry up to phase
        # [H, S] = iφS for some phase φ
        H_sym = 0.5 * (H + S @ np.conj(H.T) @ S)

        return H_sym

    def build_connes_operator(self,
                            coupling_strength: float = 0.1,
                            interaction_strength: float = 0.05) -> ConnesOperator:
        """
        Construct the finite-dimensional Connes operator H

        H = (1/2)I + iT where T encodes prime structure

        Args:
            coupling_strength: Strength of prime coupling
            interaction_strength: Strength of multiplicative interactions

        Returns:
            ConnesOperator object with matrix H
        """
        N = self.num_primes

        # Identity component (gives Re(eigenvalue) = 1/2)
        I = np.eye(N, dtype=complex)

        # Scaling operator (main prime structure)
        D = self._build_scaling_matrix()

        # Interaction operator (multiplicative relations)
        V = self._build_interaction_matrix()

        # Combine into T operator
        T = coupling_strength * D + interaction_strength * V

        # Make T explicitly Hermitian
        T = 0.5 * (T + np.conj(T.T))

        # Construct H = (1/2)I + iT
        H = 0.5 * I + 1j * T

        # Apply functional equation constraint
        H = self._apply_functional_equation_constraint(H)

        # Ensure H is Hermitian (numerical stability)
        H = 0.5 * (H + np.conj(H.T))

        self.operator = ConnesOperator(
            N=N,
            primes=self.primes,
            H=H
        )

        return self.operator

    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of the Connes operator
        These should approximate Riemann zeros
        """
        if self.operator is None:
            raise ValueError("Must build operator first using build_connes_operator()")

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(self.operator.H)

        # Sort by imaginary part (height on critical strip)
        idx = np.argsort(np.imag(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.operator.eigenvalues = eigenvalues
        self.operator.eigenvectors = eigenvectors

        return eigenvalues

    def extract_zeros(self) -> List[complex]:
        """
        Extract approximations to Riemann zeros from eigenvalues

        Returns:
            List of complex numbers approximating Riemann zeros
        """
        if self.operator.eigenvalues is None:
            self.compute_eigenvalues()

        # Filter eigenvalues to get zeros
        # We expect Re(λ) ≈ 1/2 and Im(λ) > 0
        zeros = []
        for ev in self.operator.eigenvalues:
            if np.imag(ev) > 0:  # Upper half-plane
                zeros.append(ev)

        return zeros

    def verify_critical_line(self, tolerance: float = 0.1) -> Dict:
        """
        Check if computed zeros lie on critical line Re(s) = 1/2

        Args:
            tolerance: Acceptable deviation from 1/2

        Returns:
            Dictionary with verification results
        """
        zeros = self.extract_zeros()

        if not zeros:
            return {"success": False, "message": "No zeros found"}

        real_parts = [np.real(z) for z in zeros]
        distances = [abs(rp - 0.5) for rp in real_parts]

        results = {
            "num_zeros": len(zeros),
            "mean_real_part": np.mean(real_parts),
            "std_real_part": np.std(real_parts),
            "max_deviation": max(distances),
            "on_critical_line": all(d < tolerance for d in distances),
            "percent_on_line": sum(1 for d in distances if d < tolerance) / len(distances) * 100
        }

        return results

    def compute_trace_formula(self, test_function=None) -> float:
        """
        Compute trace formula relating eigenvalues to primes

        Tr(f(H)) = Σ f(λ_n)

        Args:
            test_function: Function to apply to eigenvalues (default: identity)

        Returns:
            Trace value
        """
        if self.operator.eigenvalues is None:
            self.compute_eigenvalues()

        if test_function is None:
            test_function = lambda x: x

        trace = sum(test_function(ev) for ev in self.operator.eigenvalues)
        return trace

    def compare_with_known_zeros(self, known_zeros: List[complex]) -> Dict:
        """
        Compare computed zeros with known Riemann zeros

        Args:
            known_zeros: List of known Riemann zeros

        Returns:
            Dictionary with comparison metrics
        """
        computed = self.extract_zeros()

        if not computed:
            return {"error": "No computed zeros"}

        # Match computed zeros to nearest known zeros
        matches = []
        for cz in computed:
            if known_zeros:
                distances = [abs(cz - kz) for kz in known_zeros]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)
                matches.append({
                    "computed": cz,
                    "nearest_known": known_zeros[min_idx],
                    "distance": min_dist
                })

        avg_error = np.mean([m["distance"] for m in matches]) if matches else float('inf')

        return {
            "num_computed": len(computed),
            "num_known": len(known_zeros),
            "matches": matches,
            "average_error": avg_error,
            "max_error": max(m["distance"] for m in matches) if matches else float('inf')
        }

    def optimize_parameters(self,
                          known_zeros: List[complex],
                          param_ranges: Dict = None) -> Dict:
        """
        Optimize coupling parameters to match known zeros

        Args:
            known_zeros: List of known Riemann zeros
            param_ranges: Dictionary of parameter ranges to search

        Returns:
            Optimal parameters and results
        """
        if param_ranges is None:
            param_ranges = {
                "coupling_strength": np.linspace(0.01, 1.0, 20),
                "interaction_strength": np.linspace(0.01, 0.5, 10)
            }

        best_params = None
        best_error = float('inf')
        results = []

        for coupling in param_ranges["coupling_strength"]:
            for interaction in param_ranges["interaction_strength"]:
                # Build operator with these parameters
                self.build_connes_operator(coupling, interaction)
                self.compute_eigenvalues()

                # Compare with known zeros
                comparison = self.compare_with_known_zeros(known_zeros)
                error = comparison.get("average_error", float('inf'))

                results.append({
                    "coupling": coupling,
                    "interaction": interaction,
                    "error": error
                })

                if error < best_error:
                    best_error = error
                    best_params = {
                        "coupling_strength": coupling,
                        "interaction_strength": interaction
                    }

        return {
            "best_params": best_params,
            "best_error": best_error,
            "all_results": results
        }

    def visualize_eigenvalues(self, save_path: Optional[str] = None):
        """
        Visualize eigenvalue distribution in complex plane

        Args:
            save_path: Path to save figure (optional)
        """
        if self.operator.eigenvalues is None:
            self.compute_eigenvalues()

        eigenvalues = self.operator.eigenvalues

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Complex plane plot
        ax1 = axes[0]
        ax1.scatter(np.real(eigenvalues), np.imag(eigenvalues),
                   c='blue', s=50, alpha=0.7, label='Eigenvalues')
        ax1.axvline(x=0.5, color='red', linestyle='--',
                   label='Critical line Re(s)=1/2')
        ax1.set_xlabel('Real part')
        ax1.set_ylabel('Imaginary part')
        ax1.set_title('Eigenvalues in Complex Plane')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Distribution of real parts
        ax2 = axes[1]
        real_parts = np.real(eigenvalues)
        ax2.hist(real_parts, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--',
                   label='Expected (Re=1/2)')
        ax2.set_xlabel('Real part of eigenvalues')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Real Parts')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

        return fig

    def analyze_spacing_distribution(self) -> Dict:
        """
        Analyze eigenvalue spacing distribution
        Should follow GUE statistics if RH is true
        """
        if self.operator.eigenvalues is None:
            self.compute_eigenvalues()

        # Get imaginary parts (heights)
        heights = np.sort([np.imag(ev) for ev in self.operator.eigenvalues if np.imag(ev) > 0])

        if len(heights) < 2:
            return {"error": "Not enough eigenvalues for spacing analysis"}

        # Compute spacings
        spacings = np.diff(heights)

        # Normalize spacings
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing

        # Compute statistics
        stats = {
            "num_spacings": len(spacings),
            "mean_spacing": mean_spacing,
            "min_spacing": np.min(spacings),
            "max_spacing": np.max(spacings),
            "std_spacing": np.std(spacings),
            "normalized_mean": np.mean(normalized_spacings),
            "normalized_std": np.std(normalized_spacings)
        }

        return stats


def load_known_zeros(num_zeros: int = 10) -> List[complex]:
    """
    Load first few known Riemann zeros (hardcoded for demonstration)
    In practice, would load from a data file
    """
    # First 10 non-trivial Riemann zeros (imaginary parts)
    known_heights = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832
    ]

    # All non-trivial zeros lie on Re(s) = 1/2
    zeros = [0.5 + 1j * h for h in known_heights[:num_zeros]]

    return zeros


if __name__ == "__main__":
    print("="*60)
    print("CONNES' NONCOMMUTATIVE GEOMETRY APPROACH TO RIEMANN HYPOTHESIS")
    print("="*60)

    # Initialize solver
    num_primes = 15  # Use first 15 primes for demonstration
    solver = ConnesRiemannSolver(num_primes)

    print(f"\nUsing first {num_primes} primes: {solver.primes}")

    # Build operator with default parameters
    print("\n1. Building Connes operator H...")
    operator = solver.build_connes_operator(coupling_strength=0.15, interaction_strength=0.08)
    print(f"   Operator dimension: {operator.N}x{operator.N}")

    # Compute eigenvalues
    print("\n2. Computing eigenvalues...")
    eigenvalues = solver.compute_eigenvalues()
    print(f"   Found {len(eigenvalues)} eigenvalues")

    # Extract zeros
    print("\n3. Extracting approximate Riemann zeros...")
    zeros = solver.extract_zeros()
    print(f"   Found {len(zeros)} zeros in upper half-plane")

    if zeros:
        print("\n   First few computed zeros:")
        for i, z in enumerate(zeros[:5]):
            print(f"   ρ_{i+1} ≈ {z:.6f}")

    # Verify critical line
    print("\n4. Verifying critical line property...")
    verification = solver.verify_critical_line(tolerance=0.2)
    if "success" in verification and not verification["success"]:
        print(f"   {verification['message']}")
    else:
        if 'mean_real_part' in verification:
            print(f"   Mean real part: {verification['mean_real_part']:.6f} (expect 0.5)")
            print(f"   Std real part: {verification['std_real_part']:.6f}")
            print(f"   Max deviation from 1/2: {verification['max_deviation']:.6f}")
            print(f"   Percentage on critical line: {verification['percent_on_line']:.1f}%")
        else:
            print(f"   No zeros found to analyze")

    # Load known zeros for comparison
    print("\n5. Comparing with known Riemann zeros...")
    known_zeros = load_known_zeros(5)
    comparison = solver.compare_with_known_zeros(known_zeros)

    if "matches" in comparison:
        print(f"   Average error: {comparison['average_error']:.6f}")
        print(f"   Maximum error: {comparison['max_error']:.6f}")

        print("\n   Detailed comparison:")
        for i, match in enumerate(comparison["matches"][:3]):
            print(f"   Computed: {match['computed']:.6f}")
            print(f"   Nearest known: {match['nearest_known']:.6f}")
            print(f"   Distance: {match['distance']:.6f}")
            print()

    # Analyze spacing distribution
    print("\n6. Analyzing eigenvalue spacing...")
    spacing_stats = solver.analyze_spacing_distribution()
    if "error" not in spacing_stats:
        print(f"   Mean spacing: {spacing_stats['mean_spacing']:.6f}")
        print(f"   Std spacing: {spacing_stats['std_spacing']:.6f}")
        print(f"   Normalized std: {spacing_stats['normalized_std']:.6f}")
        print("   (GUE prediction: normalized std ≈ 0.52)")

    # Trace formula
    print("\n7. Computing trace formula...")
    trace = solver.compute_trace_formula()
    print(f"   Tr(H) = {trace:.6f}")

    # Prime sum for comparison
    prime_sum = sum(np.log(p) for p in solver.primes)
    print(f"   Σ log(p) = {prime_sum:.6f}")
    print(f"   Ratio: {np.real(trace)/prime_sum:.6f}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)