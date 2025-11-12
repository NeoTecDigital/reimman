#!/usr/bin/env python3
"""
Berry-Keating Implementation for Riemann Hypothesis
Quantum chaos approach via xp Hamiltonian quantization
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import zeta
import matplotlib.pyplot as plt
import mpmath as mp
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set high precision for mpmath
mp.mp.dps = 50

class BerryKeatingHamiltonian:
    """
    Implementation of quantum xp Hamiltonian with various orderings
    """

    def __init__(self, N: int = 500, L: float = 100.0, epsilon: float = 0.1):
        """
        Initialize the quantum system

        Parameters:
        -----------
        N : int
            Number of discretization points
        L : float
            Upper bound of position space interval [epsilon, L]
        epsilon : float
            Lower bound (regularization parameter)
        """
        self.N = N
        self.L = L
        self.epsilon = epsilon

        # Create position grid (logarithmic spacing for better resolution)
        self.x = np.logspace(np.log10(epsilon), np.log10(L), N)
        self.dx = np.diff(self.x)
        self.dx_avg = np.mean(self.dx)

        # Set ℏ = 1 for simplicity (can be restored later)
        self.hbar = 1.0

        # Initialize operators
        self._build_operators()

    def _build_operators(self):
        """Construct position and momentum operators"""

        # Position operator (diagonal)
        self.X = sp.diags(self.x, 0, shape=(self.N, self.N), format='csr')

        # Momentum operator using centered finite differences
        # p = -iℏ d/dx
        # For non-uniform grid, use appropriate finite difference coefficients
        self.P = self._build_momentum_operator()

    def _build_momentum_operator(self) -> sp.csr_matrix:
        """
        Build momentum operator with non-uniform grid spacing
        Using centered differences where possible, one-sided at boundaries
        """
        P_data = []
        P_row = []
        P_col = []

        # Use complex type for momentum operator
        factor = -1j * self.hbar

        for i in range(self.N):
            if i == 0:
                # Forward difference at left boundary
                dx = self.x[1] - self.x[0]
                # df/dx ≈ (f[i+1] - f[i])/dx
                P_data.extend([factor/dx, -factor/dx])
                P_row.extend([i, i])
                P_col.extend([i+1, i])

            elif i == self.N - 1:
                # Backward difference at right boundary
                dx = self.x[i] - self.x[i-1]
                # df/dx ≈ (f[i] - f[i-1])/dx
                P_data.extend([factor/dx, -factor/dx])
                P_row.extend([i, i])
                P_col.extend([i, i-1])

            else:
                # Centered difference in interior
                dx_forward = self.x[i+1] - self.x[i]
                dx_backward = self.x[i] - self.x[i-1]
                dx_total = self.x[i+1] - self.x[i-1]

                # Coefficients for non-uniform centered difference
                coeff_forward = factor * 2 / (dx_total * (1 + dx_forward/dx_backward))
                coeff_backward = -factor * 2 / (dx_total * (1 + dx_backward/dx_forward))
                coeff_center = -(coeff_forward + coeff_backward)

                P_data.extend([coeff_backward, coeff_center, coeff_forward])
                P_row.extend([i, i, i])
                P_col.extend([i-1, i, i+1])

        P = sp.csr_matrix((P_data, (P_row, P_col)), shape=(self.N, self.N), dtype=complex)
        return P

    def build_hamiltonian(self, ordering: str = 'weyl') -> np.ndarray:
        """
        Build quantum Hamiltonian with specified ordering

        Parameters:
        -----------
        ordering : str
            'standard' : H = xp
            'anti' : H = px
            'weyl' : H = (xp + px)/2
            'alpha' : H = α*xp + (1-α)*px with α specified

        Returns:
        --------
        H : np.ndarray
            Hamiltonian matrix
        """
        X_dense = self.X.toarray()
        P_dense = self.P.toarray()

        if ordering == 'standard':
            H = X_dense @ P_dense
        elif ordering == 'anti':
            H = P_dense @ X_dense
        elif ordering == 'weyl':
            H = 0.5 * (X_dense @ P_dense + P_dense @ X_dense)
        elif ordering.startswith('alpha'):
            # Extract alpha value, e.g., 'alpha_0.3'
            alpha = float(ordering.split('_')[1]) if '_' in ordering else 0.5
            H = alpha * X_dense @ P_dense + (1 - alpha) * P_dense @ X_dense
        else:
            raise ValueError(f"Unknown ordering: {ordering}")

        return H

    def compute_eigenvalues(self, H: np.ndarray, n_eigs: int = 100) -> np.ndarray:
        """
        Compute eigenvalues of Hamiltonian

        Parameters:
        -----------
        H : np.ndarray
            Hamiltonian matrix
        n_eigs : int
            Number of eigenvalues to compute

        Returns:
        --------
        eigenvalues : np.ndarray
            Array of eigenvalues (complex in general)
        """
        # For Hermitian matrices, use eigh for better numerical stability
        if np.allclose(H, H.conj().T):
            eigenvalues = la.eigvalsh(H)[:n_eigs]
        else:
            eigenvalues = la.eigvals(H)
            # Sort by imaginary part (corresponding to Riemann zeros)
            eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues.imag))][:n_eigs]

        return eigenvalues

    def extract_zeros(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Extract Riemann zero candidates from eigenvalues

        The connection is: ρ = 1/2 + i*E/ℏ
        where E are the eigenvalues

        Parameters:
        -----------
        eigenvalues : np.ndarray
            Eigenvalues of the Hamiltonian

        Returns:
        --------
        zeros : np.ndarray
            Estimated Riemann zeros
        """
        # Convert eigenvalues to Riemann zeros
        # Note: we expect purely imaginary eigenvalues for Hermitian H
        zeros = 0.5 + 1j * eigenvalues / self.hbar

        # Filter to positive imaginary parts (zeros come in conjugate pairs)
        zeros = zeros[zeros.imag > 0]

        return zeros


class RiemannZeroValidator:
    """
    Validate computed eigenvalues against known Riemann zeros
    """

    def __init__(self):
        """Initialize with known zeros"""
        self.known_zeros = self._load_known_zeros()

    def _load_known_zeros(self, n: int = 100) -> np.ndarray:
        """
        Load first n known Riemann zeros using mpmath

        Returns:
        --------
        zeros : np.ndarray
            Array of known zeros (imaginary parts)
        """
        zeros = []
        for k in range(1, n + 1):
            zero = mp.zetazero(k)
            zeros.append(float(zero.imag))
        return np.array(zeros)

    def compare_zeros(self, computed: np.ndarray, n_compare: int = 50) -> Dict:
        """
        Compare computed zeros with known values

        Parameters:
        -----------
        computed : np.ndarray
            Computed zero candidates (complex numbers)
        n_compare : int
            Number of zeros to compare

        Returns:
        --------
        metrics : dict
            Comparison metrics
        """
        # Extract imaginary parts and sort
        computed_im = np.sort(np.abs(computed.imag))[:n_compare]
        known_im = self.known_zeros[:n_compare]

        # Ensure same length for comparison
        min_len = min(len(computed_im), len(known_im))
        computed_im = computed_im[:min_len]
        known_im = known_im[:min_len]

        # Compute metrics
        absolute_errors = np.abs(computed_im - known_im)
        relative_errors = absolute_errors / known_im

        metrics = {
            'computed': computed_im,
            'known': known_im,
            'absolute_errors': absolute_errors,
            'relative_errors': relative_errors,
            'mean_absolute_error': np.mean(absolute_errors),
            'mean_relative_error': np.mean(relative_errors),
            'max_absolute_error': np.max(absolute_errors),
            'max_relative_error': np.max(relative_errors),
            'correlation': np.corrcoef(computed_im, known_im)[0, 1] if min_len > 1 else 0
        }

        return metrics


class QuantizationAnalyzer:
    """
    Analyze different quantization schemes
    """

    def __init__(self, N: int = 300, L: float = 50.0, epsilon: float = 0.1):
        """Initialize analyzer"""
        self.hamiltonian = BerryKeatingHamiltonian(N=N, L=L, epsilon=epsilon)
        self.validator = RiemannZeroValidator()

    def analyze_orderings(self, orderings: List[str] = None) -> Dict:
        """
        Analyze different operator orderings

        Parameters:
        -----------
        orderings : list of str
            List of ordering schemes to test

        Returns:
        --------
        results : dict
            Results for each ordering
        """
        if orderings is None:
            orderings = ['standard', 'anti', 'weyl', 'alpha_0.3', 'alpha_0.7']

        results = {}

        for ordering in orderings:
            print(f"\nAnalyzing {ordering} ordering...")

            # Build Hamiltonian
            H = self.hamiltonian.build_hamiltonian(ordering)

            # Check if Hermitian
            is_hermitian = np.allclose(H, H.conj().T)

            # Compute eigenvalues
            eigenvalues = self.hamiltonian.compute_eigenvalues(H, n_eigs=100)

            # Extract zeros
            zeros = self.hamiltonian.extract_zeros(eigenvalues)

            # Validate
            metrics = self.validator.compare_zeros(zeros, n_compare=30)

            results[ordering] = {
                'is_hermitian': is_hermitian,
                'eigenvalues': eigenvalues,
                'zeros': zeros,
                'metrics': metrics,
                'hamiltonian_norm': np.linalg.norm(H)
            }

            # Print summary
            print(f"  Hermitian: {is_hermitian}")
            print(f"  Mean absolute error: {metrics['mean_absolute_error']:.6f}")
            print(f"  Mean relative error: {metrics['mean_relative_error']:.6f}")
            print(f"  Correlation: {metrics['correlation']:.6f}")

        return results

    def optimize_parameters(self, N_range: List[int] = None,
                          L_range: List[float] = None) -> Dict:
        """
        Optimize discretization parameters

        Parameters:
        -----------
        N_range : list of int
            Range of grid sizes to test
        L_range : list of float
            Range of upper bounds to test

        Returns:
        --------
        optimization_results : dict
            Results of parameter optimization
        """
        if N_range is None:
            N_range = [100, 200, 300, 400, 500]
        if L_range is None:
            L_range = [10, 25, 50, 75, 100]

        best_params = None
        best_error = float('inf')
        all_results = []

        for N in N_range:
            for L in L_range:
                print(f"Testing N={N}, L={L}")

                # Create new Hamiltonian
                ham = BerryKeatingHamiltonian(N=N, L=L)
                H = ham.build_hamiltonian('weyl')

                # Compute eigenvalues
                eigenvalues = ham.compute_eigenvalues(H, n_eigs=50)
                zeros = ham.extract_zeros(eigenvalues)

                # Validate
                metrics = self.validator.compare_zeros(zeros, n_compare=20)

                error = metrics['mean_absolute_error']
                all_results.append({
                    'N': N,
                    'L': L,
                    'error': error,
                    'metrics': metrics
                })

                if error < best_error:
                    best_error = error
                    best_params = {'N': N, 'L': L}

        return {
            'best_params': best_params,
            'best_error': best_error,
            'all_results': all_results
        }


class Visualizer:
    """
    Visualization tools for Berry-Keating analysis
    """

    @staticmethod
    def plot_eigenvalue_comparison(results: Dict, save_path: str = None):
        """
        Plot comparison of eigenvalues for different orderings
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (ordering, data) in enumerate(results.items()):
            if idx >= 6:
                break

            ax = axes[idx]

            # Plot computed vs known
            metrics = data['metrics']
            computed = metrics['computed']
            known = metrics['known'][:len(computed)]

            ax.scatter(known, computed, alpha=0.6, label='Computed')
            ax.plot(known, known, 'r--', label='y=x (perfect match)')

            ax.set_xlabel('Known zeros (Im part)')
            ax.set_ylabel('Computed zeros (Im part)')
            ax.set_title(f'{ordering} ordering\nMAE: {metrics["mean_absolute_error"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_error_analysis(results: Dict, save_path: str = None):
        """
        Plot error analysis for different orderings
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Absolute errors
        ax = axes[0, 0]
        for ordering, data in results.items():
            errors = data['metrics']['absolute_errors']
            ax.plot(errors, label=ordering, marker='o', markersize=3)
        ax.set_xlabel('Zero index')
        ax.set_ylabel('Absolute error')
        ax.set_title('Absolute Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Relative errors
        ax = axes[0, 1]
        for ordering, data in results.items():
            errors = data['metrics']['relative_errors']
            ax.plot(errors, label=ordering, marker='o', markersize=3)
        ax.set_xlabel('Zero index')
        ax.set_ylabel('Relative error')
        ax.set_title('Relative Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Mean errors comparison
        ax = axes[1, 0]
        orderings = list(results.keys())
        mean_abs_errors = [results[o]['metrics']['mean_absolute_error'] for o in orderings]
        ax.bar(orderings, mean_abs_errors)
        ax.set_xlabel('Ordering')
        ax.set_ylabel('Mean absolute error')
        ax.set_title('Mean Absolute Error Comparison')
        ax.grid(True, alpha=0.3)

        # Hermiticity check
        ax = axes[1, 1]
        hermitian_values = [1 if results[o]['is_hermitian'] else 0 for o in orderings]
        colors = ['green' if h else 'red' for h in hermitian_values]
        ax.bar(orderings, hermitian_values, color=colors)
        ax.set_xlabel('Ordering')
        ax.set_ylabel('Is Hermitian')
        ax.set_title('Hermiticity Check')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, title: str = "Eigenvalue Spectrum",
                                save_path: str = None):
        """
        Plot the eigenvalue spectrum in complex plane
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Complex plane plot
        ax = axes[0]
        ax.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Real part')
        ax.set_ylabel('Imaginary part')
        ax.set_title(f'{title}\nComplex Plane')
        ax.grid(True, alpha=0.3)

        # Histogram of imaginary parts
        ax = axes[1]
        ax.hist(eigenvalues.imag, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Imaginary part')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Imaginary Parts')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("Berry-Keating Quantum Chaos Approach to Riemann Hypothesis")
    print("=" * 60)

    # Initialize analyzer
    print("\nInitializing quantum system...")
    analyzer = QuantizationAnalyzer(N=400, L=75.0, epsilon=0.01)

    # Analyze different orderings
    print("\n" + "=" * 40)
    print("Analyzing different quantization schemes")
    print("=" * 40)
    results = analyzer.analyze_orderings()

    # Find best ordering
    best_ordering = min(results.keys(),
                       key=lambda k: results[k]['metrics']['mean_absolute_error'])
    print(f"\nBest ordering: {best_ordering}")
    print(f"Mean absolute error: {results[best_ordering]['metrics']['mean_absolute_error']:.6f}")

    # Create visualizations
    print("\n" + "=" * 40)
    print("Creating visualizations")
    print("=" * 40)

    viz = Visualizer()

    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)

    # Plot comparisons
    viz.plot_eigenvalue_comparison(results, save_path='visualizations/eigenvalue_comparison.png')
    viz.plot_error_analysis(results, save_path='visualizations/error_analysis.png')

    # Plot spectrum for best ordering
    best_eigenvalues = results[best_ordering]['eigenvalues']
    viz.plot_eigenvalue_spectrum(best_eigenvalues,
                                 title=f"Best Ordering: {best_ordering}",
                                 save_path='visualizations/eigenvalue_spectrum.png')

    # Parameter optimization (optional - takes time)
    print("\n" + "=" * 40)
    print("Parameter Optimization (simplified)")
    print("=" * 40)
    opt_results = analyzer.optimize_parameters(
        N_range=[200, 300, 400],
        L_range=[25, 50, 75]
    )
    print(f"\nOptimal parameters: {opt_results['best_params']}")
    print(f"Best error achieved: {opt_results['best_error']:.6f}")

    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    for ordering, data in results.items():
        print(f"\n{ordering} ordering:")
        print(f"  - Hermitian: {data['is_hermitian']}")
        print(f"  - Mean absolute error: {data['metrics']['mean_absolute_error']:.6f}")
        print(f"  - Mean relative error: {data['metrics']['mean_relative_error']:.6f}")
        print(f"  - Correlation with known zeros: {data['metrics']['correlation']:.6f}")
        print(f"  - First 5 computed zeros (Im part): {data['metrics']['computed'][:5]}")
        print(f"  - First 5 known zeros (Im part): {data['metrics']['known'][:5]}")

    return results


if __name__ == "__main__":
    results = main()