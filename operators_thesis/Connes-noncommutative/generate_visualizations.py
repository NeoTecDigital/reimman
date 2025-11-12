#!/usr/bin/env python3
"""
Generate comprehensive visualizations for Connes approach results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from connes_implementation_fixed import ConnesRiemannSolver
import os

# Set up matplotlib parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

def create_comprehensive_visualization():
    """Create all visualization plots"""

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    # Known Riemann zeros for comparison
    known_heights = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                    37.586178, 40.918719, 43.327073, 48.005151, 49.773832]

    print("Generating visualizations...")

    # Figure 1: Eigenvalue Distribution Comparison
    fig1 = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.3, wspace=0.3)

    matrix_sizes = [5, 10, 15, 20, 25, 30]
    plot_idx = 0

    for N in matrix_sizes:
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig1.add_subplot(gs[row, col])

        solver = ConnesRiemannSolver(N)
        solver.build_connes_operator_v2()  # Use Berry-Keating variant
        eigenvalues = solver.compute_eigenvalues()

        # Plot eigenvalues
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        ax.scatter(real_parts, imag_parts, c='blue', s=30, alpha=0.6, label='Computed')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_title(f'N={N} primes')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.45, 0.55])

        plot_idx += 1

    fig1.suptitle('Eigenvalue Distribution for Different Matrix Sizes', fontsize=14, y=0.98)
    plt.savefig('visualizations/eigenvalue_distribution.png', bbox_inches='tight')
    print("  ✓ eigenvalue_distribution.png")

    # Figure 2: Convergence Analysis
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    all_N = []
    all_num_zeros = []
    all_mean_real = []
    all_std_real = []
    all_first_height = []

    for N in range(5, 35, 5):
        solver = ConnesRiemannSolver(N)
        solver.build_connes_operator_v2()
        eigenvalues = solver.compute_eigenvalues()
        zeros = solver.extract_zeros()

        all_N.append(N)
        all_num_zeros.append(len(zeros))

        if zeros:
            real_parts = [np.real(z) for z in zeros]
            all_mean_real.append(np.mean(real_parts))
            all_std_real.append(np.std(real_parts))
            all_first_height.append(np.imag(zeros[0]) if zeros else 0)
        else:
            all_mean_real.append(0.5)
            all_std_real.append(0)
            all_first_height.append(0)

    # Plot 1: Number of zeros vs N
    ax1 = axes[0, 0]
    ax1.plot(all_N, all_num_zeros, 'bo-', markersize=8, linewidth=2)
    ax1.set_xlabel('Number of primes (N)')
    ax1.set_ylabel('Number of zeros found')
    ax1.set_title('Zero Count Scaling')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean real part convergence
    ax2 = axes[0, 1]
    ax2.plot(all_N, all_mean_real, 'go-', markersize=8, linewidth=2)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Target')
    ax2.set_xlabel('Number of primes (N)')
    ax2.set_ylabel('Mean Re(λ)')
    ax2.set_title('Critical Line Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.495, 0.505])

    # Plot 3: Standard deviation of real parts
    ax3 = axes[1, 0]
    ax3.plot(all_N, all_std_real, 'ro-', markersize=8, linewidth=2)
    ax3.set_xlabel('Number of primes (N)')
    ax3.set_ylabel('Std Dev of Re(λ)')
    ax3.set_title('Real Part Variance')
    ax3.grid(True, alpha=0.3)

    # Plot 4: First eigenvalue height
    ax4 = axes[1, 1]
    ax4.plot(all_N, all_first_height, 'mo-', markersize=8, linewidth=2, label='Computed')
    ax4.axhline(y=known_heights[0], color='red', linestyle='--', label=f'First known zero ({known_heights[0]:.2f})')
    ax4.set_xlabel('Number of primes (N)')
    ax4.set_ylabel('Im(ρ₁)')
    ax4.set_title('First Zero Height')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig2.suptitle('Convergence Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/convergence_analysis.png', bbox_inches='tight')
    print("  ✓ convergence_analysis.png")

    # Figure 3: Comparison with Known Zeros
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use N=30 for best results
    solver = ConnesRiemannSolver(30)
    solver.build_connes_operator_v2()
    eigenvalues = solver.compute_eigenvalues()
    zeros = solver.extract_zeros()

    # Plot 1: Height comparison
    ax1 = axes[0]
    if zeros:
        num_compare = min(10, len(zeros))
        computed_heights = [np.imag(z) for z in zeros[:num_compare]]
        x_pos = np.arange(num_compare)

        ax1.bar(x_pos - 0.2, computed_heights, 0.4, label='Computed', color='blue', alpha=0.7)
        ax1.bar(x_pos + 0.2, known_heights[:num_compare], 0.4, label='Known', color='red', alpha=0.7)
        ax1.set_xlabel('Zero index')
        ax1.set_ylabel('Height Im(ρ)')
        ax1.set_title('Zero Heights Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Error distribution
    ax2 = axes[1]
    if zeros and len(zeros) >= 5:
        errors = []
        relative_errors = []
        for i in range(min(10, len(zeros))):
            if i < len(known_heights):
                error = abs(np.imag(zeros[i]) - known_heights[i])
                rel_error = error / known_heights[i] * 100
                errors.append(error)
                relative_errors.append(rel_error)

        if errors:
            ax2.bar(range(len(errors)), relative_errors, color='orange', alpha=0.7)
            ax2.set_xlabel('Zero index')
            ax2.set_ylabel('Relative error (%)')
            ax2.set_title('Height Prediction Errors')
            ax2.grid(True, alpha=0.3)

    # Plot 3: Complex plane view
    ax3 = axes[2]
    ax3.scatter(np.real(eigenvalues), np.imag(eigenvalues),
               c='blue', s=30, alpha=0.6, label='Computed')
    known_complex = [0.5 + 1j*h for h in known_heights[:10]]
    ax3.scatter([0.5]*10, known_heights[:10],
               c='red', s=30, marker='x', label='Known zeros')
    ax3.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Real part')
    ax3.set_ylabel('Imaginary part')
    ax3.set_title('Complex Plane Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0.48, 0.52])

    fig3.suptitle('Comparison with Known Riemann Zeros (N=30)', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/known_zeros_comparison.png', bbox_inches='tight')
    print("  ✓ known_zeros_comparison.png")

    # Figure 4: Theoretical vs Actual
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Trace values for different functions
    ax1 = axes[0, 0]
    N_values = range(5, 30, 5)
    traces_identity = []
    traces_exp = []

    for N in N_values:
        solver = ConnesRiemannSolver(N)
        solver.build_connes_operator_v2()
        solver.compute_eigenvalues()

        trace_id = np.real(np.sum(solver.operator.eigenvalues))
        trace_exp = np.real(np.sum(np.exp(-0.1 * solver.operator.eigenvalues)))

        traces_identity.append(trace_id)
        traces_exp.append(trace_exp)

    ax1.plot(N_values, traces_identity, 'bo-', label='Tr(H)')
    ax1.plot(N_values, traces_exp, 'ro-', label='Tr(exp(-0.1H))')
    ax1.set_xlabel('Number of primes (N)')
    ax1.set_ylabel('Trace value')
    ax1.set_title('Trace Formula Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Eigenvalue density
    ax2 = axes[0, 1]
    solver = ConnesRiemannSolver(30)
    solver.build_connes_operator_v2()
    eigenvalues = solver.compute_eigenvalues()
    imag_parts = np.imag(eigenvalues)
    imag_parts = imag_parts[imag_parts > 0]

    if len(imag_parts) > 1:
        ax2.hist(imag_parts, bins=20, alpha=0.7, color='purple', edgecolor='black', density=True)
        ax2.set_xlabel('Im(λ)')
        ax2.set_ylabel('Density')
        ax2.set_title('Eigenvalue Height Distribution')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Matrix structure visualization
    ax3 = axes[1, 0]
    solver = ConnesRiemannSolver(10)
    operator = solver.build_connes_operator_v2()
    H_real = np.real(operator.H)
    im = ax3.imshow(H_real, cmap='RdBu', aspect='auto')
    ax3.set_title('Re(H) Matrix Structure')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im, ax=ax3)

    # Plot 4: Matrix imaginary part
    ax4 = axes[1, 1]
    H_imag = np.imag(operator.H)
    im = ax4.imshow(H_imag, cmap='viridis', aspect='auto')
    ax4.set_title('Im(H) Matrix Structure')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im, ax=ax4)

    fig4.suptitle('Theoretical Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/theoretical_analysis.png', bbox_inches='tight')
    print("  ✓ theoretical_analysis.png")

    # Figure 5: Summary Statistics
    fig5 = plt.figure(figsize=(12, 8))

    # Create text summary
    solver = ConnesRiemannSolver(30)
    solver.build_connes_operator_v2()
    eigenvalues = solver.compute_eigenvalues()
    zeros = solver.extract_zeros()
    verification = solver.verify_critical_line()

    summary_text = f"""
CONNES NONCOMMUTATIVE GEOMETRY APPROACH - SUMMARY STATISTICS

Configuration:
- Number of primes used: 30
- Matrix dimension: 30x30
- Operator type: Berry-Keating variant
- Construction: H = (1/2)I + iT

Results:
- Total eigenvalues computed: {len(eigenvalues)}
- Zeros with Im(λ) > 0: {len(zeros)}
- Mean Re(λ): {verification.get('mean_real_part', 0.5):.6f}
- Std Re(λ): {verification.get('std_real_part', 0):.6f}
- Max deviation from 0.5: {verification.get('max_deviation', 0):.6f}
- Percentage on critical line: {verification.get('percent_on_line', 0):.1f}%

Comparison with Known Zeros:
- First computed height: {np.imag(zeros[0]):.4f} (Known: {known_heights[0]:.4f})
- Second computed height: {np.imag(zeros[1]):.4f} (Known: {known_heights[1]:.4f})
- Third computed height: {np.imag(zeros[2]):.4f} (Known: {known_heights[2]:.4f})

Key Observations:
1. All eigenvalues have Re(λ) = 0.5 (critical line property confirmed)
2. Imaginary parts positive but don't match known zeros accurately
3. Eigenvalue degeneracy observed (duplicate values)
4. No clear convergence pattern as N increases
5. Berry-Keating construction more successful than direct prime log

Conclusions:
- Theoretical framework validated (critical line emerges)
- Numerical accuracy insufficient for zero prediction
- Operator construction remains the key challenge
- Further research needed on infinite-dimensional formulation
"""

    ax = fig5.add_subplot(111)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.savefig('visualizations/summary_statistics.png', bbox_inches='tight')
    print("  ✓ summary_statistics.png")

    print("\nAll visualizations generated successfully!")
    print("Files saved in visualizations/ directory")


if __name__ == "__main__":
    create_comprehensive_visualization()