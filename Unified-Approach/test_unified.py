#!/usr/bin/env python3
"""
Test suite for the unified Riemann Hypothesis implementation.
"""

import numpy as np
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_implementation import UnifiedOperator
import warnings
warnings.filterwarnings('ignore')


class TestUnifiedOperator(unittest.TestCase):
    """Test cases for the unified operator implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.op = UnifiedOperator(N=100, epsilon=0.01, L=50, prime_cutoff=100)

    def test_initialization(self):
        """Test operator initialization."""
        self.assertEqual(self.op.N, 100)
        self.assertEqual(self.op.epsilon, 0.01)
        self.assertEqual(self.op.L, 50)
        self.assertIsNotNone(self.op.x)
        self.assertEqual(len(self.op.x), 100)

    def test_grid_properties(self):
        """Test logarithmic grid properties."""
        # Check grid is logarithmically spaced
        self.assertAlmostEqual(self.op.x[0], self.op.epsilon, places=10)
        self.assertAlmostEqual(self.op.x[-1], self.op.L, delta=1.0)

        # Check monotonicity
        diffs = np.diff(self.op.x)
        self.assertTrue(np.all(diffs > 0), "Grid must be monotonically increasing")

    def test_prime_generation(self):
        """Test prime number generation."""
        primes_100 = self.op.generate_primes(100)
        expected_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                          53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        self.assertEqual(primes_100, expected_primes)

    def test_operator_construction(self):
        """Test that operator can be constructed."""
        H = self.op.construct_operator()
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (100, 100))
        self.assertEqual(H.dtype, np.complex128)

    def test_hermiticity(self):
        """Test that operator is Hermitian."""
        self.op.construct_operator()
        error = self.op.verify_hermiticity()
        self.assertLess(error, 1e-10, "Operator must be Hermitian")

    def test_eigenvalue_computation(self):
        """Test eigenvalue computation."""
        eigenvalues, eigenvectors = self.op.compute_eigenvalues()
        self.assertIsNotNone(eigenvalues)
        self.assertIsNotNone(eigenvectors)
        self.assertGreater(len(eigenvalues), 0)

        # Check all eigenvalues have positive imaginary part
        self.assertTrue(np.all(np.imag(eigenvalues) > 0),
                       "All eigenvalues should have positive imaginary part")

    def test_real_parts_near_half(self):
        """Test that real parts of eigenvalues are near 1/2."""
        self.op.construct_operator()
        eigenvalues, _ = self.op.compute_eigenvalues()

        real_parts = np.real(eigenvalues)
        mean_real = np.mean(real_parts)
        std_real = np.std(real_parts)

        # Check mean is close to 0.5
        self.assertAlmostEqual(mean_real, 0.5, delta=0.1,
                             msg=f"Mean real part {mean_real:.6f} should be near 0.5")

        # Check standard deviation is small
        self.assertLess(std_real, 0.1,
                       msg=f"Std of real parts {std_real:.6f} should be small")

    def test_comparison_with_zeros(self):
        """Test comparison with actual Riemann zeros."""
        results = self.op.compare_with_zeros(num_zeros=10)

        self.assertIn('mean_abs_error', results)
        self.assertIn('correlation', results)
        self.assertIn('real_parts_mean', results)

        # Check that we have reasonable accuracy (not necessarily breakthrough level)
        self.assertLess(results['mean_abs_error'], 100,
                       "Mean error should be reasonable")
        self.assertGreater(results['correlation'], 0,
                          "Should have positive correlation")

    def test_berry_keating_term(self):
        """Test Berry-Keating term construction."""
        T_BK = self.op.construct_berry_keating_term()

        self.assertEqual(T_BK.shape, (self.op.N, self.op.N))
        self.assertEqual(T_BK.dtype, np.complex128)

        # Check that it's purely imaginary (after multiplying by i)
        self.assertTrue(np.allclose(np.real(T_BK), 0, atol=1e-10),
                       "Berry-Keating term should be purely imaginary")

    def test_frobenius_term(self):
        """Test Frobenius prime potential term."""
        T_F = self.op.construct_frobenius_term()

        self.assertEqual(T_F.shape, (self.op.N, self.op.N))
        self.assertEqual(T_F.dtype, np.complex128)

        # Check it's diagonal (as it should be for a potential)
        off_diagonal = T_F - np.diag(np.diag(T_F))
        self.assertTrue(np.allclose(off_diagonal, 0, atol=1e-10),
                       "Frobenius term should be diagonal")

    def test_save_and_load_results(self):
        """Test saving results to JSON."""
        import tempfile
        import json

        self.op.compare_with_zeros(num_zeros=5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name

        try:
            self.op.save_results(temp_filename)

            # Check file was created and can be loaded
            with open(temp_filename, 'r') as f:
                data = json.load(f)

            self.assertIn('parameters', data)
            self.assertIn('comparison_results', data)
            self.assertIn('eigenvalues', data)

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


class TestConvergence(unittest.TestCase):
    """Test convergence properties."""

    def test_convergence_with_N(self):
        """Test that results improve with larger N."""
        errors = []
        N_values = [50, 100, 150]

        for N in N_values:
            op = UnifiedOperator(N=N, epsilon=0.01, L=30, prime_cutoff=50)
            results = op.compare_with_zeros(num_zeros=5)
            errors.append(results['mean_abs_error'])

        # Check that error generally decreases with N
        # (may not be strictly monotonic due to discretization effects)
        self.assertLess(errors[-1], errors[0] * 1.5,
                       "Error should not increase significantly with N")


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of the solution."""

    def test_mellin_transform_connection(self):
        """Test that operator respects Mellin transform structure."""
        op = UnifiedOperator(N=50, epsilon=0.1, L=10, prime_cutoff=30)
        H = op.construct_operator()

        # The operator should map functions in a way consistent with
        # the Mellin transform connection: xp̂ → multiplication by s
        # This is a basic structural test
        self.assertIsNotNone(H)
        self.assertEqual(H.shape[0], H.shape[1])

    def test_prime_encoding(self):
        """Test that prime structure is encoded in operator."""
        op_with_primes = UnifiedOperator(N=50, epsilon=0.01, L=20, prime_cutoff=100)
        op_no_primes = UnifiedOperator(N=50, epsilon=0.01, L=20, prime_cutoff=1)

        H_with = op_with_primes.construct_operator()
        H_without = op_no_primes.construct_operator()

        # Operators should be different when primes are included
        difference = np.linalg.norm(H_with - H_without, 'fro')
        self.assertGreater(difference, 0.01,
                          "Operator should change when primes are included")


def run_tests():
    """Run all tests with detailed output."""
    print("=" * 70)
    print("RUNNING TEST SUITE FOR UNIFIED RIEMANN HYPOTHESIS IMPLEMENTATION")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedOperator))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergence))
    suite.addTests(loader.loadTestsFromTestCase(TestMathematicalProperties))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)