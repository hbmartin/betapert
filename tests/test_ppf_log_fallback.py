"""
Test the ppf log fallback functionality.
"""

import unittest.mock

import numpy as np
import pytest
import scipy.optimize
import scipy.stats

import betapert
from betapert import funcs


class TestPPFLogFallback:
    """Test the log-space fallback functionality for ppf calculations."""

    @classmethod
    def setup_class(cls):
        cls.original_brentq = scipy.optimize.brentq
        cls.original_ppf = scipy.stats.beta.ppf
        cls.mock_ppf_nan_calls = 0

        def mock_ppf_nan(*args, **kwargs):
            cls.mock_ppf_nan_calls += 1
            q = args[0] if args else kwargs["q"]
            if hasattr(q, "shape"):
                return np.full(q.shape, np.nan)
            return np.nan

        cls.mock_ppf_nan = mock_ppf_nan

    def teardown_method(self, method):
        scipy.optimize.brentq = TestPPFLogFallback.original_brentq
        scipy.stats.beta.ppf = TestPPFLogFallback.original_ppf
        TestPPFLogFallback.mock_ppf_nan_calls = 0
        betapert.FALLBACK = None

    def test_log_fallback_basic_functionality(self):
        """Test that log fallback works for basic cases."""
        mini, mode, maxi = 0, 1, 10
        q = np.array([0.1, 0.5, 0.9])

        # Test direct fallback function
        result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, 4)

        # Should return valid values within bounds
        assert np.all(result >= mini)
        assert np.all(result <= maxi)
        assert np.all(np.isfinite(result))

        # Should be monotonic
        assert np.all(np.diff(result) >= 0)

    def test_log_fallback_extreme_probabilities(self):
        """Test log fallback with extreme probability values."""
        mini, mode, maxi = 0, 1, 10

        # Test very small probabilities
        q_small = np.array([1e-10, 1e-15, 1e-20])
        result_small = funcs._ppf_fallback_log_space(q_small, mini, mode, maxi, 4)

        assert np.all(result_small >= mini)
        assert np.all(result_small <= maxi)
        assert np.all(np.isfinite(result_small))

        # Test very large probabilities
        q_large = np.array([1 - 1e-10, 1 - 1e-15, 1 - 1e-20])
        result_large = funcs._ppf_fallback_log_space(q_large, mini, mode, maxi, 4)

        assert np.all(result_large >= mini)
        assert np.all(result_large <= maxi)
        assert np.all(np.isfinite(result_large))

    def test_log_fallback_edge_cases(self):
        """Test log fallback with edge case probabilities."""
        mini, mode, maxi = 0, 1, 10

        # Test q = 0 and q = 1 (should be clamped internally)
        q_edge = np.array([0, 1])
        result_edge = funcs._ppf_fallback_log_space(q_edge, mini, mode, maxi, 4)

        assert np.all(result_edge >= mini)
        assert np.all(result_edge <= maxi)
        assert np.all(np.isfinite(result_edge))

    def test_log_fallback_scalar_input(self):
        """Test that log fallback works with scalar input."""
        mini, mode, maxi = 0, 1, 10
        q = 0.5

        result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, 4)

        assert isinstance(result, (int, float, np.number))
        assert mini <= result <= maxi
        assert np.isfinite(result)

    def test_log_fallback_different_parameters(self):
        """Test log fallback with different parameter combinations."""
        test_cases = [
            (0, 1, 10, 4),
            (-5, 0, 5, 2),
            (100, 150, 200, 6),
            (0.1, 0.2, 0.3, 3),
        ]

        q = np.array([0.1, 0.5, 0.9])

        for mini, mode, maxi, lambd in test_cases:
            result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, lambd)

            assert np.all(result >= mini), f"Result {result} not >= {mini}"
            assert np.all(result <= maxi), f"Result {result} not <= {maxi}"
            assert np.all(np.isfinite(result)), f"Result {result} not finite"

    def test_ppf_fallback_integration(self):
        """Test that ppf functions correctly with (unused) log fallback."""
        mini, mode, maxi = 0, 1, 10

        # Create a distribution with log fallback
        dist = betapert.pert(mini, mode, maxi)

        # Test normal probabilities
        q_normal = np.array([0.1, 0.5, 0.9])
        result_normal = dist.ppf(q_normal)

        assert np.all(result_normal >= mini)
        assert np.all(result_normal <= maxi)
        assert np.all(np.isfinite(result_normal))

    def test_ppf_fallback_not_triggered_when_not_configured(self):
        """Test that fallback is NOT triggered by default when regular ppf returns NaN."""
        mini, mode, maxi = 0, 1, 10

        with unittest.mock.patch("scipy.stats.beta.ppf", new=TestPPFLogFallback.mock_ppf_nan):
            dist = betapert.pert(mini, mode, maxi)
            result = dist.ppf(0.5)
            assert np.isnan(result)
            assert TestPPFLogFallback.mock_ppf_nan_calls == 1

    def test_ppf_fallback_triggered_by_nan_with_module_ppf(self):
        """Test that fallback is triggered when regular ppf returns NaN."""
        mini, mode, maxi = 0, 1, 10

        with unittest.mock.patch("scipy.stats.beta.ppf", new=TestPPFLogFallback.mock_ppf_nan):
            betapert.FALLBACK = "log"
            dist = betapert.pert(mini, mode, maxi)
            result = dist.ppf(0.5)
            assert TestPPFLogFallback.mock_ppf_nan_calls == 1
            assert np.isfinite(result)
            assert mini <= result <= maxi

    def test_ppf_fallback_triggered_by_nan_directly(self):
        """Test that fallback is triggered when regular ppf returns NaN."""
        mini, mode, maxi = 0, 1, 10

        result = funcs.ppf(0.5, mini, mode, maxi, fallback="log")
        assert np.isfinite(result)

        with unittest.mock.patch("scipy.stats.beta.ppf", new=TestPPFLogFallback.mock_ppf_nan):
            # This should trigger the fallback
            result = funcs.ppf(0.5, mini, mode, maxi, fallback="log")
            assert np.isfinite(result)
            assert mini <= result <= maxi
            assert TestPPFLogFallback.mock_ppf_nan_calls == 1

    def test_log_fallback_consistency_with_cdf(self):
        """Test that log fallback results are consistent with CDF."""
        mini, mode, maxi = 0, 1, 10
        lambd = 4

        # Test several probability values
        q_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Get ppf results using log fallback
        ppf_results = funcs._ppf_fallback_log_space(q_values, mini, mode, maxi, lambd)

        # Verify by computing CDF of the results
        cdf_results = funcs.cdf(ppf_results, mini, mode, maxi, lambd)

        # Should be approximately equal to original q values
        np.testing.assert_allclose(cdf_results, q_values, rtol=1e-6)

    def test_log_fallback_with_secondary_clip_fallback(self):
        """Test that multiple fallback levels work correctly."""
        mini, mode, maxi = 0, 1, 10
        q = 0.5

        # Mock both scipy functions to fail
        original_brentq = scipy.optimize.brentq
        original_ppf = scipy.stats.beta.ppf

        def mock_brentq_fail(*args, **kwargs):
            raise ValueError("Brentq failed")

        scipy.optimize.brentq = mock_brentq_fail

        try:
            # This should fall back to linear interpolation
            result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, 4)

            assert np.isfinite(result)
            assert mini <= result <= maxi

        finally:
            # Restore original functions
            scipy.optimize.brentq = original_brentq
            scipy.stats.beta.ppf = original_ppf

    def test_ppf_without_fallback_returns_nan(self):
        """Test that ppf without fallback returns NaN when appropriate."""
        mini, mode, maxi = 0, 1, 10

        with unittest.mock.patch("scipy.stats.beta.ppf", new=TestPPFLogFallback.mock_ppf_nan):
            # Without fallback, should return NaN
            result = funcs.ppf(0.5, mini, mode, maxi, fallback=None)
            assert np.isnan(result)
            assert TestPPFLogFallback.mock_ppf_nan_calls == 1

    def test_log_fallback_with_arrays(self):
        """Test log fallback with various array shapes."""
        mini, mode, maxi = 0, 1, 10

        # Test 1D array
        q_1d = np.array([0.1, 0.5, 0.9])
        result_1d = funcs._ppf_fallback_log_space(q_1d, mini, mode, maxi, 4)

        assert result_1d.shape == q_1d.shape
        assert np.all(result_1d >= mini)
        assert np.all(result_1d <= maxi)

        # Test 2D array
        q_2d = np.array([[0.1, 0.5], [0.7, 0.9]])
        result_2d = funcs._ppf_fallback_log_space(q_2d, mini, mode, maxi, 4)

        assert result_2d.shape == q_2d.shape
        assert np.all(result_2d >= mini)
        assert np.all(result_2d <= maxi)

    def test_log_fallback_monotonicity(self):
        """Test that log fallback preserves monotonicity."""
        mini, mode, maxi = 0, 1, 10

        # Create monotonic sequence of probabilities
        q_values = np.linspace(0.01, 0.99, 20)

        result = funcs._ppf_fallback_log_space(q_values, mini, mode, maxi, 4)

        # Check that results are monotonic
        assert np.all(np.diff(result) >= 0), "Results should be monotonic"

    def test_invalid_fallback_type(self):
        """Test that invalid fallback type raises appropriate error."""
        mini, mode, maxi = 0, 1, 10

        with unittest.mock.patch(  # noqa: SIM117
            "scipy.stats.beta.ppf",
            new=TestPPFLogFallback.mock_ppf_nan,
        ):
            with pytest.raises(KeyError):
                funcs.ppf(0.5, mini, mode, maxi, fallback="invalid")

    def test_log_fallback_performance_edge_case(self):
        """Test performance with extreme parameter values that might cause numerical issues."""
        # Very small range
        mini, mode, maxi = 0, 1e-10, 1e-9
        q = np.array([0.1, 0.5, 0.9])

        result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, 4)

        assert np.all(result >= mini)
        assert np.all(result <= maxi)
        assert np.all(np.isfinite(result))

        # Very large range
        mini, mode, maxi = 0, 1e6, 1e10
        result = funcs._ppf_fallback_log_space(q, mini, mode, maxi, 4)

        assert np.all(result >= mini)
        assert np.all(result <= maxi)
        assert np.all(np.isfinite(result))

    def test_log_fallback_2d_input(self):
        """Test log fallback with 2D input arrays."""
        mini, mode, maxi = 0, 1, 10

        # Test 2D array input
        q_2d = np.array([[0.1, 0.3, 0.5], [0.7, 0.9, 0.99]])
        result_2d = funcs._ppf_fallback_log_space(q_2d, mini, mode, maxi, 4)

        # Should preserve shape
        assert result_2d.shape == q_2d.shape
        assert np.all(result_2d >= mini)
        assert np.all(result_2d <= maxi)
        assert np.all(np.isfinite(result_2d))

        # Test 3D array input
        q_3d = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result_3d = funcs._ppf_fallback_log_space(q_3d, mini, mode, maxi, 4)

        assert result_3d.shape == q_3d.shape
        assert np.all(result_3d >= mini)
        assert np.all(result_3d <= maxi)
        assert np.all(np.isfinite(result_3d))

    def test_log_fallback_scalar_return(self):
        """Test that log fallback returns scalar when input is scalar."""
        mini, mode, maxi = 0, 1, 10

        # Test scalar input
        q_scalar = 0.5
        result_scalar = funcs._ppf_fallback_log_space(q_scalar, mini, mode, maxi, 4)

        # Should return scalar, not array
        assert isinstance(result_scalar, np.number)
        assert not isinstance(result_scalar, np.ndarray)
        assert mini <= result_scalar <= maxi
        assert np.isfinite(result_scalar)

        # Test single-element array input
        q_single = np.array([0.5])
        result_single = funcs._ppf_fallback_log_space(q_single, mini, mode, maxi, 4)

        # Should return single-element array
        assert isinstance(result_single, np.ndarray)
        assert mini <= result_single <= maxi
        assert np.isfinite(result_single)

    def test_log_fallback_array_return(self):
        """Test that log fallback returns array when input is array with multiple elements."""
        mini, mode, maxi = 0, 1, 10

        # Test array with multiple elements
        q_array = np.array([0.1, 0.5, 0.9])
        result_array = funcs._ppf_fallback_log_space(q_array, mini, mode, maxi, 4)

        # Should return array
        assert isinstance(result_array, np.ndarray)
        assert result_array.shape == q_array.shape
        assert np.all(result_array >= mini)
        assert np.all(result_array <= maxi)
        assert np.all(np.isfinite(result_array))

    def test_log_fallback_3d_input(self):
        """Test log fallback with 3D input arrays."""
        mini, mode, maxi = 0, 1, 10

        # Test 3D array input
        q_3d = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result_3d = funcs._ppf_fallback_log_space(q_3d, mini, mode, maxi, 4)

        # Should preserve shape
        assert result_3d.shape == q_3d.shape
        assert np.all(result_3d >= mini)
        assert np.all(result_3d <= maxi)
        assert np.all(np.isfinite(result_3d))

        # Test larger 3D array
        q_3d_large = np.array(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.95, 0.98, 0.99]]],
        )
        result_3d_large = funcs._ppf_fallback_log_space(q_3d_large, mini, mode, maxi, 4)

        assert result_3d_large.shape == q_3d_large.shape
        assert np.all(result_3d_large >= mini)
        assert np.all(result_3d_large <= maxi)
        assert np.all(np.isfinite(result_3d_large))

    def test_calc_alpha_beta_array_args_all_equal(self):
        """Test _calc_alpha_beta returns scalars when all array elements are equal."""
        from betapert.funcs import _calc_alpha_beta
        import numpy as np

        mini = np.array([0.0, 0.0, 0.0])
        mode = np.array([1.0, 1.0, 1.0])
        maxi = np.array([10.0, 10.0, 10.0])
        lambd = np.array([4.0, 4.0, 4.0])
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        assert np.isscalar(alpha)
        assert np.isscalar(beta)
        # Should match the scalar computation
        alpha_scalar, beta_scalar = _calc_alpha_beta(0.0, 1.0, 10.0, 4.0)
        assert alpha == alpha_scalar
        assert beta == beta_scalar

    def test_calc_alpha_beta_array_args_not_all_equal(self):
        """Test _calc_alpha_beta returns arrays when not all elements are equal."""
        from betapert.funcs import _calc_alpha_beta
        import numpy as np

        mini = np.array([0.0, 0.0, 0.0])
        mode = np.array([1.0, 2.0, 3.0])
        maxi = np.array([10.0, 10.0, 10.0])
        lambd = np.array([4.0, 4.0, 4.0])
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        assert isinstance(alpha, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert alpha.shape == mode.shape
        assert beta.shape == mode.shape
        # Should not be all equal
        assert not np.all(alpha == alpha[0]) or not np.all(beta == beta[0])
