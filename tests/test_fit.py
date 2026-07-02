"""
Tests for the maximum-likelihood parameter estimation helper ``betapert.fit``.
"""

import numpy as np
import pytest

import betapert


class TestFitRecoversParameters:
    def test_classic_pert(self):
        true = (2.0, 5.0, 12.0)
        data = betapert.pert(*true).rvs(size=50_000, random_state=1234)
        mini, mode, maxi = betapert.fit(data)
        assert mini == pytest.approx(true[0], abs=0.25)
        assert mode == pytest.approx(true[1], abs=0.5)
        assert maxi == pytest.approx(true[2], abs=0.5)

    def test_mpert_with_lambd(self):
        true = (-1.0, 0.0, 3.0)
        data = betapert.mpert(*true, lambd=2).rvs(size=50_000, random_state=99)
        mini, mode, maxi = betapert.fit(data, lambd=2)
        assert mini == pytest.approx(true[0], abs=0.25)
        assert mode == pytest.approx(true[1], abs=0.5)
        assert maxi == pytest.approx(true[2], abs=0.5)

    def test_endpoints_bracket_data(self):
        data = betapert.pert(0, 1, 4).rvs(size=1000, random_state=7)
        mini, mode, maxi = betapert.fit(data)
        assert mini < data.min()
        assert maxi > data.max()
        assert mini <= mode <= maxi

    def test_fitted_params_are_valid(self):
        data = betapert.pert(10, 20, 90).rvs(size=5000, random_state=3)
        params = betapert.fit(data)
        # The fitted parameters must define a usable distribution
        dist = betapert.pert(*params)
        assert np.isfinite(dist.mean())
        assert np.all(np.isfinite(dist.pdf(data)))


class TestFitValidation:
    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least"):
            betapert.fit([1.0, 2.0])

    def test_constant_data(self):
        with pytest.raises(ValueError, match="non-constant"):
            betapert.fit([3.0, 3.0, 3.0, 3.0])

    def test_non_finite_data(self):
        with pytest.raises(ValueError, match="finite"):
            betapert.fit([1.0, 2.0, np.nan, 4.0])

    def test_invalid_lambd(self):
        with pytest.raises(ValueError, match="lambd"):
            betapert.fit([1.0, 2.0, 3.0], lambd=0)
