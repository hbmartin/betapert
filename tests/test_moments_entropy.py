"""
Tests for the closed-form entropy and non-central moment implementations.
"""

import numpy as np
import pytest
import scipy.stats

import betapert
from betapert import funcs


@pytest.fixture
def mpert(params, lambd):
    mini, mode, maxi = params
    return betapert.mpert(mini, mode, maxi, lambd=lambd)


class TestEntropy:
    def test_matches_beta_closed_form(self):
        """PERT(0, 0.5, 1) is Beta(3, 3), so the entropies must agree."""
        assert betapert.pert(0, 0.5, 1).entropy() == pytest.approx(scipy.stats.beta(3, 3).entropy())

    def test_scale_shift(self):
        """Entropy of a scaled variable is the base entropy plus log of the scale."""
        base = betapert.pert(0, 0.5, 1).entropy()
        scaled = betapert.pert(0, 5, 10).entropy()
        assert scaled == pytest.approx(base + np.log(10))

    def test_matches_numerical_integration(self, mpert, params, lambd):
        closed_form = mpert.entropy()
        numerical = mpert.expect(lambda x: -np.log(mpert.pdf(x)))
        assert closed_form == pytest.approx(numerical, rel=1e-5)


class TestNonCentralMoments:
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_matches_numerical_integration(self, order):
        dist = betapert.mpert(1, 3, 10, lambd=3)
        closed_form = funcs.munp(order, 1, 3, 10, 3)
        numerical = dist.expect(lambda x: x**order)
        assert closed_form == pytest.approx(numerical, rel=1e-6)

    @pytest.mark.parametrize("order", [5, 6])
    def test_moment_method(self, order):
        """dist.moment() dispatches to _munp for orders above 4."""
        dist = betapert.pert(-2, 0, 5)
        numerical = dist.expect(lambda x: x**order)
        assert dist.moment(order) == pytest.approx(numerical, rel=1e-6)

    def test_zeroth_moment(self):
        assert funcs.munp(0, 0, 1, 3) == pytest.approx(1.0)

    def test_first_moment_is_mean(self):
        assert funcs.munp(1, 0, 1, 3) == pytest.approx(funcs.mean(0, 1, 3))

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            funcs.munp(-1, 0, 1, 3)
