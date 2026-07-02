"""
Regression tests against independent reference formulas and pinned values, guarding the
alpha/beta parameter mapping against accidental changes.
"""

import pytest

import betapert


class TestClassicPERTFormulas:
    """The classic PERT (lambd=4) has well-known closed forms independent of our implementation."""

    def test_mean(self, params):
        mini, mode, maxi = params
        dist = betapert.pert(mini, mode, maxi)
        assert dist.mean() == pytest.approx((mini + 4 * mode + maxi) / 6)

    def test_var(self, params):
        """The classic PERT variance identity: var = (μ - mini)(maxi - μ) / 7."""
        mini, mode, maxi = params
        dist = betapert.pert(mini, mode, maxi)
        mu = dist.mean()
        assert dist.var() == pytest.approx((mu - mini) * (maxi - mu) / 7)

    def test_symmetric_has_zero_skew(self):
        assert betapert.pert(0, 5, 10).stats(moments="s") == pytest.approx(0.0)

    def test_right_skewed_has_positive_skew(self):
        assert betapert.pert(0, 1, 10).stats(moments="s") > 0

    def test_left_skewed_has_negative_skew(self):
        assert betapert.pert(0, 9, 10).stats(moments="s") < 0


class TestPinnedValues:
    """Values pinned from Wolfram Mathematica's PERTDistribution.

    See https://reference.wolfram.com/language/ref/PERTDistribution.html
    """

    def test_pdf(self):
        """PDF[PERTDistribution[{1,10}, 3, 2]][5]   =>    0.15207705617310344"""
        dist = betapert.mpert(mini=1, mode=3, maxi=10, lambd=2)
        assert dist.pdf(5) == pytest.approx(0.15207705617310344)

    def test_cdf(self):
        """CDF[PERTDistribution[{10^-5, 10^-3}, 10^-4, 1]][5*10^-5]  =>  0.0588892278343665"""
        dist = betapert.mpert(mini=1e-5, mode=1e-4, maxi=1e-3, lambd=1)
        assert dist.cdf(5e-5) == pytest.approx(0.0588892278343665)

    def test_mean_mpert(self):
        """Mean[PERTDistribution[{1,10}, 3, 2]]  =>  (1 + 10 + 2*3)/(2+2) = 17/4"""
        dist = betapert.mpert(mini=1, mode=3, maxi=10, lambd=2)
        assert dist.mean() == pytest.approx(17 / 4)

    def test_kurtosis_symmetric(self):
        """A symmetric PERT (lambd=4) is Beta(3,3); excess kurtosis of Beta(a,a) is -6/(2a+3)."""
        dist = betapert.pert(0, 0.5, 1)
        assert dist.stats(moments="k") == pytest.approx(-2 / 3)
