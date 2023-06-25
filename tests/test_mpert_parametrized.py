"""
Parametrized tests for the modified PERT distribution.
Since it's a generalization of the PERT distribution, we don't need to repeat everything for the PERT distribution.
The generalization is tested in ``test_generalization.py``.
"""
import numpy as np
import pytest
import scipy

import betapert


@pytest.fixture
def mpert(params, lambd):
    """
    Provide a frozen mpert distribution. We request two parametrized fixtures, which results in the cartesian product of
    the options for each fixture, i.e. we will get a distribution for each combination of ``params`` and ``lambd``.
    """
    mini, mode, maxi = params
    return betapert.mpert(mini, mode, maxi, lambd=lambd)


class TestKnownProperties:
    def test_mode(self, mpert, params, lambd):
        """The mode is the mode parameter."""
        mini, want_mode, maxi = params

        # Get the mode by numerically maximizing the pdf
        fmin = lambda x: -mpert.pdf(x)
        x0 = mini + (maxi - mini) / 2
        optimize_result = scipy.optimize.minimize(fmin, x0=x0, bounds=[(mini, maxi)], tol=1e-12)
        if not optimize_result.success:
            raise RuntimeError("Numerical optimization failed")
        mode = optimize_result.x[0]

        assert mode == pytest.approx(want_mode, abs=1.5e-6, rel=1e-5)

    def test_support(self, mpert, params, lambd):
        """The support is the interval [mini, maxi]."""
        mini, mode, maxi = params
        assert mpert.support() == (mini, maxi)


class TestClosedFormExpressions:
    """
    Checks that our closed-form expressions for the mean, variance, and skewness match
    the results of numerical integration.

    expect() does numerical integration based on the pdf.
    """

    def test_mean(self, mpert, params, lambd):
        closed_form = mpert.mean()  # This eventually calls our closed-form formula
        numerical = mpert.expect()
        assert closed_form == pytest.approx(numerical)

    def test_var(self, mpert, params, lambd):
        closed_form = mpert.var()  # This eventually calls our closed-form formula
        numerical = mpert.expect(lambda x: (x - mpert.mean()) ** 2)
        assert closed_form == pytest.approx(numerical)

    def test_skewness(self, mpert, params, lambd):
        closed_form = mpert.stats(moments="s")  # This eventually calls our closed-form formula
        numerical = mpert.expect(
            # Fisher's skewness: third standardized moment
            lambda x: ((x - mpert.mean()) / mpert.std())
            ** 3
        )
        assert closed_form == pytest.approx(numerical)


class TestRvs:
    @pytest.fixture
    def random_seed(self):
        return np.random.seed(58672234)

    @pytest.fixture
    def rvs(self, mpert, random_seed):
        return mpert.rvs(size=100_000)

    def test_rvs_support(self, rvs, mpert, params, lambd):
        mini, mode, maxi = params
        assert np.all((rvs >= mini) & (rvs <= maxi))

    def test_rvs_moments(self, rvs, mpert, params, lambd):
        rtol = 0.05
        assert mpert.mean() == pytest.approx(rvs.mean(), rel=rtol)
        assert mpert.var() == pytest.approx(rvs.var(), rel=rtol)

    def test_rvs_quantiles(self, rvs, mpert, params, lambd):
        rtol = 0.05
        assert mpert.median() == pytest.approx(np.median(rvs), rel=rtol)

    def test_rvs_kolmogorov_smirnov(self, rvs, mpert, params, lambd):
        """Use the Kolmogorov-Smirnov test to check the entire distribution"""
        assert scipy.stats.kstest(rvs, mpert.cdf).pvalue > 0.05


def test_frozen_attrs(mpert, params, lambd):
    mini, mode, maxi = params

    numargs = mpert.dist.numargs
    assert mpert.args[:-numargs] == tuple([mini, mode, maxi, lambd][:-numargs])

    # SciPy calls the bounds "a" and "b"
    assert mpert.a == mini
    assert mpert.b == maxi
