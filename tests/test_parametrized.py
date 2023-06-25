"""Parametrized tests"""
import numpy as np
import pytest
import scipy

from betapert import betapert4, betapert3


@pytest.fixture
def random_seed():
    return np.random.seed(58672234)


@pytest.fixture(
    params=[
        (1, 2, 4),
        (-1, 0.1, 5),
        (0, 10, 12),
    ],
    ids=lambda x: f"mini={x[0]}, mode={x[1]}, maxi={x[2]}",
)
def params(request):
    """Provide required parameters (mini, mode, maxi)"""
    return request.param


@pytest.fixture(params=[1, 4, 8], ids=lambda x: f"lambd={x}")
def lambd(request):
    return request.param


@pytest.fixture
def dist(params, lambd):
    """
    Provide a frozen betapert distribution. We request two parametrized fixtures, which results in the cartesian product of
    the options for each fixture, i.e. we will get a distribution for each combination of ``params`` and ``lambd``.
    """
    mini, mode, maxi = params
    return betapert4(mini, mode, maxi, lambd)


def test_mode(dist, params, lambd):
    mini, want_mode, maxi = params
    # Get the mode by numerically maximizing the pdf
    fmin = lambda x: -dist.pdf(x)
    x0 = dist.mean()  # Start at the mean
    optimize_result = scipy.optimize.minimize(fmin, x0=x0, bounds=[(mini, maxi)], tol=1e-9)
    if not optimize_result.success:
        raise RuntimeError("Numerical optimization failed")
    mode = optimize_result.x[0]
    assert mode == pytest.approx(want_mode, abs=1.5e-6)


def test_mean(dist, params, lambd):
    closed_form = dist.mean()  # Our method with our formula
    numerical = dist.expect()  # Numerical integration based on the pdf
    assert closed_form == pytest.approx(numerical)


def test_var(dist, params, lambd):
    closed_form = dist.var()  # Our method with our formula
    numerical = dist.expect(
        lambda x: (x - dist.mean()) ** 2
    )  # Numerical integration based on the pdf
    assert closed_form == pytest.approx(numerical)


def test_skewness(dist, params, lambd):
    closed_form = dist.stats(moments="s")  # Calls our method with our formula

    numerical = dist.expect(
        # Fisher's skewness: third standardized moment
        lambda x: ((x - dist.mean()) / dist.std())
        ** 3
    )
    assert closed_form == pytest.approx(numerical)


def test_kurtoris(dist, params, lambd):
    """Does not check the values"""
    scipy_val = dist.stats(moments="k")  # Calculated by scipy's rv_continuous class
    assert isinstance(scipy_val, float)


def test_median(dist, params, lambd):
    closed_form = dist.median()  # Our method with our formula
    numerical = dist.ppf(0.5)  # Numerical integration
    assert closed_form == pytest.approx(numerical)


def test_stats(dist, params, lambd):
    """Just check that this calls the mean and var methods, doesn't check they are correct."""
    want = dist.mean(), dist.var()
    assert dist.stats(moments="mv") == pytest.approx(want)


@pytest.fixture
def rvs(dist, random_seed):
    return dist.rvs(size=100_000)


def test_rvs_support_moments(rvs, dist, params, lambd):
    mini, mode, maxi = params
    # Check that the generated random variables are within the expected range
    assert np.all((rvs >= mini) & (rvs <= maxi))

    # Check some statistics of the random variates
    rtol = 0.05
    assert dist.mean() == pytest.approx(rvs.mean(), rel=rtol)
    assert dist.var() == pytest.approx(rvs.var(), rel=rtol)
    assert dist.median() == pytest.approx(np.median(rvs), rel=rtol)


def test_rvs_kolmogorov_smirnov(rvs, dist, params, lambd):
    """Use the Kolmogorov-Smirnov test to check the entire distribution"""
    assert scipy.stats.kstest(rvs, dist.cdf).pvalue > 0.05


def test_use_default_lambd_frozen(params):
    mini, mode, maxi = params
    evaluate_at = np.arange(mini, maxi, 0.1)

    default_lambd = betapert3(mini, mode, maxi)
    provided_lambd = betapert4(mini, mode, maxi, 4)
    assert default_lambd.pdf(evaluate_at) == pytest.approx(provided_lambd.pdf(evaluate_at))


def test_use_default_lambd_non_frozen(params):
    mini, mode, maxi = params
    evaluate_at = np.arange(mini, maxi, 0.1)

    pdf_default_lambd = betapert3.pdf(evaluate_at, mini, mode, maxi)
    pdf_provided_lambd = betapert4.pdf(evaluate_at, mini, mode, maxi, 4)
    assert pdf_default_lambd == pytest.approx(pdf_provided_lambd)


def test_frozen_attrs(params, lambd):
    mini, mode, maxi = params
    dist = betapert4(mini, mode, maxi, lambd)
    2

    # SciPy calls the bounds "a" and "b"
    assert dist.a == mini
    assert dist.b == maxi


def test_non_frozen_attrs(params, lambd):
    # Recall, ``dist`` is a frozen instance of the distribution
    mini, mode, maxi = params
    evaluate_at = np.arange(mini, maxi, 0.1)
    frozen = betapert4(mini, mode, maxi, lambd)
    assert betapert4.pdf(evaluate_at, mini, mode, maxi, lambd) == pytest.approx(
        frozen.pdf(evaluate_at)
    )

    with pytest.raises(AttributeError):
        betapert4._alpha
    with pytest.raises(AttributeError):
        betapert4._beta
    with pytest.raises(AttributeError):
        betapert4.kwds
    with pytest.raises(AttributeError):
        betapert4.args
