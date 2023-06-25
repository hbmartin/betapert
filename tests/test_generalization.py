"""
Test that the PERT distribution is equivalent to the modified PERT distribution with ``lambd=4``.
"""
import numpy as np
import pytest

import betapert


@pytest.fixture
def mpert(params):
    mini, mode, maxi = params
    return betapert.mpert(mini, mode, maxi, lambd=4)


@pytest.fixture
def pert(params):
    mini, mode, maxi = params
    return betapert.pert(mini, mode, maxi)


@pytest.fixture(
    params=[
        "pdf",
        "cdf",
        "stats",
        "median",
        "support",
    ]
)
def method(request):
    """The method to test"""
    return request.param


def test(method, mpert, pert):
    x = np.linspace(*mpert.support())
    if method in ["pdf", "cdf"]:
        assert getattr(mpert, method)(x) == pytest.approx(getattr(pert, method)(x))
    else:
        assert getattr(mpert, method)() == pytest.approx(getattr(pert, method)())
