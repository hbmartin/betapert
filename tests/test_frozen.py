"""
Test that the frozen and non-frozen distributions give the same numerical results.
"""
import numpy as np
import pytest

import betapert

# Define constants
mini = 1
mode = 3
maxi = 10
lambd = 3
x = np.linspace(mini, maxi, 100)
q = 0.5
order = 2
confidence = 0.95


# Define frozen and non-frozen distributions
@pytest.fixture
def mpert_frozen():
    return betapert.mpert(mini, mode, maxi, lambd)


@pytest.fixture
def mpert_nonfrozen():
    return betapert.mpert


@pytest.fixture
def pert_frozen():
    return betapert.pert(mini, mode, maxi)


@pytest.fixture
def pert_nonfrozen():
    return betapert.pert


@pytest.fixture(params=["PERT", "ModifiedPERT"])
def dist_pair(request, mpert_frozen, mpert_nonfrozen, pert_frozen, pert_nonfrozen):
    if request.param == "PERT":
        return (mpert_frozen, mpert_nonfrozen)
    elif request.param == "ModifiedPERT":
        return (pert_frozen, pert_nonfrozen)


def test(dist_pair, method):
    frozen, nonfrozen = dist_pair

    frozen_method = getattr(frozen, method)

    if isinstance(frozen.dist, betapert.ModifiedPERT):
        nonfrozen_method = getattr(nonfrozen(mini, mode, maxi, lambd), method)
    elif isinstance(frozen.dist, betapert.PERT):
        nonfrozen_method = getattr(nonfrozen(mini, mode, maxi), method)

    if method in ["moment"]:
        frozen_value = frozen_method(order)
        nonfrozen_value = nonfrozen_method(order)
    elif method in ["ppf", "isf"]:
        frozen_value = frozen_method(q)
        nonfrozen_value = nonfrozen_method(q)
    elif method in ["interval"]:
        frozen_value = frozen_method(confidence)
        nonfrozen_value = nonfrozen_method(confidence)
    elif method in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]:
        frozen_value = frozen_method(x)
        nonfrozen_value = nonfrozen_method(x)
    else:
        frozen_value = frozen_method()
        nonfrozen_value = nonfrozen_method()

    assert frozen_value == pytest.approx(nonfrozen_value)
