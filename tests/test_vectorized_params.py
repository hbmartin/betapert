"""
Tests for array-valued (broadcast) distribution parameters, a standard SciPy idiom:
``pert.pdf(x, mini=np.array([...]), ...)``.
"""

import numpy as np
import pytest

from betapert import mpert, pert

MINI = np.array([0.0, 1.0, -5.0])
MODE = np.array([2.0, 3.0, 0.0])
MAXI = np.array([5.0, 10.0, 5.0])
LAMBD = np.array([4.0, 2.0, 6.0])
X = np.array([1.0, 5.0, 2.0])
Q = np.array([0.1, 0.5, 0.9])


@pytest.mark.parametrize("method", ["pdf", "cdf", "sf"])
def test_pert_methods_broadcast(method):
    vectorized = getattr(pert, method)(X, MINI, MODE, MAXI)
    elementwise = [
        getattr(pert, method)(x, m, mo, ma)
        for x, m, mo, ma in zip(X, MINI, MODE, MAXI, strict=True)
    ]
    assert vectorized == pytest.approx(np.array(elementwise))


@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_pert_inverse_methods_broadcast(method):
    vectorized = getattr(pert, method)(Q, MINI, MODE, MAXI)
    elementwise = [
        getattr(pert, method)(q, m, mo, ma)
        for q, m, mo, ma in zip(Q, MINI, MODE, MAXI, strict=True)
    ]
    assert vectorized == pytest.approx(np.array(elementwise))


@pytest.mark.parametrize("method", ["pdf", "cdf", "sf"])
def test_mpert_methods_broadcast(method):
    vectorized = getattr(mpert, method)(X, MINI, MODE, MAXI, LAMBD)
    elementwise = [
        getattr(mpert, method)(x, m, mo, ma, la)
        for x, m, mo, ma, la in zip(X, MINI, MODE, MAXI, LAMBD, strict=True)
    ]
    assert vectorized == pytest.approx(np.array(elementwise))


def test_mean_broadcast():
    vectorized = pert.mean(MINI, MODE, MAXI)
    elementwise = [pert.mean(m, mo, ma) for m, mo, ma in zip(MINI, MODE, MAXI, strict=True)]
    assert vectorized == pytest.approx(np.array(elementwise))


def test_invalid_element_yields_nan():
    """An invalid parameter combination poisons only its own element."""
    mini = np.array([0.0, 3.0])
    mode = np.array([2.0, 2.0])  # second element: mode < mini, invalid
    maxi = np.array([5.0, 5.0])
    result = pert.pdf(np.array([1.0, 4.0]), mini, mode, maxi)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])
