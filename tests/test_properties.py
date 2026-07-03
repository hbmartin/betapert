"""
Property-based tests using Hypothesis: invariants that must hold for any valid
parameter combination.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from betapert import funcs, mpert, pert


@st.composite
def pert_params(draw, *, allow_degenerate_mode=False):
    """Generate valid (mini, mode, maxi) with a bounded span to keep numerics honest."""
    mini = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    span = draw(st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False))
    low, high = (0.0, 1.0) if allow_degenerate_mode else (0.01, 0.99)
    frac = draw(st.floats(min_value=low, max_value=high))
    return mini, mini + frac * span, mini + span


lambds = st.floats(min_value=0.1, max_value=20)
quantiles = st.floats(min_value=0.001, max_value=0.999)


@settings(deadline=None)
@given(params=pert_params(), lambd=lambds, q=quantiles)
def test_ppf_cdf_roundtrip(params, lambd, q):
    mini, mode, maxi = params
    x = funcs.ppf(q, mini, mode, maxi, lambd, fallback="log")
    assert mini <= x <= maxi
    assert funcs.cdf(x, mini, mode, maxi, lambd) == pytest.approx(q, rel=1e-4, abs=1e-4)


@settings(deadline=None)
@given(params=pert_params(allow_degenerate_mode=True), lambd=lambds)
def test_pdf_non_negative(params, lambd):
    mini, mode, maxi = params
    x = np.linspace(mini, maxi, 50)
    density = funcs.pdf(x, mini, mode, maxi, lambd)
    assert np.all(density >= 0)


@settings(deadline=None)
@given(params=pert_params(allow_degenerate_mode=True), lambd=lambds)
def test_sf_complements_cdf(params, lambd):
    mini, mode, maxi = params
    x = np.linspace(mini, maxi, 20)
    total = funcs.cdf(x, mini, mode, maxi, lambd) + funcs.sf(x, mini, mode, maxi, lambd)
    assert total == pytest.approx(np.ones_like(x))


@settings(deadline=None)
@given(params=pert_params(allow_degenerate_mode=True), lambd=lambds)
def test_cdf_is_monotone(params, lambd):
    mini, mode, maxi = params
    x = np.linspace(mini, maxi, 50)
    probabilities = funcs.cdf(x, mini, mode, maxi, lambd)
    assert np.all(np.diff(probabilities) >= 0)
    assert probabilities[0] == pytest.approx(0.0, abs=1e-12)
    assert probabilities[-1] == pytest.approx(1.0)


@settings(deadline=None)
@given(params=pert_params(allow_degenerate_mode=True), lambd=lambds)
def test_moments_are_sane(params, lambd):
    mini, mode, maxi = params
    mu = funcs.mean(mini, mode, maxi, lambd)
    sigma2 = funcs.var(mini, mode, maxi, lambd)
    assert mini <= mu <= maxi
    # Popoviciu's inequality for a bounded variable
    assert 0 <= sigma2 <= (maxi - mini) ** 2 / 4


@settings(deadline=None)
@given(params=pert_params())
def test_pert_equals_mpert_lambd_4(params):
    mini, mode, maxi = params
    x = np.linspace(mini, maxi, 20)
    assert pert.pdf(x, mini, mode, maxi) == pytest.approx(
        mpert.pdf(x, mini, mode, maxi, 4),
    )
