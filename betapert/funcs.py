"""Functions implementing the PERT and modified PERT distributions.

This module contains the core mathematical functions used by the PERT and modified PERT distribution
classes. Each function takes the distribution parameters (minimum, mode, maximum, and optionally
lambda) and implements a specific statistical operation like pdf, cdf, etc.

All functions broadcast NumPy arrays: the distribution parameters and the evaluation points may be
scalars or arrays of compatible shapes.
"""

import sys

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats

# Avoid log(0) or log(1) which would cause -inf or 0
_CLIP_EPSILON = 1e-15
_BRENTQ_BOUND = 1e-10

# Data must strictly exceed this many observations for ``fit``
_FIT_MIN_OBSERVATIONS = 3

DEBUG = False

type FloatOrArray = float | np.floating | np.ndarray


def _ppf_fallback_log_space(
    q: FloatOrArray,
    mini: FloatOrArray,
    mode: FloatOrArray,
    maxi: FloatOrArray,
    lambd: FloatOrArray,
) -> FloatOrArray:
    """Use log-space to avoid numerical issues with extreme probabilities"""
    if DEBUG:
        sys.stderr.write(
            f"PLF: q={q!r}, mini={mini}, mode={mode}, maxi={maxi}, lambd={lambd}\n",
        )
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    if DEBUG:
        sys.stderr.write(f"PLF: alpha={alpha}, beta={beta}\n")

    # Broadcast probabilities and parameters to a common shape so that array-valued
    # parameters are handled element-wise alongside array-valued probabilities.
    _q, _alpha, _beta, _mini, _maxi = np.broadcast_arrays(
        np.atleast_1d(q),
        alpha,
        beta,
        mini,
        maxi,
    )
    results = np.zeros(_q.shape, dtype=float)

    # Define the equation to solve: log(CDF(x)) - log(q) = 0
    def make_log_cdf_eq(qi_val, alpha_i, beta_i):
        log_qi = np.log(np.clip(qi_val, _CLIP_EPSILON, 1 - _CLIP_EPSILON))

        def log_cdf_eq(x_normalized):
            # Ensure x_normalized stays in [0,1]
            x_clamped = np.clip(x_normalized, _CLIP_EPSILON, 1 - _CLIP_EPSILON)
            if DEBUG:
                sys.stderr.write(
                    f"PLF: x_clamped={x_clamped}, qi_val={qi_val}\n",
                )
            return scipy.stats.beta.logcdf(x_clamped, alpha_i, beta_i) - log_qi

        return log_cdf_eq

    for i in np.ndindex(_q.shape):
        qi = _q[i]
        try:
            # Use brentq instead of fsolve, guaranteed convergence within bounds
            x_normalized = scipy.optimize.brentq(
                make_log_cdf_eq(qi, _alpha[i], _beta[i]),
                _BRENTQ_BOUND,
                1 - _BRENTQ_BOUND,
            )
            results[i] = _mini[i] + (_maxi[i] - _mini[i]) * x_normalized

        except (ValueError, RuntimeError) as e:
            # ValueError: Invalid function values, convergence issues, or invalid bounds
            # RuntimeError: Maximum iterations exceeded, numerical problems
            if DEBUG:
                sys.stderr.write(
                    f"PLF: first try failed, i={i}, qi={qi}\n",
                )
                sys.stderr.write(f"{e}\n")
            # Fallback to clamped ppf if log-space fails
            qi_safe = np.clip(qi, _CLIP_EPSILON, 1 - _CLIP_EPSILON)
            x_normalized = scipy.stats.beta.ppf(qi_safe, _alpha[i], _beta[i])
            if DEBUG:
                sys.stderr.write(
                    f"PLF: second try: qi={qi}, qi_safe={qi_safe}, x_normalized={x_normalized}\n",
                )
            results[i] = _mini[i] + (_maxi[i] - _mini[i]) * x_normalized

    # Returns a scalar only when every input is scalar; a scalar q with array
    # parameters must still broadcast to an array result.
    all_scalar = np.broadcast(q, mini, mode, maxi, lambd).shape == ()
    return results[0] if all_scalar else results


_ppf_fallbacks = {
    "log": _ppf_fallback_log_space,
}


def _calc_alpha_beta(
    mini: FloatOrArray,
    mode: FloatOrArray,
    maxi: FloatOrArray,
    lambd: FloatOrArray,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Calculate alpha and beta parameters for the underlying beta distribution.

    Args:
        mini: Minimum value (must be <= mode and < maxi).
        mode: Most likely value (must be mini <= mode <= maxi).
        maxi: Maximum value (must be >= mode and > mini).
        lambd: Shape parameter (must be > 0, typically 2-6 for practical applications).

    Returns:
        tuple: Shape parameters alpha and beta for the beta distribution. Arrays broadcast
        element-wise when any parameter is an array.

    """
    alpha = 1 + ((mode - mini) * lambd) / (maxi - mini)
    beta = 1 + ((maxi - mode) * lambd) / (maxi - mini)
    return alpha, beta


def pdf(x, mini, mode, maxi, lambd=4):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return scipy.stats.beta.pdf((x - mini) / (maxi - mini), alpha, beta) / (maxi - mini)


def cdf(x, mini, mode, maxi, lambd=4):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return scipy.stats.beta.cdf((x - mini) / (maxi - mini), alpha, beta)


def sf(x, mini, mode, maxi, lambd=4):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return scipy.stats.beta.sf((x - mini) / (maxi - mini), alpha, beta)


def ppf(q, mini, mode, maxi, lambd=4, *, fallback=None):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    _beta_ppf = mini + (maxi - mini) * scipy.stats.beta.ppf(q, alpha, beta)
    # Use fallback if any values are NaN
    if fallback is not None and np.any(np.atleast_1d(np.isnan(_beta_ppf))):
        return _ppf_fallbacks[fallback](q, mini, mode, maxi, lambd)
    return _beta_ppf


def isf(q, mini, mode, maxi, lambd=4):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return mini + (maxi - mini) * scipy.stats.beta.isf(q, alpha, beta)


def rvs(mini, mode, maxi, lambd=4, size=None, random_state=None):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    if size is None:
        return mini + (maxi - mini) * scipy.stats.beta.rvs(alpha, beta, random_state=random_state)
    return mini + (maxi - mini) * scipy.stats.beta.rvs(
        alpha,
        beta,
        size=size,
        random_state=random_state,
    )


def mean(mini, mode, maxi, lambd=4):
    """Calculate the mean of the (modified) PERT distribution.

    This formula is equivalent to the traditional PERT mean formula
    (minimum + 4 * mode + maximum) / 6 when lambd=4.

    For the general case: μ = (mini + maxi + lambd * mode) / (2 + lambd)
    """
    return (maxi + mini + mode * lambd) / (2 + lambd)


def var(mini, mode, maxi, lambd=4):
    """Calculate the variance of the (modified) PERT distribution.

    Uses the beta distribution variance formula: αβ/[(α+β)²(α+β+1)]
    transformed to PERT parameters using: var_pert = var_beta * (maxi - mini)²
    """
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)

    # Beta distribution variance: αβ/[(α+β)²(α+β+1)]
    beta_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    # Transform to PERT scale
    return beta_var * (maxi - mini) ** 2


def skew(mini, mode, maxi, lambd=4):
    """Calculate the skewness of the (modified) PERT distribution.

    Uses the beta distribution skewness formula, which is invariant under the positive
    affine transformation to PERT parameters:
    skew = 2(β−α)√(α+β+1) / [(α+β+2)√(αβ)]
    """
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    numerator = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
    denominator = (alpha + beta + 2) * np.sqrt(alpha * beta)
    return numerator / denominator


def kurtosis(mini, mode, maxi, lambd=4):
    """Calculate the excess kurtosis of the (modified) PERT distribution.

    Uses the beta distribution kurtosis formula transformed to PERT parameters.
    Excess kurtosis = 6[(α-β)²(α+β+1) - αβ(α+β+2)] / [αβ(α+β+2)(α+β+3)]
    """
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)

    numerator = 6 * ((alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2))
    denominator = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

    return numerator / denominator


def stats(mini, mode, maxi, lambd=4):
    """Return the first four moments of the (modified) PERT distribution."""
    return (
        mean(mini, mode, maxi, lambd),
        var(mini, mode, maxi, lambd),
        skew(mini, mode, maxi, lambd),
        kurtosis(mini, mode, maxi, lambd),
    )


def entropy(mini, mode, maxi, lambd=4):
    """Calculate the differential entropy of the (modified) PERT distribution.

    The entropy of a linearly transformed random variable X = mini + (maxi - mini) * B is the
    entropy of B plus the log of the scale factor.
    """
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return scipy.stats.beta.entropy(alpha, beta) + np.log(maxi - mini)


def munp(n, mini, mode, maxi, lambd=4):
    """Calculate the nth non-central moment E[X^n] of the (modified) PERT distribution.

    Uses the binomial expansion of E[(mini + (maxi - mini) * B)^n] where B is beta-distributed,
    together with the raw beta moments E[B^k] = Π_{i=0}^{k-1} (α+i)/(α+β+i).
    """
    n = int(n)
    if n < 0:
        msg = f"Moment order must be non-negative, got {n}"
        raise ValueError(msg)
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    scale = maxi - mini

    total = 0.0
    beta_moment = 1.0  # E[B^0]
    for k in range(n + 1):
        if k > 0:
            beta_moment = beta_moment * (alpha + k - 1) / (alpha + beta + k - 1)
        total = (
            total + scipy.special.comb(n, k, exact=True) * mini ** (n - k) * scale**k * beta_moment
        )
    return total


def fit(data, lambd=4):
    """Estimate ``(mini, mode, maxi)`` from data by maximum likelihood.

    The shape parameter ``lambd`` is held fixed (default 4, the classic PERT). The support
    endpoints are constrained to lie strictly outside the observed data range, since the
    likelihood is zero (for ``lambd > 0``) when an observation falls on an endpoint.

    Args:
        data: 1-D array-like of observations.
        lambd: The fixed weight given to the mode (must be > 0).

    Returns:
        tuple[float, float, float]: The estimated ``(mini, mode, maxi)``.

    """
    data = np.asarray(data, dtype=float).ravel()
    if data.size < _FIT_MIN_OBSERVATIONS:
        msg = f"fit requires at least {_FIT_MIN_OBSERVATIONS} observations, got {data.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "fit requires all observations to be finite"
        raise ValueError(msg)
    if lambd <= 0:
        msg = f"lambd must be positive, got {lambd}"
        raise ValueError(msg)
    data_min, data_max = data.min(), data.max()
    span = data_max - data_min
    if span <= 0:
        msg = "fit requires non-constant data"
        raise ValueError(msg)

    def negative_log_likelihood(params):
        mini, mode, maxi = params
        if not (mini < data_min and maxi > data_max and mini <= mode <= maxi):
            return np.inf
        alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
        z = (data - mini) / (maxi - mini)
        log_likelihood = scipy.stats.beta.logpdf(z, alpha, beta) - np.log(maxi - mini)
        if not np.all(np.isfinite(log_likelihood)):
            return np.inf
        return -np.sum(log_likelihood)

    mini0 = data_min - 0.05 * span
    maxi0 = data_max + 0.05 * span
    # Method-of-moments starting value for the mode, from μ = (mini + maxi + λ·mode)/(2 + λ)
    mode0 = (data.mean() * (2 + lambd) - mini0 - maxi0) / lambd
    mode0 = float(np.clip(mode0, mini0 + 0.01 * span, maxi0 - 0.01 * span))

    result = scipy.optimize.minimize(
        negative_log_likelihood,
        x0=np.array([mini0, mode0, maxi0]),
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-8 * span, "fatol": 1e-10},
    )
    if not result.success:
        msg = f"fit did not converge: {result.message}"
        raise RuntimeError(msg)
    mini, mode, maxi = result.x
    return float(mini), float(mode), float(maxi)


def argcheck(mini, mode, maxi, lambd=4):
    """Check parameter validity element-wise.

    The mode may coincide with either endpoint (``mini <= mode <= maxi``), but the support
    must be non-degenerate (``mini < maxi``) and ``lambd`` must be positive.
    """
    return (mini <= mode) & (mode <= maxi) & (mini < maxi) & (lambd > 0)


def get_support(mini, mode, maxi, lambd=4):
    """SciPy requires this per the documentation:

    If either of the endpoints of the support do depend on the shape parameters, then i) the
    distribution must implement the _get_support method; ...
    """
    return mini, maxi
