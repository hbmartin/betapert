"""Functions implementing the PERT and modified PERT distributions.

This module contains the core mathematical functions used by the PERT and modified PERT distribution
classes. Each function takes the distribution parameters (minimum, mode, maximum, and optionally
lambda) and implements a specific statistical operation like pdf, cdf, etc.
"""

import sys

import numpy as np
import scipy.optimize
import scipy.stats

# Avoid log(0) or log(1) which would cause -inf or 0
_CLIP_EPSILON = 1e-15
_BRENTQ_BOUND = 1e-10

DEBUG = False


def _ppf_fallback_log_space(q, mini, mode, maxi, lambd):
    """Use log-space to avoid numerical issues with extreme probabilities"""
    if DEBUG:
        sys.stderr.write(
            f"PLF: len(q)={len(q)}, mini={mini}, mode={mode}, maxi={maxi}, lambd={lambd}\n",
        )
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    if DEBUG:
        sys.stderr.write(f"PLF: alpha={alpha}, beta={beta}\n")

    # Handle scalar and array inputs consistently
    _q = np.atleast_1d(q)
    results = np.zeros_like(_q, dtype=float)

    # Define the equation to solve: log(CDF(x)) - log(q) = 0
    def make_log_cdf_eq(qi_val):
        log_qi = np.log(np.clip(qi_val, _CLIP_EPSILON, 1 - _CLIP_EPSILON))

        def log_cdf_eq(x_normalized):
            # Ensure x_normalized stays in [0,1]
            x_clamped = np.clip(x_normalized, _CLIP_EPSILON, 1 - _CLIP_EPSILON)
            if DEBUG:
                sys.stderr.write(
                    f"PLF: x_clamped={x_clamped}, qi_val={qi_val}\n",
                )
            return scipy.stats.beta.logcdf(x_clamped, alpha, beta) - log_qi

        return log_cdf_eq

    for i, qi in np.ndenumerate(_q):
        try:
            # Use brentq instead of fsolve, guaranteed convergence within bounds
            x_normalized = scipy.optimize.brentq(
                make_log_cdf_eq(qi),
                _BRENTQ_BOUND,
                1 - _BRENTQ_BOUND,
            )
            results[i] = mini + (maxi - mini) * x_normalized

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
            x_normalized = scipy.stats.beta.ppf(qi_safe, alpha, beta)
            if DEBUG:
                sys.stderr.write(
                    f"PLF: second try: qi={qi}, qi_safe={qi_safe}, x_normalized={x_normalized}\n",
                )
            results[i] = mini + (maxi - mini) * x_normalized

    # Returns scalar for scalar input, array for array input
    return results[0] if np.isscalar(q) else results


_ppf_fallbacks = {
    "log": _ppf_fallback_log_space,
}


def _scalar_if_array_all_equal(array: np.ndarray | float) -> np.float64 | np.ndarray | float:
    if isinstance(array, np.ndarray) and array.size != 0 and np.all(array == array[0]):
        return array[0]
    return array


def _calc_alpha_beta(
    mini: np.float64 | np.ndarray | float,
    mode: np.float64 | np.ndarray | float,
    maxi: np.float64 | np.ndarray | float,
    lambd: np.float64 | np.ndarray | float,
) -> tuple[np.float64 | np.ndarray | float, np.float64 | np.ndarray | float]:
    """Calculate alpha and beta parameters for the underlying beta distribution.

    Args:
        mini: Minimum value (must be < mode).
        mode: Most likely value (must be mini < mode < maxi).
        maxi: Maximum value (must be > mode).
        lambd: Shape parameter (must be > 0, typically 2-6 for practical applications).

    Returns:
        tuple[float, float]: Shape parameters alpha and beta for the beta distribution.

    """
    alpha = 1 + ((mode - mini) * lambd) / (maxi - mini)
    beta = 1 + ((maxi - mode) * lambd) / (maxi - mini)
    # If alpha and beta are arrays and all elements are equal, return the scalar value
    if DEBUG and any(isinstance(x, np.ndarray) for x in (mini, mode, maxi, lambd)):
        sys.stderr.write("CAB: unexpected arrays in method parameters\n")
    if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
        if DEBUG:
            sys.stderr.write(f"CAB: Unexpected arrays: alpha={alpha}, beta={beta}\n")
        return _scalar_if_array_all_equal(alpha), _scalar_if_array_all_equal(beta)
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
    mini = _scalar_if_array_all_equal(mini)
    mode = _scalar_if_array_all_equal(mode)
    maxi = _scalar_if_array_all_equal(maxi)
    lambd = _scalar_if_array_all_equal(lambd)

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
    numerator = 2 * (-2 * mode + maxi + mini) * lambd * np.sqrt(3 + lambd)
    denominator_left = 4 + lambd
    denominator_middle = np.sqrt(maxi - mini - mode * lambd + maxi * lambd)
    denominator_right = np.sqrt(maxi + mode * lambd - mini * (1 + lambd))
    denominator = denominator_left * denominator_middle * denominator_right
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


def argcheck(mini, mode, maxi, lambd=4):
    return mini < mode < maxi and lambd > 0


def get_support(mini, mode, maxi, lambd=4):
    """SciPy requires this per the documentation:

    If either of the endpoints of the support do depend on the shape parameters, then i) the
    distribution must implement the _get_support method; ...
    """
    return mini, maxi
