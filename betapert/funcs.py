"""Functions implementing the PERT and modified PERT distributions.

This module contains the core mathematical functions used by the PERT and modified PERT distribution
classes. Each function takes the distribution parameters (minimum, mode, maximum, and optionally
lambda) and implementsa specific statistical operation like pdf, cdf, etc.
"""

import numpy as np
import scipy.stats


def _calc_alpha_beta(mini, mode, maxi, lambd):
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


def ppf(q, mini, mode, maxi, lambd=4):
    alpha, beta = _calc_alpha_beta(mini, mode, maxi, lambd)
    return mini + (maxi - mini) * scipy.stats.beta.ppf(q, alpha, beta)


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
