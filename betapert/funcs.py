import numpy as np
import scipy.stats


def _calc_alpha_beta(mini, mode, maxi, lambd):
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
        alpha, beta, size=size, random_state=random_state
    )


def mean(mini, mode, maxi, lambd=4):
    return (maxi + mini + mode * lambd) / (2 + lambd)


def var(mini, mode, maxi, lambd=4):
    numerator_left = maxi - mini - mode * lambd + maxi * lambd
    numerator_right = maxi + mode * lambd - mini * (1 + lambd)
    numerator = numerator_left * numerator_right
    denominator = (2 + lambd) ** 2 * (3 + lambd)
    return numerator / denominator


def skew(mini, mode, maxi, lambd=4):
    numerator = 2 * (-2 * mode + maxi + mini) * lambd * np.sqrt(3 + lambd)
    denominator_left = 4 + lambd
    denominator_middle = np.sqrt((maxi - mini - mode * lambd + maxi * lambd))
    denominator_right = np.sqrt((maxi + mode * lambd - mini * (1 + lambd)))
    denominator = denominator_left * denominator_middle * denominator_right
    return numerator / denominator


def stats(mini, mode, maxi, lambd=4):
    # Kurtosis will be calculated (inefficiently) by SciPy's generic methods
    kurt = None
    return (
        mean(mini, mode, maxi, lambd),
        var(mini, mode, maxi, lambd),
        skew(mini, mode, maxi, lambd),
        kurt,
    )


def argcheck(mini, mode, maxi, lambd=4):
    return mini < mode < maxi and lambd > 0


def get_support(mini, mode, maxi, lambd=4):
    """
    SciPy requires this per the documentation:

        If either of the endpoints of the support do depend on the shape parameters, then i) the distribution
        must implement the _get_support method; ...
    """
    return mini, maxi
