import numpy as np
import pytest
import scipy.stats

from betapert import betapert4, betapert3


def test_is_symmetric_beta():
    dist = betapert3(mini=0, mode=0.5, maxi=1)
    beta = scipy.stats.beta(3, 3)
    evaluate_at = np.linspace(0, 1, 100)
    assert dist.pdf(evaluate_at) == pytest.approx(beta.pdf(evaluate_at))


def test_wolfram_mathematica_pdf():
    """
    See https://reference.wolfram.com/language/ref/PERTDistribution.html

    PDF[PERTDistribution[{1,10}, 3, 2]][5]   =>    0.15207705617310344
    """

    dist = betapert4(mini=1, mode=3, maxi=10, lambd=2)
    assert dist.pdf(5) == pytest.approx(0.15207705617310344)


def test_wolfram_mathematica_cdf():
    """
    See https://reference.wolfram.com/language/ref/PERTDistribution.html

    CDF[PERTDistribution[{10^-5, 10^-3}, 10^-4, 1]][5*10^-5]    =>     0.0588892278343665
    """

    dist = betapert4(mini=1e-5, mode=1e-4, maxi=1e-3, lambd=1)
    assert dist.cdf(5e-5) == pytest.approx(0.0588892278343665)
