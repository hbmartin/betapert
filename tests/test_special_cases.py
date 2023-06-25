import numpy as np
import pytest
import scipy.stats

from betapert import mpert, pert


class TestIsBeta:
    """
    This is a nice strong check, because the entire PDF has to match a specific Beta exactly.
    """

    def test_is_beta_3_3(self):
        """
        A PERT (0, 0.5, 1) distribution is a Beta(3, 3) distribution.
        """
        dist = pert(mini=0, mode=0.5, maxi=1)
        beta = scipy.stats.beta(3, 3)
        x = np.linspace(0, 1, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))

    def test_is_beta_2_2(self):
        """
        A modified PERT (0, 0.5, 1) with lambda=2 is a Beta(2, 2) distribution.
        """
        dist = mpert(mini=0, mode=0.5, maxi=1, lambd=2)
        beta = scipy.stats.beta(2, 2)
        x = np.linspace(0, 1, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))


class TestMatchesWolframMathematica:
    """
    See https://reference.wolfram.com/language/ref/PERTDistribution.html
    """

    def test_pdf(self):
        """
        PDF[PERTDistribution[{1,10}, 3, 2]][5]   =>    0.15207705617310344
        """
        dist = mpert(mini=1, mode=3, maxi=10, lambd=2)
        assert dist.pdf(5) == pytest.approx(0.15207705617310344)

    def test_cdf(self):
        """
        CDF[PERTDistribution[{10^-5, 10^-3}, 10^-4, 1]][5*10^-5]    =>     0.0588892278343665
        """
        dist = mpert(mini=1e-5, mode=1e-4, maxi=1e-3, lambd=1)
        assert dist.cdf(5e-5) == pytest.approx(0.0588892278343665)
