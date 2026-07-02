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
