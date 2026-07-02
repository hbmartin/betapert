"""
Tests for degenerate modes, where the mode coincides with one of the support endpoints.
These come up frequently in risk modelling ("the best case is that nothing goes wrong").
"""

import numpy as np
import pytest
import scipy.stats

from betapert import funcs, mpert, pert


class TestModeAtMinimum:
    def test_is_beta_1_5(self):
        """A PERT (0, 0, 1) distribution is a Beta(1, 5) distribution."""
        dist = pert(mini=0, mode=0, maxi=1)
        beta = scipy.stats.beta(1, 5)
        x = np.linspace(0, 1, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))
        assert dist.cdf(x) == pytest.approx(beta.cdf(x))

    def test_scaled_and_shifted(self):
        """A PERT (2, 2, 10) is a Beta(1, 5) scaled to [2, 10]."""
        dist = pert(mini=2, mode=2, maxi=10)
        beta = scipy.stats.beta(1, 5, loc=2, scale=8)
        x = np.linspace(2, 10, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))
        assert dist.mean() == pytest.approx(beta.mean())
        assert dist.var() == pytest.approx(beta.var())

    def test_rvs_within_support(self):
        dist = pert(mini=2, mode=2, maxi=10)
        rvs = dist.rvs(size=1000, random_state=42)
        assert np.all((rvs >= 2) & (rvs <= 10))


class TestModeAtMaximum:
    def test_is_beta_5_1(self):
        """A PERT (0, 1, 1) distribution is a Beta(5, 1) distribution."""
        dist = pert(mini=0, mode=1, maxi=1)
        beta = scipy.stats.beta(5, 1)
        x = np.linspace(0, 1, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))
        assert dist.cdf(x) == pytest.approx(beta.cdf(x))

    def test_mpert_lambd_2(self):
        """A modified PERT (0, 1, 1) with lambd=2 is a Beta(3, 1) distribution."""
        dist = mpert(mini=0, mode=1, maxi=1, lambd=2)
        beta = scipy.stats.beta(3, 1)
        x = np.linspace(0, 1, 100)
        assert dist.pdf(x) == pytest.approx(beta.pdf(x))


class TestInvalidParameters:
    @pytest.mark.parametrize(
        ("mini", "mode", "maxi", "lambd"),
        [
            (0, -1, 1, 4),  # mode below mini
            (0, 2, 1, 4),  # mode above maxi
            (1, 1, 1, 4),  # degenerate support
            (2, 1.5, 1, 4),  # inverted support
            (0, 0.5, 1, 0),  # lambd not positive
            (0, 0.5, 1, -1),  # negative lambd
        ],
    )
    def test_argcheck_rejects(self, mini, mode, maxi, lambd):
        assert not funcs.argcheck(mini, mode, maxi, lambd)

    def test_pdf_nan_for_invalid_params(self):
        assert np.isnan(pert.pdf(0.5, 0, 2, 1))

    def test_argcheck_elementwise(self):
        mini = np.array([0.0, 0.0, 1.0])
        mode = np.array([0.0, -1.0, 1.0])
        maxi = np.array([1.0, 1.0, 1.0])
        result = funcs.argcheck(mini, mode, maxi, 4)
        assert result.tolist() == [True, False, False]
