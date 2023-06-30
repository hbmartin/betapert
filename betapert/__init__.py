"""
Arbitrary parameters in SciPy's ``rv_continuous`` class must be 'shape' parameters.
Optional shape parameters are not supported, and are seemingly impossible to implement without egregious hacks.
So there are two classes, one for the PERT distribution (with ``lambd=4``) and one for the modified PERT distribution
(with ``lambd`` as a shape parameter). Beyond being repetitious, this also adversely affects the user-facing API.
"""
import scipy.stats

from betapert import funcs


class PERT(scipy.stats.rv_continuous):
    """
    The `PERT distribution <https://en.wikipedia.org/wiki/PERT_distribution>`_ is defined by the
    minimum, most likely, and maximum values that a variable can take. It is commonly used to
    elicit subjective beliefs. PERT is an alternative to the triangular distribution, but has a
    smoother shape.

    :param mini: The left bound of the distribution.
    :param mode: The mode of the distribution.
    :param maxi: The right bound of the distribution.


    Examples
    --------

    >>> from betapert import pert
    >>> dist = pert(0, 3, 12)
    >>> dist.mean()
    4.0
    >>> dist.cdf(5)
    0.691229423868313

    Equivalent to:

    >>> from betapert import pert
    >>> pert.cdf(5, 0, 3, 12)
    0.691229423868313
    """

    def _get_support(self, mini, mode, maxi):
        return funcs.get_support(mini, mode, maxi)

    def _argcheck(self, mini, mode, maxi):
        return funcs.argcheck(mini, mode, maxi)

    def _pdf(self, x, mini, mode, maxi):
        return funcs.pdf(x, mini, mode, maxi)

    def _cdf(self, x, mini, mode, maxi):
        return funcs.cdf(x, mini, mode, maxi)

    def _sf(self, x, mini, mode, maxi):
        return funcs.sf(x, mini, mode, maxi)

    def _isf(self, x, mini, mode, maxi):
        return funcs.isf(x, mini, mode, maxi)

    def _stats(self, mini, mode, maxi):
        return funcs.stats(mini, mode, maxi)

    def _ppf(self, q, mini, mode, maxi):
        return funcs.ppf(q, mini, mode, maxi)

    def _rvs(self, mini, mode, maxi, size=None, random_state=None):
        return funcs.rvs(mini, mode, maxi, size=size, random_state=random_state)


class ModifiedPERT(scipy.stats.rv_continuous):
    """
    The modified PERT distribution generalizes the PERT distribution by adding a fourth parameter
    ``lambd`` that controls how much weight is given to the mode. ``lambd=4`` corresponds to the
    traditional PERT distribution.

    :param mini: The left bound of the distribution.
    :param mode: The mode of the distribution.
    :param maxi: The right bound of the distribution.
    :param lambd:
        The weight given to the mode. Relative to the PERT, values ``lambd < 4`` have the effect of flattening
        the density curve.


    Examples
    --------

    >>> from betapert import mpert
    >>> mdist = mpert(0, 3, 12, lambd=2)
    >>> mdist.mean()
    4.5

    Values of ``lambd<4`` have the effect of flattening the density curve

    >>> dist = mpert(0, 3, 12, lambd=4)
    >>> 1 - mdist.cdf(8), 1 - dist.cdf(8)
    (0.11395114580927845, 0.04526748971193417)
    """

    def _get_support(self, mini, mode, maxi, lambd):
        return funcs.get_support(mini, mode, maxi, lambd)

    def _argcheck(self, mini, mode, maxi, lambd):
        return funcs.argcheck(mini, mode, maxi, lambd)

    def _pdf(self, x, mini, mode, maxi, lambd):
        return funcs.pdf(x, mini, mode, maxi, lambd)

    def _cdf(self, x, mini, mode, maxi, lambd):
        return funcs.cdf(x, mini, mode, maxi, lambd)

    def _sf(self, x, mini, mode, maxi, lambd):
        return funcs.sf(x, mini, mode, maxi, lambd)

    def _isf(self, x, mini, mode, maxi, lambd):
        return funcs.isf(x, mini, mode, maxi, lambd)

    def _stats(self, mini, mode, maxi, lambd):
        return funcs.stats(mini, mode, maxi, lambd)

    def _ppf(self, q, mini, mode, maxi, lambd):
        return funcs.ppf(q, mini, mode, maxi, lambd)

    def _rvs(self, mini, mode, maxi, lambd, size=None, random_state=None):
        return funcs.rvs(mini, mode, maxi, lambd, size=size, random_state=random_state)


# ``pert`` and ``mpert`` being instances, not classes, is not IMO idiomatic Python, but it is core to the way SciPy's
# ``rv_continuous`` class works. See examples of how SciPy defines their distributions in
# ``scipy/stats/_continuous_distns.py``.
pert = PERT()
mpert = ModifiedPERT()
