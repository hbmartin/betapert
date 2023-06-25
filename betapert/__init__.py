import numpy as np
import scipy.stats


class BetaPertHelper:
    @staticmethod
    def get_support(mini, mode, maxi, lambd):
        """
        SciPy requires this per the documentation:

            If either of the endpoints of the support do depend on the shape parameters, then i) the distribution
            must implement the _get_support method; ...
        """
        return mini, maxi

    @staticmethod
    def argcheck(mini, mode, maxi, lambd):
        return mini < mode < maxi and lambd > 0

    @staticmethod
    def calc_alpha_beta(mini, mode, maxi, lambd):
        alpha = 1 + ((mode - mini) * lambd) / (maxi - mini)
        beta = 1 + ((maxi - mode) * lambd) / (maxi - mini)
        return alpha, beta

    @staticmethod
    def pdf(x, mini, mode, maxi, lambd):
        alpha, beta = BetaPertHelper.calc_alpha_beta(mini, mode, maxi, lambd)
        return scipy.stats.beta.pdf((x - mini) / (maxi - mini), alpha, beta) / (maxi - mini)

    @staticmethod
    def cdf(x, mini, mode, maxi, lambd):
        alpha, beta = BetaPertHelper.calc_alpha_beta(mini, mode, maxi, lambd)
        return scipy.stats.beta.cdf((x - mini) / (maxi - mini), alpha, beta)

    @staticmethod
    def mean(mini, mode, maxi, lambd):
        return (maxi + mini + mode * lambd) / (2 + lambd)

    @staticmethod
    def var(mini, mode, maxi, lambd):
        numerator_left = maxi - mini - mode * lambd + maxi * lambd
        numerator_right = maxi + mode * lambd - mini * (1 + lambd)
        numerator = numerator_left * numerator_right
        denominator = (2 + lambd) ** 2 * (3 + lambd)
        return numerator / denominator

    @staticmethod
    def skew(mini, mode, maxi, lambd):
        numerator = 2 * (-2 * mode + maxi + mini) * lambd * np.sqrt(3 + lambd)
        denominator_left = 4 + lambd
        denominator_middle = np.sqrt((maxi - mini - mode * lambd + maxi * lambd))
        denominator_right = np.sqrt((maxi + mode * lambd - mini * (1 + lambd)))
        denominator = denominator_left * denominator_middle * denominator_right
        return numerator / denominator

    @staticmethod
    def median(mini, mode, maxi, lambd):
        alpha, beta = BetaPertHelper.calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.ppf(0.5, alpha, beta)

    @staticmethod
    def ppf(q, mini, mode, maxi, lambd):
        alpha, beta = BetaPertHelper.calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.ppf(q, alpha, beta)

    @staticmethod
    def rvs(mini, mode, maxi, lambd, size=None, random_state=None):
        alpha, beta = BetaPertHelper.calc_alpha_beta(mini, mode, maxi, lambd)
        return mini + (maxi - mini) * scipy.stats.beta.rvs(
            alpha, beta, size=size, random_state=random_state
        )


class betapert3_gen(scipy.stats.rv_continuous):
    def _get_support(self, mini, mode, maxi):
        return BetaPertHelper.get_support(mini, mode, maxi, lambd=4)

    def _argcheck(self, mini, mode, maxi):
        return BetaPertHelper.argcheck(mini, mode, maxi, lambd=4)

    def _pdf(self, x, mini, mode, maxi):
        return BetaPertHelper.pdf(x, mini, mode, maxi, lambd=4)

    def _cdf(self, x, mini, mode, maxi):
        return BetaPertHelper.cdf(x, mini, mode, maxi, lambd=4)

    def _stats(self, mini, mode, maxi):
        mean = BetaPertHelper.mean(mini, mode, maxi, lambd=4)
        var = BetaPertHelper.var(mini, mode, maxi, lambd=4)
        skew = BetaPertHelper.skew(mini, mode, maxi, lambd=4)
        kurt = None
        return mean, var, skew, kurt

    def _ppf(self, q, mini, mode, maxi):
        return BetaPertHelper.ppf(q, mini, mode, maxi, lambd=4)

    def _rvs(self, mini, mode, maxi, size=None, random_state=None):
        return BetaPertHelper.rvs(mini, mode, maxi, lambd=4, size=size, random_state=random_state)


class betapert4_gen(scipy.stats.rv_continuous):
    def _get_support(self, mini, mode, maxi, lambd):
        return BetaPertHelper.get_support(mini, mode, maxi, lambd)

    def _argcheck(self, mini, mode, maxi, lambd):
        return BetaPertHelper.argcheck(mini, mode, maxi, lambd)

    def _pdf(self, x, mini, mode, maxi, lambd):
        return BetaPertHelper.pdf(x, mini, mode, maxi, lambd)

    def _cdf(self, x, mini, mode, maxi, lambd):
        return BetaPertHelper.cdf(x, mini, mode, maxi, lambd)

    def _stats(self, mini, mode, maxi, lambd):
        mean = BetaPertHelper.mean(mini, mode, maxi, lambd)
        var = BetaPertHelper.var(mini, mode, maxi, lambd)
        skew = BetaPertHelper.skew(mini, mode, maxi, lambd)
        kurt = None
        return mean, var, skew, kurt

    def _ppf(self, q, mini, mode, maxi, lambd):
        return BetaPertHelper.ppf(q, mini, mode, maxi, lambd)

    def _rvs(self, mini, mode, maxi, lambd, size=None, random_state=None):
        return BetaPertHelper.rvs(mini, mode, maxi, lambd, size=size, random_state=random_state)


betapert3 = betapert3_gen()
betapert4 = betapert4_gen()
