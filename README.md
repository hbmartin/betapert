[![PyPI](https://img.shields.io/pypi/v/beta-pert-dist-scipy.svg)](https://pypi.org/project/beta-pert-dist-scipy/)
[![Pytest, ruff, and black](https://github.com/hbmartin/betapert/actions/workflows/pytest-poetry.yml/badge.svg)](https://github.com/hbmartin/betapert/actions/workflows/pytest-poetry.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/🐧️-black-000000.svg)](https://github.com/psf/black)

This package provides the [PERT](https://en.wikipedia.org/wiki/PERT_distribution) (also known as beta-PERT) distribution.

# Background
PERT stands for "[program evaluation and review technique](https://en.wikipedia.org/wiki/Program_evaluation_and_review_technique)", the original context for which the distribution was first proposed by [Clark (1962)](https://doi.org/10.1287/opre.10.3.405).

The PERT distribution is widely used in risk and uncertainty modelling to represent uncertain quantities where one is relying on subjective estimates. This is because the three parameters defining the distribution are intuitive to the estimator.


# This package
Both the PERT distribution and its generalization, the modified PERT distribution, are provided.

The distributions work exactly like SciPy continuous probability distributions. They are subclasses of `rv_continuous`.

# Installation
```shell
uv add beta-pert-dist-scipy
```

# Usage

```python
from betapert import pert, mpert

# Define the distribution:
dist = pert(10, 30, 90)
# Or, using keyword arguments:
dist = pert(mini=10, mode=30, maxi=90)

# Call standard SciPy methods:
dist.pdf(50)
dist.cdf(50)
dist.mean()
dist.rvs(size=10)

# Or, you can directly use the methods on this object:
pert.pdf(50, mini=10, mode=30, maxi=90)
pert.cdf(50, mini=10, mode=30, maxi=90)
pert.mean(mini=10, mode=30, maxi=90)
pert.rvs(mini=10, mode=30, maxi=90, size=10)

# The modified PERT distribution is also available.
# A PERT distribution corresponds to `lambd=4`.
# Note that you cannot call `mpert` without specifying `lambd`
# (`pert` and `mpert` must have different signatures since SciPy does
# not support optional shape parameters).
mdist = mpert(10, 30, 90, lambd=2)

# Values of `lambd<4` have the effect of flattening the density curve
#       6%                 >  1.5%
assert (1 - mdist.cdf(80)) > (1 - dist.cdf(80))
```

# Tests

A thorough test suite is included.

```
❯ pytest
=============== 250 passed in 3.52s ===============
tests/test_frozen.py 
tests/test_generalization.py
tests/test_mpert_parametrized.py
tests/test_special_cases.py 
```

# Authors

- [betapert (original)](https://github.com/tadamcz/betapert) by [Tom Adamczewski](https://github.com/tadamcz)
- beta-pert-dist-scipy Python 3.13 drop-in replacement by Harold Martin
