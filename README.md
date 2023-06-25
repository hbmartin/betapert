This package provides the [PERT](https://en.wikipedia.org/wiki/PERT_distribution) (also known as beta-PERT) distribution.

The distributions work exactly like SciPy continuous probability distributions; they are subclasses of `rv_continuous`.

Both the PERT distribution and its generalization, the modified PERT distribution, are provided.

A thorough test suite is included.

# Installation
```shell
# or `poetry add betapert`
pip install betapert
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
# Note that you cannot call `mpert` without specifying `lambd`. 
# Two distributions are needed due to a limitation in SciPy.
mdist = mpert(10, 30, 90, lambd=2)

# Values less than `lambd=4` have the effect of flattening the density curve
#      6%                  >  1.5%
assert (1 - mdist.cdf(80)) > (1 - dist.cdf(80))
```