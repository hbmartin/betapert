import pytest


@pytest.fixture(
    params=[
        # A simple case
        (0, 1, 3),
        # Negative and decimal values
        (-1, 0.1, 5),
        # The mode is very close to the minimum
        (0, 1 / 300, 1),
        # The mode is very close to the maximum
        (0, 299 / 300, 1),
        # Large values
        (1e6, 1e6 + 1, 1e6 + 2),
        # Small values
        (0, 1e-6, 2e-6),
    ],
    ids=lambda x: f"mini={x[0]}, mode={x[1]}, maxi={x[2]}",
)
def params(request):
    """Provide required parameters (mini, mode, maxi)"""
    return request.param


@pytest.fixture(params=[1, 4, 8], ids=lambda x: f"lambd={x}")
def lambd(request):
    return request.param


@pytest.fixture(
    params=[
        "pdf",
        "logpdf",
        "cdf",
        "logcdf",
        "sf",
        "logsf",
        "ppf",
        "isf",
        "moment",
        "stats",
        "entropy",
        "expect",
        "median",
        "mean",
        "std",
        "var",
        "interval",
        "support",
    ]
)
def method(request):
    """The method to test"""
    return request.param
