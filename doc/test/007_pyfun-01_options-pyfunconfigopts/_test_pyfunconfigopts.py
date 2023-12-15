# Local
from cape.pyfun.options.configopts import ConfigOpts


# Sample options
OPTS1 = dict(
    Inputs={
        "comp1": 1,
        "comp2": [2, 3],
    }
)


def test_slices01():
    # Initialize options
    opts = ConfigOpts(OPTS1)
    # Test Input
    assert opts.get_ConfigInput("comp1") == 1
    # Set one
    opts.set_ConfigInput("comp3", 4)
    assert opts.get_ConfigInput("comp3") == 4

