# Local
from cape.cfdx.options.ulimitopts import ULimitOpts


# Some templates
OPTS1 = {
    "r": 0,
    "stack_size": 4200000,
}


def test_rcopts1():
    # Initialize options
    opts = ULimitOpts(OPTS1)
    # Get values
    assert opts.get_ulimit("s") == OPTS1["stack_size"]
    # Set values
    opts.set_ulimit("x")
    opts.set_ulimit("s", 420000)

