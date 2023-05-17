# Local
from cape.pyover.options import gridsysopts


# Raw test options
MYOPTS = {
    "ALL": {
        "TIMACU": {
            "CFLMIN": [2.5, 5.0, 2.0],
            "CFLMAX": [10.0, 20.0, 50.0],
        },
    },
}

# Sampled to phase 0
ALL1 = {
    "TIMACU": {
        "CFLMIN": 5.0,
        "CFLMAX": 20.0,
    },
}


def test_meshopts01():
    # Initialize options
    opts = gridsysopts.GridSystemNmlOpts(MYOPTS)
    # Sample entire namelist to phase 0
    all1 = opts.get_ALL(j=1)
    # Test it
    assert all1 == ALL1

