# Local
from cape.pyover.options.runctlopts import OverrunOpts, RunControlOpts


# Some templates
OPTS1 = {
    # Run sequence
    "PhaseSequence": [0, 1, 2, 3],
    "PhaseIters": [200, 400, 600, 800],
    "nIter": 200,
    # Inputs to command-line
    "overrun": {
        "aux": "dplace",
        "nthreads": 6,
    },
}


def test_rcopts1():
    # Initialize options
    opts = RunControlOpts(OPTS1)
    # Test extra command-line arguments
    assert isinstance(opts["overrun"], OverrunOpts)
    assert opts.get_overrun_kw() == {"nthreads": 6}

