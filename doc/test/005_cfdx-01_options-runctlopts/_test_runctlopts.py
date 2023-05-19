# Local
from cape.cfdx.options.runctlopts import RunControlOpts


# Some templates
OPTS1 = {
    # Run sequence
    "PhaseSequence": [0, 1],
    "PhaseIters": [200, 400],
    # Operation modes
    "nProc": 4,
    "MPI": False,
    # Environment variables
    "Environ": {
        "F_UFMTENDIAN": "big",
        "MPI_VERBOSE": 0,
    },
}


def test_rcopts1():
    # Initialize options
    opts = RunControlOpts(OPTS1)
    # Set env variable
    opts.set_Environ("MPI_VERBOSE", "1")
    # Check environment variables
    assert opts.get_Environ("F_UFMTENDIAN") == "big"
    assert opts.get_Environ("MPI_VERBOSE") == "1"
    # Execution sections
    assert opts.get_aflr3() is False
    assert opts.get_intersect() is False
    assert opts.get_verify() is False
    # Test last iteration
    assert opts.get_LastIter() == 400


