# Local
from cape.pyfun.options.runctlopts import RunControlOpts


# Some templates
OPTS1 = {
    # Run sequence
    "PhaseSequence": [0, 1, 2, 3],
    "PhaseIters": [200, 400, 600, 800],
    "nIter": 200,
}
OPTS2 = dict(
    OPTS1,
    Adaptive=True,
    AdaptPhase=[False, True, True, False]
)


def test_rcopts1():
    # Initialize options
    opts = RunControlOpts(OPTS1)
    # No adaptation
    assert opts.get_AdaptationNumber(j=1) is None
    # New options
    opts = RunControlOpts(OPTS2)
    # Test null output for null phase
    assert opts.get_AdaptationNumber() is None
    # Test other entries
    assert opts.get_AdaptationNumber(1) == 0
    assert opts.get_AdaptationNumber(3) == 2

