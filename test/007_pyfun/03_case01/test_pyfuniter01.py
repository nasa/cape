
# Standard library


# Third-party
import testutils

# Local imports
from cape.pyfun import casecntl


# Test iteration counters
@testutils.run_testdir(__file__)
def test_getiter01():
    # Get a runner instance
    runner = casecntl.CaseRunner()
    # Test iterations
    assert runner.get_iter() == 100
    assert runner.get_restart_iter() == 100
    # Test phase
    assert runner.get_phase() == 1

