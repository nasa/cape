
# Standard library


# Third-party
import testutils

# Local imports
from cape.pyfun import case


# Test iteration counters
@testutils.run_testdir(__file__)
def test_getiter01():
    # Get a runner instance
    runner = case.CaseRunner()
    # Test iterations
    assert runner.get_iter() == 100
    assert runner.get_restart_iter() == 100

