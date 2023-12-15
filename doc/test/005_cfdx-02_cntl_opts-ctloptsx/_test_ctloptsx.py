
# Third party
import testutils

# Local imports
from cape.cntl import Cntl


# Test using *x* in a RunControl option
@testutils.run_testdir(__file__)
def test_ctloptsx01():
    # Read case
    cntl = Cntl()
    # Test options at different conditions
    assert cntl.opts.get_nSeq(i=12) == 1
    assert cntl.opts.get_PhaseSequence(i=0) == [0, 1]

