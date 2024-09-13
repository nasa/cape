
# Third-party
import testutils

# Local imports
from cape import pltfile as pltfile


# Name of starting PLT file
PLTFILE = "arrow_plane-y0_timestep200.plt"


# Read the file, write it back, etc.
@testutils.run_sandbox(__file__, PLTFILE)
def test_pltdat01():
    # Read the file
    surf1 = pltfile.Plt(PLTFILE)
    # Write to DAT (ASCII) file and PLT file
    surf1.WriteDat("test2.dat")
    # Read that back
    surf2 = pltfile.Plt(dat="test2.dat")
    # Write DAT again
    surf2.WriteDat("test3.dat")
    # Read again
    surf3 = pltfile.Plt(dat="test3.dat")
    # Test values
    assert surf3.StrandID[0] == surf2.StrandID[0]
    assert surf3.StrandID[0] == surf1.StrandID[0]

