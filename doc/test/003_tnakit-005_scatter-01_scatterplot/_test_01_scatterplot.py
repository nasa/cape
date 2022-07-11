
# Standard library modules
import os.path as op
import sys

# Third-party modules
import numpy as np
import testutils

# Import CAPE modules
import cape.attdb.dbfm as dbfm
import cape.tnakit.plot_mpl as pmpl


# File names
FDIR = op.dirname(__file__)
MATFILE = "bullet-fm.mat"


# Test scatter plot
@testutils.run_sandbox(__file__, MATFILE)
def test_01_scatterplot():
    # Read data file
    db = dbfm.DBFM(MATFILE)
    # Creat random CA offset
    np.random.seed(100)
    dCA = 0.1 * np.random.rand(len(db["bullet.CA"]))
    # Set colormap keyword
    kw = {
        "ScatterColor": db["alpha"],
        "XLabel": "beta",
        "YLabel": "delCA",
    }
    # Plot contour
    h = pmpl.scatter(db["beta"], db["bullet.CA"] - dCA, **kw)
    # Image file name
    fimg = "python%i-bullet-deltaCA-scatter.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Compare images
    testutils.assert_png(fimg, op.join(FDIR, fimg))


if __name__ == "__main__":
    test_01_scatterplot()
