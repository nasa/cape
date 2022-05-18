
# Standard library modules
import os.path as op
import sys

# Third-party modules
import testutils

# Local imports
import cape.attdb.dbfm as dbfm
import cape.tnakit.plot_mpl as pmpl


# File names
FDIR = op.dirname(__file__)
MATFILE = "bullet-fm.mat"


# Create a contour plot
@testutils.run_sandbox(__file__, MATFILE)
def test_01_contour():
    # Read data file
    db = dbfm.DBFM("bullet-fm.mat")
    # Get cases for plotting
    mask, _ = db.find(["mach"], 0.8)
    # Set colormap keyword
    kw = {
        "ContourColorMap": "RdBu",
        "XLabel": "beta",
        "YLabel": "alpha",
    }
    # Plot contour
    h = pmpl.contour(
        db["beta"][mask], db["alpha"][mask], db["bullet.CLM"][mask], **kw)
    # Image file name
    fimg = "python%i-bullet-CLM-contour.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Compare it
    testutils.assert_png(fimg, op.join(FDIR, fimg), tol=0.93)

