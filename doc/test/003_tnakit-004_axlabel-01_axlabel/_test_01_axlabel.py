
# Standard library modules
import os.path as op
import sys

# Third-party modules
import numpy as np
import testutils

# Local imports
import cape.tnakit.plot_mpl as pmpl


# File names
FDIR = op.dirname(__file__)


# Test various axis labels
@testutils.run_sandbox(__file__)
def test_01_axlabel():
    # Circle for sample data
    th = np.linspace(0, 2*np.pi, 61)
    x = np.cos(th)
    y = np.sin(th)
    # Create plot
    h = pmpl.plot(x, y, TopSpine=True, RightSpine=True)
    # Start making labels
    pmpl.axlabel("Position 8", pos=8, AxesLabelColor="purple")
    pmpl.axlabel("Position 15", pos=15, AxesLabelOptions=dict(rotation=-90))
    # Make labels using automatic positions
    pmpl.axlabel(u"μ = 0.0000", AxesLabelColor="red")
    pmpl.axlabel(u"r = 1.0000", AxesLabelFontOptions={"weight": "bold"})
    pmpl.axlabel("Third auto position")
    # Redo margins
    pmpl.axes_adjust(h.fig)
    # Image file name
    fimg = "python%i-axlabel.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Compare
    testutils.assert_png(fimg, op.join(FDIR, fimg))

