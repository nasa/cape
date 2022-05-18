
# Third-party
import testutils

# Local imports
from cape.tnakit import rstutils
from cape.tnakit.plot_mpl import MPLOpts


# File names
RSTFILE = "pmplopts01.rst"


# Test some options
@testutils.run_testdir(__file__)
def test_01_pmplopts():
    # Try *PlotLineStyle*
    opts = MPLOpts(PlotLineStyle="--", Index=2)
    # Get "PlotOptions" for case 3
    plot_opts = opts.get_option("PlotOptions")
    # Show options for plot()
    rst = rstutils.py2rst(plot_opts, indent=4, sort=True)
    # Compare to target
    assert testutils.compare_files(rst, RSTFILE)

