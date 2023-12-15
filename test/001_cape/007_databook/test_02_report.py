# Standard imports
import os

# Third-party
import testutils

# Local imports
import cape.cntl
import cape.cfdx.dataBook as databook
import cape.cfdx.report as report

TEST_FILES = (
    "matrix.csv",
    "cape.json",
    "arrow.xml",
    "data/*"
)


# Test Report Sweeps
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_sweep():
    # Read settings
    cntl = cape.cntl.Cntl()
    # Read data book
    db = databook.DataBook(cntl)
    # Read reports
    rp = report.Report(cntl, 'mach')
    # Add databook to cntl
    cntl.DataBook = db
    # Define subfig
    sfig = "mach_arrow_CA"
    # Define sweep
    swp = "mach"
    # Figure name
    fpdf = '%s.png' % sfig
    # File names
    fabs = os.path.abspath(fpdf)
    fdir = os.path.dirname(os.getcwd())
    ftarg = os.path.join(fdir, fpdf)
    # Generate the Figure
    rp.SubfigSweepCoeff(sfig, swp, [0], True)
    # Assert if image matches
    assert testutils.assert_png(fabs, ftarg, tol=0.93)
