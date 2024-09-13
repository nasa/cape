# Standard imports
import os

# Third-party
import testutils

# Local imports
import cape.cfdx.cntl
import cape.cfdx.databook as databook
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
    cntl = cape.cfdx.cntl.Cntl()
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
    ftarg = os.path.join("..", fpdf)
    # Generate the Figure
    rp.SubfigSweepCoeff(sfig, swp, [0], True)
    # Assert if image matches
    assert testutils.assert_png(fpdf, ftarg)
