
# Standard library
import os

# Third-party imports
import testutils

# Local imports
from cape.pyfun.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "refine.json",
    "fun3d.nml",
    "matrix.csv",
    "mach-y0.lay",
    "cigar001-surf01.*",
)


# File names
I1 = 21
I2 = 22


# Test AFLR3 execution
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_plain():
    # Instantiate
    cntl = Cntl()
    # Run a case
    cntl.SubmitJobs(I=I1)
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(I1)
    # Check case exists
    assert os.path.isdir(case_folder)
    # Generate a report
    os.chdir(cntl.RootDir)
    reportname = cntl.opts.get_ReportList()[0]
    report = cntl.ReadReport(reportname)
    report.UpdateReport(I=I1)
    # Report files
    report_png = os.path.join("report", case_folder, "mach-y0.png")
    assert os.path.isfile(report_png)


# Test AFLR3 execution
@testutils.run_sandbox(__file__, fresh=False)
def test_02_refine():
    # Instantiate
    cntl = Cntl("refine.json")
    # Run a case
    cntl.SubmitJobs(I=I2)
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(I2)
    # Check case exists
    assert os.path.isdir(case_folder)
    # Generate a report
    reportname = cntl.opts.get_ReportList()[0]
    report = cntl.ReadReport(reportname)
    report.UpdateReport(I=I2)
    # Report files
    report_png = os.path.join("report", case_folder, "mach-y0.png")
    assert os.path.isfile(report_png)


if __name__ == "__main__":
    test_01_plain()
    test_02_refine()

