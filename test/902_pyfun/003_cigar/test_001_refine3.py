
# Standard library
import os

# Third-party imports
import testutils

# Local imports
import cape.pyfun.cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "fun3d.nml",
    "matrix.csv",
    "mach-y0.lay",
    "cigar001-surf01.*",
)


# File names
I = 21


# Test AFLR3 execution
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_plain():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Run a case
    cntl.SubmitJobs(I=I)
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(I)
    # Check case exists
    assert os.path.isdir(case_folder)
    # Generate a report
    reportname = cntl.opts.get_ReportList()[0]
    report = cntl.ReadReport(reportname)
    report.UpdateReport(I=I)
    # Name of report
    report_pdf = os.path.join("report", f"report-{reportname}.pdf")
    assert os.path.isfile(report_pdf)


if __name__ == "__main__":
    test_01_plain()

