
# Standard library
import os
import sys

# Third-party imports
import testutils

# Local imports
import cape.pyfun.cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "bullet.xml",
    "fun3d.nml",
    "matrix.csv",
    "bullet-*",
    "test.[0-9][0-9].out",
    os.path.join("report", "NASA_logo.pdf"),
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_run():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Run first case
    cntl.SubmitJobs(I="8")


# Test 'pyfun -c'
@testutils.run_sandbox(__file__, fresh=False)
def test_02_c():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-c", "-I", "8"]
    # Run the command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Check outout
    result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    assert result.line1 == result.line2


# Collect aero
@testutils.run_sandbox(__file__, fresh=False)
def test_03_fm():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Collect aero
    cntl.cli(fm=True, I="8")
    # Read databook
    cntl.ReadDataBook()
    # Get value
    CA = cntl.DataBook["bullet_no_base"]["CA"][0]
    # Test value
    assert abs(CA - 0.46) <= 0.005


# Run TriqPt
@testutils.run_sandbox(__file__, fresh=False)
def test_04_pt():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Run --pt
    cntl.cli(pt=True, I="8")
    # DataBook folder
    fdir = cntl.opts.get_DataBookFolder()
    # Get list of points
    ptgrp, = cntl.opts.get_DataBookByType("TriqPoint")
    pts = cntl.opts.get_DataBookPoint(ptgrp)
    # Test for files
    for pt in pts:
        ptfile = os.path.join(fdir, f"pt_{pt}.csv")
        assert os.path.isfile(ptfile)


if __name__ == "__main__":
    test_01_run()
    test_02_c()
    test_03_fm()
    test_04_pt()

