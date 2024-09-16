
# Standard library
import os
import glob

# Third-party
import testutils

# Local imports
from cape.cfdx.archivist import (
    expand_fileopt,
    getsize,
    getmtime)


# Files to copy
COPY_FILES = (
    "case.json",
    "pyfun*_fm_*.dat",
    "pyfun_tec_boundary_*.triq",
)
COPY_DIRS = (
    "lineload",
)
# Files to create
MAKE_FILES = (
    "pyfun_tec_boundary_timestep100.plt",
    "pyfun_tec_boundary_timestep200.plt",
    "pyfun_tec_boundary_timestep1000.plt",
    os.path.join("lineload", "a", "new.txt"),
)


# Test basic functions
@testutils.run_sandbox(__file__, COPY_FILES, COPY_DIRS)
def test_01_getprop():
    # Test recursive size (3 dirs + 4 2-byte files)
    assert getsize("lineload") == 4096*3 + 8
    # Create files (in correct order)
    _make_files()
    # List of surf files from old-fashioned glob
    surf_files = glob.glob("pyfun_tec_boundary_*.plt")
    latest_surf = "pyfun_tec_boundary_timestep1000.plt"
    # Check recursive mtime
    assert getmtime(*surf_files) == os.path.getmtime(latest_surf)
    # Latest file in lineload/ folder (?!)
    latest_file = os.path.join("lineload", "a", "new.txt")
    # Check recursive mtime
    assert getmtime("lineload") == os.path.getmtime(latest_file)
    # Test non-existent files
    assert getmtime("not_a_file") == 0.0
    assert getsize("not_a_file") == 0


# Test option expansion
def test_02_opt():
    # Start with a dict
    d1 = {
        "pyfun[0-9]*_fm_COMP.dat": 1,
        "run.[0-9][0-9].[0-9]+": "all",
    }
    d2 = {
        "pyfun[0-9]*_fm_COMP.dat": 1,
    }
    assert expand_fileopt(d1) == d2
    # Check a string
    s1 = "run.*"
    assert expand_fileopt(s1) == {s1: 0}
    # Check a list
    l1 = ["run.*", "pyfun.*"]
    d2 = {
        "run.*": 1,
        "pyfun.*": 1,
    }
    assert expand_fileopt(l1, vdef=1) == d2
    # List of dicts
    l1 = [
        {"run.*": 0},
        {"pyfun.*": 1},
        True,
    ]
    d2 = {
        "run.*": 0,
        "pyfun.*": 1,
    }
    assert expand_fileopt(l1) == d2


# Create files
def _make_files():
    for fname in MAKE_FILES:
        open(fname, 'w').write('1')
