
# Standard library
import os
import glob

# Third-party
import testutils

# Local imports
from cape.cfdx.archivist import (
    getsize,
    getmtime)


# Files to copy
COPY_FILES = (
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


# Create files
def _make_files():
    for fname in MAKE_FILES:
        open(fname, 'w').write('1')
