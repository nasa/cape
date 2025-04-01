
# Standard library
import os
import glob

# Third-party
import pytest
import testutils

# Local imports
from cape.cfdx.casecntl import CaseRunner
from cape.cfdx.options import RunControlOpts
from cape.cfdx.archivist import (
    CaseArchivist,
    expand_fileopt,
    getsize,
    getmtime,
    reglob,
    rematch,
    _assert_relpath,
    _disp_size,
    _posix,
    _safe_mtime,
    _validate_safety_level
)


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
    "pyfun00_plane-y0_timestep100.plt",
    "pyfun01_plane-y0_timestep200.plt",
    "pyfun02_plane-y0_timestep500.plt",
    "pyfun02_plane-y0_timestep1000.plt",
    "pyfun00_plane-z0_timestep100.plt",
    "pyfun01_plane-z0_timestep200.plt",
    "pyfun02_plane-z0_timestep500.plt",
    "pyfun02_plane-z0_timestep1000.plt",
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
    # Test non-existent files
    assert getmtime("not_a_file") == 0.0
    assert getsize("not_a_file") == 0
    # Test basic functions
    assert _safe_mtime("not_a_file") == 0.0
    assert _posix(os.path.join("lineload", "a")) == "lineload/a"


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


# Conversion of bytes to human-readable
def test_03_disp():
    assert _disp_size(3) == '3 B'
    assert _disp_size(2.1*1024) == "2.1 kB"
    assert _disp_size(21*1024*1024) == "21 MB"
    assert _disp_size(321*1024*1024*1024) == "321 GB"


# Search by regex
@testutils.run_sandbox(__file__, COPY_FILES, COPY_DIRS)
def test_04_reglob():
    # Prepare additional files
    _make_files()
    # Try a recursive regex
    search1 = [
        os.path.join("lineload", "a", "a.txt"),
        os.path.join("lineload", "b", "b.txt"),
    ]
    result1 = reglob(os.path.join("lineload", "[a-z]", "[a-z].txt"))
    result1.sort()
    assert result1 == search1


# Search by regex
@testutils.run_sandbox(__file__, COPY_FILES, COPY_DIRS)
def test_05_rematch():
    # Prepare additional files
    _make_files()
    # Use regex with group
    regex1 = "pyfun[0-9]*_fm_(?P<comp>[A-Za-z][A-Za-z0-9_-]*)\\.dat"
    search1 = {
        "comp='COMP1'": [
            "pyfun00_fm_COMP1.dat",
            "pyfun01_fm_COMP1.dat",
            "pyfun02_fm_COMP1.dat",
        ],
        "comp='COMP2'": [
            "pyfun00_fm_COMP2.dat",
            "pyfun01_fm_COMP2.dat",
            "pyfun02_fm_COMP2.dat",
        ],
    }
    # Do the search
    result1 = rematch(regex1)
    # Sort the results by file time (mtime not controlled here)
    for k, v in result1.items():
        result1[k] = sorted(v)
    assert result1 == search1
    # Do another search with multiple groups
    regex2 = "pyfun([0-9]+)_fm_(?P<comp>[A-Za-z][A-Za-z0-9_-]*)\\.dat"
    # Do the search
    result2 = rematch(regex2)
    # Expected group
    key2 = "0='00' comp='COMP1'"
    # Check one key
    assert key2 in result2


# Test unit validators
def test_06_errors():
    # Test some errors
    with pytest.raises(ValueError):
        _assert_relpath(__file__)
    with pytest.raises(ValueError):
        _validate_safety_level("made-up")


# Instantiate Archivist class
@testutils.run_sandbox(__file__, COPY_FILES, COPY_DIRS)
def test_07_case():
    # Turn sandbox into a working case
    _make_case()
    # Enter working folder
    os.chdir("case")
    # Instantiate case runner
    runner = CaseRunner()
    # Read archivist
    a = runner.get_archivist()
    assert isinstance(a, CaseArchivist)
    # Read options for case
    rc = runner.read_case_json()
    opts = rc["Archive"]
    # Instantiate maunally
    a1 = CaseArchivist(opts)
    # Check
    assert a.root_dir == a1.root_dir
    # Create archive folder
    archivedir = os.path.join(os.path.dirname(__file__), "work", "archive")
    assert a.archivedir == archivedir
    # Run the ``--clean`` command
    a.clean()
    # Make sure case folder was created
    frun = runner.get_case_name().replace('/', os.sep)
    adir = a.archivedir
    arun = os.path.join(adir, frun)
    assert os.path.isdir(arun)
    # List files in archive case folder
    alist = os.listdir(arun)
    clist = os.listdir(a.root_dir)
    # Check that some files were "archived"
    assert "pyfun00_fm_COMP1.dat" in alist
    assert "fm.tar" in alist
    # Check that some files were deleted
    assert "pyfun_tec_boundary_timestep100.plt" not in clist
    # Save modification time of one file
    fname = os.path.join(arun, "fm.tar")
    mtime = os.path.getmtime(fname)
    # Run it again to check on the updating capabilities
    a.clean()
    # There should be 0 bytes of new deletions
    assert a._last_msg.endswith("0 B")
    # Make sure mod time is unchanged
    assert os.path.getmtime(fname) == mtime
    # Make sure no new files deleted
    clist = os.listdir(a.root_dir)
    assert "pyfun_tec_boundary_timestep200.plt" in clist


@testutils.run_testdir(__file__)
def test_08_conversion():
    # Read two options
    rc1 = RunControlOpts("case.json")
    rc2 = RunControlOpts("old.json")
    # Isolate *Archive* section
    a1 = rc1["Archive"]
    a2 = rc2["Archive"]
    # Expand values
    v1 = expand_fileopt(a1["clean"]["PostDeleteDirs"])
    v2 = a2["clean"]["PostDeleteDirs"]
    # Compare sections
    assert v1 == v2


# Create files and move
def _make_case():
    # Create the files
    _make_files()
    # Move them all
    os.chdir("..")
    os.rename("work", "case")
    os.mkdir("work")
    os.rename("case", os.path.join("work", "case"))
    os.mkdir(os.path.join("work", "archive"))
    # Return to position
    os.chdir("work")


# Create files
def _make_files():
    for fname in MAKE_FILES:
        open(fname, 'w').write('1')
