
# Standard library
import os
import shutil

# Third-party imports
import testutils
import numpy as np

# CAPE
import cape.pyover.cntl as pocntl
import cape.dkit.csvfile


# Dir to case files
CASEDIR = os.path.join("poweroff", "m0.8a4.0b0.0")

# List of file globs to copy into sandbox
TEST_FILES = (
    "pyOver.json",
    "matrix.csv",
    "oflfunc",
    "test.[0-9][0-9].out"
)

TEST_FILES2 = (
    "pyOver.json",
    "matrix.csv",
    "oflfunc",
    "cap-patch.uh3d",
    "test.[0-9][0-9].out"
)

TEST_DIRS = (
    "poweroff"
)

TEST_DIRS2 = (
    "poweroff",
    "common"
)

KW1 = {
    "I": [1],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW2 = {
    "I": [1],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW3 = {
    "I": [1],
    "ll": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW4 = {
    "I": [1],
    "ll": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW5 = {
    "I": [1],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW6 = {
    "I": [1],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW7 = {
    "I": [1],
    "triqfm": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW8 = {
    "I": [1],
    "triqfm": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

FTOL = 1e-8

@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    # Get cntl
    cntl = pocntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
    # Read test.01.out
    db0 = cape.dkit.csvfile.CSVFile("test.01.out")
    db1 = cntl.read_dex("bullet_no_base")
    # Compare each column
    for col in db0.cols:
        if isinstance(db0[col], int):
            delta = db0[col] - db1[col]
            assert delta == 0
        elif isinstance(db0[col], float):
            delta = db0[col] - db1[col]
            assert abs(delta) <= FTOL
        elif isinstance(db0[col], str):
            assert db0[col] == db1[col]

@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_deletecasesfm():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "aero_bullet_no_base.csv"))
    # Get cntl
    cntl = pocntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW2)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Location of old databook
    dbold = os.path.join("data/aero_bullet_no_base.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.02.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)

