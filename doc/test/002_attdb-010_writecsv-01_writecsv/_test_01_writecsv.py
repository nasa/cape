# -*- coding: utf-8 -*-

# Standard library
import os

# Third-party
import testutils

# Local imports
import cape.attdb.rdb as rdb


# Starting database file
MAT_FILE = "CN-alpha-beta.mat"
CSV_FILE1 = "CN-dense.csv"
CSV_FILE2 = "CN-default.csv"

# This folder
FDIR = os.path.dirname(__file__)


# Write a dense CSV file
@testutils.run_sandbox(__file__, MAT_FILE)
def test_01_csv_dense():
    # Read DataKit from MAT file
    db = rdb.DataKit(MAT_FILE, DefaultWriteFormat="%.3f")
    # Write simple dense CSV file
    db.write_csv_dense(CSV_FILE1)
    # Check them
    assert testutils.compare_files(CSV_FILE1, os.path.join(FDIR, CSV_FILE1))


# Write a nominal CSV file
@testutils.run_sandbox(__file__, MAT_FILE)
def test_02_csv_write():
    # Read DataKit from MAT file
    db = rdb.DataKit(MAT_FILE, WriteFormats={"alpha": "%5.2f"})
    # Write simple dense CSV file
    db.write_csv(CSV_FILE2)
    # Check them
    assert testutils.compare_files(CSV_FILE2, os.path.join(FDIR, CSV_FILE2))

