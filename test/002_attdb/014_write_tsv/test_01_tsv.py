# -*- coding: utf-8 -*-

# Standard library
import os.path as op

# Third-party
import testutils

# Local modules
import cape.attdb.rdb as rdb


# File names
FDIR = op.dirname(__file__)
MATFILE = "CN-alpha-beta.mat"
TSVFILE1 = "CN-dense.tsv"
TSVFILE2 = "CN-default.tsv"


# Dense tab-sep
@testutils.run_sandbox(__file__, MATFILE)
def test_01_tsvdense():
    # Read DataKit from MAT file
    db = rdb.DataKit(MATFILE, DefaultWriteFormat="%.3f")
    # Write simple dense TSV file
    db.write_tsv_dense(TSVFILE1)
    # Compare
    assert testutils.compare_files(TSVFILE1, op.join(FDIR, TSVFILE1))


# Nominal tab-sep
@testutils.run_sandbox(__file__, fresh=False)
def test_02_tsv():
    # Read DataKit from MAT file
    db = rdb.DataKit(MATFILE, WriteFormats={"alpha": "%5.2f"})
    # Write normal TSV file
    db.write_tsv(TSVFILE2)
    # Compare
    assert testutils.compare_files(TSVFILE2, op.join(FDIR, TSVFILE2))

