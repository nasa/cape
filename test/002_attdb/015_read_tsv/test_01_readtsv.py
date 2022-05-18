# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports 
import cape.attdb.rdb as rdb


# Read a dense tab-separated file
@testutils.run_testdir(__file__)
def test_01_read_tsvdense():
    # Read DataKit from TSV file
    db = rdb.DataKit(tsv="CN-dense.tsv")
    assert db.get_col_type("beta") == "float64"
    assert db.get_col_type("CN") == "float64"


# Read a normal tab-separated file
@testutils.run_testdir(__file__)
def test_02_read_tsv():
    # Read DataKit from TSV file
    db = rdb.DataKit(tsv="CN-default.tsv")
    assert db.get_col_type("beta") == "float64"
    assert db.get_col_type("CN") == "float64"

