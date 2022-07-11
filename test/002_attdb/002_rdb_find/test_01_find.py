# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports
import cape.attdb.rdb as rdb


# Test basic find
@testutils.run_testdir(__file__)
def test_01_find():
    db = rdb.DataKit("mab01.mat")
    # Basic search
    I, J = db.find(["mach", "alpha", "beta"], 0.9, 2.0, 0.0)
    # Checks
    assert list(I) == [12, 15, 17]
    assert list(J) == [0]


# Test advanced find
@testutils.run_testdir(__file__)
def test_02_find2():
    db = rdb.DataKit("mab01.mat")
    # Calculate break points
    db.create_bkpts(["mach"], nmin=1)
    # All Mach numbers
    mach = db.bkpts["mach"]
    # Set values
    aoa = 2.0
    aos = 0.0
    # Basic search
    I, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos)
    assert list(I) == [2, 6, 9, 12, 15, 17]
    assert list(J) == [0, 1, 2]
    # Search each condition once
    I, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos, once=True)
    assert list(I) == [2, 6, 12]
    # Search each condition once
    Imap, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos, mapped=True)
    assert list(Imap[1]) == [6, 9]
