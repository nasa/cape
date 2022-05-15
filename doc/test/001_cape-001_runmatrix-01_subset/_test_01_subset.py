#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party
import testutils

# Import cape module
import cape


# Test a few commands
@testutils.run_testdir(__file__)
def test_01_subset():
    # Load interface
    cntl = cape.Cntl()
    # Test --filter
    cases = list(cntl.x.FilterString("b2"))
    assert cases == [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
    # Test --glob
    cases = list(cntl.x.FilterWildcard("poweroff/m0*"))
    assert cases == [0, 1, 2, 3, 4, 5, 6, 7]
    # Test --re
    cases = list(cntl.x.FilterRegex(r"m.\.5.*b2"))
    assert cases == [2, 3, 14, 15, 18, 19]
    # Test -I and --cons
    cases = list(cntl.x.GetIndices(I=range(15,20), cons=["Mach%1==0.5"]))
    assert cases == [15, 16, 17, 18, 19]

