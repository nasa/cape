# -*- coding: utf-8 -*-

# Local imports
import cape.convert


# Test fsteps
def test_01_fsteps():
    # Test fstep negative
    assert cape.convert.fstep(-10.0) == -1
    # Test fstep1 positive
    assert cape.convert.fstep1(10.0) == 1
