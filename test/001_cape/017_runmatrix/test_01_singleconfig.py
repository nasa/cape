#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os

# Third-party
import testutils

# Import cape module
from cape.cfdx.cntl import Cntl


# Test a few commands
@testutils.run_testdir(__file__)
def test_01_config():
    # Load interface
    cntl = Cntl()
    # Get name of second (i=1) case
    frun = cntl.x.GetFullFolderNames(1)
    # Get parameters
    config = cntl.x["config"][0]
    # Check it
    assert frun.startswith(config)
    # Check folder name
    assert frun == os.path.join("poweroff", "m0.5a2.0b0.0")

