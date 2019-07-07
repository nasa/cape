#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.testresults`: Results from one test
===========================================================

This module primarily provides a simple class, :class:`TestResults`,
which stores the results from a singe CAPE unit test.  This class is
based on :class:`dict` but only accepts a fixed list of keys.  These
keys and their default values are contained in the variable *rc* from
this module.  They are tabulated below:

    =============  ===============  ========================================
    Option         Default Value    Description
    =============  ===============  ========================================
    *Status*       ``False``        Overall test pass/fail
    *stdout*       ``"STDOUT"``     Name of file to capture standard out
    *stderr*       ``"STDERR"``     Name of file to capture standard error
    *CopyFiles*    ``[]``           Files to copy into test folder
    *LinkFiles*    ``[]``           Files to link into test folder
    *Commands*     ``[]``           List of commands to run during test
    =============  ===============  ========================================
"""

# Standard library modules
import os


# Default attributes
rc = {
    "Status": False,
    "Name": "",
    "Path": "",
    "Time": 0.0,
    "RunTimes": [],
    "ReturnCodes": [],
}