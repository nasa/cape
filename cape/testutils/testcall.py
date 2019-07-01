#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.testcall`: Shell utilities for CAPE testing system
=======================================================================

This module contains various methods to perform system calls using the
:mod:`subprocess` module.
"""

# Standard library modules
import subprocess as sp


# Call a function and capture the output
def callo(cmd, stderr=None):
    # Create a Popen process
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=stderr)
    # Run the command
    out, err = proc.communicate()
    # Check for error
    
    # Output
    return out
    