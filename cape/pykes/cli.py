#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.cli`: Interface to ``pykes`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pykes`` is used.
"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import text as textutils
from .cli_doc import PYKES_HELP


# Primary interface
def main():
    r"""Main interface to ``pykes``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pykes.cntl.Cntl.cli`. 

    :Call:
        >>> main()
    :Versions:
        * 2021-10-19 ``@ddalle``: Version 1.0
    """
    # Parse inputs
    a, kw = argread.readflagstar(sys.argv)
    # Check for args
    if len(a) == 0:
        # No command, doing class pyfun behavior
        cmd = None
    else:
        cmd = a[0]

    # Check for "run_fun3d"
    if cmd in {"run_kestrel", "run"}:
        # Run case in this folder
        from .case import run_kestrel
        # Run and exit
        return run_kestrel()

    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(PYKES_HELP))
        return
        
    # Get file name
    fname = kw.get('f', "pyKes.json")
    
    # Try to read it
    from .cntl import Cntl
    cntl = Cntl(fname)

    # Call the command-line interface
    cntl.cli(*a, **kw)

