#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyover.cli`: Interface to ``pyover`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pyover`` is used.
"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import text as textutils
from .cntl import Cntl
from .cli_doc import PYOVER_HELP


# Primary interface
def main():
    r"""Main interface to ``pyover``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pyover.cntl.Cntl.cli`. 

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: Version 1.0
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
    if cmd.lower() in {"run_overflow", "run"}:
        # Run case in this folder
        from .case import run_overflow
        # Run and exit
        return run_overflow()
    
    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(PYOVER_HELP))
        return
        
    # Get file name
    fname = kw.get('f', "pyOver.json")
    
    # Try to read it
    cntl = Cntl(fname)
    
    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

