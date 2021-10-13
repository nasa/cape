#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pycart.cli`: Interface to ``pycart`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pycart`` is used.
"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import text as textutils
from .cntl import Cntl
from .cli_doc import PYCART_HELP


# Primary interface
def main():
    r"""Main interface to ``pycart``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pycart.cntl.Cntl.cli`. 

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
    if cmd.lower() in {"run_flowcart", "run_cart3d", "run"}:
        # Run case in this folder
        from .case import run_flowCart
        # Run and exit
        return run_flowCart()
    
    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(PYCART_HELP))
        return
        
    # Get file name
    fname = kw.get('f', "pyCart.json")
    
    # Try to read it
    cntl = Cntl(fname)
    
    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

