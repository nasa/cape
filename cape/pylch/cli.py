#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylch.cli`: Interface to ``pylch`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pylch`` is used.
"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import text as textutils
from .cntl import Cntl
from .cli_doc import PYLAVA_HELP


# Primary interface
def main():
    r"""Main interface to ``pylch``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pylch.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2024-11-07 ``@ddalle``: v1.0
    """
    # Parse inputs
    a, kw = argread.readflagstar(sys.argv)
    # Check for args
    if len(a) == 0:
        # No command, doing class pylava behavior
        cmd = None
    else:
        cmd = a[0]

    # Check for "run_lavacurv"
    if cmd and cmd.lower() in {"run_locichem", "run"}:
        # Run case in this folder
        from .casecntl import run_lavacurv
        # Run and exit
        return run_lavacurv()

    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(PYLAVA_HELP))
        return

    # Get file name
    fname = kw.get('f', "pyLCH.json")

    # Try to read it
    cntl = Cntl(fname)

    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

