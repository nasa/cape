#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyus.cli`: Command-line interface to ``pyus``
===========================================================

This module provides the :func:`main` function that is used by the
executable called ``pyus``.

"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import pyus
from .. import text
from . import cli_doc


# Primary interface
def main():
    r"""Main interface to ``pyus``

    This is basically an interface to :func:`cape.pyus.cntl.Cntl.cli`. 

    :Call:
        >>> main()
    :Versions:
        * 2021-03-04 ``@ddalle``: Version 1.0
    """
    # Parse inputs
    a, kw = argread.readflagstar(sys.argv)
    
    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Get help message
        HELP_MSG = cli_doc.PYUS_HELP
        # Display help
        print(text.markdown(HELP_MSG))
        return
        
    # Get file name
    fname = kw.get('f', "pyUS.json")
    
    # Try to read it
    cntl = pyus.Cntl(fname)
    
    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

