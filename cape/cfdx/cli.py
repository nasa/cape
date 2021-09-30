#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cfdx.cli`: Command-line interface to ``cape`` (executable)
======================================================================

This module provides the :func:`main` function that is used by the
executable called ``cape``.

"""

# Standard library modules
import sys

# CAPE modules
from .. import argread
from .. import text as textutils
from .cfdx_doc import CAPE_HELP
from ..cntl import Cntl


# Primary interface
def main():
    r"""Main interface to ``pyfun``

    This is basically an interface to :func:`cape.cntl.Cntl.cli`. 

    :Call:
        >>> main()
    :Versions:
        * 2021-03-04 ``@ddalle``: Version 1.0
    """
    # Parse inputs
    a, kw = argread.readflagstar(sys.argv)
    
    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(CAPE_HELP))
        return
        
    # Get file name
    fname = kw.get('f', "cape.json")
    
    # Try to read it
    cntl = Cntl(fname)
    
    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

