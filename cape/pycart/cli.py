#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pycart.cli`: Command-line interface to ``pycart``
=============================================================

This module provides the :func:`main` function that is used by the
executable called ``pycart``.

"""

# Standard library modules
import sys

# CAPE modules
import cape.argread
import cape.pycart
import cape.pycart.cli_doc
import cape.text


# Primary interface
def main():
    r"""Main interface to ``pycart``

    This is basically an interface to :func:`cape.pycart.cntl.Cntl.cli`. 

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: Version 1.0
    """
    # Parse inputs
    a, kw = cape.argread.readflagstar(sys.argv)
    
    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Get help message
        HELP_MSG = cape.pycart.cli_doc.PYCART_HELP
        # Display help
        print(cape.text.markdown(HELP_MSG))
        return
        
    # Get file name
    fname = kw.get('f', "pyCart.json")
    
    # Try to read it
    cntl = cape.pycart.Cntl(fname)
    
    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

