#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Run FUN3D for one phase: :file:`run_fun3d.py`
================================================

This script determines the appropriate index to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:
    
    .. code-block:: console
    
        $ run_fun3d.py [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: Version 1.0 (pycart)
    * 2015-10-19 ``@ddalle``: Version 1.0
"""

# Standard library
import sys

# CAPE modules
import cape.argread
import cape.pyfun.case


# Simple function to call the main function of that module.
def run_fun3d():
    """Calls :func:`cape.pyfun.case.run_fun3d`
    
    :Call:
        >>> run_fun3d()
    :Versions:
        * 2015-10-19 ``@ddalle``: Version 1.0
    """
    cape.pyfun.case.run_fun3d()


# Check if run as a script.
if __name__ == "__main__":
    # Parse arguments
    a, kw = cape.argread.readkeys(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run FUN3D
    run_fun3d()

