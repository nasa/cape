#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Run US3D for one phase: :file:`run_us3d.py`
================================================

This script determines the appropriate index to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: bash

        $ run_us3d.py [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: Version 1.0 (pycart)
    * 2020-04-16 ``@ddalle``: Version 1.0
"""

# Standard library
import sys

# CAPE modules
import cape.argread
import cape.text
import cape.pyus.case


# Simple function to call the main function of that module.
def run_us3d():
    r"""Call :func:`cape.pyus.case.run_us3d`
    
    :Call:
        >>> run_fun3d()
    :Versions:
        * 2020-04-16 ``@ddalle``: First version
    """
    cape.pyus.case.run_us3d()


# Check if run as a script.
if __name__ == "__main__":
    # Parse arguments
    a, kw = cape.argread.readflags(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run US3D
    run_us3d()

