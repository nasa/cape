#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Run Cart3D for one index of one case: :file:`run_flowCart.py`
=============================================================

This script determines the appropriate index to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:
    
    .. code-block:: console
    
        $ run_flowCart.py [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit
        
    --verify
        Run `verify` if starting from iteration 0
        
    --intersect
        Run `intersect` if starting from iteration 0

:Versions:
    * 2014-10-02 ``@ddalle``: Version 1.0
    * 2015-02-14 ``@ddalle``: Version 1.1: verify and intersect checks
"""

# Standard library modules
import sys

# CAPE modules
import cape.argread
import cape.pycart.case


# Simple function to call the main function of that module.
def run_flowCart(verify=False, isect=False):
    r"""Calls :func:`cape.pycart.case.run_flowCart`
    
    :Call:
        >>> run_flowCart(verify=False, isect=False)
    :Inputs:
        *verify*: :class:`bool`
            Whether or not to run ``verify`` before ``flowCart``
        *isect*: :class:`bool`
            Whether or not to run ``intersect`` before ``flowCart``
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
        * 2015-02-14 ``@ddalle``: Version 1.1: verify and intersect
    """
    cape.pycart.case.run_flowCart(verify, isect)

    
# Check if run as a script.
if __name__ == "__main__":
    # Parse arguments
    a, kw = cape.argread.readflags(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Check verify and intersect flags
    vfy   = kw.get('verify', False)
    isect = kw.get('intersect', False) 
    # Run ``flowCart``
    run_flowCart(verify=vfy, isect=isect)
