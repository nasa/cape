#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Expand tarred Cart3D ``adapt??.tar`` archives: `pc_ExpandAdapt.py``
====================================================================

Expand all ``adapt??.tar`` files and delete the tar balls.

    * :file:`adapt00.tar` --> ``adapt00/``
    * :file:`adapt01.tar` --> ``adapt01/``
    * ``adapt02/`` --> ``adapt02/`

:Usage:
    .. code-block:: console
    
        $ pc_ExpandAdapt.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-12-30 ``@ddalle``: First version
"""

# Standard library
import sys

# CAPE Modules
import cape.argread
import cape.pycart.manage
        
    
# Check for running as a script.
if __name__ == "__main__":
    # Process inputs.
    a, kw = cape.argread.readkeys(sys.argv)
    # Check for help flag.
    if kw.get('h') or kw.get('help'):
        # Display help message and exit.
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Do the plotting.
    cape.pycart.manage.ExpandAdapt()
    
