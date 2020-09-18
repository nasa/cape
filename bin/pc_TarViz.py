#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Tar Cart3D time-accurate visualization files: ``pc_TarViz.py``
===============================================================

Archive all ``adapt??`` folders into (uncompressed) tar balls except for
the folder pointed to by ``BEST/``.  For instance if there are 2
Cart3D adaptation cycles, the following changes are made.

    * ``Components.i.*.stats.{plt,dat}`` --> ``Components.i.stats.tar``
    * ``Components.i.*.{plt,dat}`` --> ``Components.i.tar``
    * ``cutPlanes.*.{plt,dat}`` --> ``cutPlanes.tar``

:Usage:
    .. code-block:: console
    
        $ pc_TarViz.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-12-18 ``@ddalle``: Version 1.0
"""

# Standard library
import sys

# CAPE modules
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
    cape.pycart.manage.TarViz()
    
