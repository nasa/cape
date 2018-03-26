#!/usr/bin/env python
"""
Tar time-accurate visualization files: :file:`pc_TarViz.py`
===========================================================

Archive all ``adapt??`` folders into (uncompressed) tar balls except for the
folder pointed to by ``BEST/``.  For instance if there are 2 adaptation cycles,
the following changes are made.

    * ``Components.i.*.stats.{plt,dat}`` --> :file:`Components.i.stats.tar`
    * ``Components.i.*.{plt,dat}`` --> :file:`Components.i.tar`
    * ``cutPlanes.*.{plt,dat}`` --> :file:`cutPlanes.tar`

:Call:

    .. code-block:: console
    
        $ pc_TarViz.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-12-18 ``@ddalle``: First version
"""

# Modules
import cape.argread
import pyCart.manage
# System interface.
import sys
        
    
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
    pyCart.manage.TarViz()
    
