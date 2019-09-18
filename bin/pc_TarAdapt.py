#!/usr/bin/env python
"""
Tar early ``adapt??/`` folders: :file:`pc_TarAdapt.py`
======================================================

Archive all ``adapt??`` folders into (uncompressed) tar balls except for the
folder pointed to by ``BEST/``.  For instance if there are 2 adaptation cycles,
the following changes are made.

    * ``adapt00/`` --> :file:`adapt00.tar`
    * ``adapt01/`` --> :file:`adapt01.tar`
    * ``adapt02/`` --> ``adapt02/`

:Call:

    .. code-block:: console
    
        $ pc_TarAdapt.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-11-17 ``@ddalle``: First version
"""

# Modules
import cape.argread
import cape.pycart.manage
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
    pyCart.manage.TarAdapt()
    
