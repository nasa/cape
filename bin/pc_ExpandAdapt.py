#!/usr/bin/env python
"""
Expand tarred ``adapt??.tar`` archives: :file:`pc_ExpandAdapt.py`
=================================================================

Expand all :file:`adapt??.tar` files and delete the tar balls.

    * :file:`adapt00.tar` --> ``adapt00/``
    * :file:`adapt01.tar` --> ``adapt01/``
    * ``adapt02/`` --> ``adapt02/`

:Call:

    .. code-block:: console
    
        $ pc_ExpandAdapt.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-12-30 ``@ddalle``: First version
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
    pyCart.manage.ExpandAdapt()
    
