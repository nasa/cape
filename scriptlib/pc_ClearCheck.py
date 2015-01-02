#!/usr/bin/env python
"""
Clear early ``check.*/`` files: :file:`pc_ClearCheck.py`
========================================================

Delete all files in the globs ``check.?????`` and ``check.??????.td`` except
for the most recent one.

:Call:

    .. code-block:: console
    
        $ pc_ClearCheck.py [OPTIONS]
    
:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2014-12-31 ``@ddalle``: First version
"""

# Modules
import pyCart.argread
import pyCart.manage
# System interface.
import sys
        
    
# Check for running as a script.
if __name__ == "__main__":
    # Process inputs.
    a, kw = pyCart.argread.readkeys(sys.argv)
    # Check for help flag.
    if kw.get('h') or kw.get('help'):
        # Display help message and exit.
        print(__doc__)
        sys.exit()
    # Do the plotting.
    pyCart.manage.ClearCheck()
    
