#!/usr/bin/env python
"""
Attempt to assimilate owners on all objects
============================================

This script searches all files and checks for any that don't have the same
owner as the user running the script. In order to correct this, it will move
each such file to a temporary file, copy it (thereby creating a new file as the
current user), and then deleting the temporary file.

:Call:
    
    .. code-block:: console
    
        $ assim_user.py [OPTIONS]

:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2018-04-03 ``@ddalle``: First version
"""

# System modules
import os
import glob
import shutil

# Primary function
def assimilate_user():
    """Attempt to map all files to the current user
    
    :Call:
        >>> assimilate_user()
    :Versions:
        * 2018-04-03 ``@ddalle``: First version
    """
    # Get the executor's UID
    uid = os.getuid()

# Check if run as script
if __name__ == "__main__":
    # Get inputs
    argv = os.sys.argv
    # Check for help option
    if ("-h" in argv) or ("--help" in argv):
        print(__doc__)
        sys.exit()
    # Run main function
    assimilate_user()

