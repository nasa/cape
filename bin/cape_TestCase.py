#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAPE single-case test driver

The test driver executes a test in the current folder.  This consists
of several steps that run the test in a subfolder (which is called
work/ by default).  If that subfolder exists, it is deleted at the
beginning of the test.

:Usage:
    .. code-block:: console
        
        $ pc_TestCase.py [OPTIONS]

:Options:
    -f, --json FNAME
        Read settings from file *FNAME* {cape-test.json}

:Versions:
    * 2019-07-06 ``@ddalle``: First version
"""

# Standard library modules
import sys

# Import test modules
import cape.testutils.argread
import cape.testutils.driver


# Primary function
def driver(*a, **kw):
    """Test driver command-line interface
    
    :Call:
        >>> driver(*a, **kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file for crawler
    :Versions:
        * 2019-07-06 ``@ddalle``: First version
    """
    # Run the command-line interface from the module
    cape.testutils.driver.cli(*a, **kw)
    
    
# Check if called as a script
if __name__ == "__main__":
    # Process inputs
    a, kw = cape.testutils.argread.readkeys(sys.argv)
    # Check for help flags
    if kw.get("h") or kw.get("help"):
        # Show docstring and exit
        print(__doc__)
        sys.exit(1)
    # Call the function
    driver(*a, **kw)

