#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPE test crawler
===================

The test crawler completes a simple task, namely that enters zero or
more folders and runs the main CAPE test utility.  This list of
folders is set by the user either directly or by pattern.  The default
is to run a test in each subfolder that exists.

:Usage:
    .. code-block:: console
        
        $ pc_TestCrawler.py [OPTIONS]

:Options:
    -f, --json FNAME
        Read settings from file *FNAME* {cape-test.json}

:Versions:
    * 2019-07-03 ``@ddalle``: Version 1.0
"""

# Standard library modules
import sys

# Import test modules
import cape.testutils.argread
import cape.testutils.crawler


# Primary function
def crawler(*a, **kw):
    """Test crawler command-line interface
    
    :Call:
        >>> crawler(*a, **kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file for crawler
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    # Run the command-line interface from the module
    cape.testutils.crawler.cli(*a, **kw)
    
    
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
    crawler(*a, **kw)

