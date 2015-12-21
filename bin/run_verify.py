#!/usr/bin/env python
"""
Run Cart3D for one index of one case: :file:`run_flowCart.py`
=============================================================

This script determines the appropriate index to run for an individual case (e.g.
if a restart is appropriate, etc.), sets that case up, and runs it.

:Versions:
    * 2014-02-14 ``@ddalle``: First version
"""

# Import the module specifically for this task.
import pyCart.bin

# Simple function to call the main function of that module.
def run_verify():
    """Calls :func:`pyCart.bin.verify`
    
    :Call:
        >>> run_verify()
    :Versions:
        * 2015-02-13 ``@ddalle``: First version
    """
    pyCart.bin.verify()
    
# Check if run as a script.
if __name__ == "__main__":
    run_verify()
