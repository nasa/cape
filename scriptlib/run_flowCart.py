#!/usr/bin/env python
"""
Run Cart3D for one index of one case: :file:`run_flowCart.py`
=============================================================

This script determines the appropriate index to run for an individual case (e.g.
if a restart is appropriate, etc.), sets that case up, and runs it.

:Versions:
    * 2014.10.02 ``@ddalle``: First version
"""

# Import the module specifically for this task.
import pyCart.case

# Simple function to call the main function of that module.
def run_flowCart():
    """Calls :func:`pyCart.case.run_flowCart`
    
    :Call:
        >>> run_flowCart()
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
    """
    pyCart.case.run_flowCart()
    
# Check if run as a script.
if __name__ == "__main__":
    run_flowCart()
