#!/usr/bin/env python
"""
Run FUN3D for one squence of one case: :file:`run_fun3d.py`
===========================================================

This script determines the appropriate index to run for an individual case (e.g.
if a restart is appropriate, etc.), sets that case up, and runs it.

:Call:
    
    .. code-block:: bash
    
        $ run_fun3d.py [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: First version
    * 2015-10-19 ``@ddalle``: FUN3D fork
"""

# Import the module specifically for this task.
import cape.pyfun.case
# Argument parsing
import sys, cape.argread

# Simple function to call the main function of that module.
def run_fun3d():
    """Calls :func:`pyFun.case.run_fun3d`
    
    :Call:
        >>> run_fun3d()
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-02-14 ``@ddalle``: Added `verify` and `intersect` checks
    """
    pyFun.case.run_fun3d()
    
# Check if run as a script.
if __name__ == "__main__":
    # Parse arguments
    a, kw = cape.argread.readflags(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run FUN3D
    run_fun3d()

