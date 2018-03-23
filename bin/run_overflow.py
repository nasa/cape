#!/usr/bin/env python
"""
Run OVERFLOW for one squence of one case: :file:`run_overflow.py`
=================================================================

This script determines the appropriate index to run for an individual case (e.g.
if a restart is appropriate, etc.), sets that case up, and runs it.

:Call:
    
    .. code-block:: bash
    
        $ run_overflow.py [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: First version
    * 2016-02-02 ``@ddalle``: OVERFLOW fork
"""

# Import the module specifically for this task.
import pyOver.case
# Argument parsing
import sys, cape.argread

# Simple function to call the main function of that module.
def run_overflow():
    """Calls :func:`pyOver.case.run_overflow`
    
    :Call:
        >>> run_overflow()
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-02-14 ``@ddalle``: Added `verify` and `intersect` checks
    """
    pyOver.case.run_overflow()
    
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
    run_overflow()

