#!/usr/bin/env python
"""
Extend a pyOver Job and Resubmit
================================

This function increases the maximum number of iterations and restarts the case
to run until the new iteration number is reached.  The function does nothing
unless the job is not currently running and the current requested number of
iterations has been reached.

The number of additional iterations is determined by reading the namelist file
for the last phase and reading *NSTEPS* from the *GLOBAL* section.

:Usage:
    .. code-block:: bash
    
        $ po_ExtendCase.py
        $ po_ExtendCase.py [OPTIONS]
    
:Options:

    -h, --help
        Display this help message and quit
        
    -m M
        Add *M* times *NSTEPS* additional iterations (default: 1)
        
    --no-run
        Update the ``case.json`` file but don't resubmit

:Versions:
    * 2014-10-06 ``@ddalle``: First version
    * 2015-10-16 ``@ddalle``: Copied from ``pycart``
"""

# Import tools from pyOver
import pyOver.case
# Input parsing
import cape.argread

# Check if run as script
if __name__ == "__main__":
    # Read input flags
    a, kw = cape.argread.readkeys(pyOver.case.os.sys.argv)
    # Check for help flag
    if kw.get('h') or kw.get('help'):
        print(__doc__)
        pyOver.case.os.sys.exit()
    
    # Get number of phases
    m = int(kw.get('m', '1'))
    # Get run/no-run setting
    if kw.get('no-run'):
        # Do not run
        run = False
    else:
        # Do run/resubmit
        run = True
        
    # Call the function
    pyOver.case.ExtendCase(m=m, run=run)
    
