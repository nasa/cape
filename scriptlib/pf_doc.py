#!/usr/bin/env python
"""
Python interface for FUN3D: :file:`pyfun`
=========================================

This function provides a master interface for pyFun.  All of the functionality
from this script is also accessible from the :mod:`pyFun` module using
relatively simple commands.

:Usage:
    .. code-block:: bash
    
        $ pyfun [options]
        
:Examples:
    
    The basic call submits all jobs prescribed in the file :file:`pyCart.json`
    
        .. code-block:: bash
            
            $ pycart
            
    This command uses the inputs from :file:`poweron.json` and only displays
    statuses.  No jobs are submitted.
    
        .. code-block:: bash
        
            $ pycart -f poweron.json -c
            
    This command submits at most 5 jobs, but only cases with "Mach" greater
    than 1.5 and "alpha" equal to 0.0 are considered as candidates.
    
        .. code-block:: bash
        
            $ pycart -n 5 --cons "Mach>0.5, alpha==0.0"
    
:Options:

    -h, --help
        Display this help message and quit
        
    -c
        Check status and don't submit new jobs
    
    -j
        Show the PBS job numbers as well
        
    -f FNAME
        Use pyCart input file *FNAME* (defaults to 'pyCart.json')

    -n NJOB
        Submit at most *NJOB* PBS scripts (defaults to unlimited)
        
    -q QUEUE
        Submit to a specific queue (defaults to "sls_aero1")
        
    --cons CNS
        Only consider cases that pass a list of inequalities separated by
        commas.  Constraints must use variable names (not abbreviations) from
        the trajectory described in *FNAME*.
        
    -I INDS
        Specify a list of cases to consider directly by index
        
    --re REGEX
        Restrict to cases whose name matches regular expression *REGEX*
        
    --filter TXT
        Restrict to cases whose name contains the test *TXT*
        
    --glob TXT
        Restrict to cases whose name matches the glob *TXT*
        
    --report RP
        Update report named *RP* or first report if *RP* is not specified
        
    --aero
        Loop through cases and extract force and moment coefficients and
        statistics for components described in the "Plot" section of *FNAME*
        
    --apply
        Apply the settings in *FNAME* to all cases; way to quickly change
        settings for a set of runs

    -a
        Archive folders according to settings in "Management" section of
        *FNAME*

    --expand
        Unarchive :file:`adapt??.tar` files in run folders

:Versions:
    * 2014-10-06 ``@ddalle``: First version
    * 2015-10-16 ``@ddalle``: Copied from ``pycart``
"""

# Print help if appropriate.
if __name__ == "__main__":
    print(__doc__)
