#!/usr/bin/env python
"""
CAPE interface for CAPE: **cape**
=======================================

This function provides a template executable interface for CAPE. The
functionality from this script is also accessible from the :mod:`cape` module
using relatively simple commands.

:Usage:
    .. code-block:: bash
    
        $ cape [options]
        
:Examples:
    
    The basic call submits all jobs prescribed in the file :file:`cape.json`
    
        .. code-block:: bash
            
            $ cape
            
    This command uses the inputs from :file:`poweron.json` and only displays
    statuses.  No jobs are submitted.
    
        .. code-block:: bash
        
            $ cape -f poweron.json -c
            
    This command submits at most 5 jobs, but only cases with "Mach" greater
    than 1.5 and "alpha" equal to 0.0 are considered as candidates.
    
        .. code-block:: bash
        
            $ cape -n 5 --cons "Mach>0.5, alpha==0.0"
    
:Options:

    -h, --help
         Display this help message and quit
        
    -c
         Check status and don't submit new jobs
    
    -j
         Show the PBS job numbers as well
        
    -f FNAME
         Use pyCart input file *FNAME* (defaults to ``'pyCart.json'``)
        
    -x FPY
        Executes Python script *FPY* using ``execfile()`` prior to taking other
        actions; can be stacked using ``-x FPY1 -x FPY2 -x FPY3``, etc.

    -n NJOB
         Submit at most *NJOB* PBS scripts (defaults to unlimited)
        
    -q QUEUE
         Submit to a specific queue (defaults to ``'normal'``)
        
    --cons CONS
         Only consider cases that pass a list of inequalities separated by
         commas.  Constraints must use variable names (not abbreviations) from
         the trajectory described in *FNAME*.
        
    -I INDS
         Specify a list of cases to consider directly by index
        
    --re REGEX
         Restrict to cases whose name matches regular expression *REGEX*
        
    --filter TXT
         Restrict to cases whose name contains the text *TXT*
        
    --glob WC
         Restrict to cases whose name matches the glob/wildcard *WC*

    --kill, --qdel
         Remove jobs from the queue and stop them abruptly
        
    --report REP
         Update report named *REP* or first report if *REP* is not specified
        
    --archive
        Create archive according to options in "Archive" section of *FNAME* and
        clean up run folder if case is marked PASS
        
    --clean
        Delete any files as described by *ProgressDeleteFiles* in "Archive"
        section of *FNAME*; can be run at any time
        
    --aero, --aero GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for force & moment components described in the "DataBook"
        section of *FNAME*; only process components matching wildcard *GLOB* if
        if is specified
        
    --ll, --ll GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for LineLoad components described in the "DataBook"
        section of *FNAME*; only process components matching wildcard *GLOB* if
        if is specified
        
    --triqfm, --triqfm GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for TriqFM components described in the "DataBook"
        section of *FNAME*; only process components matching wildcard *GLOB* if
        if is specified
        
    --apply
        Apply the settings in *FNAME* to all cases; way to quickly change
        settings for a set of runs

    --extend, --extend E
        Add another run of the current last phase *E* times (default is 1)
        
    --imax M
        Do not extend a case (when using --extend) beyond iteration *M*
        
    --no-start
        When running a command that would otherwise submit jobs, set them up
        but do not start (or submit) them
        
    --no-restart
        When submitting new jobs, only submit new cases (status '---')

    -a
         Archive folders according to settings in "Management" section of
         *FNAME*

    --expand
         Unarchive :file:`adapt??.tar` files in run folders

:Versions:
    * 2014-10-03 ``@ddalle``: Started
    * 2014-10-06 ``@ddalle``: First version
    * 2014-12-11 ``@ddalle``: Version 2.0 with constraints, plot, and F&M
"""

# Print help if appropriate.
if __name__ == "__main__":
    import cape.text
    print(cape.text.markdown(__doc__))
