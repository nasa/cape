"""
Case Control Module: :mod:`cape.case`
=====================================

This module contains templates for interacting with individual cases.  Since
this is one of the most highly customized modules of the Cape system, there are
few functions here, and the functions that are present are mostly templates.

In general, the :mod:`case` module is used for actually running the CFD solver
(and any additional binaries that may be required as part of the run process),
and it contains other capabilities for renaming files and determining the
settings for a particular case.  Cape saves many settings for the CFD solver and
archiving in a file called :file:`case.json` within each case folder, which
prevents changes to the master JSON file from unpredictably affecting cases that
have already been initialized or are already running.

Actual functionality is left to individual modules listed below.

    * :mod:`pyCart.case`
    * :mod:`pyFun.case`
    * :mod:`pyOver.case`

Several of the key methods for this API module are described below.
"""

# Import options class
from options.runControl import RunControl
# Interface for writing commands
from . import queue
from . import bin

# Need triangulations for cases with `intersect`
from .tri import Tri, Triq

# Read the local JSON file.
import json
# Timing
from datetime import datetime
# File control
import os, resource, glob, shutil
# Basic numerics
from numpy import nan, isnan


# Function to intersect geometry if appropriate
def CaseIntersect(rc, proj='Components', n=0):
    """Run ``intersect`` to combine geometries if appropriate
    
    This is a multistep process in order to preserve all the component IDs of
    the input triangulations.  Normally ``intersect`` requires each intersecting
    component to have a single component ID, and each component must be a
    water-tight surface.
    
    Cape utilizes two input files, ``Components.c.tri``, which is the original
    triangulation file with intersections and the original component IDs, and
    ``Components.tri``, which maps each individual original ``tri`` file to a
    single component.  The files involved are tabulated below.
    
        * ``Components.tri``: Intersecting components, each with own compID
        * ``Components.c.tri``: Intersecting triangulation with original compIDs
        * ``Components.o.tri``: Output of ``intersect``, only a few compIDs
        * ``Components.i.tri``: Original compIDs mapped onto intersected tris
    
    :Call:
        >>> CaseIntersect(rc, proj='Components', n=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *proj*: {``'Components'``} | :class:`str`
            Project root name
        *n*: :class:`int`
            Iteration number
    :Versions:
        * 2015-09-07 ``@ddalle``: Split from :func:`run_flowCart`
        * 2016-04-05 ``@ddalle``: Generalized to :mod:`cape`
    """
    # Check for intersect status.
    if not rc.get_intersect(): return
    # Check for initial run
    if n > 0: return
    # Check for triangulation file.
    if os.path.isfile('%s.i.tri' % proj):
        # Note this.
        print("File '%s.i.tri' already exists; aborting intersect."%proj)
        return
    # Set file names
    rc.set_intersect_i('%s.tri' % proj)
    rc.set_intersect_o('%s.o.tri' % proj) 
    # Run intersect.
    bin.intersect(opts=rc)
    # Read the original triangulation.
    tric = Tri('%s.c.tri' % proj)
    # Read the intersected triangulation.
    trii = Tri('%s.o.tri' % proj)
    # Read the pre-intersection triangulation.
    tri0 = Tri('%s.tri' % proj)
    # Map the Component IDs.
    trii.MapCompID(tric, tri0)
    # Write the triangulation.
    trii.Write('%s.i.tri' % proj)
    
# Function to verify if requested
def CaseVerify(rc, proj='Components', n=0):
    """Run ``verify`` to check triangulation if appropriate
    
    :Call:
        >>> CaseVerify(rc, proj='Components', n=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *proj*: {``'Components'``} | :class:`str`
            Project root name
        *n*: :class:`int`
            Iteration number
    :Versions:
        * 2015-09-07 ``@ddalle``: Split from :func:`run_flowCart`
        * 2016-04-05 ``@ddalle``: Generalized to :mod:`cape`
    """
    # Check for verify
    if not rc.get_verify(): return
    # Check for initial run
    if n > 0: return
    # Set file name
    rc.set_verify_i('%s.i.tri' % proj)
    # Run it.
    bin.verify(opts=rc)
    
# Mesh generation
def CaseAFLR3(rc, proj='Components', fmt='b8.ugrid', n=0):
    """Create volume mesh using ``aflr3``
    
    :Call:
        >>> CaseAFLR3(rc, proj="Components", fmt='b8.ugrid', n=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *proj*: {``"Components"``} | :class:`str`
            Project root name
        *fmt*: {``"b8.ugrid"``} | :class:`str`
            AFLR3 volume mesh format
        *n*: :class:`int`
            Iteration number
    :Versions:
        * 2016-04-05 ``@ddalle``: First version
    """
    # Check for option to run AFLR3
    if not rc.get_aflr3(): return
    # Check for initial run
    if n > 0: return
    # File names
    ftri  = '%s.i.tri'   % proj
    fsurf = '%s.surf'    % proj
    fbc   = '%s.aflr3bc' % proj
    fvol  = '%s.%s'      % (proj, fmt)
    # Check for file availability
    if not os.path.isfile(fsurf):
        # Check for the triangulation to provide a nice error message if app.
        if not os.path.isfile(ftri):
            raise ValueError("User has requested AFLR3 volume mesh.\n" +
                ("But found neither Cart3D tri file '%s' " % ftri) +
                ("nor AFLR3 surf file '%s'" % fsurf))
        # Read the triangulation
        tri = Tri(ftri)
        # Check for boundary condition flags
        if os.path.isfile(fbc):
            tri.ReadBCs_AFLR3(fbc)
        # Write the surface file
        tri.WriteSurf(fsurf)
    # Set file names
    rc.set_aflr3_i(fsurf)
    rc.set_aflr3_o(fvol)
    # Run AFLR3
    bin.aflr3(opts=rc)
    
def GetRestartIter():
    print("poop!")
    raise IOError()
    

# Function to call script or submit.
def StartCase():
    """Empty template for starting a case
    
    The function is empty 
    
    :Call:
        >>> cape.case.StartCase()
    :Versions:
        * 2015-09-27 ``@ddalle``: Skeleton
    """
    # Get the config.
    fc = ReadCaseJSON()
    # Determine the run index.
    i = GetInputNumber(fc)
    # Check qsub status.
    if fc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        pass
        
# Function to delete job and remove running file.
def StopCase():
    """Stop a case by deleting its PBS job and removing :file:`RUNNING` file
    
    :Call:
        >>> case.StopCase()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Get the job number.
    jobID = queue.pqjob()
    # Try to delete it.
    queue.qdel(jobID)
    # Check if the RUNNING file exists.
    if os.path.isfile('RUNNING'):
        # Delete it.
        os.remove('RUNNING')
    

# Function to read the local settings file.
def ReadCaseJSON():
    """Read Cape settings for local case
    
    :Call:
        >>> rc = cape.case.ReadCaseJSON()
    :Outputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Options interface for run control and command-line inputs
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
    """
    # Read the file, fail if not present.
    f = open('case.json')
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a Cape options object.
    fc = RunControl(**opts)
    # Output
    return fc
    
# Function to set the environment
def PrepareEnvironment(rc, i=0):
    """Set environment variables and alter any resource limits (``ulimit``)
    
    :Call:
        >>> case.PrepareEnvironment(rc, i=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Options interface for run control and command-line inputs
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2015-11-10 ``@ddalle``: First version
    """
    # Loop through environment variables.
    for key in rc.get('Environ', {}):
        # Get the environment variable
        val = rc.get_Environ(key, i)
        # Check if it stars with "+"
        if val.startswith("+"):
            # Check if it's present
            if key in os.environ:
                # Append to path
                os.environ[key] += (os.path.pathsep + val.lstrip('+'))
            else:
                # New variable
                os.environ[key] = val
        else:
            # Set the environment variable.
            os.environ[key] = val
    # Get ulimit parameters
    ulim = rc['ulimit']
    # Block size
    block = resource.getpagesize()
    # Set the stack size
    SetResourceLimit(resource.RLIMIT_STACK,   ulim, 's', i, 1024)
    SetResourceLimit(resource.RLIMIT_CORE,    ulim, 'c', i, block)
    SetResourceLimit(resource.RLIMIT_DATA,    ulim, 'd', i, 1024)
    SetResourceLimit(resource.RLIMIT_FSIZE,   ulim, 'f', i, block)
    SetResourceLimit(resource.RLIMIT_MEMLOCK, ulim, 'l', i, 1024)
    SetResourceLimit(resource.RLIMIT_NOFILE,  ulim, 'n', i, 1)
    SetResourceLimit(resource.RLIMIT_CPU,     ulim, 't', i, 1)
    SetResourceLimit(resource.RLIMIT_NPROC,   ulim, 'u', i, 1)
    
    
# Set resource limit
def SetResourceLimit(r, ulim, u, i=0, unit=1024):
    """Set resource limit
    
    :Call:
        >>> SetResourceLimit(r, ulim, u, i=0, unit=1024)
    :Inputs:
        *r*: :class:`int`
            Integer code of particular limit, usually from :mod:`resource`
        *ulim*: :class:`cape.options.ulimit.ulimit`
            System resource options interface
        *u*: :class:`str`
            Name of limit to set
        *i*: :class:`int`
            Phase number
        *unit*: :class:`int`
            Multiplier, usually for a kbyte
    :Versions:
        * 2016-03-13 ``@ddalle``: First version
    """
    # Check if the limit has been set
    if u not in ulim: return
    # Get the value of the limit
    l = ulim.get_ulimit(u, i)
    # Get the type of the input
    t = type(l).__name__
    # Check the type
    if t in ['int', 'float'] and l > 0:
        # Set the value numerically
        try:
            resource.setrlimit(r, (unit*l, unit*l))
        except ValueError:
            pass
    else:
        # Set unlimited
        resource.setrlimit(r, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    
# Function to get most recent L1 residual
def GetCurrentIter():
    """
    Skeleton function to report the current most recent iteration for which
    force and moment data can be found

    :Call:
        >>> n = cape.case.GetCurrentIter()
    :Outputs:
        *n*: ``0``
            Most recent index, customized for each solver
    :Versions:
        * 2015-09-27 ``@ddalle``: First version
    """
    return 0
    
# Write time used
def WriteUserTimeProg(tic, rc, i, fname, prog):
    """Write time usage since time *tic* to file
    
    :Call:
        >>> toc = WriteUserTime(tic, rc, i, fname, prog)
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: :class:`str`
            Name of file containing CPU usage history
        *prog*: :class:`str`
            Name of program to write in history
    :Outputs:
        *toc*: :class:`datetime.datetime`
            Time at which time delta was measured
    :Versions:
        * 2015-12-09 ``@ddalle``: First version
        * 2015-12-22 ``@ddalle``: Copied from :mod:`pyCart.case`
    """
    # Check if the file exists
    if not os.path.isfile(fname):
        # Create it.
        f = open(fname, 'w')
        # Write header line
        f.write("# TotalCPUHours, nProc, program, date, PBS job ID\n")
    else:
        # Append to the file
        f = open(fname, 'a')
    # Check for job ID
    if rc.get_qsub(i):
        try:
            # Try to read it and convert to integer
            jobID = open('jobID.dat').readline().split()[0]
        except Exception:
            jobID = ''
    else:
        # No job ID
        jobID = ''
    # Get the time.
    toc = datetime.now()
    # Time difference
    t = toc - tic
    # Number of processors
    nProc = rc.get_nProc(i)
    # Calculate CPU hours
    CPU = nProc * (t.days*24 + t.seconds/3600.0)
    # Write the data.
    f.write('%8.2f, %4i, %-20s, %s, %s\n' % (CPU, nProc, prog,
        toc.strftime('%Y-%m-%d %H:%M:%S %Z'), jobID))
    # Cleanup
    f.close()
    
