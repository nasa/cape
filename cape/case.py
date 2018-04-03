"""
:mod:`cape.case`: Case Control Module
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
from .options.runControl import RunControl
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
import numpy as np
from numpy import nan, isnan, argmax


# Function to intersect geometry if appropriate
def CaseIntersect(rc, proj='Components', n=0, fpre='run'):
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
        * ``Components.c.tri``: Intersecting triangulation, original compIDs
        * ``Components.o.tri``: Output of ``intersect``, only a few compIDs
        * ``Components.i.tri``: Original compIDs mapped onto intersected tris
        
    More specifically, these files are ``"%s.i.tri" % proj``, etc.; the default
    project name is ``"Components"``.  This function also calls the Chimera
    Grid Tools program ``triged`` to remove unused nodes from the intersected
    triangulation and optionally remove small triangles.
    
    :Call:
        >>> CaseIntersect(rc, proj='Components', n=0, fpre='run')
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *proj*: {``'Components'``} | :class:`str`
            Project root name
        *n*: :class:`int`
            Iteration number
        *fpre*: {``'run'``} | :class:`str`
            Standard output file name prefix
    :See also:
        * :class:`cape.tri.Tri`
        * :func:`cape.bin.intersect`
    :Versions:
        * 2015-09-07 ``@ddalle``: Split from :func:`run_flowCart`
        * 2016-04-05 ``@ddalle``: Generalized to :mod:`cape`
    """
    # Check for phase number
    j = GetPhaseNumber(rc, n, fpre=fpre)
    # Exit if not phase zero
    if j > 0: return
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
    # Name of farfield/source tri (if any)
    ftrif = '%s.f.tri' % proj
    # Read it
    if os.path.isfile(ftrif):
        # Read the farfield, sources, and other non-intersected surfaces
        trif = Tri(ftrif)
        # Add it to the mapped triangulation
        trii.AddRawCompID(trif)
    # Names of intermediate steps
    fatri = '%s.a.tri' % proj
    futri = '%s.u.tri' % proj
    fitri = '%s.i.tri' % proj
    # Write the triangulation.
    trii.Write(fatri)
    # Remove unused nodes
    infix = "RemoveUnusedNodes"
    fi = open('triged.%s.i' % infix, 'w')
    # Write inputs to the file
    fi.write('%s\n' % fatri)
    fi.write('10\n')
    fi.write('%s\n' % futri)
    fi.write('1\n')
    fi.close()
    # Run triged to remove unused nodes
    print(" > triged < triged.%s.i > triged.%s.o" % (infix, infix))
    os.system("triged < triged.%s.i > triged.%s.o" % (infix, infix))
    # Check options
    if rc.get_intersect_rm():
        # Input file to remove small tris
        infix = "RemoveSmallTris"
        fi = open('triged.%s.i' % infix, 'w')
        # Write inputs to file
        fi.write('%s\n' % futri)
        fi.write('19\n')
        fi.write('%f\n' % rc.get("SmallArea", rc.get_intersect_smalltri()))
        fi.write('%s\n' % fitri)
        fi.write('1\n')
        fi.close()
        # Run triged to remove small tris
        print(" > triged < triged.%s.i > triged.%s.o" % (infix, infix))
        os.system("triged < triged.%s.i > triged.%s.o" % (infix, infix))
    else:
        # Rename file
        os.rename(futri, fitri)
    # Clean up
    if os.path.isfile(fitri):
        if os.path.isfile(fatri): os.remove(fatri)
        if os.path.isfile(futri): os.remove(futri)

    
# Function to verify if requested
def CaseVerify(rc, proj='Components', n=0, fpre='run'):
    """Run ``verify`` to check triangulation if appropriate
    
    This function checks the validity of triangulation in file 
    ``"%s.i.tri" % proj``.  It calls :func:`cape.bin.verify`.
    
    :Call:
        >>> CaseVerify(rc, proj='Components', n=0, fpre='run')
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *proj*: {``'Components'``} | :class:`str`
            Project root name
        *n*: :class:`int`
            Iteration number
        *fpre*: {``'run'``} | :class:`str`
            Standard output file name prefix
    :Versions:
        * 2015-09-07 ``@ddalle``: Split from :func:`run_flowCart`
        * 2016-04-05 ``@ddalle``: Generalized to :mod:`cape`
    """
    # Check for phase number
    j = GetPhaseNumber(rc, n, fpre=fpre)
    # Exit if not phase zero
    if j > 0: return
    # Check for verify
    if not rc.get_verify(): return
    # Check for initial run
    if n > 0: return
    # Set file name
    rc.set_verify_i('%s.i.tri' % proj)
    # Run it.
    bin.verify(opts=rc)
    
# Mesh generation
def CaseAFLR3(rc, proj='Components', fmt='lb8.ugrid', n=0):
    """Create volume mesh using ``aflr3``
    
    This function looks for several files to determine the most appropriate
    actions to take, replacing ``Components`` with the value from *proj* for
    each file name and ``lb8.ugrid`` with the value from *fmt*:
    
        * ``Components.i.tri``: Triangulation file
        * ``Components.surf``: AFLR3 surface file
        * ``Components.aflr3bc``: AFLR3 boundary conditions
        * ``Components.xml``: Surface component ID mapping file
        * ``Components.lb8.ugrid``: Output volume mesh
        * ``Components.FAIL.surf``: AFLR3 surface indicating failure
        
    If the volume grid file already exists, this function takes no action.  If
    the ``surf`` file does not exist, the function attempts to create it by
    reading the ``tri``, ``xml``, and ``aflr3bc`` files using
    :class:`cape.tri.Tri`.  The function then calls :func:`cape.bin.aflr3` and
    finally checks for the ``FAIL`` file.
    
    :Call:
        >>> CaseAFLR3(rc, proj="Components", fmt='lb8.ugrid', n=0)
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
    fxml  = '%s.xml'     % proj
    fvol  = '%s.%s'      % (proj, fmt)
    ffail = "%s.FAIL.surf" % proj
    # Exit if volume exists
    if os.path.isfile(fvol): return
    # Check for file availability
    if not os.path.isfile(fsurf):
        # Check for the triangulation to provide a nice error message if app.
        if not os.path.isfile(ftri):
            raise ValueError("User has requested AFLR3 volume mesh.\n" +
                ("But found neither Cart3D tri file '%s' " % ftri) +
                ("nor AFLR3 surf file '%s'" % fsurf))
        # Read the triangulation
        if os.path.isfile(fxml):
            # Read with configuration
            tri = Tri(ftri, c=fxml)
        else:
            # Read without config
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
    # Check for failure; aflr3 returns 0 status even on failure
    if os.path.isfile(ffail):
        # Remove RUNNING file
        if os.path.isfile("RUNNING"):
            os.remove("RUNNING")
        # Create failure file
        f = open('FAIL', 'w')
        f.write("aflr3\n")
        f.close()
        # Error message
        raise RuntimeError("Failure during AFLR3 run:\n" +
            ("File '%s' exists." % ffail))
   
# Function for the most recent available restart iteration
def GetRestartIter():
    """Get the restart iteration

    This is a placeholder function and is only called in error.

    :Call:
        >>> cape.case.GetRestartIter()
    :Raises:
        *RuntimeError*: :class:`Exception`
            Error regarding where this was called
    :Versions:
        * 2016-04-14 ``@ddalle``: First version
    """
    raise IOError("Called cape.GetRestartIter()")

# Function to call script or submit.
def StartCase():
    """Empty template for starting a case
    
    The function is empty but does not raise an error
    
    :Call:
        >>> cape.case.StartCase()
    :See also:
        * :func:`pyCart.case.StartCase`
        * :func:`pyFun.case.StartCase`
        * :func:`pyOver.case.StartCase`
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
def ReadCaseJSON(fjson='case.json'):
    """Read Cape settings for local case
    
    :Call:
        >>> rc = cape.case.ReadCaseJSON()
    :Inputs:
        *fjson*: {``"case.json"``} | :class:`str`
            Name of JSON settings file
    :Outputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Options interface for run control and command-line inputs
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
    """
    # Read the file, fail if not present.
    f = open(fjson)
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a Cape options object.
    fc = RunControl(**opts)
    # Output
    return fc
    
# Read variable from conditions file
def ReadConditions(k=None):
    """Read run matrix variable value in the current folder
    
    :Call:
        >>> conds = cape.case.ReadConditions()
        >>> v = cape.case.ReadConditions(k)
    :Inputs:
        *k*: :class:`str`
            Name of run matrix variable/trajectory key
    :Outputs:
        *conds*: :class:`dict` (:class:`any`)
            Dictionary of run matrix conditions
        *v*: :class:`any`
            Run matrix conditions of key *k*
    :Versions:
        * 2017-03-28 ``@ddalle``: First version
    """
    # Read the file
    try:
        # Open the file
        f = open('conditions.json')
        # REad the settings
        conds = json.load(f)
        # Close the file
        f.close()
    except Exception:
        return None
    # Check for trajectory key
    if k is None:
        # Return full set
        return conds
    else:
        # Return the trajectory value
        return conds.get(k)
    
# Function to set the environment
def PrepareEnvironment(rc, i=0):
    """Set environment variables and alter any resource limits (``ulimit``)
    
    This function relies on the system module :mod:`resource`.
    
    :Call:
        >>> case.PrepareEnvironment(rc, i=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Options interface for run control and command-line inputs
        *i*: :class:`int`
            Phase number
    :See also:
        * :func:`SetResourceLimit`
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
    """Set resource limit for one variable
    
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
    :See also:
        * :mod:`cape.options.ulimit`
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

# Function to chose the correct input to use from the sequence.
def GetPhaseNumber(rc, n=None, fpre='run'):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> j = cape.case.GetPhaseNumber(rc, n=None, fpre='run')
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Options interface for run control
        *n*: :class:`int`
            Iteration number
        *fpre*: {``"run"``} | :class:`str`
            Prefix for output files
    :Outputs:
        *j*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
        * 2016-04-14 ``@ddalle``: Cape version
    """
    # Loop through possible input numbers.
    for i in range(rc.get_nSeq()):
        # Get the actual run number
        j = rc.get_PhaseSequence(i)
        # Check for output files.
        if len(glob.glob('%s.%02i.*' % (fpre, j))) == 0:
            # This run has not been completed yet.
            return j
        # Check the iteration number.
        if n < rc.get_PhaseIters(i):
            # This case has been run, but hasn't reached the min iter cutoff
            return j
    # Case completed; just return the last value.
    return j
    
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
    
# Write current time use
def WriteStartTimeProg(tic, rc, i, fname, prog):
    """Write the time to file at which a program or job started
    
    :Call:
        >>> WriteStartTimeProg(tic, rc, i, fname, prog)
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
    :Versions:
        * 2016-08-30 ``@ddalle``: First version
    """
    # Check if the file exists
    if not os.path.isfile(fname):
        # Create it.
        f = open(fname, 'w')
        # Write header line
        f.write("# nProc, program, date, PBS job ID\n")
    else:
        # Append to file
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
    # Number of processors
    nProc = rc.get_nProc(i)
    # Write the data.
    f.write('%4i, %-20s, %s, %s\n' % (nProc, prog,
        tic.strftime('%Y-%m-%d %H:%M:%S %Z'), jobID))
    # Cleanup
    f.close()
    
# Read most recent start time from file
def ReadStartTimeProg(fname):
    """Read the most recent start time to file
    
    :Call:
        >>> nProc, tic = ReadStartTimeProg(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file containing CPU usage history
    :Outputs:
        *nProc*: :class:`int`
            Number of cores
        *tic*: :class:`datetime.datetime`
            Time at which most recent run was started
    :Versions:
        * 2016-08-30 ``@ddalle``: First version
    """
    # Check for the file
    if not os.path.isfile(fname):
        # No time of start
        return None, None
    # Avoid failures
    try:
        # Read the last line and split on commas
        V = bin.tail(fname).split(',')
        # Get the number of processors
        nProc = int(V[0])
        # Split date and time
        dtxt, ttxt = V[2].strip().split()
        # Get year, month, day
        year, month, day = [int(v) for v in dtxt.split('-')]
        # Get hour, minute, second
        hour, minute, sec = [int(v) for v in ttxt.split(':')]
        # Construct date
        tic = datetime(year, month, day, hour, minute, sec)
        # Output
        return nProc, tic
    except Exception:
        # Fail softly
        return None, None
# WriteStartTimeProg

# Function to determine newest triangulation file
def GetTriqFile(proj='Components'):
    """Get most recent ``triq`` file and its associated iterations
    
    This is a template version with specific implementations for each solver.
    The :mod:`cape` version simply returns the most recent ``triq`` file in the
    current folder with no iteration information.
    
    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile(proj='Components')
    :Inputs:
        *proj*: {``"Components"``} | :class:`str`
            File root name
    :Outputs:
        *ftriq*: :class:`str`
            Name of most recently modified ``triq`` file
        *n*: {``None``}
            Number of iterations included
        *i0*: {``None``}
            First iteration in the averaging
        *i1*: {``None``}
            Last iteration in the averaging
    :Versions:
        * 2016-12-19 ``@ddalle``: First version
    """
    # Get the glob of numbered files.
    fglob = glob.glob('*.triq')
    # Check it.
    if len(fglob) > 0:
        # Get modification times
        t = [os.path.getmtime(f) for f in fglob]
        # Extract file with maximum index
        ftriq = fglob[t.index(max(t))]
        # Output
        return ftriq, None, None, None
    else:
        # No TRIQ files
        return None, None, None, None
# GetTriqFile

