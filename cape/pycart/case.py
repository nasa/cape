"""
:mod:`cape.pycart.case`: Case Control Module
=============================================

This module contains the important function :func:`case.run_flowCart`, which
actually runs ``flowCart`` or ``aero.csh``, along with the utilities that
support it.

For instance, it contains function to determine how many iterations have been
run, what the working folder is (e.g. ``.``, ``adapt00``, etc.), and what
command-line options to run.

It also contains Cart3D-specific versions of some of the generic methods from
:mod:`cape.case`.  All of the functions in that module are also available here.

"""

# Standard library modules
import glob
import json
import os
import re
import resource
import shutil
import sys

# Third-party modules
import numpy as np

# Standard library direct imports
from datetime import datetime

# Template class
import cape.cfdx.case as cc

# Direct CAPE imports
from cape.cfdx.case import CaseIntersect, CaseVerify

# Direct local imports
from .tri import Tri, Triq
from .options.runControl import RunControl

# Local modules
from . import cmd
from . import manage
from . import bin
from . import pointSensor
from .. import argread
from .. import text as textutils
from ..cfdx import queue


# Help message for CLI
HELP_RUN_FLOWCART = """
``run_flowCart.py``: Run Cart3D for one phase
================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:
    
    .. code-block:: console
    
        $ run_flowCart.py [OPTIONS]
        $ python -m cape.pycart run [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: Version 1.0
    * 2015-02-14 ``@ddalle``: Version 1.1; ``verify`` and ``intersect``
    * 2021-10-01 ``@ddalle``: Version 2.0; part of :mod:`case`
"""


# Function to setup and call the appropriate flowCart file.
def run_flowCart():
    r"""Setup and run ``flowCart``, ``mpi_flowCart`` command
    
    :Call:
        >>> run_flowCart()
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
        * 2014-12-18 ``@ddalle``: Version 1.1; Added :func:`TarAdapt`
        * 2021-10-08 ``@ddalle``: Version 1.2; removed args
    """
    # Parse arguments
    a, kw = argread.readkeys(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        print(textutils.markdown(HELP_RUN_FLOWCART))
        return
    # Check for RUNNING file.
    if os.path.isfile('RUNNING'):
        # Case already running
        raise SystemError('Case already running!')
    # Touch the running file.
    os.system('touch RUNNING')
    # Start timer
    tic = datetime.now()
    # Get the settings.
    rc = ReadCaseJSON()
    # Run intersect and verify
    cc.CaseIntersect(rc)
    cc.CaseVerify(rc)
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Write start time
    WriteStartTime(tic, rc, i)
    # Prepare all files
    PrepareFiles(rc, i)
    # Prepare environment variables (other than OMP_NUM_THREADS)
    cc.PrepareEnvironment(rc, i)
    # Run the appropriate commands
    RunPhase(rc, i)
    # Clean up the folder
    FinalizeFiles(rc, i)
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Save time usage
    WriteUserTime(tic, rc, i)
    # Check for bomb/early termination
    CheckSuccess(rc, i)
    # Run full restart command, including qsub if appropriate
    RestartCase(i)
    
    
# Write time used
def WriteUserTime(tic, rc, i, fname="pycart_time.dat"):
    """Write time usage since time *tic* to file
    
    :Call:
        >>> toc = WriteUserTime(tic, rc, i, fname="pycart_time.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: :class:`str`
            Name of file containing CPU usage history
    :Outputs:
        *toc*: :class:`datetime.datetime`
            Time at which time delta was measured
    :Versions:
        * 2015-12-09 ``@ddalle``: Version 1.0
    """
    # Call the function from :mode:`cape.case`
    cc.WriteUserTimeProg(tic, rc, i, fname, 'run_flowCart.py')
    
# Write start time
def WriteStartTime(tic, rc, i, fname="pycart_start.dat"):
    """Write the start time in *tic*
    
    :Call:
        >>> WriteStartTime(tic, rc, i, fname="pycart_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: {``"pycart_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2016-08-31 ``@ddalle``: Version 1.0
    """
    # Call the function from :mod:`cape.case`
    cc.WriteStartTimeProg(tic, rc, i, fname, 'run_flowCart.py')

# Run cubes if necessary
def CaseCubes(rc, j=0):
    """Run ``cubes`` and ``mgPrep`` to create multigrid volume mesh
    
    :Call:
        >>> CaseCubes(rc, j=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *j*: {``0``} | :class:`int`
            Phase number
    :Versions:
        * 2016-04-06 ``@ddalle``: Version 1.0
    """
    # Check for previous iterations
    # TODO: This will need an edit for 'remesh'
    if GetRestartIter() > 0: return
    # Check for mesh file
    if os.path.isfile('Mesh.mg.c3d'): return
    # Check for cubes option
    if not rc.get_cubes(): return
    # If adaptive, check for jumpstart
    if rc.get_Adaptive(j) and not rc.get_jumpstart(j): return
    # Run cubes
    bin.cubes(opts=rc, j=j)
    # Run mgPrep
    bin.mgPrep(opts=rc, j=j)
    
# Run autoInputs if appropriate
def CaseAutoInputs(rc, j=0):
    """Run ``autoInputs`` if necessary
    
    :Call:
        >>> CaseAutoInputs(rc)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``cape.json``
        *j*: {``0``} | :class:`int`
            Phase number
    :Versions:
        * 2016-04-06 ``@ddalle``: Version 1.0
    """
    # Check for previous iterations
    if GetRestartIter() > 0:
        return
    # Check for output files
    if os.path.isfile('input.c3d') and os.path.isfile('preSpec.c3d.cntl'):
        return
    # Check for cubes option
    if not rc.get_autoInputs():
        return
    # Run autoInputs
    bin.autoInputs(opts=rc, j=j)
    
# Prepare the files of the case
def PrepareFiles(rc, i=None):
    """Prepare file names appropriate to run phase *i* of Cart3D
    
    :Call:
        >>> PrepareFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Get the phase number if necessary
    if i is None:
        # Get the phase number.
        i = GetPhaseNumber(rc)
    # Create a restart file if appropriate.
    if not rc.get_Adaptive(i):
        # Automatically determine the best check file to use.
        SetRestartIter()
    # Delete any input file.
    if os.path.isfile('input.cntl') or os.path.islink('input.cntl'):
        os.remove('input.cntl')
    # Create the correct input file.
    os.symlink('input.%02i.cntl' % i, 'input.cntl')
    # Extra prep for adaptive --> non-adaptive
    if (i>0) and (not rc.get_Adaptive(i)) and (os.path.isdir('BEST')
            and (not os.path.isfile('history.dat'))):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Find all *.dat files and Mesh files
        fglob = glob.glob('*.dat') + glob.glob('Mesh.*')
        # Go back up one folder.
        os.chdir('..')
        # Copy all the important files.
        for fname in fglob:
            # Check for the file.
            if os.path.isfile(fname): continue
            # Copy the file.
            shutil.copy(os.path.join('BEST',fname), fname)
    # Convince aero.csh to use the *new* input.cntl
    if (i>0) and (rc.get_Adaptive(i)) and (rc.get_Adaptive(i-1)):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Check for an input.cntl file
        if os.path.isfile('input.cntl'):
            # Move it to a representative name.
            os.rename('input.cntl', 'input.%02i.cntl' % (i-1))
        # Go back up.
        os.chdir('..')
        # Copy the new input file.
        shutil.copy('input.%02i.cntl' % i, 'BEST/input.cntl')
    # Get rid of linked Tecplot files
    if os.path.islink('Components.i.plt'): os.remove('Components.i.plt')
    if os.path.islink('Components.i.dat'): os.remove('Components.i.dat')
    if os.path.islink('cutPlanes.plt'):    os.remove('cutPlanes.plt')
    if os.path.islink('cutPlanes.dat'):    os.remove('cutPlanes.dat')

# Run one phase appropriately
def RunPhase(rc, i):
    """Run one phase using appropriate commands
    
    :Call:
        >>> RunPhase(rc, i)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Mesh generation
    CaseAutoInputs(rc, i)
    CaseCubes(rc, i)
    # Check for flowCart vs. mpi_flowCart
    if not rc.get_MPI(i):
        # Get the number of threads, which may be irrelevant.
        nProc = rc.get_nProc(i)
        # Set it.
        os.environ['OMP_NUM_THREADS'] = str(nProc)
    # Check for adaptive runs.
    if rc.get_Adaptive(i):
        # Run 'aero.csh'
        RunAdaptive(rc, i)
    elif rc.get_it_avg(i):
        # Run a few iterations at a time
        RunWithRestarts(rc, i)
    else:
        # Run with the nominal inputs
        RunFixed(rc, i)

# Run one phase adaptively
def RunAdaptive(rc, i):
    """Run one phase using adaptive commands
    
    :Call:
        >>> RunAdaptive(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Delete the existing aero.csh file
    if os.path.islink('aero.csh'): os.remove('aero.csh')
    # Create a link to this run.
    os.symlink('aero.%02i.csh' % i, 'aero.csh')
    # Call the aero.csh command
    if i > 0 or GetCurrentIter() > 0:
        # Restart case.
        cmdi = ['./aero.csh', 'restart']
    elif rc.get_jumpstart():
        # Initial case
        cmdi = ['./aero.csh', 'jumpstart']
    else:
        # Initial case and create grid
        cmdi = ['./aero.csh']
    # Verbosity option
    v_fc = rc.get_Verbose()
    # Run the command.
    bin.callf(cmdi, f='flowCart.out', v=v_fc)
    # Check for point sensors
    if os.path.isfile(os.path.join('BEST', 'pointSensors.dat')):
        # Collect point sensor data
        PS = pointSensor.CasePointSensor()
        PS.UpdateIterations()
        PS.WriteHist()

# Run one phase with *it_avg*
def RunWithRestarts(rc, i):
    """Run ``flowCart`` a few iterations at a time for averaging purposes
    
    :Call:
        >>> RunWithRestarts(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Check how many iterations by which to offset the count.
    if rc.get_unsteady(i):
        # Get the number of previous unsteady steps.
        n = GetUnsteadyIter()
    else:
        # Get the number of previous steady steps.
        n = GetSteadyIter()
    # Initialize triq.
    if rc.get_clic(i): triq = Triq('Components.i.tri', n=0)
    # Initialize point sensor
    PS = pointSensor.CasePointSensor()
    # Requested iterations
    it_fc = rc.get_it_fc(i)
    # Start and end iterations
    n0 = n
    n1 = n + it_fc
    # Get verbose option
    v_fc = rc.get_Verbose()
    # Loop through iterations.
    for j in range(it_fc):
        # flowCart command automatically accepts *it_avg*; update *n*
        if j==0 and rc.get_it_start(i)>0:
            # Save settings.
            it_avg = rc.get_it_avg()
            # Startup iterations
            rc.set_it_avg(rc.get_it_start(i))
            # Increase reference for averaging.
            n0 += rc.get_it_start(i)
            # Modified command
            cmdi = cmd.flowCart(fc=rc, i=i, n=n)
            # Reset averaging settings
            rc.set_it_avg(it_avg)
        else:
            # Normal stops every *it_avg* iterations.
            cmdi = cmd.flowCart(fc=rc, i=i, n=n)
        # Run the command for *it_avg* iterations.
        bin.callf(cmdi, f='flowCart.out', v=v_fc)
        # Automatically determine the best check file to use.
        SetRestartIter()
        # Get new iteration count.
        if rc.get_unsteady(i):
            # Get the number of previous unsteady steps.
            n = GetUnsteadyIter()
        else:
            # Get the number of previous steady steps.
            n = GetSteadyIter()
        # Process triq files
        if rc.get_clic(i):
            # Read the triq file
            triqj = Triq('Components.i.triq')
            # Weighted average
            triq.WeightedAverage(triqj)
        # Update history
        PS.UpdateIterations()
        # Check for completion
        if (n>=n1) or (j+1==it_fc): break
        # Clear check files as appropriate.
        manage.ClearCheck_iStart(nkeep=1, istart=n0)
    # Write the averaged triq file
    if rc.get_clic(i):
        triq.Write('Components.%i.%i.%i.triq' % (j+1, n0, n))
    # Write the point sensor history file.
    try:
        if PS.nIter > 0:
            PS.WriteHist()
    except Exception: pass

# Run the nominal mode
def RunFixed(rc, i):
    """Run ``flowCart`` the nominal way
    
    :Call:
        >>> RunFixed(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Check how many iterations by which to offset the count.
    if rc.get_unsteady(i):
        # Get the number of previous unsteady steps.
        n = GetUnsteadyIter()
    else:
        # Get the number of previous steady steps.
        n = GetSteadyIter()
    # Get verbosity option
    v_fc = rc.get_Verbose()
    # Call flowCart directly.
    cmdi = cmd.flowCart(fc=rc, i=i, n=n)
    # Run the command.
    bin.callf(cmdi, f='flowCart.out', v=v_fc)
    # Check for point sensors
    if os.path.isfile('pointSensors.dat'):
        # Collect point sensor data
        PS = pointSensor.CasePointSensor()
        PS.UpdateIterations()
        PS.WriteHist()
            
# Check if a case was run successfully
def CheckSuccess(rc=None, i=None):
    """Check iteration counts and residual change for most recent run
    
    :Call:
        >>> CheckSuccess(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Last reported iteration number
    n = GetHistoryIter()
    # Check status
    if n % 1 != 0:
        # Ended with a failed unsteady cycle!
        f = open('FAIL', 'w')
        # Write the failure type.
        f.write('# Ended with failed unsteady cycle at iteration:\n')
        f.write('%13.6f\n' % n)
        # Quit
        f.close()
        raise SystemError("Failed unsteady cycle at iteration %.3f" % n)
    # First and last reported residual
    L1i = GetFirstResid()
    L1f = GetCurrentResid()
    # Check for bad (large or NaN) values.
    if np.isnan(L1f) or L1f/(0.1+L1i)>1.0e+6:
        # Exploded.
        f = open('FAIL', 'w')
        # Write the failure type.
        f.write('# Bombed at iteration %.6f with residual %.2E.\n' % (n, L1f))
        f.write('%13.6f\n' % n)
        # Quit
        f.close()
        raise SystemError("Bombed at iteration %s with residual %.2E" %
            (n, L1f))
    # Check for a hard-to-detect failure present in the output file.
    if CheckFailed():
        # Some other failure
        f = open('FAIL', 'w')
        # Copy the last line of flowCart.out
        f.write('# %s' % bin.tail('flowCart.out'))
        # Quit
        f.close()
        raise SystemError("flowCart failed to exit properly")
    
# Clean up immediately after running
def FinalizeFiles(rc, i=None):
    """Clean up files names after running one cycle of phase *i*
    
    :Call:
        >>> FinalizeFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Get the phase number if necessary
    if i is None:
        # Get the phase number.
        i = GetPhaseNumber(rc)
    # Clean up the folder as appropriate.
    manage.ManageFilesProgress(rc)
    # Tar visualization files.
    if rc.get_unsteady(i):
        manage.TarViz(rc)
    # Tar old adaptation folders.
    if rc.get_Adaptive(i):
        manage.TarAdapt(rc)
    # Get the new restart iteration.
    n = GetCheckResubIter()
    # Assuming that worked, move the temp output file.
    os.rename('flowCart.out', 'run.%02i.%i' % (i, n))
    # Check for TecPlot files to save.
    if os.path.isfile('cutPlanes.plt'):
        os.rename('cutPlanes.plt', 'cutPlanes.%05i.plt' % n)
    if os.path.isfile('Components.i.plt'):
        os.rename('Components.i.plt', 'Components.i.%05i.plt' % n)
    if os.path.isfile('cutPlanes.dat'):
        os.rename('cutPlanes.dat', 'cutPlanes.%05i.dat' % n)
    if os.path.isfile('Components.i.dat'):
        os.rename('Components.i.dat', 'Components.i.%05i.dat' % n)

# Function to call script or submit.
def StartCase():
    """Start a case by either submitting it or calling with a system command
    
    :Call:
        >>> pyCart.case.StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: Version 1.0
        * 2015-11-08 ``@ddalle``: Added resubmit/continue functionality
        * 2015-12-28 ``@ddalle``: Split :func:`RestartCase`
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Check qsub status.
    if rc.get_sbatch(i):
        # Get the name of the PBS file
        fpbs = GetPBSScript(i)
        # Submit the Slurm case
        pbs = queue.psbatch(fpbs)
        return pbs
    elif rc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Run the case.
        run_flowCart()
        
# Function to call script or submit.
def RestartCase(i0=None):
    """Restart a case by either submitting it or calling with a system command
    
    This version of the command is called within :func:`run_flowCart` after
    running a phase or attempting to run a phase.
    
    :Call:
        >>> pyCart.case.RetartCase(i0=None)
    :Inputs:
        *i0*: :class:`int` | ``None``
            Run sequence index of the previous run
    :Versions:
        * 2014-10-06 ``@ddalle``: Version 1.0
        * 2015-11-08 ``@ddalle``: Added resubmit/continue functionality
        * 2015-12-28 ``@ddalle``: Split from :func:`StartCase`
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Get the new restart iteration.
    n = GetCheckResubIter()
    # Task manager
    qpbs = rc.get_qsub(i)
    qslr = rc.get_sbatch(i)
    # Check current iteration count.
    if n >= rc.get_LastIter():
        return
    # Check qsub status.
    if not (qpbs or qslr):
        # Run the case.
        run_flowCart()
    elif rc.get_Resubmit(i):
        # Check for continuance
        if (i0 is None) or (i>i0) or (not rc.get_Continue(i)):
            # Get the name of the PBS file.
            fpbs = GetPBSScript(i)
            # Submit the case.
            if qslr:
                # Slurm
                pbs = queue.psbatch(fpbs)
            elif qpbs:
                # PBS
                pbs = queue.pqsub(fpbs)
            else:
                # No task manager
                raise NotImplementedError("Could not determine task manager")
            return pbs
        else:
            # Continue on the same job
            run_flowCart()
    else:
        # Simply run the case. Don't reset modules either.
        run_flowCart()
        
# Function to delete job and remove running file.
def StopCase():
    """Stop a case by deleting its PBS job and removing :file:`RUNNING` file
    
    :Call:
        >>> pyCart.case.StopCase()
    :Versions:
        * 2014-12-27 ``@ddalle``: Version 1.0
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Get the job number.
    jobID = queue.pqjob()
    # Try to delete it.
    if rc.get_sbatch(i):
        # Delete Slurm job
        queue.scancel(jobID)
    elif rc.get_qsub(i):
        # Delete PBS job
        queue.qdel(jobID)
    # Check if the RUNNING file exists.
    if os.path.isfile('RUNNING'):
        # Delete it.
        os.remove('RUNNING')
        
# Function to check output file for some kind of failure.
def CheckFailed():
    """Check the :file:`flowCart.out` file for a failure
    
    :Call:
        >>> q = pyCart.case.CheckFailed()
    :Outputs:
        *q*: :class:`bool`
            Whether or not the last line of `flowCart.out` contains 'fail'
    :Versions:
        * 2015-01-02 ``@ddalle``: Version 1.0
    """
    # Check for the file.
    if os.path.isfile('flowCart.out'):
        # Read the last line.
        if 'fail' in bin.tail('flowCart.out', 1):
            # This is a failure.
            return True
        else:
            # Normal completed run.
            return False
    else:
        # No flowCart.out file
        return False

# Function to determine which PBS script to call
def GetPBSScript(i=None):
    """Determine the file name of the PBS script to call
    
    This is a compatibility function for cases that do or do not have multiple
    PBS scripts in a single run directory
    
    :Call:
        >>> fpbs = pyCart.case.GetPBSScript(i=None)
    :Inputs:
        *i*: :class:`int`
            Phase number
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: Version 1.0
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if i is not None:
        # Create the name.
        fpbs = 'run_cart3d.%02i.pbs' % i
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return 'run_cart3d.pbs'
    else:
        # Do not search for numbered PBS script if *i* is None
        return 'run_cart3d.pbs'
    

# Function to read the local settings file.
def ReadCaseJSON():
    """Read `flowCart` settings for local case
    
    :Call:
        >>> rc = pyCart.case.ReadCaseJSON()
    :Outputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface for run
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
    """
    # Read the file, fail if not present.
    f = open('case.json')
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a RunControl object.
    rc = RunControl(**opts)
    # Output
    return rc
    

# Function to get the most recent check file.
def GetSteadyIter():
    """Get iteration number of most recent steady check file
    
    :Call:
        >>> n = pyCart.case.GetSteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
        * 2014-11-28 ``@ddalle``: Renamed from :func:`GetRestartIter`
    """
    # List the check.* files.
    fch = glob.glob('check.*[0-9]') + glob.glob('BEST/check.*')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname.split('.')[-1])
        # Use the running maximum.
        n = max(i, n)
    # Output
    return n
    
# Function to get the most recent time-domain check file.
def GetUnsteadyIter():
    """Get iteration number of most recent unsteady check file
    
    :Call:
        >>> n = pyCart.case.GetUnsteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: Version 1.0
    """
    # Check for td checkpoints
    fch = glob.glob('check.*.td')
    # Initialize unsteady count
    n = 0
    # Loop through matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname.split('.')[1])
        # Use the running maximum.
        n = max(i, n)
    # Output.
    return n
    
# Function to get total iteration number
def GetRestartIter():
    """Get total iteration number of most recent check file
    
    This is the sum of the most recent steady iteration and unsteady iteration.
    
    :Call:
        >>> n = pyCart.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: Version 1.0
    """
    # Get the unsteady iteration number based on available check files.
    ntd = GetUnsteadyIter()
    # Check for an unsteady iteration number.
    if ntd:
        # If there's an unsteady iteration, use that step directly.
        return ntd
    else:
        # Use the steady-state iteration number.
        return GetSteadyIter()
    
# Function to get total iteration number
def GetCheckResubIter():
    """Get total iteration number of most recent check file
    
    This is the sum of the most recent steady iteration and unsteady iteration.
    
    :Call:
        >>> n = pyCart.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: Version 1.0
        * 2014-11-29 ``@ddalle``: This was renamed from :func:`GetRestartIter`
    """
    # Get the two numbers
    nfc = GetSteadyIter()
    ntd = GetUnsteadyIter()
    # Output
    return nfc + ntd
    
    
# Function to set up most recent check file as restart.
def SetRestartIter(n=None, ntd=None):
    """Set a given check file as the restart point
    
    :Call:
        >>> pyCart.case.SetRestartIter(n=None, ntd=None)
    :Inputs:
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
        *ntd*: :class:`int`
            Unsteady iteration number
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
        * 2014-11-28 ``@ddalle``: Added time-accurate compatibility
    """
    # Check the input.
    if n   is None: n = GetSteadyIter()
    if ntd is None: ntd = GetUnsteadyIter()
    # Remove the current restart file if necessary.
    if os.path.isfile('Restart.file') or os.path.islink('Restart.file'):
        os.remove('Restart.file')
    # Quit if no check point.
    if n == 0 and ntd == 0:
        return None
    # Create a link to the most appropriate file.
    if os.path.isfile('check.%06i.td' % ntd):
        # Restart from time-accurate check point
        os.symlink('check.%06i.td' % ntd, 'Restart.file')
    elif os.path.isfile('BEST/check.%05i' % n):
        # Restart file in adaptive folder
        os.symlink('BEST/check.%05i' % n, 'Restart.file')
    elif os.path.isfile('check.%05i' % n):
        # Restart file in current folder
        os.symlink('check.%05i' % n, 'Restart.file')
    
    
# Function to chose the correct input to use from the sequence.
def GetPhaseNumber(rc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyCart.case.GetPhaseNumber(rc)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface for `flowCart`
    :Outputs:
        *i*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: Version 1.0
    """
    # Get the run index.
    n = GetCheckResubIter()
    # Loop through possible input numbers.
    for j in range(rc.get_nSeq()):
        # Get the actual run number
        i = rc.get_PhaseSequence(j)
        # Check for output files.
        if len(glob.glob('run.%02i.*' % i)) == 0:
            # This run has not been completed yet.
            return i
        # Check the iteration number.
        if n < rc.get_PhaseIters(j):
            # This case has been run, but hasn't reached the min iter cutoff
            return i
    # Case completed; just return the last value.
    return i
    
# Function to read last line of 'history.dat' file
def GetHistoryIter(fname='history.dat'):
    """Get the most recent iteration number from a :file:`history.dat` file
    
    :Call:
        >>> n = pyCart.case.GetHistoryIter(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`float`
            Last iteration number
    :Versions:
        * 2014-11-24 ``@ddalle``: Version 1.0
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return 0
    # Check the file.
    try:
        # Try to tail the last line.
        txt = bin.tail(fname)
        # Try to get the integer.
        return float(txt.split()[0])
    except Exception:
        # If any of that fails, return 0
        return 0
        
# Get last residual from 'history.dat' file
def GetHistoryResid(fname='history.dat'):
    """Get the last residual in a :file:`history.dat` file
    
    :Call:
        >>> L1 = pyCart.case.GetHistoryResid(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: Version 1.0
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return nan
    # Check the file.
    try:
        # Try to tail the last line.
        txt = bin.tail(fname)
        # Try to get the integer.
        return float(txt.split()[3])
    except Exception:
        # If any of that fails, return 0
        return nan
        

# Function to check if last line is unsteady
def CheckUnsteadyHistory(fname='history.dat'):
    """Check if the current history ends with an unsteady iteration

    :Call:
        >>> q = pyCart.case.CheckUnsteadyHistory(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *q*: :class:`float`
            Whether or not the last iteration of *fname* has a '.' in it
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return False
    # Check the file's contents.
    try:
        # Try to tail the last line.
        txt = bin.tail(fname)
        # Check for a dot.
        return ('.' in txt.split()[0])
    except Exception:
        # Something failed; invalid history
        return False

    
# Function to get the most recent working folder
def GetWorkingFolder():
    """Get the most recent working folder, either '.' or 'adapt??/'
    
    This function must be called from the top level of a case run directory.
    
    :Call:
        >>> fdir = pyCart.case.GetWorkingFolder()
    :Outputs:
        *fdir*: :class:`str`
            Name of the most recently used working folder with a history file
    :Versions:
        * 2014-11-24 ``@ddalle``: Version 1.0
    """
    # Try to get iteration number from working folder.
    n0 = GetCurrentIter()
    # Initialize working directory.
    fdir = '.'
    # Implementation of returning to adapt after startup turned off
    if os.path.isfile('history.dat') and not os.path.islink('history.dat'):
        return fdir
    # Check for adapt?? folders
    for fi in glob.glob('adapt??'):
        # Attempt to read it.
        ni = GetHistoryIter(os.path.join(fi, 'history.dat'))
        # Check it.
        if ni >= n0:
            # Current best estimate
            fdir = fi
    # Output
    return fdir
       
# Function to get most recent adaptive iteration
def GetCurrentResid():
    """Get the most recent iteration including unsaved progress

    Iteration numbers from time-accurate restarts are corrected to match the
    global iteration numbering.

    :Call:
        >>> L1 = pyCart.case.GetCurrentResid()
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: Version 1.0
    """
    # Get the working folder.
    fdir = GetWorkingFolder()
    # Get the residual.
    return GetHistoryResid(os.path.join(fdir, 'history.dat'))

# Function to get first recent adaptive iteration
def GetFirstResid():
    """Get the first iteration

    :Call:
        >>> L1 = pyCart.case.GetFirstResid()
    :Outputs:
        *L1*: :class:`float`
            First L1 residual
    :Versions:
        * 2015-07-22 ``@ddalle``: Version 1.0
    """
    # Get the working folder.
    fdir = GetWorkingFolder()
    # File name
    fname = os.path.join(fdir, 'history.dat')
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return nan
    # Check the file.
    try:
        # Try to open the file.
        f = open(fname, 'r')
        # Initialize line.
        txt = '#'
        # Read the lines until it's not a comment.
        while txt.startswith('#'):
            # Read the next line.
            txt = f.readline()
        # Try to get the integer.
        return float(txt.split()[3])
    except Exception:
        # If any of that fails, return 0
        return nan
    
# Function to get most recent L1 residual
def GetCurrentIter():
    """Get the residual of the most recent iteration including unsaved progress

    :Call:
        >>> n = pyCart.case.GetCurrentIter()
    :Outputs:
        *n*: :class:`int`
            Most recent index written to :file:`history.dat`
    :Versions:
        * 2014-11-28 ``@ddalle``: Version 1.0
    """
    # Try to get iteration number from working folder.
    ntd = GetHistoryIter()
    # Check it.
    if ntd and (not CheckUnsteadyHistory()):
        # Don't read adapt??/ history
        return ntd
    # Initialize adaptive iteration number
    n0 = 0
    # Check for adapt?? folders
    for fi in glob.glob('adapt??'):
        # Attempt to read it.
        ni = GetHistoryIter(os.path.join(fi, 'history.dat'))
        # Check it.
        if ni > n0:
            # Update best estimate.
            n0 = ni
    # Output the total.
    return n0 + ntd

# Function to determine newest triangulation file
def GetTriqFile():
    """Get most recent ``triq`` file and its associated iterations
    
    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile()
    :Outputs:
        *ftriq*: :class:`str`
            Name of ``triq`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2015-09-16 ``@ddalle``: Version 1.0
        * 2021-12-09 ``@ddalle``: Version 1.1
            - Check for ``adapt??/`` folder w/o ``triq`` file
    """
    # Find all possible TRIQ files
    triqglob0 = sorted(glob.glob("Components.*.triq"))
    triqglob1 = sorted(glob.glob("adapt??/Components.*.triq"))
    # Determine best folder
    if len(triqglob0) > 0:
        # Use parent folder
        fwrk = "."
    elif len(triqglob1) > 0:
        # Use latest adapt??/ folder
        fwrk = os.path.dirname(triqglob1[-1])
    else:
        # None available
        return None, None, None, None
    # Go to best folder
    fpwd = os.getcwd()
    os.chdir(fwrk)
    # Get the glob of numbered files.
    fglob3 = glob.glob('Components.*.*.*.triq')
    fglob2 = glob.glob('Components.*.*.triq')
    fglob1 = glob.glob('Components.[0-9]*.triq')
    # Check it.
    if len(fglob3) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[3]) for f in fglob3]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob3[j].split('.')[2])
        # Count
        n = int(fglob3[j].split('.')[1])
        # File name
        ftriq = fglob3[j]
    elif len(fglob2) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[2]) for f in fglob2]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob2[j].split('.')[1])
        # File name
        ftriq = fglob2[j]
    # Check it.
    elif len(fglob1) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[1]) for f in fglob1]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = I0[j]
        # Count
        n = i1 - i0 + 1
        # File name
        ftriq = fglob1[j]
    # Plain file
    elif os.path.isfile('Components.i.triq'):
        # Iteration counts: assume it's most recent iteration
        i1 = GetCurrentIter()
        i0 = i1
        # Count
        n = 1
        # file name
        ftriq = 'Components.i.triq'
    # Return to original location
    os.chdir(fpwd)
    # Prepend name of folder if appropriate
    if fwrk != '.' and ftriq is not None:
        ftriq = os.path.join(fwrk, ftriq)
    # Output
    return ftriq, n, i0, i1
    
# Link best file based on name and glob
def LinkFromGlob(fname, fglb, isplit=-2, csplit='.'):
    """Link the most recent file to a basic unmarked file name
    
    The function will attempt to map numbered or adapted file names using the
    most recent iteration or adaptation.  The following gives examples of links
    that could be created using ``Components.i.plt`` for *fname* and
    ``Components.[0-9]*.plt`` for *fglb*.
    
        * ``Components.i.plt`` (no link)
        * ``Components.01000.plt`` --> ``Components.i.plt``
        * ``adapt03/Components.i.plt`` --> ``Components.i.plt``
    
    :Call:
        >>> pyCart.case.LinkFromGlob(fname, fglb, isplit=-2, csplit='.')
    :Inputs:
        *fname*: :class:`str`
            Name of unmarked file, like ``Components.i.plt``
        *fglb*: :class:`str`
            Glob for marked file names
        *isplit*: :class:`int`
            Which value of ``f.split()`` to use to get index number
        *csplit*: :class:`str`
            Character on which to split to find indices, usually ``'.'``
    :Versions:
        * 2015-11-20 ``@ddalle``: Version 1.0
    """
    # Check for already-existing regular file.
    if os.path.isfile(fname) and not os.path.islink(fname): return
    # Remove the link if necessary.
    if os.path.isfile(fname) or os.path.islink(fname):
        os.remove(fname)
    # Get the working directory.
    fdir = GetWorkingFolder()
    # Check it.
    if fdir == '.':
        # List files that match the requested glob.
        fglob = glob.glob(fglb)
        # Check for empty glob.
        if len(fglob) == 0: return
        # Get indices from those files.
        n = [int(f.split(csplit)[isplit]) for f in fglob]
        # Extract file with maximum index.
        fsrc = fglob[n.index(max(n))]
    else:
        # File from the working folder (if it exists)
        fsrc = os.path.join(fdir, fname)
        # Check for the file.
        if not os.path.isfile(fsrc):
            # Get the adaptation number of the working folder
            nadapt = int(fdir[-2:])
            # Try the previous adaptation file.
            fdir = 'adapt%02i' % (nadapt-1)
            # Use that folder.
            fsrc = os.path.join(fdir, fname)
        # Check for the file again.
        if not os.path.isfile(fsrc): return
    # Create the link if possible
    if os.path.isfile(fsrc): os.symlink(fsrc, fname)
            
    
# Link best tecplot files
def LinkPLT():
    """Link the most recent Tecplot files to fixed file names
    
    Uses file names :file:`Components.i.plt` and :file:`cutPlanes.plt`
    
    :Call:
        >>> pyCart.case.LinkPLT()
    :Versions:
        * 2015-03-10 ``@ddalle``: Version 1.0
        * 2015-11-20 ``@ddalle``: Delegate work and support ``*.dat`` files
    """
    # Surface file
    if len(glob.glob('Components.i.[0-9]*.plt')) > 0:
        # Universal format; mpix_flowCart
        LinkFromGlob('Components.i.plt', 'Components.i.[0-9]*.plt', -2)
    elif len(glob.glob('Components.i.[0-9]*.dat')) > 0:
        # Universal format without -binaryIO
        LinkFromGlob('Components.i.dat', 'Components.i.[0-9]*.dat', -2)
    else:
        # Special pyCart format renamed from flowcart outputs
        LinkFromGlob('Components.i.plt', 'Components.[0-9]*.plt', -2)
        LinkFromGlob('Components.i.dat', 'Components.[0-9]*.dat', -2)
    # Cut planes
    LinkFromGlob('cutPlanes.plt',    'cutPlanes.[0-9]*.plt', -2)
    LinkFromGlob('cutPlanes.dat',    'cutPlanes.[0-9]*.dat', -2)
            
    
