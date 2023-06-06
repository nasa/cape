"""
:mod:`cape.pycart.case`: Case Control Module
=============================================

This module contains the important function :func:`case.run_flowCart`,
which actually runs ``flowCart`` or ``aero.csh``, along with the
utilities that support it.

For instance, it contains function to determine how many iterations have
been run, what the working folder is (e.g. ``.``, ``adapt00``, etc.),
and what command-line options to run.

It also contains Cart3D-specific versions of some of the generic methods
from :mod:`cape.case`.  All of the functions in that module are also
available here.

"""

# Standard library modules
import glob
import os
import shutil
import sys

# Third-party modules
import numpy as np


# Local imports
from ..cfdx import case as cc
from .tri import Triq
from .options.runctlopts import RunControlOpts
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
    * 2014-10-02 ``@ddalle``: v1.0
    * 2015-02-14 ``@ddalle``: v1.1; ``verify`` and ``intersect``
    * 2021-10-01 ``@ddalle``: v2.0; part of :mod:`case`
"""

# Maximum number of calls to run_phase()
NSTART_MAX = 1000


# Function to setup and call the appropriate flowCart file.
def run_flowCart():
    r"""Setup and run ``flowCart``, ``mpi_flowCart`` command

    :Call:
        >>> run_flowCart()
    :Versions:
        * 2014-10-02 ``@ddalle``: v1.0
        * 2014-12-18 ``@ddalle``: v1.1; Added :func:`TarAdapt`
        * 2021-10-08 ``@ddalle``: v1.2; removed args
    """
    # Parse arguments
    a, kw = argread.readkeys(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        print(textutils.markdown(HELP_RUN_FLOWCART))
        return cc.IERR_OK
    # Start RUNNING and timer (checks if already running)
    tic = cc.init_timer()
    # Get the run control settings
    rc = read_case_json()
    # Initialize FUN3D start counter
    nstart = 0
    # Loop until case complete, new job submitted, or timeout
    while nstart < NSTART_MAX:
        # Run intersect and verify
        cc.CaseIntersect(rc)
        cc.CaseVerify(rc)
        # Determine the run index.
        j = GetPhaseNumber(rc)
        # Write start time
        WriteStartTime(tic, rc, j)
        # Prepare all files
        PrepareFiles(rc, j)
        # Prepare environment variables (other than OMP_NUM_THREADS)
        cc.prepare_env(rc, j)
        # Run the appropriate commands
        try:
            run_phase(rc, j)
        except Exception:
            # Failure
            cc.mark_failure("run_phase")
            # Stop running marker
            cc.mark_stopped()
            # Return code
            return cc.IERR_RUN_PHASE
        # Clean up the folder
        FinalizeFiles(rc, j)
        # Save time usage
        WriteUserTime(tic, rc, j)
        # Check for bomb/early termination
        CheckSuccess(rc, j)
        # Update start counter
        nstart += 1
        # Check for explicit exit
        if check_complete(rc):
            break
        # Submit new PBS/Slurm job if appropriate
        q = resubmit_case(rc, j)
        # If new job started, this one should stop
        if q:
            break
    # Remove the RUNNING file
    cc.mark_stopped()
    # Return code
    return cc.IERR_OK


# Write time used
def WriteUserTime(tic, rc, i, fname="pycart_time.dat"):
    r"""Write time usage since time *tic* to file

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
        * 2015-12-09 ``@ddalle``: v1.0
    """
    # Call the function from :mode:`cape.case`
    cc.WriteUserTimeProg(tic, rc, i, fname, 'run_flowCart.py')


# Write start time
def WriteStartTime(tic, rc, i, fname="pycart_start.dat"):
    r"""Write the start time in *tic*

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
        * 2016-08-31 ``@ddalle``: v1.0
    """
    # Call the function from :mod:`cape.case`
    cc.WriteStartTimeProg(tic, rc, i, fname, 'run_flowCart.py')


# Run cubes if necessary
def CaseCubes(rc, j=0):
    r"""Run ``cubes`` and ``mgPrep`` to create multigrid volume mesh

    :Call:
        >>> CaseCubes(rc, j=0)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``case.json``
        *j*: {``0``} | :class:`int`
            Phase number
    :Versions:
        * 2016-04-06 ``@ddalle``: v1.0
    """
    # Check for previous iterations
    # TODO: This will need an edit for 'remesh'
    if GetRestartIter() > 0: return
    # Check for mesh file
    if os.path.isfile('Mesh.mg.c3d'): return
    # Check for cubes option
    if not rc.get_cubes_run():
        return
    # If adaptive, check for jumpstart
    if rc.get_Adaptive(j) and not rc.get_jumpstart(j): return
    # Run cubes
    bin.cubes(opts=rc, j=j)
    # Run mgPrep
    bin.mgPrep(opts=rc, j=j)


# Run autoInputs if appropriate
def CaseAutoInputs(rc, j=0):
    r"""Run ``autoInputs`` if necessary

    :Call:
        >>> CaseAutoInputs(rc)
    :Inputs:
        *rc*: :class:`cape.options.runControl.RunControl`
            Case options interface from ``cape.json``
        *j*: {``0``} | :class:`int`
            Phase number
    :Versions:
        * 2016-04-06 ``@ddalle``: v1.0
    """
    # Check for previous iterations
    if GetRestartIter() > 0:
        return
    # Check for output files
    if os.path.isfile('input.c3d') and os.path.isfile('preSpec.c3d.cntl'):
        return
    # Check for cubes option
    if not rc.get_autoInputs_run():
        return
    # Run autoInputs
    bin.autoInputs(opts=rc, j=j)


# Prepare the files of the case
def PrepareFiles(rc, i=None):
    r"""Prepare file names appropriate to run phase *i* of Cart3D

    :Call:
        >>> PrepareFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
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
    if (i > 0) and (not rc.get_Adaptive(i)) and (
            os.path.isdir('BEST') and (not os.path.isfile('history.dat'))):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Find all *.dat files and Mesh files
        fglob = glob.glob('*.dat') + glob.glob('Mesh.*')
        # Go back up one folder.
        os.chdir('..')
        # Copy all the important files.
        for fname in fglob:
            # Check for the file.
            if os.path.isfile(fname):
                continue
            # Copy the file.
            shutil.copy(os.path.join('BEST', fname), fname)
    # Convince aero.csh to use the *new* input.cntl
    if (i > 0) and (rc.get_Adaptive(i)) and (rc.get_Adaptive(i-1)):
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
def run_phase(rc, i):
    r"""Run one phase using appropriate commands

    :Call:
        >>> run_phase(rc, i)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0 (``RunPhase()``
        * 2023-06-02 ``@ddalle``: v1.1
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
    r"""Run one phase using adaptive commands

    :Call:
        >>> RunAdaptive(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
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
    r"""Run ``flowCart`` a few iters at a time for averaging purposes

    :Call:
        >>> RunWithRestarts(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
    """
    # Check how many iterations by which to offset the count.
    if rc.get_unsteady(i):
        # Get the number of previous unsteady steps.
        n = GetUnsteadyIter()
    else:
        # Get the number of previous steady steps.
        n = GetSteadyIter()
    # Initialize triq
    if rc.get_clic(i):
        triq = Triq('Components.i.tri', n=0)
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
        if j == 0 and rc.get_it_start(i) > 0:
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
        if (n >= n1) or (j + 1 == it_fc):
            break
        # Clear check files as appropriate.
        manage.ClearCheck_iStart(nkeep=1, istart=n0)
    # Write the averaged triq file
    if rc.get_clic(i):
        triq.Write('Components.%i.%i.%i.triq' % (j+1, n0, n))
    # Write the point sensor history file.
    try:
        if PS.nIter > 0:
            PS.WriteHist()
    except Exception:
        pass


# Run the nominal mode
def RunFixed(rc, i):
    r"""Run ``flowCart`` the nominal way

    :Call:
        >>> RunFixed(rc, i)
    :Inputs:
        *rc*: :Class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
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
    r"""Check iteration counts and residual change for most recent run

    :Call:
        >>> CheckSuccess(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
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
    if np.isnan(L1f) or L1f/(0.1+L1i) > 1.0e+6:
        # Exploded.
        f = open('FAIL', 'w')
        # Write the failure type.
        f.write('# Bombed at iteration %.6f with residual %.2E.\n' % (n, L1f))
        f.write('%13.6f\n' % n)
        # Quit
        f.close()
        raise SystemError(
            "Bombed at iteration %s with residual %.2E" %
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
    r"""Clean up files names after running one cycle of phase *i*

    :Call:
        >>> FinalizeFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-03-04 ``@ddalle``: v1.0
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
    r"""Start a case by either submitting it or calling with a system command

    :Call:
        >>> StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: v1.0
        * 2015-11-08 ``@ddalle``: Added resubmit/continue functionality
        * 2015-12-28 ``@ddalle``: Split :func:`RestartCase`
    """
    # Get the config.
    rc = read_case_json()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Check qsub status.
    if rc.get_slurm(i):
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


def resubmit_case(rc, j0):
    r"""Resubmit a case as a new job if appropriate

    :Call:
        >>> q = resubmit_case(rc, j0)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
        *j0*: :class:`int`
            Index of phase most recently run prior
            (may differ from :func:`get_phase` now)
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not a new job was submitted to queue
    :Versions:
        * 2022-01-20 ``@ddalle``: v1.0 (:mod:`cape.pykes.case`)
        * 2023-06-02 ``@ddalle``: v1.0
    """
    # Get *current* phase
    j1 = GetPhaseNumber(rc)
    # Get name of run script for next case
    fpbs = GetPBSScript(j1)
    # Call parent function
    return cc.resubmit_case(rc, fpbs, j0, j1)


def check_complete(rc):
    r"""Check if case is complete as described

    :Call:
        >>> q = check_complete(rc)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
    :Outputs:
        *q*: ``True`` | ``False``
            Whether case has reached last phase w/ enough iters
    :Versions:
        * 2023-06-02 ``@ddalle``: v1.0
    """
    # Determine current phase
    j = GetPhaseNumber(rc)
    # Check if last phase
    if j < rc.get_PhaseSequence(-1):
        return False
    # Get restart iteration
    n = GetCheckResubIter()
    # Check iteration number
    if n is None:
        # No iterations complete
        return False
    elif n < rc.get_LastIter():
        # Not enough iterations complete
        return False
    else:
        # All criteria met
        return True


# Function to delete job and remove running file.
def StopCase():
    r"""Stop a case by deleting its PBS job and removing :file:`RUNNING` file

    :Call:
        >>> StopCase()
    :Versions:
        * 2014-12-27 ``@ddalle``: v1.0
    """
    # Get the config.
    rc = read_case_json()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Get the job number.
    jobID = queue.pqjob()
    # Try to delete it.
    if rc.get_slurm(i):
        # Delete Slurm job
        queue.scancel(jobID)
    elif rc.get_qsub(i):
        # Delete PBS job
        queue.qdel(jobID)
    # Check if the RUNNING file exists
    cc.mark_stopped()


# Function to check output file for some kind of failure.
def CheckFailed():
    r"""Check the :file:`flowCart.out` file for a failure

    :Call:
        >>> q = CheckFailed()
    :Outputs:
        *q*: :class:`bool`
            Whether or not the last line of `flowCart.out` contains 'fail'
    :Versions:
        * 2015-01-02 ``@ddalle``: v1.0
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
    r"""Determine the file name of the PBS script to call

    This is a compatibility function for cases that do or do not have
    multiple PBS scripts in a single run directory

    :Call:
        >>> fpbs = GetPBSScript(i=None)
    :Inputs:
        *i*: :class:`int`
            Phase number
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: v1.0
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
def read_case_json():
    r"""Read `flowCart` settings for local case

    :Call:
        >>> rc = read_case_json()
    :Outputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface for run
    :Versions:
        * 2014-10-02 ``@ddalle``: v1.0 (``ReadCaseJSON()``)
        * 2023-06-02 ``@ddalle``: v2.0; use :mod:`cape.cfdx`
    """
    # Use generic version, but w/ correct class
    return cc.read_case_json(RunControlOpts)


# Function to get the most recent check file.
def GetSteadyIter():
    r"""Get iteration number of most recent steady check file

    :Call:
        >>> n = GetSteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-10-02 ``@ddalle``: v1.0
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
    r"""Get iteration number of most recent unsteady check file

    :Call:
        >>> n = GetUnsteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: v1.0
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
    r"""Get total iteration number of most recent check file

    This is the sum of the most recent steady iteration and unsteady iteration.

    :Call:
        >>> n = GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: v1.0
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
    r"""Get total iteration number of most recent check file

    This is the sum of the most recent steady iteration and unsteady iteration.

    :Call:
        >>> n = GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: v1.0
        * 2014-11-29 ``@ddalle``: This was renamed from :func:`GetRestartIter`
    """
    # Get the two numbers
    nfc = GetSteadyIter()
    ntd = GetUnsteadyIter()
    # Output
    return nfc + ntd


# Function to set up most recent check file as restart.
def SetRestartIter(n=None, ntd=None):
    r"""Set a given check file as the restart point

    :Call:
        >>> SetRestartIter(n=None, ntd=None)
    :Inputs:
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
        *ntd*: :class:`int`
            Unsteady iteration number
    :Versions:
        * 2014-10-02 ``@ddalle``: v1.0
        * 2014-11-28 ``@ddalle``: Added time-accurate compatibility
    """
    # Check the input.
    if n is None: n = GetSteadyIter()
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
    r"""Determine the appropriate input number based on results available

    :Call:
        >>> i = GetPhaseNumber(rc)
    :Inputs:
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface for `flowCart`
    :Outputs:
        *i*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: v1.0
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
    r"""Get the most recent iteration number from a :file:`history.dat` file

    :Call:
        >>> n = GetHistoryIter(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`float`
            Last iteration number
    :Versions:
        * 2014-11-24 ``@ddalle``: v1.0
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
    r"""Get the last residual in a :file:`history.dat` file

    :Call:
        >>> L1 = GetHistoryResid(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: v1.0
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return np.nan
    # Check the file.
    try:
        # Try to tail the last line.
        txt = bin.tail(fname)
        # Try to get the integer.
        return float(txt.split()[3])
    except Exception:
        # If any of that fails, return 0
        return np.nan


# Function to check if last line is unsteady
def CheckUnsteadyHistory(fname='history.dat'):
    r"""Check if the current history ends with an unsteady iteration

    :Call:
        >>> q = CheckUnsteadyHistory(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *q*: :class:`float`
            Whether or not the last iteration of *fname* has a '.' in it
    :Versions:
        * 2014-12-17 ``@ddalle``: v1.0
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
    r"""Get working folder, ``.``,  ``adapt??/``, or ``adapt??/FLOW/``

    This function must be called from the top level of a case.

    :Call:
        >>> fdir = GetWorkingFolder()
    :Outputs:
        *fdir*: :class:`str`
            Most recently used working folder with a history file
    :Versions:
        * 2014-11-24 ``@ddalle``: v1.0
        * 2023-06-05 ``@ddalle``: v2.0; support ``adapt??/FLOW/``
    """
    # Search three possible patterns for ``history.dat``
    glob1 = glob.glob("history.dat")
    glob2 = glob.glob(os.path.join("adapt??", "history.dat"))
    glob3 = glob.glob(os.path.join("adapt??", "FLOW", "history.dat"))
    # Combine
    hist_files = glob1 + glob2 + glob3
    # Get modification times for each
    mtimes = [os.path.getmtime(hist_file) for hist_file in hist_files]
    # Get index of most recent
    i_latest = mtimes.index(max(mtimes))
    # Latest modified history.dat file
    hist_latest = hist_files[i_latest]
    # Return folder from whence most recent ``history.dat`` file came
    return os.path.dirname(hist_latest)


# Function to get most recent adaptive iteration
def GetCurrentResid():
    r"""Get the most recent iteration including unsaved progress

    Iteration numbers from time-accurate restarts are corrected to match
    the global iteration numbering.

    :Call:
        >>> L1 = GetCurrentResid()
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: v1.0
    """
    # Get the working folder.
    fdir = GetWorkingFolder()
    # Get the residual.
    return GetHistoryResid(os.path.join(fdir, 'history.dat'))


# Function to get first recent adaptive iteration
def GetFirstResid():
    r"""Get the first iteration

    :Call:
        >>> L1 = GetFirstResid()
    :Outputs:
        *L1*: :class:`float`
            First L1 residual
    :Versions:
        * 2015-07-22 ``@ddalle``: v1.0
    """
    # Get the working folder.
    fdir = GetWorkingFolder()
    # File name
    fname = os.path.join(fdir, 'history.dat')
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return np.nan
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
        return np.nan


# Function to get most recent L1 residual
def GetCurrentIter():
    r"""Get the residual of the most recent iteration

    :Call:
        >>> n = GetCurrentIter()
    :Outputs:
        *n*: :class:`int`
            Most recent index written to :file:`history.dat`
    :Versions:
        * 2014-11-28 ``@ddalle``: v1.0
        * 2023-06-06 ``@ddalle``: v1.1; check ``adapt??/FLOW/``
    """
    # Try to get iteration number from working folder
    ntd = GetHistoryIter()
    # Check it
    if ntd and (not CheckUnsteadyHistory()):
        # Don't read adapt??/ history
        return ntd
    # Initialize adaptive iteration number
    n0 = 0
    # Check for adapt?? folders
    for fi in glob.glob('adapt??'):
        # Two candidates
        f1 = os.path.join(fi, "FLOW", "history.dat")
        f2 = os.path.join(fi, "history.dat")
        # Attempt to read it
        if os.path.isfile(f1):
            # Read from FLOW/
            ni = GetHistoryIter(f1)
        else:
            # Fall back to adapt??/
            ni = GetHistoryIter(f2)
        # Check it
        if ni > n0:
            # Update best estimate.
            n0 = ni
    # Output the total.
    return n0 + ntd


# Function to determine newest triangulation file
def GetTriqFile():
    r"""Get most recent ``triq`` file and its associated iterations

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
        * 2015-09-16 ``@ddalle``: v1.0
        * 2021-12-09 ``@ddalle``: v1.1
            - Check for ``adapt??/`` folder w/o ``triq`` file

        * 2022-06-06 ``@ddalle``: v1.2; check ``adapt??/FLOW/``
    """
    # Find all possible TRIQ files
    pat0 = "Components.*.triq"
    pat1 = os.path.join("adapt??", pat1)
    pat2 = os.path.join("adapt??", "FLOW", pat1)
    # Search them
    triqglob0 = sorted(glob.glob(pat0))
    triqglob1 = sorted(glob.glob(pat1))
    triqglob2 = sorted(glob.glob(pat2))
    # Determine best folder
    if len(triqglob0) > 0:
        # Use parent folder
        fwrk = "."
    elif len(triqglob2) > 0:
        # Use latest adapt??/FLOW/ folder
        fwrk = os.path.dirname(triqglob2[-1])
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
    r"""Link the most recent file to a basic unmarked file name

    The function will attempt to map numbered or adapted file names using the
    most recent iteration or adaptation.  The following gives examples of links
    that could be created using ``Components.i.plt`` for *fname* and
    ``Components.[0-9]*.plt`` for *fglb*.

        * ``Components.i.plt`` (no link)
        * ``Components.01000.plt`` --> ``Components.i.plt``
        * ``adapt03/Components.i.plt`` --> ``Components.i.plt``

    :Call:
        >>> LinkFromGlob(fname, fglb, isplit=-2, csplit='.')
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
        * 2015-11-20 ``@ddalle``: v1.0
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
    r"""Link the most recent Tecplot files to fixed file names

    Uses file names :file:`Components.i.plt` and :file:`cutPlanes.plt`

    :Call:
        >>> LinkPLT()
    :Versions:
        * 2015-03-10 ``@ddalle``: v1.0
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


