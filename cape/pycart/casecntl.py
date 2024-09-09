"""
:mod:`cape.pycart.case`: Case Control Module
=============================================

This module contains the important function :func:`casecntl.run_flowCart`,
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

# Third-party modules
import numpy as np


# Local imports
from . import cmdrun
from . import cmdgen
from . import manage
from . import pointsensor
from .. import fileutils
from .options.runctlopts import RunControlOpts
from .trifile import Triq
from .util import GetAdaptFolder, GetWorkingFolder
from ..cfdx import casecntl


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
        * 2023-07-08 ``@ddalle``: v2.0; use CaseRunner
    """
    # Get a case reader
    runner = CaseRunner()
    # Run it
    return runner.run()


# Case runner
class CaseRunner(casecntl.CaseRunner):
   # --- Clas attributes ---
    # Help message
    _help_msg = HELP_RUN_FLOWCART

    # Names
    _modname = "pycart"
    _progname = "cart3d"

    # Specific classes
    _rc_cls = RunControlOpts

   # --- Runners ---
    # Run one phase appropriately
    def run_phase(self, j: int):
        r"""Run one phase using appropriate commands

        :Call:
            >>> runner.run_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``RunPhase``)
            * 2023-06-02 ``@ddalle``: v1.1
            * 2023-07-09 ``@ddalle``: v1.2; rename, instance method
        """
        # Mesh generation
        self.run_intersect(j)
        self.run_verify(j)
        self.run_autoInputs(j)
        self.run_cubes(j)
        # Read settings
        rc = self.read_case_json()
        # Check for flowCart vs. mpi_flowCart
        if not rc.get_MPI(j):
            # Get the number of threads, which may be irrelevant.
            nProc = rc.get_nProc(j)
            # Set it.
            os.environ['OMP_NUM_THREADS'] = str(nProc)
        # Check for adaptive runs.
        if rc.get_Adaptive(j):
            # Run 'aero.csh'
            return self.run_phase_adaptive(j)
        elif rc.get_it_avg(j):
            # Run a few iterations at a time
            return self.run_phase_with_restarts(j)
        else:
            # Run with the nominal inputs
            return self.run_phase_fixed(j)

    # Run cubes if necessary
    @casecntl.run_rootdir
    def run_cubes(self, j: int):
        r"""Run ``cubes`` and ``mgPrep`` to create multigrid volume mesh

        :Call:
            >>> runner.run_cubes(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-04-06 ``@ddalle``: v1.0 (``CaseCubes``)
            * 2023-07-09 ``@ddalle``: v2.0; rename, instance method
        """
        # Check for subsequent phase
        if j != 0:
            return
        # Check for previous iterations
        # TODO: This will need an edit for 'remesh'
        if self.get_iter() > 0:
            return
        # Check for mesh file
        if os.path.isfile('Mesh.mg.c3d'):
            return
        # Read settings
        rc = self.read_case_json()
        # Check for cubes option
        if not rc.get_cubes_run():
            return
        # If adaptive, check for jumpstart
        if rc.get_Adaptive(j) and not rc.get_jumpstart(j):
            return
        # Run cubes
        cmdrun.cubes(opts=rc, j=j)
        # Run mgPrep
        cmdrun.mgPrep(opts=rc, j=j)

    # Run autoInputs if appropriate
    @casecntl.run_rootdir
    def run_autoInputs(self, j: int):
        r"""Run ``autoInputs`` if necessary

        :Call:
            >>> runner.run_autoInputs(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-04-06 ``@ddalle``: v1.0 (``CaseAutoInputs``)
            * 2023-07-09 ``@ddalle``: v2.0; rename, instance method
        """
        # Check for subsequent phase
        if j != 0:
            return
        # Check for previous iterations
        if self.get_iter() > 0:
            return
        # Check for output files
        if os.path.isfile('input.c3d') and os.path.isfile('preSpec.c3d.cntl'):
            return
        # Read settings
        rc = self.read_case_json()
        # Check for cubes option
        if not rc.get_autoInputs_run():
            return
        # Run autoInputs
        cmdrun.autoInputs(opts=rc, j=j)

    # Run one phase adaptively
    def run_phase_adaptive(self, j: int) -> int:
        r"""Run one phase using adaptive commands

        :Call:
            >>> ierr = runner.run_phase_adaptive(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``RunAdaptive``)
            * 2023-07-09 ``@ddalle``: v1.1; rename; instance method
        """
        # Delete the existing aero.csh file
        if os.path.islink('aero.csh'):
            os.remove('aero.csh')
        # Create a link to this run.
        os.symlink('aero.%02i.csh' % j, 'aero.csh')
        # Read settings
        rc = self.read_case_json()
        # Call the aero.csh command
        if j > 0 or self.get_iter() > 0:
            # Restart casecntl.
            cmdi = ['./aero.csh', 'restart']
        elif rc.get_jumpstart():
            # Initial case
            cmdi = ['./aero.csh', 'jumpstart']
        else:
            # Initial case and create grid
            cmdi = ['./aero.csh']
        # Run the command.
        ierr = self.callf(cmdi, f='flowCart.out', e="flowCart.err")
        # Get adaptive folder
        adaptdir = GetAdaptFolder()
        if adaptdir is not None:
            # Check if point sensor data exists within
            if os.path.isfile(os.path.join(adaptdir, 'pointSensors.dat')):
                # Collect point sensor data
                PS = pointsensor.CasePointSensor()
                PS.UpdateIterations()
                PS.WriteHist()
        # Output
        return ierr

    # Run one phase with *it_avg*
    def run_phase_with_restarts(self, j):
        r"""Run ``flowCart`` a few iters at a time for averaging purposes

        :Call:
            >>> RunWithRestarts(rc, i)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Check how many iterations by which to offset the count.
        if rc.get_unsteady(j):
            # Get the number of previous unsteady steps.
            n = self.get_unsteady_iter()
        else:
            # Get the number of previous steady steps.
            n = self.get_steady_iter()
        # Initialize triq
        if rc.get_clic(j):
            triq = Triq('Components.i.tri', n=0)
        # Initialize point sensor
        PS = pointsensor.CasePointSensor()
        # Requested iterations
        it_fc = rc.get_it_fc(j)
        # Start and end iterations
        n0 = n
        n1 = n + it_fc
        # Loop through iterations.
        for i in range(it_fc):
            # flowCart command accepts *it_avg*; update *n*
            if i == 0 and rc.get_it_start(j) > 0:
                # Save settings.
                it_avg = rc.get_it_avg()
                # Startup iterations
                rc.set_it_avg(rc.get_it_start(j))
                # Increase reference for averaging.
                n0 += rc.get_it_start(j)
                # Modified command
                cmdi = cmdgen.flowCart(opts=rc, i=j, n=n)
                # Reset averaging settings
                rc.set_it_avg(it_avg)
            else:
                # Normal stops every *it_avg* iterations.
                cmdi = cmdgen.flowCart(opts=rc, i=j, n=n)
            # Run the command for *it_avg* iterations.
            ierr = self.callf(cmdi, f='flowCart.out', e="flowCart.err")
            # Automatically determine the best check file to use.
            self.set_restart_iter()
            # Get new iteration count.
            if rc.get_unsteady(j):
                # Get the number of previous unsteady steps.
                n = self.get_unsteady_iter()
            else:
                # Get the number of previous steady steps.
                n = self.get_steady_iter()
            # Process triq files
            if rc.get_clic(j):
                # Read the triq file
                triqj = Triq('Components.i.triq')
                # Weighted average
                triq.WeightedAverage(triqj)
            # Update history
            PS.UpdateIterations()
            # Check for completion
            if (n >= n1) or (i + 1 == it_fc):
                break
            # Clear check files as appropriate.
            manage.ClearCheck_iStart(nkeep=1, istart=n0)
        # Write the averaged triq file
        if rc.get_clic(j):
            triq.Write('Components.%i.%i.%i.triq' % (i+1, n0, n))
        # Write the point sensor history file.
        try:
            if PS.nIter > 0:
                PS.WriteHist()
        except Exception:
            pass
        # Output
        return ierr

    # Run the nominal mode
    def run_phase_fixed(self, j: int) -> int:
        r"""Run ``flowCart`` the nominal way

        :Call:
            >>> ierr = runner.run_phase_fixed(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``RunFixed``)
            * 2023-07-09 ``@ddalle``: v1.1; rename, instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Check how many iterations by which to offset the count
        if rc.get_unsteady(j):
            # Get the number of previous unsteady steps
            n = self.get_unsteady_iter()
        else:
            # Get the number of previous steady steps
            n = self.get_steady_iter()
        # Call flowCart directly.
        cmdi = cmdgen.flowCart(opts=rc, i=j, n=n)
        # Run the command.
        ierr = self.callf(cmdi, f='flowCart.out', e="flowCart.err")
        # Check for point sensors
        if os.path.isfile('pointsensors.dat'):
            # Collect point sensor data
            PS = pointsensor.CasePointSensor()
            PS.UpdateIterations()
            PS.WriteHist()
        # Return code
        return ierr

   # --- Run status ---
    # Check if a case was run successfully
    @casecntl.run_rootdir
    def get_returncode(self):
        r"""Check iteration counts and residual change for most recent run

        :Call:
            >>> runner.get_returncode()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (was ``CheckSuccess()``)
            * 2024-06-16 ``@ddalle``: v1.1; was ``check_error()``
        """
        # Last reported iteration number
        n = self.get_history_iter()
        # Check status
        if n % 1 != 0:
            # Write the failure type
            msg = "Ended with failed unsteady cycle at iter %13.6f" % n
            # Mark failure
            self.mark_failure(msg)
            # STDOUT
            print(f"    {msg}")
            return casecntl.IERR_INCOMPLETE_ITER
        # First and last reported residual
        L1i = self.get_first_resid()
        L1f = self.get_current_resid()
        # Check for bad (large or NaN) values.
        if np.isnan(L1f) or L1f/(0.1 + L1i) > 1.0e+6:
            # Message for failure type
            msg = "Bombed at iter %.2f with resid %.2E" % (n, L1f)
            # Mark failure
            self.mark_failure(msg)
            # STDOUT
            print(f"    {msg}")
            return casecntl.IERR_BOMB
        # Check for a hard-to-detect failure present in the output file
        # Check for the file.
        if os.path.isfile('flowCart.out'):
            # Read the last line
            line = fileutils.tail('flowCart.out', 1)
            # Check if failure is mentioned
            if 'fail' in line:
                # Use last line as message
                self.mark_failure(line)
                # STDOUT
                print("    Unknown failure; last line of 'flowCart.out':")
                print(f"      {line}")
                # Return code
                return casecntl.IERR_UNKNOWN
        # Everything ok
        return casecntl.IERR_OK

   # --- File control ---
    # Prepare the files of the case
    def prepare_files(self, j: int):
        r"""Prepare file names appropriate to run phase *i* of Cart3D

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``PrepareFiles``)
            * 2023-07-09 ``@ddalle``: v1.1; rename, instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Create a restart file if appropriate.
        if not rc.get_Adaptive(j):
            # Automatically determine the best check file to use.
            self.set_restart_iter()
        # Delete any input file.
        if os.path.isfile('input.cntl') or os.path.islink('input.cntl'):
            os.remove('input.cntl')
        # Create the correct input file.
        os.symlink('input.%02i.cntl' % j, 'input.cntl')
        # Get adaptive
        adaptdir = GetAdaptFolder()
        # Extra prep for adaptive --> non-adaptive
        if (j > 0) and (not rc.get_Adaptive(j)) and (
                os.path.isdir(adaptdir) and
                (not os.path.isfile('history.dat'))):
            # Find all *.dat files and Mesh files
            pat1 = glob.glob(os.path.join(adaptdir, "*.dat"))
            pat2 = glob.glob(os.path.join(adaptdir, "Mesh.*"))
            # Find matches for both
            fglob = (pat1) + (pat2)
            # Copy all the important files.
            for fname in fglob:
                # Localize file name
                flocal = os.path.basename(fname)
                # Check for the file already in working dir
                if os.path.isfile(flocal):
                    continue
                # Copy the file to working parent dir
                shutil.copy(fname, flocal)
        # Convince aero.csh to use the *new* input.cntl
        if (j > 0) and (rc.get_Adaptive(j)) and (rc.get_Adaptive(j - 1)):
            # Go to the best adaptive result
            os.chdir(adaptdir)
            # Check for an input.cntl file
            if os.path.isfile('input.cntl'):
                # Move it to a representative name.
                os.rename('input.cntl', 'input.%02i.cntl' % (j - 1))
            # Go back up
            os.chdir(self.root_dir)
            # Copy the new input file
            shutil.copy(
                'input.%02i.cntl' % j,
                os.path.join(adaptdir, "input.cntl"))
        # Get rid of linked Tecplot files
        if os.path.islink('Components.i.plt'):
            os.remove('Components.i.plt')
        if os.path.islink('Components.i.dat'):
            os.remove('Components.i.dat')
        if os.path.islink('cutPlanes.plt'):
            os.remove('cutPlanes.plt')
        if os.path.islink('cutPlanes.dat'):
            os.remove('cutPlanes.dat')

    # Clean up immediately after running
    def finalize_files(self, j: int):
        r"""Clean up files names after running one cycle of phase *j*

        :Call:
            >>> runner.finalize_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``FinalizeFiles``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Clean up the folder as appropriate.
        manage.ManageFilesProgress(rc)
        # Tar visualization files.
        if rc.get_unsteady(j):
            manage.TarViz(rc)
        # Tar old adaptation folders.
        if rc.get_Adaptive(j):
            manage.TarAdapt(rc)
        # Get the new restart iteration.
        n = self.get_check_resub_iter()
        # Assuming that worked, move the temp output file.
        os.rename('flowCart.out', 'run.%02i.%i' % (j, n))
        # Check for TecPlot files to save.
        if os.path.isfile('cutPlanes.plt'):
            os.rename('cutPlanes.plt', 'cutPlanes.%05i.plt' % n)
        if os.path.isfile('Components.i.plt'):
            os.rename('Components.i.plt', 'Components.i.%05i.plt' % n)
        if os.path.isfile('cutPlanes.dat'):
            os.rename('cutPlanes.dat', 'cutPlanes.%05i.dat' % n)
        if os.path.isfile('Components.i.dat'):
            os.rename('Components.i.dat', 'Components.i.%05i.dat' % n)

    # Function to set up most recent check file as restart.
    def set_restart_iter(self, n=None, ntd=None):
        r"""Set a given check file as the restart point

        :Call:
            >>> runner.set_restart_iter(n=None, ntd=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: {``None``} | :class:`int`
                Restart iteration number, defaults to most recent available
            *ntd*: {``None``} | :class:`int`
                Unsteady iteration number
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0 (``SetRestartIter``)
            * 2014-11-28 ``@ddalle``: v1.1; support time-accurate
            * 2023-07-10 ``@ddalle``: v1.2; rename, instance method
            * 2024-06-22 ``@jmeeroff``: v1.3; support FLOW directories
            * 2025-06-24 ``@ddalle``: v1.4; use ``GetWorkingDir()``
        """
        # Check the input.
        if n is None:
            n = self.get_steady_iter()
        if ntd is None:
            ntd = self.get_unsteady_iter()
        # Remove the current restart file if necessary.
        if os.path.isfile('Restart.file') or os.path.islink('Restart.file'):
            os.remove('Restart.file')
        # Quit if no check point.
        if n == 0 and ntd == 0:
            return None
        # Find working dir
        fdir = GetWorkingFolder()
        # Map '.' -> ''
        fdir = '' if fdir == '.' else fdir
        # Time-accurate checkpoint file
        fcheck_td = os.path.join(fdir, "check.%06i.td" % ntd)
        # Steady-state checkpoint file
        fcheck = os.path.join(fdir, "check.%05i" % n)
        # Create a link to the most appropriate file
        if os.path.isfile(fcheck_td):
            # Restart from time-accurate checkpoint
            os.symlink(fcheck_td, "Restart.file")
        elif os.path.isfile(fcheck):
            # Restart from steady-state checkpoint
            os.symlink(fcheck, "Restart.file")

   # --- Case status ---
   # Function to get most recent iteration
    def getx_iter(self):
        r"""Get the residual of the most recent iteration

        :Call:
            >>> n = runner.getx_iter()
        :Outputs:
            *n*: :class:`int`
                Most recent index written to :file:`history.dat`
        :Versions:
            * 2014-11-28 ``@ddalle``: v1.0 (``GetCurrentIter``)
            * 2023-06-06 ``@ddalle``: v1.1; check ``adapt??/FLOW/``
            * 2023-07-10 ``@ddalle``: v1.2; rename, instance method
        """
        # Try to get iteration number from working folder
        ntd = self.get_history_iter()
        # Check it
        if ntd and (not self.check_unsteady_history()):
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
                ni = self.get_history_iter(f1)
            else:
                # Fall back to adapt??/
                ni = self.get_history_iter(f2)
            # Check it
            if ni > n0:
                # Update best estimate.
                n0 = ni
        # Output the total.
        return n0 + ntd

    # Function to get total iteration number
    def getx_restart_iter(self):
        r"""Get total iteration number of most recent check file

        This is the sum of the most recent steady iteration and the most
        recent unsteady iteration.

        :Call:
            >>> n = runner.getx_restart_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Index of most recent check file
        :Versions:
            * 2014-11-28 ``@ddalle``: v1.0 (``GetRestartIter``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Get unsteady iteration number based on available check files
        ntd = self.get_unsteady_iter()
        # Check for an unsteady iteration number
        if ntd:
            # If there's an unsteady iteration, use that step directly
            return ntd
        else:
            # Use the steady-state iteration number
            return self.get_steady_iter()

    # Function to get the most recent check file
    @casecntl.run_rootdir
    def get_steady_iter(self):
        r"""Get iteration number of most recent steady check file

        :Call:
            >>> n = runner.get_steady_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Index of most recent check file
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0 (``GetRestartIter``)
            * 2014-11-28 ``@ddalle``: v1.1 (``GetSteadyIter``)
            * 2023-06-06 ``@ddalle``: v1.2; support ``BEST/FLOW/``
            * 2023-07-10 ``@ddalle``: v2.3; rename, instance method
        """
        # List the check.* files
        fch = (
            glob.glob('check.*[0-9]') +
            glob.glob('adapt??/check.*') +
            glob.glob('adapt??/FLOW/check.*'))
        # Initialize iteration number until informed otherwise
        n = 0
        # Loop through the matches
        for fname in fch:
            # Get the integer for this file
            i = int(fname.split('.')[-1])
            # Use the running maximum
            n = max(i, n)
        # Output
        return n

    # Function to get the most recent time-domain check file
    @casecntl.run_rootdir
    def get_unsteady_iter(self):
        r"""Get iteration number of most recent unsteady check file

        :Call:
            >>> n = runner.get_unsteady_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Index of most recent check file
        :Versions:
            * 2014-11-28 ``@ddalle``: v1.0 (``GetUnsteadyIter``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
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
    @casecntl.run_rootdir
    def get_check_resub_iter(self):
        r"""Get total iteration number of most recent check file

        This is the sum of the most recent steady iteration number and
        unsteady iteration number.

        :Call:
            >>> n = self.get_check_resub_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Index of most recent check file
        :Versions:
            * 2014-11-28 ``@ddalle``: v1.0 (``GetRestartIter``)
            * 2014-11-29 ``@ddalle``: v1.1 (``GetCheckResubIter``)
            * 2023-07-10 ``@ddalle``: v1.2; rename, instance method
        """
        # Get the two numbers
        nfc = self.get_steady_iter()
        ntd = self.get_unsteady_iter()
        # Output
        return nfc + ntd

    # Function to read last line of 'history.dat' file
    @casecntl.run_rootdir
    def get_history_iter(self, fname='history.dat') -> float:
        r"""Read last iteration number from a ``history.dat`` file

        :Call:
            >>> n = runner.get_history_iter(fname='history.dat')
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: {``"history.dat"``} | :class:`str`
                Name of file to read
        :Outputs:
            *n*: :class:`float`
                Last iteration number
        :Versions:
            * 2014-11-24 ``@ddalle``: v1.0 (``GetHistoryIter``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Check the file beforehand.
        if not os.path.isfile(fname):
            # No history
            return 0.0
        # Check the file.
        try:
            # Try to tail the last line.
            txt = fileutils.tail(fname)
            # Try to get the integer.
            return float(txt.split()[0])
        except Exception:
            # If any of that fails, return 0
            return 0.0

   # --- Local data ---
    # Get last residual from 'history.dat' file
    @casecntl.run_rootdir
    def get_history_resid(self, fname='history.dat'):
        r"""Get the last residual in a :file:`history.dat` file

        :Call:
            >>> L1 = runner.get_history_resid(fname='history.dat')
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *L1*: :class:`float`
                Last L1 residual
        :Versions:
            * 2015-01-02 ``@ddalle``: v1.0 (``GetHistoryResid``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Check the file beforehand
        if not os.path.isfile(fname):
            # No history
            return np.nan
        # Check the file.
        try:
            # Try to tail the last line.
            txt = fileutils.tail(fname)
            # Try to get the value
            return float(txt.split()[3])
        except Exception:
            # If any of that fails, return 0
            return np.nan

    # Function to check if last line is unsteady
    @casecntl.run_rootdir
    def check_unsteady_history(self, fname='history.dat') -> bool:
        r"""Check if the current history ends with an unsteady iteration

        :Call:
            >>> q = runner.check_unsteady_history(fname='history.dat')
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *q*: ``True`` | ``False``
                Whether the last iteration of *fname* has a '.' in it
        :Versions:
            * 2014-12-17 ``@ddalle``: v1.0 (``CheckUnsteadyHistory``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Check the file beforehand.
        if not os.path.isfile(fname):
            # No history
            return False
        # Check the file's contents.
        try:
            # Try to tail the last line.
            txt = fileutils.tail(fname)
            # Check for a dot.
            return ('.' in txt.split()[0])
        except Exception:
            # Something failed; invalid history
            return False

    # Function to get the most recent working folder
    @casecntl.run_rootdir
    def get_working_folder(self) -> str:
        r"""Get working folder, ``.``,  ``adapt??/``, or ``adapt??/FLOW/``

        This function must be called from the top level of a casecntl.

        :Call:
            >>> fdir = runner.get_working_folder()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fdir*: :class:`str`
                Most recently used working folder with a history file
        :Versions:
            * 2014-11-24 ``@ddalle``: v1.0 (``GetWorkingFolder``)
            * 2023-06-05 ``@ddalle``: v2.0; support ``adapt??/FLOW/``
            * 2023-07-10 ``@ddalle``: v2.1; rename, instance method
        """
        # Search three possible patterns for ``history.dat``
        glob1 = glob.glob("history.dat")
        glob2 = glob.glob(os.path.join("adapt??", "history.dat"))
        glob3 = glob.glob(os.path.join("adapt??", "FLOW", "history.dat"))
        # Combine
        hist_files = glob1 + glob2 + glob3
        # Check for starting out
        if len(hist_files) == 0:
            return "."
        # Get modification times for each
        mtimes = [os.path.getmtime(hist_file) for hist_file in hist_files]
        # Get index of most recent
        i_latest = mtimes.index(max(mtimes))
        # Latest modified history.dat file
        hist_latest = hist_files[i_latest]
        # Return folder from whence most recent ``history.dat`` file came
        fdir = os.path.dirname(hist_latest)
        # Check for empty
        fdir = "." if fdir == "" else fdir
        # Output
        return fdir

    # Function to get most recent adaptive iteration
    @casecntl.run_rootdir
    def get_current_resid(self):
        r"""Get the most recent iteration including unsaved progress

        Iteration numbers from time-accurate restarts are corrected to match
        the global iteration numbering.

        :Call:
            >>> L1 = runner.get_current_resid()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *L1*: :class:`float`
                Last L1 residual
        :Versions:
            * 2015-01-02 ``@ddalle``: v1.0 (``GetCurrentResid``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Get the working folder
        fdir = self.get_working_folder()
        # History file
        fhist = os.path.join(fdir, 'history.dat')
        # Get the residual.
        return self.get_history_iter(fhist)

    # Function to get first recent adaptive iteration
    @casecntl.run_rootdir
    def get_first_resid(self):
        r"""Get the first iteration

        :Call:
            >>> L1 = GetFirstResid()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *L1*: :class:`float`
                First L1 residual
        :Versions:
            * 2015-07-22 ``@ddalle``: v1.0 (``GetFirstResid``)
            * 2023-07-10 ``@ddalle``: v1.1; rename, instance method
        """
        # Get the working folder.
        fdir = self.get_working_folder()
        # File name
        fname = os.path.join(fdir, 'history.dat')
        # Check the file beforehand.
        if not os.path.isfile(fname):
            # No history
            return np.nan
        # Check the file.
        try:
            # Try to open the file
            with open(fname, 'r') as fp:
                # Initialize line
                txt = '#'
                # Read the lines until it's not a comment.
                while txt.startswith('#'):
                    # Read the next line.
                    txt = fp.readline()
            # Try to get the integer.
            return float(txt.split()[3])
        except Exception:
            # If any of that fails, return 0
            return np.nan


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
    pat1 = os.path.join("adapt??", pat0)
    pat2 = os.path.join("adapt??", "FLOW", pat0)
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
        runner = CaseRunner()
        i1 = runner.get_iter()
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
    if os.path.isfile(fname) and not os.path.islink(fname):
        return
    # Remove the link if necessary
    if os.path.isfile(fname) or os.path.islink(fname):
        os.remove(fname)
    # Get the working directory
    runner = CaseRunner()
    fdir = runner.get_working_folder()
    # Check it.
    if fdir == '.':
        # List files that match the requested glob
        fglob = glob.glob(fglb)
        # Check for empty glob
        if len(fglob) == 0:
            return
        # Get indices from those files.
        n = [int(f.split(csplit)[isplit]) for f in fglob]
        # Extract file with maximum index.
        fsrc = fglob[n.index(max(n))]
    else:
        # File from the working folder (if it exists)
        fsrc = os.path.join(fdir, fname)
        # Check for the file
        if not os.path.isfile(fsrc):
            # Get the adaptation number of the working folder
            # This assumes that folder name is 'adapt'
            nadapt = int(fdir[5:7])
            # Try the previous adaptation file
            if fdir.endswith("FLOW"):
                fdir = 'adapt%02i/FLOW' % (nadapt-1)
            else:
                fdir = 'adapt%02i' % (nadapt-1)
            # Use that folder
            fsrc = os.path.join(fdir, fname)
        # Check for the file again
        if not os.path.isfile(fsrc):
            return
    # Create the link if possible
    if os.path.isfile(fsrc):
        os.symlink(fsrc, fname)


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


