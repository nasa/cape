r"""
:mod:`cape.cfdx.cntl`: Base module for CFD operations and processing
=====================================================================

This module provides tools and templates for tools to interact with
various CFD codes and their input files. The base class is
:class:`cape.cfdx.cntl.Cntl`, and the derivative classes include
:class:`cape.pycart.cntl.Cntl`. This module creates folders for cases,
copies files, and can be used as an interface to perform most of the
tasks that Cape can accomplish except for running individual cases.

The control module is set up as a Python interface for thec master
JSON file, which contains the settings to be used for a given CFD
project.

The derivative classes are used to read input files, set up cases,
submit and/or run cases, and be an interface for the various Cape
options as they are customized for the various CFD solvers. The
individualized modules are below.

    * :mod:`cape.pycart.cntl`
    * :mod:`cape.pyfun.cntl`
    * :mod:`cape.pyover.cntl`

:See also:
    * :mod:`cape.cfdx.casecntl`
    * :mod:`cape.cfdx.options`
    * :mod:`cape.cfdx.runmatrix`

"""

# Standard library modules
import functools
import os
from typing import Any, Optional, Union

# Third-party modules
import numpy as np

# Local imports
from . import casecntl
from . import databook
from . import queue
from . import report
from .casecntl import CaseRunner
from .cntlbase import CntlBase
from .logger import CntlLogger
from .options import Options
from .dex import DataExchanger
from .runmatrix import RunMatrix
from ..optdict import WARNMODE_WARN, WARNMODE_QUIET


# Constants
DEFAULT_WARNMODE = WARNMODE_WARN
MATRIX_CHUNK_SIZE = 1000
UGRID_EXTS = (
    "b4",
    "b8",
    "b8l",
    "lb4",
    "lb8",
    "lb8l",
    "lr4",
    "lr8",
    "r4",
    "r8",
)
COL_HEADERS = {
    "case": "Case Folder",
    "cpu-abbrev": "CPU Hours",
    "cpu-hours": "CPU Time",
    "frun": "Config/Run Directory",
    "gpu-abbrev": "GPU Hours",
    "gpu-hours": "GPU Hours",
    "group": "Group Folder",
    "i": "Case",
    "job": "Job ID",
    "job-id": "Job ID",
    "phase": "Phase",
    "progress": "Iterations",
    "queue": "Que",
    "status": "Status",
}
DEG = np.pi / 180.0
JOB_STATUSES = (
    'PASS',
    'PASS*',
    '---',
    'INCOMP',
    'RUN',
    'DONE',
    'QUEUE',
    'ERROR',
    'ERROR*',
    'FAIL',
    'ZOMBIE',
    'THIS_JOB',
)


# Decorator for moving directories
def run_rootdir(func):
    r"""Decorator to run a function within a specified folder

    :Call:
        >>> func = run_rootdir(func)
    :Wrapper Signature:
        >>> v = cntl.func(*a, **kw)
    :Inputs:
        *func*: :class:`func`
            Name of function
        *cntl*: :class:`Cntl`
            Control instance from which to use *cntl.RootDir*
        *a*: :class:`tuple`
            Positional args to :func:`cntl.func`
        *kw*: :class:`dict`
            Keyword args to :func:`cntl.func`
    :Versions:
        * 2018-11-20 ``@ddalle``: v1.0
        * 2020-02-25 ``@ddalle``: v1.1: better exceptions
        * 2023-06-16 ``@ddalle``: v1.2; use ``finally``
    """
    # Declare wrapper function to change directory
    @functools.wraps(func)
    def wrapper_func(self, *args, **kwargs):
        # Recall current directory
        fpwd = os.getcwd()
        # Go to specified directory
        os.chdir(self.RootDir)
        # Run the function with exception handling
        try:
            # Attempt to run the function
            v = func(self, *args, **kwargs)
        except Exception:
            # Raise the error
            raise
        except KeyboardInterrupt:
            # Raise the error
            raise
        finally:
            # Go back to original folder (always)
            os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func


# Convert ``a,b,c`` -> ``['a', 'b', 'c']``
def _split(v: Union[str, list]) -> list:
    # Check type
    if isinstance(v, str):
        return [vj.strip() for vj in v.split(',')]
    else:
        return v


# Cache of one property for each case
class CaseCache(dict):
    r"""Cache of one property for cases in a run matrix

    :Call:
        >>> cache = CaseCache(prop)
    :Inputs:
        *prop*: :class:`str`
            Name of property being cached
    :Outputs:
        *cache*: :class:`CaseCache`
            Cache of property for each case, like a :class:`dict`
    """
    # Properties
    __slots__ = (
        "prop"
    )

    # Initialization
    def __init__(self, prop: str):
        #: :class:`str`
        #: Name of property being cached
        self.prop = prop

    # Get value
    def get_value(self, i: int) -> Any:
        r"""Get a value for a case, if any

        :Call:
            >>> val = cache.get_value(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Outputs:
            *val*: :class:`object` | ``None``
                Value, if present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        # Get value if any
        return self.get(i)

    # Save value
    def save_value(self, i: int, val: Any):
        r"""Save a value for a case

        :Call:
            >>> cache.save_value(i, val)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
            *val*: :class:`object` | ``None``
                Value, if present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        # Save value
        self[i] = val

    # Clear value
    def clear_case(self, i: int):
        r"""Clear cache for one case, if present

        :Call:
            >>> cache.clear_case(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        self.pop(i, None)

    # Check case
    def check_case(self, i: int) -> bool:
        r"""Save a value for a case

        :Call:
            >>> q = cache.check_case(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Outputs:
            *q*: :class:`bool`
                Whether or not case *i* is present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        return i in self


# Class to read input files
class Cntl(CntlBase):
    r"""Class to handle options, setup, and execution of CFD codes

    :Call:
        >>> cntl = cape.Cntl(fname="cape.json")
    :Inputs:
        *fname*: :class:`str`
            Name of JSON settings file from which to read options
    :Outputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            Instance of Cape control interface
        *cntl.opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *cntl.x*: :class:`cape.runmatrix.RunMatrix`
            Run matrix interface
        *cntl.RootDir*: :class:`str`
            Working directory from which the class was generated
    :Versions:
        * 2015-09-20 ``@ddalle``: Started
        * 2016-04-01 ``@ddalle``: v1.0
    """
  # *** CLASS ATTRIBUTES ***
   # --- Names ---
    #: Name of this CAPE module
    #: :class:`str`
    _name = "cfdx"

    #: Name of CFD solver for this module
    #: :class:`str`
    _solver = "cfdx"

   # --- Specific modules ---
    #: Module for case control
    _case_mod = casecntl

    #: Solver-specific module for DataBook
    _databook_mod = databook

    #: Solver-specific module for automated reports
    _report_mod = report

   # --- Specific  classes ---
    #: Solver-specific class for running cases
    #: :class:`type`
    _case_cls = casecntl.CaseRunner

    #: Solver-specific class for CAPE options
    #: :class:`type`
    _opts_cls = Options

   # --- Other settings ---
    #: Name of default JSON file
    #: :class:`str`
    _fjson_default = "cape.json"

    #: Warning mode
    #: :class:`int`
    _warnmode_default = WARNMODE_WARN

    #: Environment variable to read warning mode from
    #: :class:`str`
    _warnmode_envvar = "CAPE_WARNMODE"

    #: List of files to check for zombie status
    #: :class:`list`\ [:class:`str`]
    _zombie_files = ["*.out"]

  # *** DUNDER ***
    # Initialization method
    def __init__(self, fname: Optional[str] = None):
        r"""Initialization method for :mod:`cape.cfdx.cntl.Cntl`

        :Versions:
            * 2015-09-20 ``ddalle``: v1.0
        """
        # Default file name
        fname = self._fjson_default if fname is None else fname
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No cape control file '%s' found" % fname)
        #: :class:`str`
        #: Root folder for this run matrix
        self.RootDir = os.getcwd()
        #: :class:`CaseRunner`
        #: Slot for the current case runner
        self.caserunner = None
        #: :class:`int`
        #: Case index of the current case runner
        self.caseindex = None
        #: :class:`Options`
        #: Options interface for this run matrix
        self.opts = None
        # Read options
        self.read_options(fname)
        #: :class:`dict`
        #: Dictionary of imported custom modules
        self.modules = {}
        #: :class:`CntlLogger`
        #: Run matrix logger instacnce
        self.logger = None
        #: :class:`RunMatrix`
        #: Run matrix instance
        self.x = RunMatrix(**self.opts['RunMatrix'])
        # Set run matrix w/i options
        self.opts.save_x(self.x)
        # Set initial index
        self.opts.setx_i(0)
        #: :class:`str`
        #: Job name to check
        self.job = None
        #: :class:`dict`\ [:class:`str`]
        #: Dictionary of jobs 
        self.jobs = {}
        #: :class:`list`\ [:class:`str`]
        #: List of queues that have been checked
        self.jobqueues = []
        # Run cntl init functions, customize for py{x}
        self.init_post()
        # Run any initialization functions
        self.InitFunction()
        #: :class:`DataBook`
        #: Interface to post-processed data
        self.DataBook = None
        #: :class:`dict`\ [:class:`DataExchanger`]
        #: Data extraction classes for each component of DataBook
        self.data = {}
        #: :class:`CaseCache`
        #: Cache of current iteration for each case
        self.cache_iter = CaseCache("iter")

    # Output representation
    def __repr__(self) -> str:
        r"""Output representation method for Cntl class

        :Versions:
            * 2015-09-20 ``@ddalle``: v1.0
        """
        # Get class handle
        cls = self.__class__
        # Display basic information
        return "<%s.%s(nCase=%i)>" % (
            cls.__module__,
            cls.__name__,
            self.x.nCase)

    __str__ = __repr__

   # --- Input Readers ---
    # Read the data book
    @run_rootdir
    def ReadDataBook(self, comp=None):
        r"""Read the current data book

        :Call:
            >>> cntl.ReadDataBook()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2016-09-15 ``@ddalle``: v1.0
            * 2023-05-31 ``@ddalle``: v2.0; universal ``cape.cntl``
        """
        # Test if already read
        if self.DataBook is not None:
            return
        # Ensure list of components
        if comp is not None and not isinstance(comp, list):
            comp = [comp]
        # Get DataBook class
        databookmod = self.__class__._databook_mod
        # Instantiate class
        self.DataBook = databookmod.DataBook(self, comp=comp)
        # Call any custom functions
        self.ReadDataBookPost()

    # Call special post-read DataBook functions
    def ReadDataBookPost(self):
        r"""Do ``py{x}`` specific init actions after reading DataBook

        :Call:
            >>> cntl.ReadDataBookPost()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-05-31 ``@ddalle``: v1.0
        """
        pass

    # Read report
    @run_rootdir
    def ReadReport(self, rep):
        r"""Read a report interface

        :Call:
            >>> R = cntl.ReadReport(rep)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *R*: :class:`pyFun.report.Report`
                Report interface
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Read the report
        R = self.__class__._report_mod.Report(self, rep)
        # Output
        return R

   # --- Run Interface ---
    # Get case runner from a folder
    @run_rootdir
    def ReadFolderCaseRunner(self, fdir: str) -> CaseRunner:
        r"""Read a ``CaseRunner`` from a folder by name

        :Call:
            >>> runner = cntl.ReadFolderCaseRunner(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2024-11-05 ``@ddalle``: v1.0
        """
        # Check if folder exists
        if not os.path.isdir(fdir):
            raise ValueError(f"Cannot read CaseRunner: no folder '{fdir}'")
        # Read case runner
        return self._case_cls(fdir)

    # Instantiate a case runner
    @run_rootdir
    def ReadCaseRunner(self, i: int) -> CaseRunner:
        r"""Read CaseRunner into slot

        :Call:
            >>> runner = cntl.ReadCaseRunner(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-22 ``@ddalle``: v1.0
        """
        # Check the slot
        if (self.caseindex == i) and (self.caserunner is not None):
            # Check if folder still exists
            if os.path.isdir(self.caserunner.root_dir):
                # Already in the slot
                return self.caserunner
        # Get case name
        frun = self.x.GetFullFolderNames(i)
        fabs = os.path.join(self.RootDir, frun)
        # Check if case is present
        if not os.path.isdir(fabs):
            return
        # Instantiate
        self.caserunner = self._case_cls(fabs)
        self.caseindex = i
        # Save jobs
        self.caserunner.jobs = self.jobs
        self.caserunner.job = self.job
        # Save *cntl* so it doesn't have to read it
        self.caserunner.cntl = self
        # Output
        return self.caserunner

    # Function to start a case: submit or run
    @run_rootdir
    def StartCase(self, i: int):
        r"""Start a case by either submitting it or running it

        This function checks whether or not a case is submittable.  If
        so, the case is submitted via :func:`cape.cfdx.queue.pqsub`,
        and otherwise the case is started using a system call.

        Before starting case, this function checks the folder using
        :func:`cape.cfdx.cntl.CheckCase`; if this function returns ``None``,
        the case is not started.  Actual starting of the case is done
        using :func:`CaseStartCase`, which has a specific version for
        each CFD solver.

        :Call:
            >>> pbs = cntl.StartCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *pbs*: :class:`int` | ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2014-10-06 ``@ddalle``: v1.0
            * 2023-06-27 ``@ddalle``: v2.0; use *CaseRunner*
        """
        # Set case index
        self.opts.setx_i(i)
        # Get case name
        frun = self.x.GetFullFolderNames(i)
        # Check status.
        if self.CheckCase(i) is None:
            # Case not ready
            print("    Attempted to start case '%s'." % frun)
            print("    However, case failed initial checks.")
            # Check again with verbose option
            self.CheckCase(i, v=True)
            return
        elif self.CheckRunning(i):
            # Case already running!
            return
        # Print status
        print("     Starting case '%s'" % frun)
        # Get the case runner
        runner = self.ReadCaseRunner(i)
        # Remove case from cache
        self.cache_iter.clear_case(i)
        # Start the case by either submitting or calling it.
        ierr, pbs = runner.start()
        # Check for error
        if ierr:
            print(f"     Job failed with return code {ierr}")
        # Display the PBS job ID if that's appropriate.
        if pbs:
            print(f"     Submitted job: {pbs}")
        # Output
        return pbs

    # Function to terminate a case: qdel and remove RUNNING file
    @run_rootdir
    def StopCase(self, i: int):
        r"""Stop a case if running

        This function deletes a case's PBS job and removes the
        ``RUNNING`` file if it exists.

        :Call:
            >>> cntl.StopCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-27 ``@ddalle``: v1.0
            * 2023-06-27 ``@ddalle``: v2.0; use ``CaseRunner``
        """
        # Check status
        if self.CheckCase(i) is None:
            # Case not ready
            return
        # Read runner
        runner = self.ReadCaseRunner(i)
        # Stop the job if possible
        runner.stop_case()

   # --- Case Preparation ---
    # Prepare ``CAPE-STOP-PHASE`` file
    def _prepare_incremental(self, i: int, j: Union[bool, int] = False):
        r"""Prepare a case to stop at end of specified phase

        :Call:
            >>> cntl._prepare_incremental(i, j=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
            *j*: ``True`` | {``False``} | :class:`int`
                Option to stop at end of any phase (``True``) or
                specific phase number
        :Versions:
            * 2024-05-26 ``@ddalle``: v1.0
        """
        # Check option
        if j is False:
            # Normal case; no incremental option
            return
        # Run folder
        frun = self.x.GetFullFolderNames(i)
        # Absolutize
        fstop = os.path.join(self.RootDir, frun, casecntl.STOP_PHASE_FILE)
        # Create file
        with open(fstop, 'w') as fp:
            # Write phase number if *j* is an int
            if isinstance(j, (int, np.int32, np.int64)):
                fp.write(f"{j}\n")

   # --- Cases ---
    # Check if a case is running.
    @run_rootdir
    def CheckRunning(self, i):
        r"""Check if a case is currently running

        :Call:
            >>> q = cntl.CheckRunning(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`RUNNING` file in it
        :Versions:
            * 2014-10-03 ``@ddalle``: v1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, casecntl.RUNNING_FILE))
        # Output
        return q

    # Check for a failure
    @run_rootdir
    def CheckError(self, i: int) -> bool:
        r"""Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has ``FAIL`` file in it
        :Versions:
            * 2015-01-02 ``@ddalle``: v1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the error file
        q = os.path.isfile(os.path.join(frun, casecntl.FAIL_FILE))
        # Check ERROR flag
        q = q or self.x.ERROR[i]
        # Output
        return q

    def check_case_job(self, i: int, active: bool = True) -> str:
        r"""Get queue status of the PBS/Slurm job from case *i*

        :Call:
            >>> s = cntl.check_case_job(i, active=True)
        :Inputs:
            *cntl*: :class:`Cntl`
                Controller for one CAPE run matrix
            *i*: :class:`int`
                Case index
            *active*: {``True``} | ``False``
                Whether or not to allow new calls to ``qstat``
        :Outputs:
            *s*: :class:`str`
                Job queue status

                * ``-``: not in queue
                * ``Q``: job queued (not running)
                * ``R``: job running
                * ``H``: job held
                * ``E``: job error status
        :Versions:
            * 2025-05-04 ``@ddalle``: v1.0
        """
        # Get case runner
        runner = self._read_runner(i, active)
        # Check for null case
        if runner is None:
            return '.'
        # Check
        return runner.check_queue()

    def check_case_status(self, i: int, active: bool = True) -> str:
        r"""Get queue status of the PBS/Slurm job from case *i*

        :Call:
            >>> sts = cntl.check_case_job(i, active=True)
        :Inputs:
            *cntl*: :class:`Cntl`
                Controller for one CAPE run matrix
            *i*: :class:`int`
                Case index
            *active*: {``True``} | ``False``
                Whether or not to allow new calls to ``qstat``
        :Outputs:
            *sts*: :class:`str`
                Case status

                * ``---``: folder does not exist
                * ``INCOMP``: case incomplete but [partially] set up
                * ``QUEUE``: case waiting in PBS/Slurm queue
                * ``RUNNING``: case currently running
                * ``DONE``: all phases and iterations complete
                * ``PASS``: *DONE* and marked by user as passed
                * ``ZOMBIE``: seems running; but no recent file updates
                * ``PASS*``: case marked by user but not *DONE*
                * ``ERROR``: case marked as error
                * ``FAIL``: case failed but not marked by user
        :Versions:
            * 2025-05-04 ``@ddalle``: v1.0
        """
        # Get case runner
        runner = self._read_runner(i, active)
        # Check for empty case
        if runner is None:
            # Check for mark
            if self.x.ERROR[i]:
                return 'ERROR'
            elif self.x.PASS[i]:
                return 'PASS*'
            return '---'
        # Check
        return runner.get_status()

    def _read_runner(self, i: int, active: bool = True) -> CaseRunner:
        r"""Read case runner and synch PBS jobs tracker

        :Call:
            >>> runner = cntl._read_runner(i, active=True)
        """
        # Get case runner
        runner = self.ReadCaseRunner(i)
        # Check for null case
        if runner is None:
            return
        # Check for active job trackers
        if isinstance(runner.jobs, queue.QStat):
            if isinstance(self.jobs, queue.QStat):
                # Both runner and cntl are active; combine
                self.jobs.update(runner.jobs)
                runner.jobs = self.jobs
            else:
                # Save the runner's version here
                self.jobs = runner.jobs
        elif isinstance(self.jobs, queue.QStat):
            # Save local instance in *runner*
            runner.jobs = self.jobs
        else:
            # Initialize
            self.jobs = runner._get_qstat()
        # Apply *active* setting
        runner.jobs.active = active
        # Output
        return runner

   # --- Case Modification ---
    # Extend a case
    def ExtendCase(
            self,
            i: int,
            n: int = 1,
            j: Optional[int] = None,
            imax: Optional[int] = None):
        r"""Add iterations to case *i* by repeating the last phase

        :Call:
            >>> cntl.ExtendCase(i, n=1, j=None, imax=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Run index
            *n*: {``1``} | positive :class:`int`
                Add *n* times *steps* to the total iteration count
            *j*: {``None``} | :class:`int`
                Optional phase to extend
            *imax*: {``None``} | nonnegative :class:`int`
                Use *imax* as the maximum iteration count
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
            * 2024-08-27 ``@ddalle``: v2.0; move code to ``CaseRunner``
            * 2024-09-28 ``@ddalle``: v2.1; add *j*
        """
        # Ignore cases marked PASS
        if self.x.PASS[i] or self.x.ERROR[i]:
            return
        # Get the runner
        runner = self.ReadCaseRunner(i)
        # Extend the case
        runner.extend_case(m=n, j=j, nmax=imax)

   # --- Data Exchange ---
    def update_dex_case(self, comp: str, i: int) -> int:
        r"""Update one case of a *DataBook* component

        :Call:
            >>> n = cntl.update_dex_case(comp, i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *comp*: :class:`str`
                Name of component to read
            *i*: :class:`int`
                Case to index
        :Outputs:
            *n*: ``0`` | ``1``
                Number of updates made
        :Versions:
            * 2025-07-25 ``@ddalle``: v1.0
        """
        ...

    def read_dex(self, comp: str, force: bool = False) -> DataExchanger:
        r"""Read a *DataBook* component using :class:`DataExchanger`

        :Call:
            >>> db = cntl.read_dex(comp, force=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *comp*: :class:`str`
                Name of component to read
            *force*: ``True`` | {``False``}
                Option to re-read even if present in database
        :Outputs:
            *db*: :class:`DataExchanger`
                Data extracted from run matrix for comp *comp*
        :Versions:
            * 2025-07-25 ``@ddalle``: v1.0
        """
        # Get data dictionary
        dex = self.data
        dex = {} if self.data is None else dex
        # Check for component
        if (not force) and comp in dex:
            return dex[comp]
        # Read
        db = DataExchanger(self, comp)
        # Save it
        dex[comp] = db
        self.data = dex
        # Output
        return db

   # --- Archiving ---
    # Run ``--archive`` on one case
    def ArchiveCase(self, i: int, test: bool = False):
        r"""Perform ``--archive`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.CleanCase(i, test=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.archive(test)

    # Run ``--skeleton`` on one case
    def SkeletonCase(self, i: int, test: bool = False):
        r"""Perform ``--skeleton`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.SkeletonCase(i, test=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.skeleton(test)

    # Run ``--clean`` on one case
    def CleanCase(self, i: int, test: bool = False):
        r"""Perform ``--clean`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.CleanCase(i, test=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.clean(test)

    # Unarchive cases
    def UnarchiveCases(self, **kw):
        r"""Unarchive a list of cases

        :Call:
            >>> cntl.UnarchiveCases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
        :Versions:
            * 2017-03-13 ``@ddalle``: v1.0
            * 2023-10-20 ``@ddalle``: v1.1; arbitrary-depth *frun*
            * 2024-09-20 ``@ddalle``: v2.0; use CaseArchivist
        """
        # Test status
        test = kw.get("test", False)
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Print case name
            print(self.x.GetFullFolderNames(i))
            # Create the case folder
            self.make_case_folder(i)
            # Read case runner
            runner = self.ReadCaseRunner(i)
            # Unarchive!
            runner.unarchive(test)

   # --- DataBook Updaters ---
    # Databook updater
    @run_rootdir
    def UpdateFM(self, **kw):
        r"""Collect force and moment data

        :Call:
            >>> cntl.UpdateFM(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-12-12 ``@ddalle``: v1.0
            * 2014-12-22 ``@ddalle``: v2.0
                - Complete rewrite of DataBook class
                - Eliminate "Aero" class

            * 2017-04-25 ``@ddalle``: v2.1, add wildcards
            * 2018-10-19 ``@ddalle``: v3.0, rename from Aero()
        """
        # Get component option
        comp = kw.get("fm", kw.get("aero"))
        # If *comp* is ``True``, process all options
        if comp is True:
            comp = None
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("FM", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Read the existing data book.
            self.ReadDataBook(comp=comp)
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read an empty data book
            self.ReadDataBook(comp=[])
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Function to collect statistics from generic-property component
    @run_rootdir
    def UpdateCaseProp(self, **kw):
        r"""Update generic-property databook for one or more comp

        :Call:
            >>> cntl.UpdateCaseProp(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *prop*: {``None``} | :class:`str`
                Wildcard to subset list of ``"Prop"`` components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2022-04-08 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("prop")
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("CaseProp", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Make sure databook is present
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Function to collect statistics from generic-property component
    @run_rootdir
    def UpdatePyFuncDataBook(self, **kw):
        r"""Update Python function databook for one or more comp

        :Call:
            >>> cntl.UpdatePyFuncDataBook(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *prop*: {``None``} | :class:`str`
                Wildcard to subset list of ``"PyFunc"`` components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2022-04-10 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("dbpyfunc")
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("PyFunc", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Make sure databook is present
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update line loads
    @run_rootdir
    def UpdateLL(self, **kw):
        r"""Update one or more line load data books

        :Call:
            >>> cntl.UpdateLL(ll=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *ll*: {``None``} | :class:`str`
                Optional name of line load component to update
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *pbs*: ``True`` | {``False``}
                Whether or not to calculate line loads with PBS scripts
        :Versions:
            * 2016-06-07 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: v1.1, Add *pbs* flag
            * 2017-04-25 ``@ddalle``: v1.2
                - Removed *pbs*
                - Added ``--delete``
        """
        # Get component option
        comp = kw.get("ll")
        # Check for True or False
        if comp is True:
            # Update all components
            comp = None
        elif comp is False:
            # Exit
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        self.ReadConfig()
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteLineLoad(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateLineLoad(I, comp=comp, conf=self.config)

    @run_rootdir
    def UpdateSurfCp(self, **kw):
        r"""Collect surface pressure data

        :Call:
            >>> cntl.UpdateSurfCp(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-12-12 ``@ddalle``: v1.0
            * 2014-12-22 ``@ddalle``: v2.0
                - Complete rewrite of DataBook class
                - Eliminate "Aero" class

            * 2017-04-25 ``@ddalle``: v2.1, add wildcards
            * 2018-10-19 ``@ddalle``: v3.0, rename from Aero()
        """
        # Get component option
        comp = kw.get("surfcp")
        # If *comp* is ``True``, process all options
        if comp is True:
            comp = None
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("surfcp", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Read the existing data book.
            self.ReadDataBook(comp=comp)
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read an empty data book
            self.ReadDataBook(comp=[])
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update time series
    @run_rootdir
    def UpdateTS(self, **kw):
        r"""Update one or more time series data books

        :Call:
            >>> cntl.UpdateTS(ts=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *ts*: {``None``} | :class:`str`
                Optional name of time series component to update
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *pbs*: ``True`` | {``False``}
                Whether or not to calculate line loads with PBS scripts
        :Versions:
            * 2016-06-07 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: v1.1, Add *pbs* flag
            * 2017-04-25 ``@ddalle``: v1.2
                - Removed *pbs*
                - Added ``--delete``
        """
        # Get component option
        comp = kw.get("ts")
        # Check for True or False
        if comp is True:
            # Update all components
            comp = None
        elif comp is False:
            # Exit
            return
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("TimeSeries", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        self.ReadConfig()
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Read the existing data book.
            self.ReadDataBook(comp=comp)
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            self.ReadDataBook(comp=[])
            # self.ReadDataBook(comp=[])
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update TriqFM data book
    @run_rootdir
    def UpdateTriqFM(self, **kw):
        r"""Update one or more TriqFM data books

        :Call:
            >>> cntl.UpdateTriqFM(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("triqfm")
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteTriqFM(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateTriqFM(I, comp=comp)

    # Update TriqPointGroup data book
    @run_rootdir
    def UpdateTriqPoint(self, **kw):
        r"""Update one or more TriqPoint point sensor data books

        :Call:
            >>> cntl.UpdateTriqPoint(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("pt")
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteTriqPoint(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateTriqPoint(I, comp=comp)

   # --- DataBook Checkers ---
    # Function to check FM component status
    def CheckFM(self, **kw):
        r"""Display missing force & moment components

        :Call:
            >>> cntl.CheckFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "fm", kw.get(
                "aero", kw.get(
                    "checkFM", kw.get("check-fm", kw.get("check-db")))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("FM", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=comps)
        # Loop through the components
        for comp in comps:
            # Restrict the trajectory to cases in the databook
            self.DataBook[comp].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip marked errors
            if self.x.ERROR[i]:
                continue
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked":
                    continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook[comp]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook[comp]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += (
                        "    Extra case: %s\n" % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check LL component status
    def CheckLL(self, **kw):
        r"""Display missing line load components

        :Call:
            >>> cntl.CheckLL(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "ll", kw.get("checkLL", kw.get("check-ll", kw.get("check-db"))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("LineLoad", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadLineLoad(comp)
            # Restrict the trajectory to cases in the databook
            self.DataBook.LineLoads[comp].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked":
                    continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook.LineLoads[comp]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook.LineLoads[comp]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += (
                        "    Extra case: %s\n"
                        % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqFM(self, **kw):
        r"""Display missing TriqFM components

        :Call:
            >>> cntl.CheckTriqFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "triqfm", kw.get(
                "checkTriqFM", kw.get("check-triqfm", kw.get("check-db"))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("TriqFM", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadTriqFM(comp)
            # Restrict the trajectory to cases in the databook
            self.DataBook.TriqFM[comp][None].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked":
                    continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook.TriqFM[comp][None]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook.TriqFM[comp][None]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += (
                        "    Extra case: %s\n"
                        % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqPoint(self, **kw):
        r"""Display missing TriqPoint components

        :Call:
            >>> cntl.CheckTriqPoint(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "pt", kw.get("checkPt", kw.get("check-pt", kw.get("check-db"))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("TriqPoint", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Component list for text
        complist = []
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadTriqPoint(comp)
            # Get point group
            DBG = self.DataBook.TriqPoint[comp]
            # Loop through points
            for pt in DBG.pts:
                # Restrict the trajectory to cases in the databook
                DBG[pt].UpdateRunMatrix()
                # Add to the list
                complist.append("%s/%s" % (comp, pt))

        # Longest component name (plus room for the '/' char)
        maxcomp = max(map(len, complist)) + 1
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked":
                    continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get point group
                DBG = self.DataBook.TriqPoint[comp]
                # Loop through points
                for pt in DBG.pts:
                    # Get interface to component
                    DBc = DBG[pt]
                    # See if it's missing
                    j = DBc.x.FindMatch(self.x, i, **kw)
                    # Check for missing case
                    if j is None:
                        # Missing case
                        txt += (fmtc % ("%s/%s" % (comp, pt)))
                        txt += "missing\n"
                        continue
                    # Otherwise, check iteration
                    try:
                        # Get the recorded iteration number
                        nIter = DBc["nIter"][j]
                    except KeyError:
                        # No iteration number found
                        nIter = nLast
                    # Check for out-of date iteration
                    if nIter < nLast:
                        # Out-of-date case
                        txt += (fmtc % ("%s/%s" % (comp, pt)))
                        txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get group
            DBG = self.DataBook.TriqPoint[comp]
            # Loop through points
            for pt in DBG.pts:
                # Get component handle
                DBc = DBG[pt]
                # Initialize text
                txt = ""
                # Loop through database entries
                for j in range(DBc.x.nCase):
                    # Check for a find in master matrix
                    i = self.x.FindMatch(DBc.x, j, **kw)
                    # Check for a match
                    if i is None:
                        # This case is not in the run matrix
                        txt += (
                            "    Extra case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
                        continue
                    # Check for a user filter
                    if ku:
                        # Get the user value
                        uj = DBc[ku][j]
                        # Strip it
                        uj = uj.lstrip('@').lower()
                        # Check if it's blocked
                        if uj == "blocked":
                            # Blocked case
                            txt += (
                                "    Blocked case: %s\n"
                                % DBc.x.GetFullFolderNames(j))
                # If there is text, display the info
                if txt:
                    # Header
                    print("Checking point sensor '%s/%s'" % (comp, pt))
                    print(txt[:-1])

