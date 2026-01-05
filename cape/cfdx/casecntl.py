r"""
:mod:`cape.cfdx.casecntl`: Case control module
===============================================

This module contains templates for interacting with and executing
individual cases. Since this is one of the most highly customized
modules of the CAPE system, there are few functions here, and the
functions that are present are mostly templates.

In general, the :mod:`case` module is used for actually running the CFD
solver (and any additional binaries that may be required as part of the
run process), and it contains other capabilities for renaming files and
determining the settings for a particular casecntl. CAPE saves many settings
for the CFD solver and archiving in a file called ``case.json`` within
each case folder, which allows for the settings of one case to diverge
from the other cases in the same run matrix.

Actual functionality is left to individual modules listed below.

    * :mod:`cape.pycart.case`
    * :mod:`cape.pyfun.case`
    * :mod:`cape.pyover.case`
"""

# Standard library modules
import fnmatch
import importlib
import getpass
import glob
import json
import os
import pwd
import re
import shlex
import shutil
import signal
import sys
import time
import traceback
from collections import namedtuple
from datetime import datetime
from typing import Any, Optional, Tuple, Union

# System-dependent standard library
if os.name == "nt":
    resource = None
else:
    import resource

# Third-party
import numpy as np

# Local imports
from . import archivist
from . import cmdgen
from . import cmdrun
from . import queue
from .. import fileutils
from .archivist import CaseArchivist
from .casecntlbase import CaseRunnerBase
from .casedata import CaseFM, CaseResid, CaseSurfCp
from .caseutils import run_rootdir
from .cntlbase import CntlBase
from .ll import CaseLineLoad
from .logger import CaseLogger
from .triqfm import CaseTriqFM, CaseTriqPoint
from .options import RunControlOpts, ulimitopts
from .options.archiveopts import ArchiveOpts
from .options.funcopts import UserFuncOpts
from ..config import ConfigXML, SurfConfig
from ..dkit.rdb import DataKit
from ..errors import CapeRuntimeError
from ..gruvoc import umesh
from ..optdict import _NPEncoder
from ..trifile import Tri
from ..util import RangeString


# Constants

#: Regular expression for run log files written by CAPE
REGEX_RUNFILE = re.compile("run.([0-9][0-9]+).([0-9]+)")

# Log folder
LOGDIR = "cape"
# Name of file that marks a case as currently running
RUNNING_FILE = "RUNNING"
ACTIVE_FILE = os.path.join(LOGDIR, "ACTIVE")
# Name of file marking a case as in a failure status
FAIL_FILE = "FAIL"
# Name of file to stop at end of phase
STOP_PHASE_FILE = "CAPE-STOP-PHASE"
# Case settings
RC_FILE = "case.json"
# Run matrix conditions
CONDITIONS_FILE = "conditions.json"
# Logger files
LOGFILE_MAIN = "cape-main.log"
LOGFILE_VERBOSE = "cape-verbose.log"
# PBS/Slurm job ID file
JOB_ID_FILE = "jobID.dat"
JOB_ID_FILES = (
    os.path.join("cape", "jobID.dat"),
    "jobID.dat",
)
# Max number of IDs allowed per case
MAX_JOB_IDS = 20
# Constants
DEFAULT_SLEEPTIME = 10.0

# Return codes
IERR_OK = 0
IERR_CALL_RETURNCODE = 1
IERR_BOMB = 2
IERR_EXEC_NOT_FOUND = 8
IERR_PERMISSION = 13
IERR_UNKNOWN = 14
IERR_NANS = 32
IERR_INCOMPLETE_ITER = 65
IERR_RUN_PHASE = 128
IERR_RERUN_PHASE = 16

#: Class for beginning and end iter of an averaging window
#:
#: :Call:
#:      >>> window = IterWindow(ia, ib)
#: :Attributes:
#:      *ia*: :class:`int`| ``None``
#:          Iteration at start of window
#:      *ib*: :class:`int` | ``None``
#:          Iteration at end of window
IterWindow = namedtuple("IterWindow", ("ia", "ib"))

#: Class for file search regex and iteration status
#:
#: :Call:
#:     >>> fstat = FileSearchStatus(mtch, n)
#: :Attributes:
#:     *mtch*: :class:`re.Match` | ``None``
#:         Regular expression match object for file name
#:     *n*: :class:`int` | ``None``
#:         (Inferred) iteration for any file found
FileSearchStatus = namedtuple("FileSearchStatus", ("mtch", "n"))

#: Class for file name and iteration status
#:
#: :Call:
#:     >>> fstat = FileStatus(fname, n)
#: :Attributes:
#:     *fname*: :class:`str` | ``None``
#:         File name
#:     *n*: :class:`int` | ``None``
#:         (Inferred) iteration for any file found
FileStatus = namedtuple("FileStatus", ("fname", "n"))


# Help message for CLI
HELP_RUN_CFDX = r"""
Run solver CFD{X} for one case

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: console

        $ python -m cape.cfdx run [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit
"""


# Case runner class
class CaseRunner(CaseRunnerBase):
    r"""Class to handle running of individual CAPE cases

    :Call:
        >>> runner = CaseRunner(fdir=None)
    :Inputs:
        *fdir*: {``None``} | :class:`str`
            Optional case folder (by default ``os.getcwd()``)
    :Outputs:
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
    :Class attributes:
        * :attr:`_archivist_cls`
        * :attr:`_dex_cls`
        * :attr:`_logprefix`
        * :attr:`_modname`
        * :attr:`_nstart_max`
        * :attr:`_progname`
        * :attr:`_rc_cls`
        * :attr:`_zombie_files`
    :Attributes:
        * :attr:`archivist`
        * :attr:`cntl`
        * :attr:`j`
        * :attr:`job`
        * :attr:`logger`
        * :attr:`n`
        * :attr:`nr`
        * :attr:`rc`
        * :attr:`resid`
        * :attr:`returncode`
        * :attr:`root_dir`
        * :attr:`tic`
        * :attr:`workers`
    """
  # *** CONFIG ***
   # --- Class attributes ---
    # Attributes
    __slots__ = (
        "cntl",
        "j",
        "logger",
        "archivist",
        "child",
        "n",
        "nr",
        "job",
        "jobs",
        "rc",
        "resid",
        "returncode",
        "root_dir",
        "tic",
        "workers",
        "xi",
        "_mtime_case_json",
        "_dex_n_iter",
        "_dex_n_orders",
        "_dex_n_stats",
        "_dex_triqfile",
    )

    #: :class:`int`
    #: Maximum number of cases to start
    _nstart_max = 100

    #: :class:`list`\ [:class:`str`]
    #: List of file name patterns (glob) to check mod time for zombies
    _zombie_files = ["*.out"]

    #: :class:`str`
    #: Name of module
    _modname = "cfdx"
    #: :class:`str`
    #: Name of program or CFD solver
    _progname = "cfdx"
    #: :class:`str`
    #: Prefix for STDOUT files
    _logprefix = "run"

    #: :class:`type`
    #: Module-specific options class
    _rc_cls = RunControlOpts
    #: :class:`type`
    #: Class for archiving cases
    _archivist_cls = CaseArchivist
    #: :class:`type`
    #: Class for reading residual histories
    _resid_cls = CaseResid
    #: :class:`dict`\ [:class:`type`]
    #: Classes for reading data of various DataBook component types
    _dex_cls = {
        "fm": CaseFM,
        "lineload": CaseLineLoad,
        "surfcp": CaseSurfCp,
        "triqfm": CaseTriqFM,
        "triqpoint": CaseTriqPoint,
    }

   # --- __dunder__ ---
    def __init__(self, fdir: Optional[str] = None):
        r"""Initialization method

        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
        """
        # Default root folder
        if fdir is None:
            # Use current directory (usual case)
            fdir = os.getcwd()
        elif not os.path.isabs(fdir):
            # Absolutize relative to PWD
            fdir = os.path.abspath(fdir)
        # Save root folder
        self.root_dir = fdir
        #: :class:`cape.cfdx.cntl.Cntl` | ``None``
        #: Run matrix controller that owns this case
        self.cntl = None
        #: :class:`int` | ``None``
        #: Current phase number
        self.j = None
        #: :class:`cape.cfdx.logger.CaseLogger`
        #: Logger instance for this case
        self.logger = None
        #: :class:`cape.cfdx.archivist.CaseArchivist` | ``None``
        #: Archiver instance
        self.archivist = None
        #: :class:`int` | ``None``
        #: Current iteration number
        self.n = None
        #: :class:`int`
        #: Current restart iteration
        self.nr = None
        #: :class:`cape.cfdx.casedata.CaseResid`
        #: Residual history instance
        self.resid = None
        #: :class:`cape.cfdx.options.runctlopts.RunControlOpts`
        #: *RunControl* options for this case
        self.rc = None
        #: :class:`str` | ``None``
        #: Name of current PBS/Slurm job
        self.job = None
        #: :class:`dict`
        #: Dictionary of job statuses
        self.jobs = None
        #: :class:`bool`
        #: Whether or not to check PBS/Slurm job status
        self.qstat = True
        #: :class:`float`
        #: Reference timer
        self.tic = None
        #: :class:`dict` | ``None``
        #: Dictionary of conditions for this case
        self.xi = None
        #: :class:`int`
        #: Return code of last system command
        self.returncode = IERR_OK
        #: :class:`list`\ [:class:`int`]
        #: List of concurrent worker Process IDs
        self.workers = []
        # Set private slots
        self._mtime_case_json = 0.0
        self._dex_n_iter = None
        self._dex_n_orders = None
        self._dex_n_stats = None
        self._dex_triqfile = None
        self._dex_sourcefile = None
        self._dex_source = None
        # Other inits
        self.init_post()

    def __str__(self) -> str:
        r"""String method

        :Versions:
            * 2024-08-26 ``@ddalle``: v1.0
        """
        # Get the case name
        frun = self.get_case_name()
        # Get class handle
        cls = self.__class__
        # Include module
        return f"<{cls.__module__}.{cls.__name__} '{frun}'>"

    def __repr__(self) -> str:
        r"""Representation method

        :Versions:
            * 2024-08-26 ``@ddalle``: v1.0
        """
        # Get the case name
        frun = self.get_case_name()
        # Get class handle
        cls = self.__class__
        # Literal representation
        return f"{cls.__module__}('{frun}')"

   # --- Init hooks ---
    def init_post(self):
        r"""Custom initialization hook

        :Call:
            >>> runner.init_post()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-28 ``@ddalle``: v1.0
        """
        pass

  # *** RUN CONTROL ***
   # --- Start ---
    # Start case or submit
    @run_rootdir
    def start(self):
        r"""Start or submit initial job

        :Call:
            >>> ierr, job_id = runner.start()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code; ``0`` for success
            *job_id*: ``None`` | :class:`int`
                PBS/Slurm job ID number if appropriate
        :Versions:
            * 2023-06-23 ``@ddalle``: v1.0
        """
        rc = self.read_case_json()
        # Get phase index
        j = self.get_phase_next()
        # Log progress
        self.log_verbose(f"phase={j}")
        # Get script name
        fpbs = self.get_pbs_script(j)
        # Check submission options
        if rc.get_slurm(j):
            # Verbose log
            self.log_verbose("submitting slurm job")
            # Submit case
            job_id = queue.psbatch(fpbs)
            # Log
            self.log_both(f"submitted slurm job {job_id}")
            # Output
            return IERR_OK, job_id
        elif rc.get_qsub(j):
            # Verbose log
            self.log_verbose("submitting PBS job")
            # Submit case
            job_id = queue.pqsub(fpbs)
            # Log
            self.log_both(f"submitted PBS job {job_id}")
            # Output
            return IERR_OK, job_id
        else:
            # Log
            self.log_both("running in same process")
            # Start case
            ierr = self.run()
            # Output
            return ierr, None

    # Main loop
    @run_rootdir
    def run(self):
        r"""Setup and run appropriate solver commands

        :Call:
            >>> ierr = runner.run()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code; ``0`` for success
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0
            * 2021-10-08 ``@ddalle``: v1.1 (``run_overflow``)
            * 2023-06-21 ``@ddalle``: v2.0; instance method
            * 2024-05-26 ``@ddalle``: v2.1; more exit causes
            * 2025-06-24 ``@ddalle``: v2.2; add rerunable-phase check
        """
        # Log startup
        self.log_verbose(f"start {self._cls()}.run()")
        # Check if case is already running
        self.assert_not_running()
        # Mark case running
        self.mark_running()
        # Start a timer
        self.init_timer()
        # Log beginning
        self.log_main(f"{self._cls()}.run()")
        self.log_verbose(f"{self._cls()}.run() phase loop")
        # Initialize start counter
        nstart = 0
        # Loop until case exits, fails, or reaches start count limit
        while nstart < self._nstart_max:
            # Determine the phase
            j = self.get_phase_next()
            # Write start time
            self.write_start_time(j)
            # Prepare files as needed
            self.prepare_files(j)
            # Prepare environment variables
            self.prepare_env(j)
            # Run *PreShellCmds* hook
            self.run_pre_shell_cmds(j)
            self.run_pre_pyfuncs(j)
            # Run appropriate commands
            try:
                # Log
                self.log_both(f"running phase {j}")
                # Run primary
                self.run_phase_main(j)
            except Exception as e:
                # Log failure encounter
                self.log_both(f"error during phase {j}")
                self.log_verbose(f"{e.__class__}: " + ' '.join(e.args))
                self.log_verbose(traceback.format_exc())
                # Failure
                self.mark_failure("run_phase")
                # Stop running marker
                self.mark_stopped()
                # Return code
                return IERR_RUN_PHASE
            # Run *PostShellCmds* hook
            self.run_post_shell_cmds(j)
            self.run_post_pyfuncs(j)
            # Clean up files
            self.finalize_files(j)
            # Save time usage
            self.write_user_time(j)
            # Check for other errors
            ierr = self.get_returncode()
            # If nonzero
            if ierr != IERR_OK:
                # Log return code
                self.log_both("unsuccessful exit")
                self.log_both(f"returncode={ierr}")
                # Stop running case
                self.mark_stopped()
                # Return code
                return ierr
            # Update start counter
            nstart += 1
            # Check for non-rerunable phase
            if (not self.check_phase(j)) and not self.check_phase_rerunable(j):
                # Log message
                self.log_both(f"Phase {j} not complete and not rerunable")
                # Failure
                self.mark_failure(f"Rerun phase {j}")
                self.mark_stopped()
                # Pretend that case has been restarted
                return IERR_RERUN_PHASE
            # Check for explicit exit
            if self.check_exit(j):
                # Log
                self.log_verbose("exiting case loop")
                break
            # Submit new PBS/Slurm job if appropriate
            if self.resubmit_case(j):
                # If new job started, this one should stop
                self.log_verbose("exiting phase loop b/c new job submitted")
                break
        # Remove the RUNNING file
        self.mark_stopped()
        # Check for completion
        if self.check_complete():
            # Log
            self.log_both("case completed")
            # Submit additional jobs if appropriate
            self.run_more_cases()
        # Return code
        return IERR_OK

   # --- Main phase loop ---
    # Run a phase
    def run_phase(self, j: int) -> int:
        r"""Run one phase using appropriate commands

        :Call:
            >>> ierr = runner.run_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2023-06-05 ``@ddalle``: v1.0 (``pyover``)
            * 2023-06-14 ``@ddalle``: v1.0
        """
        # Generic version
        return IERR_OK

   # --- Job control ---
    # Resubmit a case, if appropriate
    def resubmit_case(self, j0: int):
        r"""Resubmit a case as a new job if appropriate

        :Call:
            >>> q = runner.resubmit_case(j0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j0*: :class:`int`
                Index of phase most recently run prior
                (may differ from current phase)
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not a new job was submitted to queue
        :Versions:
            * 2022-01-20 ``@ddalle``: v1.0 (:mod:`cape.pykes.case`)
            * 2023-06-02 ``@ddalle``: v1.0
            * 2024-05-25 ``@ddalle``: v1.1; rename options
        """
        # Read settings
        rc = self.read_case_json()
        # Get current phase (after run_phase())
        j1 = self.get_phase_next()
        # Get name of script for next phase
        fpbs = self.get_pbs_script(j1)
        # Job submission options
        qsub0 = rc.get_qsub(j0) or rc.get_slurm(j0)
        qsub1 = rc.get_qsub(j1) or rc.get_slurm(j1)
        # Trivial case if phase *j* is not submitted
        if not qsub1:
            return False
        # Check if *j1* is submitted and not *j0*
        if not qsub0:
            # Submit new phase
            _submit_job(rc, fpbs, j1)
            return True
        # If rerunning same phase, check the *Continue* option
        if j0 == j1:
            if rc.get_RunControlOpt("ResubmitSamePhase", j0):
                # Rerun same phase as new job
                _submit_job(rc, fpbs, j1)
                return True
            else:
                # Don't submit new job (continue current one)
                self.log_verbose(
                    f"continuing phase {j0} in same job " +
                    "because ResubmitSamePhase=False")
                return False
        # Now we know we're going to new phase; check the *Resubmit* opt
        if rc.get_RunControlOpt("ResubmitNextPhase", j0):
            # Submit phase *j1* as new job
            _submit_job(rc, fpbs, j1)
            return True
        else:
            # Continue to next phase in same job
            self.log_verbose(
                f"continuing to phase {j1} in same job " +
                "because ResubmitNextPhase=")
            return False

    # Delete job and remove running file
    def stop_case(self):
        r"""Stop a case by deleting PBS job and removing ``RUNNING`` file

        :Call:
            >>> runner.stop_case()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2014-12-27 ``@ddalle``: v1.0 (``StopCase()``)
            * 2023-06-20 ``@ddalle``: v1.1; instance method
            * 2023-07-05 ``@ddalle``: v1.2; use CaseRunner.get_job_id()
        """
        # Get the config
        rc = self.read_case_json()
        # Delete RUNNING file if appropriate
        self.mark_stopped()
        # Get the job number
        jobID = self.get_job_id()
        # Try to delete it
        if rc.get_slurm(self.j):
            # Log message
            self.log_verbose(f"scancel {jobID}")
            # Delete Slurm job
            queue.scancel(jobID)
        elif rc.get_qsub(self.j):
            # Log message
            self.log_verbose(f"qdel {jobID}")
            # Delete PBS job
            queue.qdel(jobID)

   # --- Case markers ---
    # Mark a cases as running
    @run_rootdir
    def mark_running(self):
        r"""Check if cases already running and create ``RUNNING`` otherwise

        :Call:
            >>> runner.mark_running()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-02 ``@ddalle``: v1.0
            * 2023-06-20 ``@ddalle``: v1.1; instance method, no check()
        """
        # Log message
        self.log_verbose("case running")
        # Create RUNNING file
        fileutils.touch(RUNNING_FILE)

    # Mark a case as "active"
    @run_rootdir
    def mark_active(self):
        r"""Mark a case as active, a subset of RUNNING

        :Call:
            >>> runner.mark_active()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Conroller to run one case of solver
        :Versions:
            * 2025-03-11 ``@ddalle``: v1.0
        """
        # Log message
        self.log_verbose("case active")
        # Create log dir
        fileutils.touch(ACTIVE_FILE)

    # General function to mark failures
    @run_rootdir
    def mark_failure(self, msg: str = "no details"):
        r"""Mark the current folder in failure status using ``FAIL`` file

        :Call:
            >>> runner.mark_failure(msg="no details")
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *msg*: ``{"no details"}`` | :class:`str`
                Error message for output file
        :Versions:
            * 2023-06-02 ``@ddalle``: v1.0
            * 2023-06-20 ``@ddalle``: v1.1; instance method
        """
        # Ensure new line
        txt = msg.rstrip("\n") + "\n"
        # Log message
        self.log_both(f"error: {txt}")
        # Append message to failure file
        open(FAIL_FILE, "a+").write(txt)

    # Delete running file if appropriate
    @run_rootdir
    def mark_stopped(self):
        r"""Delete the ``RUNNING`` file if it exists

        :Call:
            >>> runner.mark_stopped()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-02 ``@ddalle``: v1.0
            * 2024-08-03 ``@ddalle``: v1.1; add log message
        """
        # Log
        self.log_verbose("case stopped")
        # Check if file exists
        if os.path.isfile(RUNNING_FILE):
            # Delete it
            os.remove(RUNNING_FILE)

    # Delete active file if appropriate
    @run_rootdir
    def mark_inactive(self):
        r"""Delete the ``ACTIVE`` file if it exists

        :Call:
            >>> runner.mark_inactive()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2025-03-11 ``@ddalle``: v1.0
        """
        # Check if file exists
        if os.path.isfile(ACTIVE_FILE):
            # Log
            self.log_verbose("case no longer active")
            # Delete it
            os.remove(ACTIVE_FILE)

    # Check if case already running
    @run_rootdir
    def assert_not_running(self):
        r"""Check if a case is already running, raise exception if so

        :Call:
            >>> runner.assert_not_running()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-02 ``@ddalle``: v1.0
            * 2023-06-20 ``@ddalle``: v1.1; instance method
            * 2024-06-16 ``@ddalle``: v2.0; was ``check_running()``
        """
        # Check if case is running
        if self.check_running():
            # Log message
            self.log_verbose("case already running")
            # Case already running
            raise CapeRuntimeError('Case already running!')

   # --- Hooks ---
    # Run "PostPythonFuncs" hook
    def run_post_pyfuncs(self, j: int):
        r"""Run *PostPythonFuncs* before :func:`run_phase`

        :Call:
            >>> v = runner.run_post_pyfuncs(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *v*: **any**
                Output of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Get *PostPythonFuncs*
        funclist = rc.get_opt("PostPythonFuncs", j=j, vdef=[])
        # Log
        self.log_verbose(f"running {len(funclist)} PostPythonFuncs")
        # Loop through functions
        for funcspec in funclist:
            # Run function
            self.exec_modfunc(funcspec)

    # Run "PostShellCmds" hook
    def run_post_shell_cmds(self, j: int):
        r"""Run *PostShellCmds* after successful :func:`run_phase` exit

        :Call:
            >>> runner.run_post_shell_cmds(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2023-07-17 ``@ddalle``: v1.0
            * 2024-08-21 ``@ddalle``: v1.1; log messages
        """
        # Read settings
        rc = self.read_case_json()
        # Get "PostShellCmds"
        cmdlist = rc.get_opt("PostShellCmds", j=j, vdef=[])
        cmdlist = [] if cmdlist is None else cmdlist
        # Log counter
        self.log_verbose(f"running {len(cmdlist)} PostShellCmds")
        # Get new status
        j1 = self.get_phase_next()
        n1 = self.get_iter()
        # Run post commands
        for cmdj, cmdv in enumerate(cmdlist):
            # Create log file name
            flogbase = "postcmd%i.%02i.%i." % (cmdj, j1, n1)
            fout = flogbase + "out"
            ferr = flogbase + "err"
            # Check if we were given a string
            is_str = isinstance(cmdv, str)
            # Execute command
            self.callf(cmdv, f=fout, e=ferr, shell=is_str)

    # Run *PrePythonFuncs* hook
    def run_pre_pyfuncs(self, j: int):
        r"""Run *PrePythonFuncs* before :func:`run_phase`

        :Call:
            >>> v = runner.run_pre_pyfuncs(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *v*: **any**
                Output of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Get *PrePythonFuncs*
        funclist = rc.get_opt("PrePythonFuncs", j=j, vdef=[])
        # Log
        self.log_verbose(f"running {len(funclist)} PrePythonFuncs")
        # Loop through functions
        for funcspec in funclist:
            # Run function
            self.exec_modfunc(funcspec)

    # Run "PreShellCmds" hook
    def run_pre_shell_cmds(self, j: int):
        r"""Run *PreShellCmds* before :func:`run_phase`

        :Call:
            >>> runner.run_pre_shell_cmds(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-02-28 ``@aburkhea``: v1.0;
        """
        # Read settings
        rc = self.read_case_json()
        # Get "PreShellCmds"
        cmdlist = rc.get_opt("PreShellCmds", j=j, vdef=[])
        cmdlist = [] if cmdlist is None else cmdlist
        # Log counter
        self.log_verbose(f"running {len(cmdlist)} PreShellCmds")
        # Get new status
        j1 = self.get_phase_next()
        n1 = self.get_iter()
        n1 = 0 if n1 is None else n1
        # Run pre commands
        for cmdj, cmdv in enumerate(cmdlist):
            # Create log file name
            flogbase = "precmd%i.%02i.%i." % (cmdj, j1, n1)
            fout = flogbase + "out"
            ferr = flogbase + "err"
            # Check if we were given a string
            is_str = isinstance(cmdv, str)
            # Execute command
            self.callf(cmdv, f=fout, e=ferr, shell=is_str)

   # --- Functions ---
    def exec_modfunc(self, funcspec: Union[dict, str]) -> Any:
        r"""Call an arbitrary Python function

        :Call:
            >>> v = runner.exec_modfunc(funcspec)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *v*: **any**
                Output of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Read *Cntl* instance
        cntl = self.read_cntl()
        # Get name
        if isinstance(funcspec, dict):
            # Parse
            funcspec = UserFuncOpts(funcspec)
            # Get name
            funcname = funcspec.get("name")
        else:
            # Only given name
            funcname = funcspec
        # Log message
        self.log_verbose(f"running PythonFunc {funcname}()")
        # Call the thing
        try:
            v = cntl.exec_cntlfunction(funcspec)
            return v
        except Exception as e:
            msg = f"PythonFunc {funcname} failed\n{e.args}\n"
            print(msg)
            msg += traceback.format_exc()
            self.log_verbose(msg)

  # *** SYSTEM INTERFACE ***
   # --- Runners (multiple-use) ---
    # Mesh generation
    def run_aflr3(self, j: int, proj: str, fmt='lb8.ugrid'):
        r"""Create volume mesh using ``aflr3``

        This function looks for several files to determine the most
        appropriate actions to take:

            * ``{proj}.i.tri``: Triangulation file
            * ``{proj}.surf``: AFLR3 surface file
            * ``{proj}.aflr3bc``: AFLR3 boundary conditions
            * ``{proj}.xml``: Surface component ID mapping file
            * ``{proj}.{fmt}``: Output volume mesh
            * ``{proj}.FAIL.surf``: AFLR3 surface indicating failure

        If the volume grid file already exists, this function takes no
        action. If the ``surf`` file does not exist, the function
        attempts to create it by reading the ``tri``, ``xml``, and
        ``aflr3bc`` files using :class:`cape.trifile.Tri`.  The function
        then calls :func:`cape.bin.aflr3` and finally checks for the
        ``FAIL`` file.

        :Call:
            >>> runner.run_aflr3(j, proj, fmt='lb8.ugrid')
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
            *proj*: :class:`str`
                Project root name
            *fmt*: {``"lb8.ugrid"``} | :class:`str`
                AFLR3 volume mesh format
        :Versions:
            * 2016-04-05 ``@ddalle``: v1.0 (``CaseAFLR3()``)
            * 2023-06-02 ``@ddalle``: v1.1; use ``get_aflr3_run()``
            * 2023-06-20 ``@ddalle``: v1.1; instance method
            * 2024-08-22 ``@ddalle``: v1.2; add log messages
        """
        # Get iteration
        n = self.get_iter()
        # Check for initial run
        if j > 0:
            # Don't run AFLR3 if >0 iterations already complete
            return
        # Log message
        self.log_verbose("checking for ``aflr3`` settings")
        # Read settings
        rc = self.read_case_json()
        # Check for option to run AFLR3
        if not rc.get_aflr3_run(j=0):
            # AFLR3 not requested for this run
            return
        # Get iteration number
        n = self.get_iter()
        # Check for initial run
        if n:
            return
        # Log message
        self.log_verbose(f"preparing to run ``aflr3`` at phase {j}")
        # File names
        ftri = '%s.i.tri' % proj
        fsurf = '%s.surf' % proj
        fbc = '%s.aflr3bc' % proj
        fxml = '%s.xml' % proj
        fvol = '%s.%s' % (proj, fmt)
        ffail = "%s.FAIL.surf" % proj
        # Exit if volume exists
        if os.path.isfile(fvol):
            return
        # Check for file availability
        if not os.path.isfile(fsurf):
            # Check for the triangulation to provide a nice error msg
            if not os.path.isfile(ftri):
                msg = (
                    "missing AFLR3 input file candidates: " +
                    f"{ftri} or {fsurf}")
                self.log_both(msg)
                raise ValueError(msg)
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
        # Generate AFLR3 command
        cmdi = cmdgen.aflr3(opts=rc)
        # Run it
        self.callf(cmdi, f="aflr3.out", e="aflr3.out")
        # Check for failure; aflr3 returns 0 status even on failure
        if os.path.isfile(ffail):
            # Remove RUNNING file
            self.mark_stopped()
            # Create failure file
            self.mark_failure("aflr3")
            # Error message
            msg = f"aflr3 failure: found '{ffail}'"
            self.log_both(msg)
            raise RuntimeError(msg)

    # Function to run ``comp2tri``, if appropriate
    def run_comp2tri(self, j: int) -> int:
        r"""Run ``comp2tri`` to manipulate Cart3D geometries

        :Call:
            >>> ierr = runner.run_comp2tri(j, proj="Components")
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
            *proj*: {``'Components'``} | :class:`str`
                Project root name
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Check if this command should be run
        if not rc.get_comp2tri_run():
            return 0
        # Generate AFLR3 command
        cmdi = cmdgen.comp2tri(opts=rc, j=j)
        # Run program
        ierr = self.callf(cmdi)
        # Return code
        return ierr

    # Function to intersect geometry if appropriate
    def run_intersect(self, j: int, proj: str = "Components"):
        r"""Run ``intersect`` to combine surface triangulations

        This is a multi-step process in order to preserve all the
        component IDs of the input triangulations. Normally
        ``intersect`` requires each intersecting component to have a
        single component ID, and each component must be a water-tight
        surface.

        Cape utilizes two input files, ``Components.c.tri``, which is
        the original triangulation file with intersections and the
        original component IDs, and ``Components.tri``, which maps each
        individual original ``tri`` file to a single component. The
        files involved are tabulated below.

        * ``Components.tri``: Intersecting comps, each w/ single compID
        * ``Components.c.tri``: Original intersecting tris and compIDs
        * ``Components.o.tri``: Output of ``intersect`` w/ few compIDs
        * ``Components.i.tri``: Orig compIDs mapped to intersected tris

        More specifically, these files are ``{proj}.i.tri``, etc.; the
        default project name is ``"Components"``.  This function also
        calls the Chimera Grid Tools program ``triged`` to remove unused
        nodes from the intersected triangulation and optionally remove
        small triangles.

        :Call:
            >>> runner.run_intersect(j, proj="Components")
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
            *proj*: {``'Components'``} | :class:`str`
                Project root name
        :See also:
            * :class:`cape.trifile.Tri`
            * :func:`cape.cfdx.cmdgen.intersect`
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0 (``CaseIntersect``)
            * 2016-04-05 ``@ddalle``: v1.1; generalize to ``cfdx``
            * 2023-06-21 ``@ddalle``: v1.2; update name, instance method
            * 2024-08-22 ``@ddalle``: v1.3; add log messages
        """
        # Exit if not phase zero
        if j > 0:
            return
        # Log message
        self.log_verbose("checking for ``intersect`` settings")
        # Read settings
        rc = self.read_case_json()
        # Check for intersect status.
        if not rc.get_intersect():
            return
        # Get iteration number
        n = self.get_iter()
        # Check for initial run
        if n:
            return
        # Log message
        self.log_verbose(f"preparing to run ``intersect`` at phase {j}")
        # Triangulation file names
        ftri  = "%s.tri" % proj
        fftri = "%s.f.tri" % proj
        fotri = "%s.o.tri" % proj
        fctri = "%s.c.tri" % proj
        fatri = "%s.a.tri" % proj
        futri = "%s.u.tri" % proj
        fitri = "%s.i.tri" % proj
        # Check for triangulation file.
        if os.path.isfile(fitri):
            # Note this.
            self.log_verbose(f"'{fitri}' exists; aborting intersect")
            return
        # Set file names
        rc.set_intersect_i(ftri)
        rc.set_intersect_o(fotri)
        # Run intersect
        if os.path.isfile(fotri):
            # Status update
            self.log_verbose(f"'{fotri}' exists; skipping to post-processing")
        else:
            # Get command
            cmdi = cmdgen.intersect(rc)
            # Runn it
            self.callf(cmdi)
        # Read the original triangulation.
        tric = Tri(fctri)
        # Read the intersected triangulation.
        trii = Tri(fotri)
        # Read the pre-intersection triangulation.
        tri0 = Tri(ftri)
        # Map the Component IDs
        if os.path.isfile(fatri):
            # Just read the mapped file
            trii = Tri(fatri)
        elif os.path.isfile(futri):
            # Just read the mapped file w/o unused nodes
            trii = Tri(futri)
        else:
            # Perform the mapping
            trii.MapCompID(tric, tri0)
            # Add in far-field, sources, non-intersect comps
            if os.path.isfile(fftri):
                # Read the tri file
                trif = Tri(fftri)
                # Add it to the mapped triangulation
                trii.AddRawCompID(trif)
        # Intersect post-process options
        o_rm = rc.get_intersect_rm()
        o_triged = rc.get_intersect_triged()
        o_smalltri = rc.get_intersect_smalltri()
        # Trim unused trianlges (internal)
        trii.RemoveUnusedNodes(v=True)
        # Write trimmed triangulation
        trii.Write(futri)
        # Check if we should remove small triangles
        if o_rm and o_triged:
            # Status update
            self.log_verbose("removing small tris after intersect")
            # Input file to remove small tris
            infix = "RemoveSmallTris"
            fi = open('triged.%s.i' % infix, 'w')
            # Write inputs to file
            fi.write('%s\n' % futri)
            fi.write('19\n')
            fi.write('%f\n' % rc.get("SmallArea", o_smalltri))
            fi.write('%s\n' % fitri)
            fi.write('1\n')
            fi.close()
            # Run triged to remove small tris
            self.callf(
                f"triged < triged.{infix}.i > triged.{infix}.o", shell=True)
        else:
            # Rename file
            os.rename(futri, fitri)

    # Function to verify if requested
    def run_verify(self, j: int, proj='Components'):
        r"""Run ``verify`` to check triangulation if appropriate

        This function checks the validity of triangulation in file
        ``"%s.i.tri" % proj``.  It calls :func:`cape.bin.verify`.

        :Call:
            >>> runner.run_verify(j, proj='Components', fpre='run')
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
            *proj*: {``'Components'``} | :class:`str`
                Project root name
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0; from :func:`run_flowCart`
            * 2016-04-05 ``@ddalle``: v1.1; generalize to :mod:`cape`
            * 2023-06-21 ``@ddalle``: v2.0; instance method
            * 2024-08-22 ``@ddalle``: v2.1; add log messages
        """
        # Exit if not phase zero
        if j > 0:
            return
        # Log message
        self.log_verbose("checking for ``verify`` settings")
        # Read settings
        rc = self.read_case_json()
        # Check for verify
        if not rc.get_verify():
            return
        # Get number of iterations
        n = self.get_iter()
        # Check for initial run
        if n:
            return
        # Log message
        self.log_verbose(f"preparing to run ``verify`` at phase {j}")
        # Set file name
        rc.set_verify_i('%s.i.tri' % proj)
        # Create command
        cmdi = cmdgen.verify(rc)
        # Run it
        self.callf(cmdi)

    # Function to run triload
    def run_triload(self, ifile: str, ofile: str) -> int:
        r"""Run the ``triloadCmd`` executable

        :Call:
            >>> ierr = runner.run_triload(ifile, ofile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ifile*: :class:`str`
                Name of ``triload`` input file
            *ofile*: :class:`str`
                Name of file for ``triload`` STDOUT
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-01-22 ``@ddalle``: v1.0
        """
        # Check for executable
        cmdrun.find_executable("triloadCmd", "CGT triload executable")
        # Run function and return its exit code
        return self.callf(["triloadCmd"], i=ifile, f=ofile)

    # Function to run mixsur
    def run_mixsur_tri(self, ifile: str, ofile: str) -> int:
        r"""Create ``grid.i.tri`` using `mixsur` executable

        :Call:
            >>> ierr = runner.run_mixsur_tri(ifile, ofile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ifile*: :class:`str`
                Name of ``mixsur`` input file
            *ofile*: :class:`str`
                Name of file for ``usurp`` STDOUT
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check for executable
        cmdrun.find_executable("mixsur", "CGT mixsur executable")
        # Run function and use its return code
        return self.callf(["mixsur"], i=ifile, f=ofile)

    # Function to run overint
    def run_overint_triq(self, ifile: str, ofile: str) -> int:
        r"""Create ``grid.i.triq`` using `overint` executable

        :Call:
            >>> ierr = runner.run_overint_triq(ifile, ofile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ifile*: :class:`str`
                Name of ``usurp`` input file
            *ofile*: :class:`str`
                Name of file for ``usurp`` STDOUT
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check for executable
        cmdrun.find_executable("overint", "CGT overint executable")
        # Run function and use its return code
        return self.callf(["overint"], i=ifile, f=ofile)

    # Function to run usurp
    def run_usurp_tri(self, ifile: str, ofile: str) -> int:
        r"""Create ``grid.i.tri`` using `usurp` executable

        :Call:
            >>> ierr = runner.run_usurp_tri(ifile, ofile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ifile*: :class:`str`
                Name of ``usurp`` input file
            *ofile*: :class:`str`
                Name of file for ``usurp`` STDOUT
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check for executable
        cmdrun.find_executable("usurp", "CGT usurp executable")
        # Construct function call
        cmdlist = ["usurp", "-v", "--watertight", "disjoin=yes"]
        # Run function and use its return code
        return self.callf(cmdlist, i=ifile, f=ofile)

    # Function to run usurp
    def run_usurp_triq(self, ifile: str, ofile: str) -> int:
        r"""Create ``grid.i.triq`` using `usurp` executable

        :Call:
            >>> ierr = runner.run_usurp_triq(ifile, ofile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ifile*: :class:`str`
                Name of ``usurp`` input file
            *ofile*: :class:`str`
                Name of file for ``usurp`` STDOUT
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Check for executable
        cmdrun.find_executable("usurp", "CGT usurp executable")
        # Construct function call
        cmdlist = ["usurp", "-v", "--use-map"]
        # Run function and use its return code
        return self.callf(cmdlist, i=ifile, f=ofile)

   # --- Shell/System ---
    # Run a function
    def callf(
            self,
            cmdi: list,
            f: Optional[str] = None,
            e: Optional[str] = None,
            i: Optional[str] = None,
            shell: bool = False) -> int:
        r"""Execute a function and save returncode

        :Call:
            >>> ierr = runner.callf(cmdi, f=None, e=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``None``} | :class:`str`
                Name of file to write STDOUT
            *e*: {*f*} | :class:`str`
                Name of file to write STDERR
            *i*: {``None``} | :class:`str`
                Name of file from which to read STDIN
            *shell*: ``True`` | {``False``}
                Option to run subprocess in shell
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2024-07-16 ``@ddalle``: v1.0
            * 2024-08-03 ``@ddalle``: v1.1; add log messages
            * 2025-01-22 ``@ddalle``: v1.2; add *i* option
        """
        # Log command
        self.log_main("> " + _shjoin(cmdi), parent=1)
        self.log_data(
            {
                "cmd": _shjoin(cmdi),
                "stdin": i,
                "stdout": f,
                "stderr": e,
                "cwd": os.getcwd()
            }, parent=1)
        # Run command
        ierr = cmdrun.callf(cmdi, f=f, e=e, i=i, shell=shell, check=False)
        # Save return code
        self.log_both(f"returncode={ierr}", parent=1)
        # Save return code
        self.returncode = ierr
        # Output
        return ierr

  # *** FILES/ENVIRONMENT ***
   # --- File prep and cleanup ---
    def prepare_files(self, j: int):
        r"""Prepare files for phase *j*

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase index
        :Versions:
            * 2021-10-21 ``@ddalle``: v1.0 (abstract ``cfdx`` method)
        """
        pass

    # Clean up after case
    def finalize_files(self, j: int):
        r"""Clean up files after running one cycle of phase *j*

        :Call:
            >>> runner.finalize_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0 (abstract method)
        """
        pass

    # Process the STDOUT file
    def finalize_stdoutfile(self, j: int):
        r"""Move the ``{_progname}.out`` file to ``run.{j}.{n}``

        :Call:
            >>> runner.finalize_stdoutfile(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-04-07 ``@ddalle``: v1.0
        """
        # STDOUT file
        fout = self.get_stdout_filename()
        # Iteration number
        n = self.get_iter()
        # History remains in present folder
        fhist = f"{self._logprefix}.{j:02d}.{n}"
        # Assuming that worked, move the temp output file.
        if os.path.isfile(fout):
            if not os.path.isfile(fhist):
                # Move the file
                os.rename(fout, fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)

   # --- Environment ---
    # Function to set the environment
    def prepare_env(self, j: int):
        r"""Set environment vars, alter resource limits (``ulimit``)

        This function relies on the system module :mod:`resource`

        :Call:
            >>> runner.prepare_env(rc, i=0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :See also:
            * :func:`set_rlimit`
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0 (``PrepareEnvironment()``)
            * 2023-06-02 ``@ddalle``: v1.1; fix logic for appending
                - E.g. ``"PATH": "+$HOME/bin"``
                - This is designed to append to path

            * 2023-06-20 ``@ddalle``: v1.2; instance mthod
        """
        # Do nothing on Windows
        if resource is None:
            return
        # Read settings
        rc = self.read_case_json()
        # Loop through environment variables
        for key in rc.get('Environ', {}):
            # Get the environment variable
            val = rc.get_Environ(key, j)
            # Check if it stars with "+"
            if val.startswith("+"):
                # Remove preceding '+' signs
                val = val.lstrip('+')
                # Check if it's present
                if key in os.environ:
                    # Append to path
                    os.environ[key] += (os.path.pathsep + val.lstrip('+'))
                    continue
            # Log
            self.log_verbose(f'{key}="{val}"')
            # Set the environment variable from scratch
            os.environ[key] = val
        # Block size
        block = resource.getpagesize()
        # Set the stack size
        self.set_rlimit(resource.RLIMIT_STACK,   's', j, 1024)
        self.set_rlimit(resource.RLIMIT_CORE,    'c', j, block)
        self.set_rlimit(resource.RLIMIT_DATA,    'd', j, 1024)
        self.set_rlimit(resource.RLIMIT_FSIZE,   'f', j, block)
        self.set_rlimit(resource.RLIMIT_MEMLOCK, 'l', j, 1024)
        self.set_rlimit(resource.RLIMIT_NOFILE,  'n', j, 1)
        self.set_rlimit(resource.RLIMIT_CPU,     't', j, 1)
        self.set_rlimit(resource.RLIMIT_NPROC,   'u', j, 1)

    # Limit
    def set_rlimit(
            self,
            r: int,
            u: str,
            j: int = 0,
            unit: int = 1024):
        r"""Set resource limit for one variable

        :Call:
            >>> runner.set_rlimit(r, u, j=0, unit=1024)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *r*: :class:`int`
                Integer code of particular limit, from :mod:`resource`
            *u*: :class:`str`
                Name of limit to set
            *j*: {``0``} | :class:`int`
                Phase number
            *unit*: {``1024``} | :class:`int`
                Multiplier, usually for a kbyte
        :See also:
            * :mod:`cape.options.ulimit`
        :Versions:
            * 2016-03-13 ``@ddalle``: v1.0
            * 2021-10-21 ``@ddalle``: v1.1; check if Windows
            * 2023-06-20 ``@ddalle``: v1.2; was ``SetResourceLimit()``
        """
        # Get settings
        rc = self.read_case_json()
        # Get ``ulimit`` parameters
        ulim = rc.get("ulimit", {})
        # Get the value of the limit
        l = ulim.get_ulimit(u, j)
        # Log
        if l is not None:
            self.log_verbose(f"ulimit -{r} {l} (phase={j})")
        # Apply setting
        set_rlimit(r, ulim, u, j, unit)

   # --- Folders ---
    @run_rootdir
    def mkdir(self, dirname: str):
        r"""Create a folder (if needed); set permissions to match root

        :Call:
            >>> runner.mkdir(dirname)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *dirname*: :class:`str`
                Name of folder to create and/or modify
        :Versions:
            * 2025-07-28 ``@ddalle``: v1.0
        """
        # Check if folder exists
        if not os.path.isdir(dirname):
            # Create folder
            os.mkdir(dirname)
        # Get permissions of parent folder and current subfolder perms
        p1 = os.stat(self.root_dir).st_mode
        p2 = os.stat(dirname).st_mode
        # Match permissions to parent folder
        os.chmod(dirname, p1 | p2)

   # --- File manipulation ---
    # Copy a file
    def copy_file(self, src: str, dst: str, f: bool = False):
        r"""Copy a file and log results

        :Call:
            >>> runner.copy_file(src, dst, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *src*: :class:`str`
                Name of input file, before renaming
            *dst*: :class:`str`
                Name of renamed file
            *f*: ``True`` | {``False``}
                Option to overwrite existing *dst*
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Relative paths
        src_rel = self.relpath(src)
        dst_rel = self.relpath(dst)
        # Initial log
        self.log_verbose(f"copy '{src_rel}' -> '{dst_rel}'")
        # Validate source
        self.validate_srcfile(src)
        # Remove existing target, if any
        self.remove_link(dst, f=True)
        # Rename
        shutil.copy(src, dst)

    # Create a link
    def link_file(self, src: str, dst: str, f: bool = False):
        r"""Copy a link and log results

        :Call:
            >>> runner.link_file(src, dst, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *src*: :class:`str`
                Name of input file, before renaming
            *dst*: :class:`str`
                Name of renamed file
            *f*: ``True`` | {``False``}
                Option to overwrite existing *dst*
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Relative paths
        src_rel = self.relpath(src)
        dst_rel = self.relpath(dst)
        # Initial log
        self.log_verbose(f"link '{src_rel}' -> '{dst_rel}'", parent=1)
        # Validate source
        self.validate_srcfile(src)
        # Remove existing target, if any
        self.remove_link(dst, f=True)
        # Rename
        os.symlink(src, dst)

    # Rename a file
    def rename_file(self, src: str, dst: str, f: bool = False):
        r"""Rename a file and log results

        :Call:
            >>> runner.rename_file(src, dst, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *src*: :class:`str`
                Name of input file, before renaming
            *dst*: :class:`str`
                Name of renamed file
            *f*: ``True`` | {``False``}
                Option to overwrite existing *dst*
        :Versions:
            * 2024-08-13 ``@ddalle``: v1.0
        """
        # Relative paths
        src_rel = self.relpath(src)
        dst_rel = self.relpath(dst)
        # Initial log
        self.log_verbose(f"rename '{src_rel}' -> '{dst_rel}'", parent=1)
        # Validate source
        self.validate_srcfile(src)
        # Remove existing target, if any
        self.remove_link(dst, f=True)
        # Rename
        os.rename(src, dst)

    # Create empty file
    def touch_file(self, fname: str):
        r"""Create an empty file if necessary, or update mtime

        :Call:
            >>> runner.touch_file(fname)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file to "touch"
        :Version:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Create header for log message
        msg = "touch" if os.path.isfile(fname) else "create empty"
        # Log message
        self.log_verbose(f"{msg} '{fname}'", parent=1)
        # Action
        fileutils.touch(fname)

    # Remove a link
    def remove_link(self, dst: str, f: bool = False):
        r"""Delete a link [file if *f*] if it exists

        :Call:
            >>> runner.remove_link(dst, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *dst*: :class:`str`
                Name of link to delete
            *f*: ``True`` | {``False``}
                Option to overwrite *dst*, even if not a link
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Validate destination
        self.validate_dstfile(dst)
        # Relative to root
        dst_rel = self.relpath(dst)
        # Check for existing link
        if os.path.islink(dst):
            # Remove link
            self.log_verbose(f"removing link '{dst_rel}'", parent=1)
            os.remove(dst)
        # Check for existing file
        if os.path.isfile(dst):
            # Check for overwrite
            if f:
                # Replace (overwriten later)
                self.log_verbose(f"overwriting file '{dst_rel}'", parent=1)
                # Delete file
                os.remove(dst)
            else:
                msg = f"cannot overwrite '{dst_rel}'; file exists"
                self.log_verbose(msg, parent=1)
                raise FileExistsError(msg)

    # Check source file exists
    def validate_srcfile(self, src: str):
        r"""Check that *src* exists and is a file

        Checks that *src* is a file or a valid link.

        :Call:
            >>> runner.validate_srcfile(src)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *src*: :class:`str`
                Name of input file, before renaming
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Relative to root
        src_rel = self.relpath(src)
        # Check if existing file exists
        if os.path.isfile(src):
            # File exists
            return
        elif os.path.islink(src):
            # Bad link
            msg = f"file '{src_rel}' is a broken link"
            self.log_verbose(msg, parent=1)
            raise FileNotFoundError(msg)
        elif os.path.isdir(src):
            # Folder instead of file
            msg = f"'{src_rel}' is a folder instead of file"
            self.log_verbose(msg, parent=1)
            raise ValueError(msg)
        else:
            # File does not exists
            msg = f"file '{src_rel}' does not exist"
            self.log_verbose(msg, parent=1)
            raise FileNotFoundError(msg)

    # Check appropriate destination file
    def validate_dstfile(self, dst: str):
        r"""Check that *dst* is a valid destination for rename/copy

        Checks that *dst* is inside case folder

        :Call:
            >>> runner.validate_dstfile(dst)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *src*: :class:`str`
                Name of input file, before renaming
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Relative to root
        dst_rel = self.relpath(dst)
        # Check if outside root
        if dst_rel.startswith(".."):
            # Cannot copy/link outside of root
            msg = (
                f"invalid destination '{os.path.abspath(dst)}'; " +
                f"outside of root dir '{self.root_dir}'")
            self.log_verbose(msg, parent=1)
            raise ValueError(msg)

    # Get path relative to root
    def relpath(self, fname: str) -> str:
        r"""Get path to file relative to case root directory

        :Call:
            >>> frel = runner.relpath(fname)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                File name, relative to *PWD* or absolute
        :Outputs:
            *frel*: :class:`str`
                Path to *fname* relative to *runner.root_dir*
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Absolutize
        fabs = os.path.abspath(fname)
        # Relative to case root
        return os.path.relpath(fabs, self.root_dir)

   # --- File/Folder names ---
    @run_rootdir
    def get_pbs_script(self, j=None):
        r"""Get file name of PBS script

        ... or Slurm script or execution script

        :Call:
            >>> fpbs = runner.get_pbs_script(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fpbs*: :class:`str`
                Name of main script to run case
        :Versions:
            * 2014-12-01 ``@ddalle``: v1.0 (``pycart``)
            * 2015-10-19 ``@ddalle``: v1.0 (``pyfun``)
            * 2023-06-18 ``@ddalle``: v1.1; instance method
        """
        # Get file prefix
        prefix = f"run_{self._progname}."
        # Check phase
        if j is None:
            # Base file name; no search if *j* is None
            return prefix + "pbs"
        else:
            # Create phase-dependent file name
            fpbs = prefix + ("%02i.pbs" % j)
            # Check if file is present
            if os.path.isfile(fpbs):
                # Use file test to see if PBS depends on
                return fpbs
            else:
                # No phase-dependent script found
                return prefix + "pbs"

    # Function to get restart file
    def get_restart_file(self, j: Optional[int] = None) -> str:
        r"""Get the most recent restart file for phase *j*

        :Call:
            >>> restartfile = runner.get_restart_file(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2024-11-05 ``@ddalle``: v1.0
        """
        # Let's use the OVERFLOW value for default (it's the simplest)
        return "q.save"

    # Function to get working folder relative to root
    def get_working_folder(self) -> str:
        r"""Get working folder, ``.`` for generic solver

        :Call:
            >>> fdir = runner.get_working_folder()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fdir*: ``"."``
                Working folder relative to roo, where next phase is run
        :Versions:
            * 2024-08-11 ``@ddalle``: v1.0
        """
        return '.'

    # Function to get working folder, but '' instead of '.'
    def get_working_folder_(self) -> str:
        r"""Get working folder, but replace ``'.'`` with ``''``

        This results in cleaner results with :func:`os.path.join`.

        :Call:
            >>> fdir = runner.get_working_folder()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fdir*: ``""`` | :class:`str`
                Working folder relative to roo, where next phase is run
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
        """
        # Get working folder
        fdir = self.get_working_folder()
        # Replace "." with "" (otherwise leave *fdir* alone)
        return "" if fdir == "." else fdir

   # --- Search ---
    @run_rootdir
    def link_from_search(
            self,
            fname: str,
            pat_or_pats: Union[list, str],
            workdir: bool = True,
            regex: bool = False):
        # Convert to list
        pats = [pat_or_pats] if isinstance(pat_or_pats, str) else pat_or_pats
        # Perform search
        file_list = self._search(pats, workdir=workdir, regex=regex)
        # Exit if no match
        if len(file_list) == 0:
            return
        # Create link (overwrite if necessary)
        self.link_file(file_list[-1], fname, f=True)

    @run_rootdir
    def search(
            self,
            pat: str,
            workdir: bool = False,
            regex: bool = False,
            links: bool = False) -> list:
        r"""Search for files by glob and sort them by modification time

        :Call:
            >>> file_list = runner.search(pat, workdir=False, **kw)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pat*: :class:`str`
                File name pattern
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat*, sorted by mtime
        :Versions:
            * 2025-01-23 ``@ddalle``: v1.0
            * 2025-02-01 ``@ddalle``: v1.1; use _search()
        """
        return self._search([pat], workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_multi(
            self,
            pats: list,
            workdir: bool = False,
            regex: bool = False,
            links: bool = False) -> list:
        r"""Search for files by glob and sort them by modification time

        :Call:
            >>> file_list = runner.search_multi(pats)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pats*: :class:`list`\ [:class:`str`]
                List of file name patterns
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching any *pat* in *pats*, sorted by mtime
        :Versions:
            * 2025-01-23 ``@ddalle``: v1.0
            * 2025-02-01 ``@ddalle``: v1.1; use _search()
        """
        return self._search(pats, workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_workdir(
            self,
            pat: str,
            workdir: bool = True,
            regex: bool = False,
            links: bool = False) -> list:
        r"""Search for files by glob and sort them by modification time

        :Call:
            >>> file_list = runner.search_workdir(pat)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pat*: :class:`str`
                File name pattern
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat* in working folder, sorted by mtime
        :Versions:
            * 2025-01-23 ``@ddalle``: v1.0
            * 2025-02-01 ``@ddalle``: v1.1; use _search()
        """
        return self._search([pat], workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_workdir_multi(
            self,
            pats: list,
            workdir: bool = True,
            regex: bool = False,
            links: bool = False) -> list:
        r"""Search for files by glob and sort them by modification time

        :Call:
            >>> file_list = runner.search_workdir_multi(pats)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pats*: :class:`list`\ [:class:`str`]
                List of file name patterns
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat* in working folder, sorted by mtime
        :Versions:
            * 2025-01-23 ``@ddalle``: v1.0
            * 2025-02-01 ``@ddalle``: v1.1; use _search()
        """
        return self._search(pats, workdir=workdir, regex=regex)

    @run_rootdir
    def match_regex(
            self,
            pat: str,
            workdir: bool = False,
            regex: bool = True,
            links: bool = False):
        r"""Search for files by regex; return the match info for latest

        :Call:
            >>> mtch = runner.match_regex(pat)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pat*: :class:`str`
                File name pattern
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *mtch*: :class:`re.Match` | ``None``
                Match object for latest match to *pat*, if any
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
        """
        # Get matches
        fileset = self.search_regex(pat, workdir, regex, links)
        # Check for any
        if len(fileset) == 0:
            return None
        # Mattch the latest
        return re.fullmatch(pat, fileset[-1])

    @run_rootdir
    def search_regex(
            self,
            pat: str,
            workdir: bool = False,
            regex: bool = True,
            links: bool = False) -> list:
        r"""Search for files by regex and sort them by modification time

        :Call:
            >>> file_list = runner.search_regex(pat)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pat*: :class:`str`
                File name pattern
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat*, sorted by mtime
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
        """
        return self._search([pat], workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_regex_multi(
            self,
            pats: list,
            workdir: bool = False,
            regex: bool = True,
            links: bool = False) -> list:
        r"""Search for files by regex and sort them by modification time

        :Call:
            >>> file_list = runner.search_regex_multi(pats)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pats*: :class:`list`\ [:class:`str`]
                List of file name patterns
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching any *pat* in *pats*, sorted by mtime
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
        """
        return self._search(pats, workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_regex_workdir(
            self,
            pat: str,
            workdir: bool = True,
            regex: bool = True,
            links: bool = False) -> list:
        r"""Search for files by regex and sort them by modification time

        :Call:
            >>> file_list = runner.search_regex_workdir(pat)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pat*: :class:`str`
                File name pattern
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat* in working folder, sorted by mtime
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
        """
        return self._search([pat], workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def search_regex_workdir_multi(
            self,
            pats: list,
            workdir: bool = True,
            regex: bool = True,
            links: bool = False) -> list:
        r"""Search for files by regex and sort them by modification time

        :Call:
            >>> file_list = runner.search_regex_workdir_multi(pats)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pats*: :class:`list`\ [:class:`str`]
                List of file name patterns
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                Files matching *pat* in working folder, sorted by mtime
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
        """
        return self._search(pats, workdir=workdir, regex=regex, links=links)

    @run_rootdir
    def _search(
            self,
            pats: list,
            workdir: bool = False,
            regex: bool = False,
            links: bool = False) -> list:
        r"""Generic file search function

        :Call:
            >>> file_list = runner._search(pats, workdir, regex)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *pats*: :class:`list`\ [:class:`str`]
                List of file name patterns
            *workdir*: ``True`` | {``False``}
                Whether or not to search in work directory
            *regex*: ``True`` | {``False``}
                Whether or not to treat *pats* as regular expressions
            *links*: ``True`` | {``False``}
                Option to allow links in output
        :Outputs:
            *file_list*: :class:`list`\ [:class:`str`]
                List of matching files sorted by modification time
        :Versions:
            * 2025-02-01 ``@ddalle``: v1.0
            * 2025-02-05 ``@ddalle``: v1.1; remove linked from results
            * 2025-04-04 ``@ddalle``: v1.2; add *links* option
        """
        # Check *workdir* option
        if workdir:
            # Enter it
            os.chdir(self.get_working_folder())
        # Initialize set of matches
        fileset = set()
        # Get search function
        searchfunc = archivist.reglob if regex else glob.glob
        # Loop through patterns
        for pat in pats:
            # Find list of files matching pattern
            raw_list = searchfunc(pat)
            # Extend
            fileset.update(raw_list)
        # Remove links
        if not links:
            for fname in tuple(fileset):
                if os.path.islink(fname):
                    fileset.remove(fname)
        # Return sorted by mod time
        return archivist.sort_by_mtime(list(fileset))

   # --- STDOUT files ---
    # Get name of STDOUT file
    def get_stdout_filename(self) -> str:
        r"""Get standard STDOUT file name, e.g. ``fun3d.out``

        :Call:
            >>> fname = runner.get_stdout_filename()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fname*: :class:`str`
                Name of file
        :Versions:
            * 2025-04-07 ``@ddalle``: v1.0
        """
        return f"{self._progname}.out"

    # Get name of STDOUT file
    def get_stderr_filename(self) -> str:
        r"""Get standard STDERR file name, e.g. ``fun3d.err``

        :Call:
            >>> fname = runner.get_stderr_filename()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fname*: :class:`str`
                Name of file
        :Versions:
            * 2025-04-07 ``@ddalle``: v1.0
        """
        return f"{self._progname}.err"

  # *** FLOW VIZ ***
   # --- Iteration number ---
    def infer_file_niter(self, mtch) -> int:
        return self.get_restart_iter()

   # --- Iterations in a time-averaging window ---
    def infer_tavg_nstats(
            self,
            n: Optional[int] = None,
            fname: Optional[str] = None) -> int:
        r"""Infer num of iters averaged for output at iteration *n*

        :Call:
            >>> nstats = runner.infer_tavg_nstats(n)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: :class:`int`
                Iteration number
        :Outputs:
            *nstats*: :class:`int`
                Number of iterations averaged in viz file
        :Versions:
            * 2025-08-10 ``@ddalle``: v1.0
        """
        return 1

  # *** MESH ***
   # --- Volume ---
    def match_grid_file(self):
        r"""Get latest volume grid and regex match instance

        :Call:
            >>> re_match = runner.match_grid_file()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *re_match*: :class:`re.Match` | ``None``
                Regular expression groups, if any
        :Versions:
            * 2025-08-16 ``@ddalle``: v1.0
        """
        # Get regular expression of volume file matches
        pat = self.get_grid_regex()
        # Perform search
        mtch = self.match_regex(pat)
        # Output
        return mtch

    def match_surfgrid_file(self):
        r"""Get latest surface grid and regex match instance

        :Call:
            >>> re_match = runner.match_surfgrid_file()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *re_match*: :class:`re.Match` | ``None``
                Regular expression groups, if any
        :Versions:
            * 2025-08-16 ``@ddalle``: v1.0
        """
        # Get regular expression of volume file matches
        pat = self.get_surfgrid_regex()
        # Perform search
        mtch = self.match_regex(pat)
        # Output
        return mtch

    def get_grid_regex(self) -> str:
        r"""Return regular expression for volume grid files

        :Call:
            >>> pat = runner.get_grid_regex()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *pat*: :class:`str`
                Regular expression search pattern
        :Versions:
            * 2025-08-16 ``@ddalle``: v1.0
        """
        return r".*\.l?[br][48]l?\.ugrid"

   # --- Surface ---
    def get_surfgrid_regex(self) -> str:
        r"""Return regular expression for surface grid files

        :Call:
            >>> pat = runner.get_surfgrid_regex()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *pat*: :class:`str`
                Regular expression search pattern
        :Versions:
            * 2025-08-16 ``@ddalle``: v1.0
        """
        return r"x(\.([0-9]+))?\.(sur,srf,surf)"

  # *** VOLUME DATA ***
   # --- Volume ---
    def match_vol_file(self):
        r"""Get latest volume file and regex match instance

        :Call:
            >>> re_match = runner.match_vol_file()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *re_match*: :class:`re.Match` | ``None``
                Regular expression groups, if any
        :Versions:
            * 2025-01-24 ``@ddalle``: v1.0
        """
        # Get regular expression of volume file matches
        pat = self.get_vol_regex()
        # Perform search
        mtch = self.match_regex(pat)
        # Save it
        if mtch is not None:
            self._dex_source = "volume"
            self._dex_sourcefile = mtch.group()
        # Output
        return mtch

   # --- File name patterns ---
    def get_vol_regex(self) -> str:
        r"""Get regular expression that all volume output files match

        :Call:
            >>> regex = runner.get_vol_regex()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *regex*: :class:`str`
                Regular expression that all surface sol'n files match
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        raise NotImplementedError(
            "CaseRunner.get_vol_regex() is not implemented for "
            f"{self._modname}.CaseRunner")

  # *** SURFACE DATA ***
   # --- Surface ---
    def find_surf_file(self) -> FileSearchStatus:
        r"""Find CFD output file that contains the most recent surface data

        :Call:
            >>> fstat = runner.find_surf_file()
            >>> mtch, n = runner.find_surf_file()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *mtch*: :class:`re.Match` | ``None``
                Regular expression match instance
            *n*: :class:`int`
                Latest iteration contained in file
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Find the file
        mtch = self.match_surf_file()
        # Infer iteration number
        n = self.infer_file_niter(mtch)
        # Cache it
        self._dex_n_iter = n
        # Output
        return FileSearchStatus(mtch, n)

    def match_surf_file(self):
        r"""Get latest surface file and regex match instance

        :Call:
            >>> re_match = runner.match_surf_file()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *re_match*: :class:`re.Match` | ``None``
                Regular expression groups, if any
        :Versions:
            * 2025-01-24 ``@ddalle``: v1.0
        """
        # Get regular expression of exact matches
        pat = self.get_surf_regex()
        # Perform search
        mtch = self.match_regex(pat)
        # Save file name
        if mtch is not None:
            self._dex_source = "surface"
            self._dex_sourcefile = mtch.group()
        # Output
        return mtch

    def read_surf_data(self):
        ...

   # --- File name patterns ---
    def get_surf_regex(self) -> str:
        r"""Get regular expression that all surface output files match

        :Call:
            >>> regex = runner.get_surf_regex()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *regex*: :class:`str`
                Regular expression that all surface sol'n files match
        :Versions:
            * 2025-01-24 ``@ddalle``: v1.0
        """
        raise NotImplementedError(
            "CaseRunner.get_surf_regex() is not implemented for "
            f"{self._modname}.CaseRunner")

   # --- TriQ ---
    def prepare_triq(self) -> FileStatus:
        r"""Prepare the surface ``.triq`` file and return metadata

        :Call:
            >>> triqstats = runner.prepare_triq()
            >>> fname, n = runner.prepare_triq()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *triqstats*: :class:`FileStatus`
                Surface ``.triq`` file name
            *triqstats.fname*: :class:`str`
                Name of ``.triq`` file (created or found)
            *triqstats.n*: :class:`int`
                Iteration number for *fname*
        :Versions:
            * 2025-08-11 ``@ddalle``: v1.0
        """
        # Search for data file
        mtch = self.match_surf_file()
        # Get iteration thereof
        n = self.infer_file_niter(mtch)
        # Get name of ``.triq`` file
        ftriq = self.genr8_triq_filename(mtch, n)
        # Cache it
        self._dex_n_iter = n
        self._dex_triqfile = ftriq
        # Check if file exists
        if not os.path.isfile(ftriq):
            # Log this action
            self.log_verbose(f"Writing surface {mtch.group()} -> {ftriq}")
            # Write triq file from surface data
            self.write_triq(mtch.group(), ftriq)
        # Return file name and status
        return FileStatus(ftriq, n)

    def genr8_triq_filename(self, mtch=None, n: Optional[int] = None) -> str:
        r"""Find the name of the expected ``.triq`` file based on status

        :Call:
            >>> ftriq = runner.genr8_triq_filename(mtch=None, n=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *mtch*: {``None``} | :mod:`re.Match`
                Regex match object for surface data file (else call
                :func:`CaseRunner.match_surf_file`)
            *n*: {``None``} | :class:`int`
                Iteration number of surface data (else call
                :func:`CaseRunner.infer_file_niter`)
        :Outputs:
            *ftriq*: :class:`str`
                Name of ``.triq`` file expected (or created) by CAPE
        :Versions:
            * 2025-08-10 ``@ddalle``: v1.0
        """
        # Get default
        mtch = mtch if mtch is not None else self.match_surf_file()
        n = n if n is not None else self.infer_file_niter(mtch)
        # Phase number
        j = self.get_phase()
        # Get project name
        proj = self.get_project_rootname(j)
        # Get surface file infix
        stem = self.get_triq_filename_stem()
        # Form pattern
        ftriq = f"{proj}_{stem}_timestep{n}.triq"
        # Relative to working folder
        workdir = self.get_working_folder_()
        return os.path.join(workdir, ftriq)

    def get_triq_filename_stem(self) -> str:
        r"""Get the infix for surface ``.triq``  files

        :Call:
            >>> stem = runner.get_triq_filename_stem()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *stem*: :class:`str`
                File name infix, e.g. ``"surf"`` or ``"tec_boundary"``
        :Versions:
            * 2025-08-11 ``@ddalle``: v1.0
        """
        return "surf"

    def write_triq(self, rawdatafile: str, triqfile: str):
        r"""Convert CFD data file to ``.triq`` format

        This enables use with Chimera Grid Tools capabilities and
        integrates with CAPE's TriqFM tools.

        :Call:
            >>> runner.write_triq(rawdatafile, triqfile)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2025-08-11 ``@ddalle``: v1.0
        """
        # Read the raw data file
        mesh = umesh.Umesh(rawdatafile)
        # Write as ``.triq``
        mesh.write_triq(triqfile, fmt="lr4")

    def get_triq_filename(self) -> str:
        r"""Get latest ``.triq`` file

        :Call:
            >>> ftriq = runner.get_triq_filename()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ftriq*: :class:`str`
                Name of latest ``.triq`` annotated triangulation file
        :Versions:
            * 2025-01-29 ``@ddalle``: v1.0
        """
        return self.search_workdir("*.triq")

    def get_triq_file(self):
        # Get working folder
        workdir = self.get_working_folder()
        # Get the glob of all such files
        triqfiles = self.search(os.path.join(workdir, "*.triq"))
        # Check for matches
        if len(triqfiles) == 0:
            return None, None, None, None
        # Use latest
        return triqfiles[-1], None, None, None

  # *** RESIDUALS ***
   # --- Convergence ---
    def get_dex_n_orders(
            self,
            nstats: int = 1,
            nlast: Optional[int] = None,
            f: bool = False) -> float:
        r"""Get current residual order of magnitude drop

        :Call:
            >>> norders = runner.get_dex_n_orders(nstatus=1, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nstats*: {``1``} | :class:`int`
                Number of iterations over which to average current resid
            *nlast*: {``None``} | :class:`int`
                Last iteration to use in window
            *f*: ``True``| { ``False``}
                Read residual even if one is cached
        :Outputs:
            *h*: :class:`cape.cfdx.casedata.CaseResid`
                Iterative residual history
        :Versions:
            * 2055-08-04 ``@ddalle``: v1.0
        """
        # Check for cache
        n = self._dex_n_orders
        if n is not None:
            return n
        # Read residual history
        h = self.read_resid(f, meta=True)
        # Return the convergence depth
        self._dex_n_orders = h.GetNOrders(nstats, nLast=nlast)
        return self._dex_n_orders

   # --- Readers ---
    @run_rootdir
    def read_resid(self, f: bool = False, meta: bool = False) -> CaseResid:
        r"""Read the current residual history

        :Call:
            >>> h = runner.read_resid(f=False, meta=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: ``True``| { ``False``}
                Read residual even if one is cached
            *meta*: ``True`` | {``False``}
                Option to only read first and last iteration
        :Outputs:
            *h*: :class:`cape.cfdx.casedata.CaseResid`
                Iterative residual history
        :Versions:
            * 2055-08-04 ``@ddalle``: v1.0
        """
        # Check for existing residual
        h = getattr(self, "resid", None)
        if (not f) and (h is not None):
            return h
        # Get class
        cls = self._resid_cls
        # Get arguments
        args = self.genr8_resid_args()
        # Call
        h = cls(*args, meta=meta)
        # Save
        self.resid = h
        # output
        return h

    def genr8_resid_args(self) -> tuple:
        r"""Generate list of arguments to case residual class

        :Call:
            >>> args = runner.genr8_resid_args()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *args*: :class:`tuple`
                Arguments to :class:`CaseResid` instantiation
        :Versions:
            * 2025-08-04 ``@ddalle``: v1.0
        """
        return ()

  # *** DATABOOK ***
   # --- Sampling ---
    def sample_dex(self, comp: str) -> DataKit:
        r"""Sample a DataBook component (e.g. to a specific iteration)

        :Call:
            >>> db = runner.sample_dex(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read and sample
        :Outputs:
            *db*: :class:`DataKit`
                Sampled DataBook component or raw data
        :Versions:
            * 2025-08-04 ``@ddalle``: v1.0
            * 2025-08-12 ``@ddalle``: v1.1; sample *nStats*, *XMRP*, etc
        """
        # Get component type
        typ = self.get_dex_type(comp)
        # Check for custom sampling function
        samplefunc = getattr(self, f"sample_dex_{typ}", None)
        # Check if found
        if samplefunc is None:
            # Read the raw data (e.g. iterative history)
            db = self.read_dex(comp)
        else:
            # Call sample
            db = samplefunc(comp)
        # Check special cols
        self._sample_n_iter(comp, db)
        self._sample_dex_n_orders(comp, db)
        self._sample_n_stats(comp, db)
        self._sample_xmrp(comp, db)
        self._sample_ymrp(comp, db)
        self._sample_zmrp(comp, db)
        # Output
        return db


    def sample_dex_surfcp(self, comp: str) -> DataKit:
        r"""Sample a force & moment iterative history

        :Call:
            >>> db = runner.sample_dex_fm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read and sample
        :Outputs:
            *db*: :class:`cape.dkit.rdb.DataKit`
                Sampled DataBook component or raw data
        :Versions:
            * 2025-07-29 ``@ddalle``: v1.0
        """
        # Read the raw data (e.g. iterative history)
        surfcp = self.read_dex(comp)
        # Output
        return surfcp


    def sample_dex_fm(self, comp: str) -> DataKit:
        r"""Sample a force & moment iterative history

        :Call:
            >>> db = runner.sample_dex_fm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read and sample
        :Outputs:
            *db*: :class:`cape.dkit.rdb.DataKit`
                Sampled DataBook component or raw data
        :Versions:
            * 2025-07-29 ``@ddalle``: v1.0
        """
        # Read the raw data (e.g. iterative history)
        fm = self.read_dex(comp)
        # Get run matrix instance
        cntl = self.read_cntl()
        # Get relevant options
        na = cntl.opts.get_DataBookOpt(comp, "NStats")
        nb = cntl.opts.get_DataBookOpt(comp, "NMaxStats")
        # Sample
        s = fm.GetStats(na, nb)
        # Eliminate *_n* cols
        for col in list(s.keys()):
            if col.endswith("_n"):
                s.pop(col)
        # Convert to DataKit
        db = DataKit()
        db.link_data(s)
        # Output
        return db

    def _sample_dex_n_orders(self, comp: str, db: dict):
        # Check if unncessary
        if "nOrders" in db:
            return
        # Check if present
        if self._check_dex_fcol(comp, "nOrders"):
            db.save_col("nOrders", self.get_dex_n_orders())

    def _sample_n_iter(self, comp: str, db: dict):
        # Check if necessary
        if "nIter" in db:
            return
        # Check if requested
        if self._check_dex_icol(comp, "nIter"):
            db.save_col("nIter", self.get_dex_iter(comp))

    def _sample_n_stats(self, comp: str, db: dict):
        # Check if necessary
        if "nStats" in db:
            return
        # Check if requested
        if self._check_dex_icol(comp, "nStats"):
            db.save_col("nStats", self.get_dex_nstats(comp))

    def _sample_xmrp(self, comp: str, db: dict):
        # Check if necessary
        if "XMRP" in db:
            return
        # Check if requested
        if self._check_dex_fcol(comp, "XMRP"):
            # Get *cntl*
            cntl = self.read_cntl()
            # Get *CompID* for this component
            compid = self.get_dex_opt(comp, "CompID")
            compid = _listify(compid)[0]
            # Get *MRP*
            mrp = cntl.opts.get_RefPoint(compid)
            # Return first coord
            db.save_col("XMRP", mrp[0])

    def _sample_ymrp(self, comp: str, db: dict):
        # Check if necessary
        if "YMRP" in db:
            return
        # Check if requested
        if self._check_dex_fcol(comp, "YMRP"):
            # Get *cntl*
            cntl = self.read_cntl()
            # Get *CompID* for this component
            compid = self.get_dex_opt(comp, "CompID")
            compid = _listify(compid)[0]
            # Get *MRP*
            mrp = cntl.opts.get_RefPoint(compid)
            # Return second coord
            db.save_col("YMRP", mrp[1])

    def _sample_zmrp(self, comp: str, db: dict):
        # Check if necessary
        if "ZMRP" in db:
            return
        # Check if requested
        if self._check_dex_fcol(comp, "ZMRP"):
            # Get *cntl*
            cntl = self.read_cntl()
            # Get *CompID* for this component
            compid = self.get_dex_opt(comp, "CompID")
            compid = _listify(compid)[0]
            # Get *MRP*
            mrp = cntl.opts.get_RefPoint(compid)
            # Return third coord
            db.save_col("ZMRP", mrp[2])

    def _check_dex_fcol(self, comp: str, col: str) -> bool:
        # Read *cntl* instance
        cntl = self.read_cntl()
        # Get list of floating-point columns
        return col in cntl.opts.get_DataBookFloatCols(comp)

    def _check_dex_icol(self, comp: str, col: str) -> bool:
        # Read *cntl* instance
        cntl = self.read_cntl()
        # Check list of integer columns
        return col in cntl.opts.get_DataBookIntCols(comp)

   # --- Readers ---
    # Read raw data for DataBook component
    def read_dex(self, comp: str) -> DataKit:
        r"""Read a data component

        :Call:
            >>> db = runner.read_dex(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
            * 2025-08-12 ``@ddalle``: v1.1; by-element; preprocess
        """
        # Get component type
        typ = self.get_dex_type(comp)
        # Perform preprocessing if needed
        self.prep_dex(comp)
        # Check it
        if typ in ("fm", "surfcp"):
            return self.read_dex_by_element(comp)
        else:
            return self.read_dex_element(comp, comp)

    # Read a DEx by element
    def read_dex_by_element(self, comp: str) -> DataKit:
        r"""Read a data component with multiple elements

        :Call:
            >>> db = runner.read_dex_by_element(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        # Get component(s)
        compid = self.get_dex_opt(comp, "CompID", vdef=comp)
        # Convert to list if necessary
        compids = _listify(compid)
        # Loop through components
        for j, compj in enumerate(compids):
            # Check for negative
            negj = compj.startswith("-")
            # Strip negative sign
            compj = compj.lstrip('-')
            # Read component
            dbj = self.read_dex_element(comp, compj)
            # Transformations
            self.transform_dex(comp, dbj)
            # Add or initialize
            if j == 0:
                db = dbj
            elif negj:
                db -= dbj
            else:
                db += dbj
        # Output
        return db

    # Read one element of a data extraction
    @run_rootdir
    def read_dex_element(self, comp: str, compid: str) -> DataKit:
        r"""Read one element of a data extracter component

        :Call:
            >>> db = runner.read_dex_element(comp, compid)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
            *compid*: :class:`str`
                Name of component ID (usually matches *comp*)
        :Versions:
            * 2025-07-30 ``@ddalle``: v1.0
        """
        # Get component type
        typ = self.get_dex_type(comp)
        # Create extra args
        args0 = self.genr8_dex_args_pre(typ)
        args1 = self.genr8_dex_args_post(typ)
        # Get class
        cls = self._dex_cls[typ]
        # Enter working folder
        os.chdir(self.get_working_folder())
        # Use custom clas to instantiate
        db = cls(*args0, compid, *args1)
        # Output
        return db

    # Preprocessing
    def prep_dex(self, comp: str):
        r"""Perform preprocessing tasks as needed before reading DEx

        :Call:
            >>> db = runner.prep_dex(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        # Get component type
        typ = self.get_dex_type(comp)
        # Create function name
        funcname = f"prep_dex_{typ}"
        # Get the function if applicable
        func = getattr(self, funcname, None)
        # Call it
        if callable(func):
            func(comp)

    # Create tuple of args prior to *comp*
    def genr8_dex_args_pre(self, typ: str) -> tuple:
        r"""Generate tuple of args before *comp* when reading data cls

        This will call the function ``runner.get_dex_args_pre_{typ}()``
        if possible, or ``runner.get_dex_args_pre()`` as a fall back. If
        neither function exists, an empty :class:`tuple` will be
        returned.

        :Call:
            >>> args = runner.genr8_dex_args_pre(typ)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *typ*: :class:`str`
                DataBook component type
        :Outputs:
            *args*: :class:`tuple`
                Arguments for ``_dex_cls`` instantiation
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Generate function names
        name0 = "get_dex_args_pre"
        name1 = f"get_dex_args_pre_{typ}"
        # Get functions if possible
        f0 = getattr(self, name0, None)
        f1 = getattr(self, name1, f0)
        # Call function if possible
        args = () if not callable(f1) else f1()
        # Convert scalaars
        args = args if isinstance(args, (tuple, list)) else (args,)
        # Output
        return args

    # Create tuple of args after *comp*
    def genr8_dex_args_post(self, typ: str) -> tuple:
        r"""Generate tuple of args after *comp* when reading data cls

        This will call the function ``runner.get_dex_args_post_{typ}()``
        if possible, or ``runner.get_dex_args_post()`` as a fall back.
        If neither function exists, an empty :class:`tuple` will be
        returned.

        :Call:
            >>> args = runner.genr8_dex_args_post(typ)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *typ*: :class:`str`
                DataBook component type
        :Outputs:
            *args*: :class:`tuple`
                Arguments for ``_dex_cls`` instantiation
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Generate function names
        name0 = "get_dex_args_post"
        name1 = f"get_dex_args_post_{typ}"
        # Get functions if possible
        f0 = getattr(self, name0, None)
        f1 = getattr(self, name1, f0)
        # Call function if possible
        args = () if not callable(f1) else f1()
        args = () if not callable(f1) else f1()
        # Output
        return args

    # Apply transformations
    def transform_dex(self, comp: str, db: DataKit):
        r"""Apply transformations for a data extraction component

        :Call:
            >>> runner.transform_dex(comp, db)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
            *db*: :class:`cape.dkit.rdb.DataKit`
                Untransformed data extraction
        :Versions:
            * 2025-08-05 ``@ddalle``: v1.0
        """
        # Get component type
        typ = self.get_dex_type(comp)
        # Get name of transformation function
        name = f"transform_dex_{typ}"
        # Get function if any
        fn = getattr(self, name, None)
        # Call it if appropriate
        if callable(fn):
            fn(comp, db)

   # --- Force & Moment ---
    def transform_dex_fm(self, comp: str, fm: CaseFM):
        r"""Apply transformations for a force & moment data extraction

        :Call:
            >>> runner.transform_dex_fm(comp, fm)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
            *fm*: :class:`cape.cfdx.casedata.CasseFM`
                Untransformed force & moment residual history
        :Versions:
            * 2025-07-31 ``@ddalle``: v1.0
        """
        # Get transformations
        transforms = self.get_dex_opt(comp, "Transformations")
        # Exit if none
        if not transforms:
            return
        # Read run matrix controller and case
        cntl = self.read_cntl()
        i = self.get_case_index()
        # Loop through transformations
        for topts in transforms:
            fm.TransformFM(topts, cntl.x, i)

   # --- Status ---
    # Get the current iteration number as applies to
    def get_dex_iter(self, comp: str) -> int:
        r"""Get number of iterations available for a DataBook component

        :Call:
            >>> n = runner.get_dex_iter(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2025-08-07 ``@ddalle``: v1.0
        """
        # Get type
        typ = self.get_dex_type(comp)
        # Get specialized function name
        funcname = f"get_dex_iter_{typ}"
        # Check for such a function
        func = getattr(self, funcname, None)
        # Call it
        if callable(func):
            # Call specialized function
            return func(comp)
        else:
            # Fall back to *restart* iteration
            return self.get_restart_iter()

    def get_dex_iter_fm(self, comp: str) -> int:
        r"""Get number of iterations available for ``"FM"`` comps

        :Call:
            >>> n = runner.get_dex_iter_fm()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2025-08-07 ``@ddalle``: v1.0
        """
        # Check option
        nlast = self.get_dex_opt(comp, "NLastStats")
        # Default to last iteration
        n = nlast if (nlast) else self.get_iter()
        # Use last iteration
        return n

    def get_dex_iter_lineload(self, comp: str) -> int:
        r"""Get number of iterations for ``"LineLoad"`` comp

        :Call:
            >>> n = runner.get_dex_iter_lineload()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        return self._get_dex_iter_surf(comp)

    def get_dex_iter_triqfm(self, comp: str) -> int:
        r"""Get number of iterations for ``"TriqFM"`` comp

        :Call:
            >>> n = runner.get_dex_iter_triqfm()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        return self._get_dex_iter_surf(comp)

    def get_dex_iter_triqpoint(self, comp: str) -> int:
        r"""Get number of iterations for ``"TriqPoint"`` comp

        :Call:
            >>> n = runner.get_dex_iter_triqpoint()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        return self._get_dex_iter_surf(comp)

    def _get_dex_iter_surf(self, comp: str) -> int:
        # Check cache .. from previous call of prepare_triq()
        n = self._dex_n_iter
        # Usit if appropriate
        if n is not None:
            return n
        # Search for data file
        mtch = self.match_surf_file()
        # Check for a match
        if mtch is None:
            return 0
        # Get iteration thereof
        n = self.infer_file_niter(mtch)
        # Save it
        self._dex_n_iter = n
        # Output
        return n

    def _get_dex_iter_vol(self, comp: str) -> int:
        # Check cache .. from previous call of prepare_triq()
        n = self._dex_n_iter
        # Usit if appropriate
        if n is not None:
            return n
        # Search for data file
        mtch = self.match_vol_file()
        # Check for a match
        if mtch is None:
            return 0
        # Get iteration thereof
        n = self.infer_file_niter(mtch)
        # Save it
        self._dex_n_iter = n
        # Output
        return n

   # --- Window size ---
    def get_dex_nstats(self, comp: str) -> int:
        r"""Get iters in averaging window for a DataBook component

        :Call:
            >>> n = db.get_nstats(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iterations/timesteps included in averaging window
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        # Get type
        typ = self.get_dex_type(comp)
        # Get specialized function name
        funcname = f"get_dex_nstats_{typ}"
        # Check for such a function
        func = getattr(self, funcname, None)
        # Call it
        if callable(func):
            # Call specialized function
            return func(comp)
        else:
            # Fall back to inference from averaging/restart options
            n = self.get_dex_iter(comp)
            # Infer *nStats* based on iteration
            return self.infer_tavg_nstats(n=n)

    def get_dex_nstats_fm(self, comp: str) -> int:
        r"""Get averaging window iters for a ``"FM"`` DataBook component

        :Call:
            >>> n = db.get_nstats(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iterations/timesteps included in averaging window
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        return self.get_dex_opt(comp, "NStats")

    def get_dex_nstats_lineload(self, comp: str) -> int:
        r"""Get window size for a ``"LineLoad"`` DataBook component

        :Call:
            >>> n = db.get_dex_nstats_lineload(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iterations/timesteps included in averaging window
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        # Get current iteration
        n = self.get_dex_iter(comp)
        # Infer *nStats* based on restart/averaging settings
        nstats = self.infer_tavg_nstats(n)
        # Cache it
        self._dex_n_stats = nstats
        return nstats

    def get_dex_nstats_triqfm(self, comp: str) -> int:
        r"""Get window size for a ``"TriqFM"`` DataBook component

        :Call:
            >>> n = db.get_dex_nstats_triqfm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iterations/timesteps included in averaging window
        :Versions:
            * 2025-08-29 ``@ddalle``: v1.0
        """
        # Same as lineload
        return self.get_dex_nstats_lineload(comp)

    def get_dex_nstats_triqpoint(self, comp: str) -> int:
        r"""Get window size for a ``"TriqPoint"`` DataBook component

        :Call:
            >>> n = db.get_dex_nstats_triqpoint(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iterations/timesteps included in averaging window
        :Versions:
            * 2025-08-29 ``@ddalle``: v1.0
        """
        # Same as lineload
        return self.get_dex_nstats_lineload(comp)

   # --- Options ---
    # Get DataBook component type
    def get_dex_type(self, comp: str) -> str:
        r"""Get category for a DataBook component

        :Call:
            >>> typ = runner.get_dex_type(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *typ*: :class:`str`
                DataBook component type
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Read *cntl*
        cntl = self.read_cntl()
        # Get component type
        typ = cntl.opts.get_DataBookType(comp).lower()
        # Output
        return typ

    # General option
    def get_dex_opt(self, comp: str, opt: str, vdef: Any = None) -> Any:
        r"""Get general option for a DataBook component

        :Call:
            >>> v = runner.get_dex_opt(comp, opt, vdef=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
            *opt*: :class:`str`
                Name of option to access
            *vdef*: {``None``} | :class:`object`
                Default value
        :Outputs:
            *v*: :class:`object`
                DataBook option value
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Read *cntl*
        cntl = self.read_cntl()
        # Get component option
        return cntl.opts.get_DataBookOpt(comp, opt, vdef=vdef)

    def get_dex_transformation(self, comp: str) -> np.ndarray:
        ...

   # --- Line Loads ---
    @run_rootdir
    def prep_dex_lineload(self, comp: str):
        r"""Prepare surface, run ``triload``, before reading line loads

        :Call:
            >>> db = runner.prep_dex_lineload(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-12 ``@ddalle``: v1.0
        """
        # Enter working folder
        os.chdir(self.get_working_folder())
        # Get name of ``.triq`` file
        ftriq, _ = self.prepare_triq()
        # Section type
        sec = self.get_dex_opt(comp, "SectionType")
        # Get name of expected output file
        fname = os.path.join("lineload", f"LineLoad_{comp}.{sec}")
        # Cannot proceed if *ftriq* does not exist
        if not os.path.isfile(ftriq):
            return
        # Check if it exists or is up-to-date
        if os.path.isfile(fname):
            if os.path.getmtime(fname) > os.path.getmtime(ftriq):
                return
        # Otherwise, create and enter ``lineload/`` folder
        self.mkdir("lineload")
        os.chdir("lineload")
        # Prepare triload.i
        self.write_triload_input(comp, ftriq)
        # Run ``triloadCmd``
        self.run_triload(f"triload.{comp}.i", f"triload.{comp}.o")

   # --- Surf Cp ---
    @run_rootdir
    def prep_dex_surfcp(self, comp: str):
        r"""Prepare surface ``.triq`` file for SurfCp processing

        :Call:
            >>> db = runner.prep_dex_triqfm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Enter working folder
        os.chdir(self.get_working_folder())
        # Get name of ``.triq`` file and create it if needed
        self.prepare_triq()

    # Create tuple of TriqFM dex args after to *comp*
    def get_dex_args_post_surfcp(self) -> tuple:
        r"""Get list of args after *comp* in :class:`CaseTriqFM`

        :Call:
            >>> args = runner.get_dex_args_post_triqfm()
            >>> ftriq, cntl, i = runner.get_dex_args_post_triqfm()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *args*: :class:`tuple`
                Tuple of args
            *ftriq*: :class:`str`
                Name of ``.triq`` file with surface solution data
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Run matrix controller with definitions for dex comp
            *i*: :class:`int`
                Index of this case in run matrix
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Read run matrix controller
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Use name of TriqFM file
        return (self._dex_triqfile, cntl, i)


    def post_delete_dex_surfcp(self, dex, **kw):
        fname = kw.get("fname", None)
        casename = self.get_case_name().split("/")[-1]
        # Delete data cdb
        dex.delete_datafile(fname, casename)

    def post_sample_dex_surfcp(self, dex, **kw):
        fname = kw.get("fname", None)
        casename = self.get_case_name().split("/")[-1]
        # Write data cdb
        dex.write_datafile(fname, casename)

   # --- TriqFM ---
    @run_rootdir
    def prep_dex_triqfm(self, comp: str):
        r"""Prepare surface ``.triq`` file for TriqFM processing

        :Call:
            >>> db = runner.prep_dex_triqfm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-13 ``@ddalle``: v1.0
        """
        # Enter working folder
        os.chdir(self.get_working_folder())
        # Get name of ``.triq`` file and create it if needed
        self.prepare_triq()

    # Create tuple of TriqFM dex args after to *comp*
    def get_dex_args_post_triqfm(self) -> tuple:
        r"""Get list of args after *comp* in :class:`CaseTriqFM`

        :Call:
            >>> args = runner.get_dex_args_post_triqfm()
            >>> ftriq, cntl, i = runner.get_dex_args_post_triqfm()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *args*: :class:`tuple`
                Tuple of args
            *ftriq*: :class:`str`
                Name of ``.triq`` file with surface solution data
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Run matrix controller with definitions for dex comp
            *i*: :class:`int`
                Index of this case in run matrix
        :Versions:
            * 2025-07-24 ``@ddalle``: v1.0
        """
        # Read run matrix controller
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Use name of TriqFM file
        return (self._dex_triqfile, cntl, i)

   # --- TriqPoint ---
    def prep_dex_triqpoint(self, comp: str):
        r"""Prepare surface ``.triq`` file for TriqPoint processing

        :Call:
            >>> db = runner.prep_dex_triqfm(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component to read
        :Versions:
            * 2025-08-29 ``@ddalle``: v1.0
        """
        self.prep_dex_triqfm(comp)

    # Create tuple of TriqPoint dex args after to *comp*
    def get_dex_args_post_triqpoint(self) -> tuple:
        r"""Get list of args after *comp* in :class:`CaseTriqPoint`

        :Call:
            >>> args = runner.get_dex_args_post_triqpoint()
            >>> ftriq, cntl, i = runner.get_dex_args_post_triqpoint()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *args*: :class:`tuple`
                Tuple of args
            *ftriq*: :class:`str`
                Name of ``.triq`` file with surface solution data
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Run matrix controller with definitions for dex comp
            *i*: :class:`int`
                Index of this case in run matrix
        :Versions:
            * 2025-08-30 ``@ddalle``: v1.0
        """
        # Read run matrix controller
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Use name of TriqFM file
        return (self._dex_triqfile, cntl, i)

   # --- Triload ---
    def write_triload_input(self, comp: str, ftriq: str):
        r"""Write input file for ``triloadCmd``

        :Call:
            >>> runner.write_triload_input(comp, ftriq)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component
            *ftriq*: :class:`str`
                Name of appropriate ``.triq`` file to read from
        :Versions:
            * 2025-01-29 ``@ddalle``: v1.0
            * 2025-08-12 ``@ddalle``: v1.1; add *ftriq* as input
        """
        # Read control instance
        cntl = self.read_cntl()
        # Setting for output triq file
        write_slice_triq = cntl.opts.get_DataBookTrim(comp)
        slice_opt = 1 if write_slice_triq else 0
        # Momentum setting
        include_momentum = cntl.opts.get_DataBookMomentum(comp)
        momentum_opt = 'y' if include_momentum else 'n'
        # Number of cuts
        ncut = cntl.opts.get_DataBookNCut(comp)
        # Cut direction
        cutdir = cntl.opts.get_DataBookOpt(comp, "CutPlaneNormal")
        # Get components and type of the input
        compID = self.expand_compids(comp)
        # Get triload input conditions
        mach = self.get_mach()
        rey = self.get_reynolds()
        gam = self.get_gamma()
        # Check for NaNs
        mach = 1.0 if mach is None else mach
        rey = 1.0 if rey is None else rey
        gam = 1.4 if gam is None else gam
        # Reference quantities
        Aref = cntl.opts.get_RefArea(comp)
        Lref = cntl.opts.get_RefLength(comp)
        xmrp = cntl.opts.get_RefPoint(comp)
        # Check for missing values
        if Aref is None:
            raise ValueError(f"No reference area specified for {comp}")
        if Lref is None:
            raise ValueError(f"No reference length specified for {comp}")
        if xmrp is None:
            raise ValueError(f"No moment reference point specified for {comp}")
        if not isinstance(compID, (list, tuple, np.ndarray)):
            raise TypeError(
                f"Unable to find compID list for {self.comp}; got {compID}")
        # File name
        with open(f"triload.{comp}.i", 'w') as fp:
            # Write the triq file name
            fp.write(f"../{ftriq}\n")
            # Write the prefix name
            fp.write("LineLoad\n")
            # Write the Mach number, reference Reynolds number, and gamma
            fp.write('%s %s %s\n' % (mach, rey, gam))
            # Moment center
            fp.write('%s %s %s\n' % (xmrp[0], xmrp[1], xmrp[2]))
            # Setting for gauge pressure and non-dimensional output
            fp.write('0 0\n')
            # Reference length and area
            fp.write('%s %s\n' % (Lref, Aref))
            # Whether or not to include momentum
            fp.write(f"{momentum_opt}\n")
            # Group name and list of component IDs
            # e.g. COMP_part 3-10,12-15,17,19
            fp.write(f"{comp} {RangeString(compID)}\n")
            # Number of cuts
            fp.write(f"{ncut} {slice_opt}\n")
            # Write min and max; use '1 1' to make it automatically detect
            fp.write('1 1\n')
            # Write the cut type
            fp.write('const %s\n' % cutdir)
        # Write coordinate transform
        self.write_triload_transformations(comp)

    # Write triload transformations
    def write_triload_transformations(self, comp: str):
        r"""Write transformation aspect of ``triloadCmd`` input file

        :Call:
            >>> runner.write_triload_transformations(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of component
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Read control instance
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Get the raw option from the data book
        db_transforms = cntl.opts.get_DataBookTransformations(comp)
        # Initialize transformations
        rotation_matrix = None
        # Loop through transformations
        for topts in db_transforms:
            # Check for rotation matrix
            ri = cntl.get_transformation_matrix(topts, i)
            # Exit if none
            if ri is None:
                continue
            if rotation_matrix is None:
                # First transformation
                rotation_matrix = ri
            else:
                # Compound
                rotation_matrix = np.dot(rotation_matrix, ri)
        # Append to input file
        with open(f"triload.{comp}.i", 'a') as fp:
            # Check if no transformations
            if rotation_matrix is None:
                fp.write('n\n')
                return
            # Yes, we are doing transformations
            fp.write('y\n')
            # Write the transformation
            for row in rotation_matrix:
                fp.write("%9.6f %9.6f %9.6f\n" % tuple(row))

  # *** OPTIONS ***
   # --- Case options ---
    # Get project root name
    def get_project_rootname(self, j: Optional[int] = None) -> str:
        r"""Read namelist and return project namelist

        :Call:
            >>> rname = runner.get_project_rootname(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *rname*: :class:`str`
                Project rootname
        :Versions:
            * 2024-11-05 ``@ddalle``: v1.0
        """
        # Default
        return self._modname

   # --- Settings: Read  ---
    # Read ``case.json``
    def read_case_json(self) -> RunControlOpts:
        r"""Read ``case.json`` if not already

        :Call:
            >>> rc = runner.read_case_json()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *rc*: :class:`RunControlOpts`
                Options interface from ``case.json``
        :Versions:
            * 2023-06-15 ``@ddalle``: v1.0
            * 2024-07-18 ``@ddalle``: v2.0; remove *f* option, use mtime
        """
        # Absolute path
        fjson = os.path.join(self.root_dir, RC_FILE)
        # Check current file
        if os.path.isfile(fjson):
            # Get modification time for *fjson*
            mtime = os.path.getmtime(fjson)
        else:
            # Default modification time for missing file
            mtime = 1.0
        # Check if we need to read
        if mtime > self._mtime_case_json:
            # Check for file
            if os.path.isfile(fjson):
                # Read the file
                self.rc = self._rc_cls(fjson, _warnmode=0)
            else:
                # Try to read Cntl
                cntl = self.read_cntl()
                # Check if that worked
                if cntl is None:
                    # Create default
                    self.rc = self._rc_cls()
                else:
                    # Isolate subsection
                    self.rc = cntl.opts["RunControl"]
                # Write settings to avoid repeating this situation
                self.write_case_json(self.rc)
            # Save modification time
            self._mtime_case_json = mtime
        # Output
        return self.rc

    # Get "archive" options
    def read_archive_opts(self) -> ArchiveOpts:
        r"""Read the *Archive* options for this case

        This prefers the parent-folder JSON settings to those found in
        ``case.json``. If no run matrix JSON settings can be read, the
        ``case.json`` settings will be used.

        :Call:
            >>> opts = runner.read_archive_opts()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *opts*: :class:`ArchiveOpts`
                Options interface from ``case.json``
        :Versions:
            * 2024-08-28 ``@ddalle``: v1.0
            * 2024-12-09 ``@ddalle``: v2.0; prefer top-level JSON
        """
        # Read parent folder
        cntl = self.read_cntl()
        # Check if that worked
        if cntl is None:
            # Read case settings
            rc = self.read_case_json()
            # Return the "Archive" section
            return rc["Archive"]
        else:
            # Use run-matrix-level settings
            return cntl.opts["RunControl"]["Archive"]

   # --- Conditions ---
    # Read ``conditions.json``
    def read_conditions(self, f: bool = False) -> dict:
        r"""Read ``conditions.json`` if not already

        :Call:
            >>> xi = runner.read_conditions(f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: ``True`` | {``False``}
                Option to force re-read
        :Outputs:
            *xi*: :class:`dict`
                Run matrix conditions for this case
        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
        """
        # Check if present
        if (not f) and isinstance(self.xi, dict):
            # Already read
            return self.xi
        # Absolute path
        fconds = os.path.join(self.root_dir, CONDITIONS_FILE)
        # Check if file exists
        if os.path.isfile(fconds):
            # Open file for handling by json
            with open(fconds) as fp:
                # Read it and save
                self.xi = json.load(fp)
        else:
            # No conditions to read
            self.xi = {}
        # Output
        return self.xi

    # Get condition of a single run matrix key
    def read_condition(self, key: str, f: bool = False):
        r"""Read ``conditions.json`` if not already

        :Call:
            >>> v = runner.read_condition(key, f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *key*: :class:`str`
                Name of run matrix key to query
            *f*: ``True`` | {``False``}
                Option to force re-read
        :Outputs:
            *v*: :class:`int` | :class:`float` | :class:`str`
                Value of run matrix key *key*
        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
        """
        # Read conditions
        xi = self.read_conditions(f)
        # Get single key
        return xi.get(key)

    # Get user
    def get_user(self) -> str:
        r"""Get user name who is running this case

        :Call:
            >>> user = runner.get_user()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *user*: :class:`str`
                Name of specified user or owner of PBS file
        :Versions:
            * 2025-05-02 ``@ddalle``: v1.0
        """
        # Read conditions
        x = self.read_conditions()
        # Check for a user
        u = x.get("user")
        # Use it if appropriate
        if u is not None:
            return u.lstrip("@")
        # Name of main PBS script
        pbscript = self.get_pbs_script()
        pbsabs = os.path.join(self.root_dir, pbscript)
        # Check for PBS script
        if not os.path.isfile(pbsabs):
            # Use current user
            return getpass.getuser()
        # Get user ID of owner
        uid = os.stat(pbsabs).st_uid
        # Get username
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    # Get Mach number
    def get_mach(self) -> float:
        r"""Get Mach number even if *mach* is not a run matrix key

        This uses :func:`cape.cfdx.runmatrix.RunMatrix.GetMach` to
        combine information from all run matrix keys.

        :Call:
            >>> mach = runner.get_mach()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2024-12-03 ``@ddalle``: v1.0
            * 2025-01-29 ``@ddalle``: v1.1; attempt read_condition()
        """
        # Try local condition
        mach = self.read_condition("mach")
        # Use it if possible
        if mach is not None:
            return mach
        # Read *cntl*
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Get Mach number
        return cntl.x.GetMach(i)

    # Get Reynolds number per grid unit
    def get_reynolds(self) -> float:
        r"""Get Reynolds number per grid unit if possible

        :Call:
            >>> rey = runner.get_reynolds()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *rey*: :class:`float`
                Reynolds number per grid nuit
        :Versions:
            * 2025-01-29 ``@ddalle``: v1.0
        """
        # Try local condition
        conds = self.read_conditions()
        # Attempt a few names
        rey = conds.get("Re", conds.get("Rey", conds.get("rey")))
        # Use it if possible
        if rey is not None:
            return rey
        # Read *cntl*
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Get Mach number
        return cntl.x.GetReynoldsNumber(i)

    # Get ratio of specific heats
    def get_gamma(self) -> float:
        r"""Get freestream ratio of specific heats if possible

        :Call:
            >>> gam = runner.get_gamma()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *gam*: :class:`float`
                Freestream ratio of specific heats
        :Versions:
            * 2025-01-29 ``@ddalle``: v1.0
        """
        # Try local condition
        conds = self.read_conditions()
        # Attempt a few names
        gam = conds.get("gamma", conds.get("gam"))
        # Use it if possible
        if gam is not None:
            return gam
        # Read *cntl*
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Get Mach number
        return cntl.x.GetGamma(i)

   # --- Settings: Write ---
    # Write case settings to ``case.json``
    def write_case_json(self, rc: RunControlOpts):
        r"""Write the current settinsg to ``case.json``

        :Call:
            >>> runner.write_case_json(rc)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *rc*: :class:`RunControlOpts`
                Options interface from ``case.json``
        :Versions:
            * 2024-08-24 ``@ddalle``: v1.0
        """
        # Absolute path
        fjson = os.path.join(self.root_dir, RC_FILE)
        # Write file
        try:
            with open(fjson, 'w') as fp:
                # Dump the run settings
                json.dump(rc, fp, indent=1, cls=_NPEncoder)
        except PermissionError:
            print(f"  permission to write '{fjson}' denied")

   # --- Settings: modify ---
    # Extend the case by one run of last phase
    def extend_case(
            self,
            m: int = 1,
            j: Optional[int] = None,
            nmax: Optional[int] = None) -> Optional[int]:
        r"""Extend the case by one execution of final phase

        :Call:
            >>> nnew = runner.extend_case(m=1, nmax=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *m*: {``1``} | :class:`int`
                Number of additional times to execute final phase
            *j*: {``None``} | :class:`int`
                Phase to extend
            *nmax*: {``None``} | :class:`int`
                Do not exceed this iteration
        :Outputs:
            *nnew*: ``None`` | :class:`int`
                Number of iters after extension, if changed
        :Versions:
            * 2024-08-26 ``@ddalle``: v1.0
        """
        # Read ``case.json`` file
        rc = self.read_case_json()
        # Last phase
        jlast = self.get_last_phase()
        # Current phase
        jcur = self.get_phase_next()
        # Use last phase if not specified
        j = jlast if j is None else j
        # Don't extend previous phases
        j = max(j, jcur)
        # Get current iter and projected last iter
        ncur = self.get_iter()
        nend = self.get_last_iter()
        ncur = 0 if ncur is None else ncur
        # Get number of steps in one execution of final phase
        nj = rc.get_nIter(j, vdef=100)
        # Get highest estimate of current last iter
        na = max(ncur, nend)
        # Extension
        nb = na + m*nj
        # Apply *nmax*, but don't go backwards!
        if nmax is None:
            # No limit on final iter
            nc = nb
        else:
            # User limit
            nc = min(nmax, nb)
        # But don't go backwards!
        nnew = max(na, nc)
        # Check for null extension
        if nnew <= na:
            return
        # Additional iters
        dn = nnew - na
        # Loop through phases from *j* to final
        for jj in range(j, jlast + 1):
            # Get iters for that phase
            njj = rc.get_PhaseIters(jj)
            # Extend at least to current iter
            nold = max(ncur, njj)
            nout = nold + dn
            # Status update
            msg = f"  extend phase {jj}: {njj} -> {nout}"
            self.log_both(msg)
            print(msg)
            # Update settings
            rc.set_PhaseIters(nout, j=jj)
            # Write new options
            self.write_case_json(rc)
        # Return the new iter
        return nnew

   # --- DataBook ---
    def get_databook_opt(self, comp: str, opt: str):
        r"""Retrieve a databook option for a given component

        :Call:
            >>> v = runner.get_databook_opt(comp, opt)
        :Inputs:
            *comp*: :class:`str`
                Name of component
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *v*: :class:`Any`
                Value of option *opt* for component *comp*
        :Versions:
            * 2025-01-23 ``@ddalle``: v1.0
        """
        # Read *cntl*
        cntl = self.read_cntl()
        # Assert component
        if comp not in cntl.opts.get_DataBookComponents():
            raise KeyError(f"No DataBook component named '{comp}'")
        # Return option
        return cntl.opts.get_DataBookOpt(comp, opt)

   # --- Job IDs ---
    # Get PBS/Slurm job ID
    @run_rootdir
    def get_job_id(self) -> str:
        r"""Get PBS/Slurm job ID, if any

        :Call:
            >>> job_id = runner.get_job_id()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *job_id*: :class:`str`
                Text form of job ID; ``''`` if no job found
        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
            * 2023-07-05 ``@ddalle``: v1.1; eliminate *j* arg
            * 2024-06-10 ``@ddalle``: v2.0; use get_job_ids()
        """
        # Read full list
        job_ids = self.get_job_ids()
        # Check if empty
        if len(job_ids):
            # Return the principal one
            return job_ids[0]
        else:
            # Use empty string
            return ''

    # Get list of PBS/Slurm job IDs
    def get_job_ids(self) -> list:
        r"""Get list of PBS/Slurm job IDs, if appropriate

        :Call:
            >>> job_id = runner.get_job_id()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *job_ids*: :class:`list`\ [:class:`str`]
                List of PBS/Slurm job IDs, as text
        :Versions:
            * 2024-06-10 ``@ddalle``: v1.0
        """
        # Unpack options
        rc = self.read_case_json()
        # Get phase
        j = self.get_phase_next()
        # Check for a job ID to locate
        if not (rc.get_qsub(j) or rc.get_slurm(j)):
            # No submission options
            return []
        # Loop through candidate job ID file names
        for jobfile in JOB_ID_FILES:
            # Read job IDs
            job_ids = self._read_job_id(jobfile)
            # Check if any found
            if len(job_ids):
                return job_ids
        # If reaching this point, no IDs found
        return []

    # Check for if we are running inside a batch job
    def get_env_jobid(self) -> str:
        r"""Check to see if we are running inside a PBS/Slurm job

        This looks for environment variables to see if this is running
        inside a batch job.  Currently supports slurm and PBS.

        :Call:
            >>> jobid = runner.get_env_jobid()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *jobid*: :class:`str`
                Name of current job, ``0`` if no batch environment
        :Versions:
            * 2023-12-13 ``@dvicker``: v1.0 (Cntl)
            * 2023-12-18 ``@ddalle``: v1.1; debug
            * 2025-06-01 ``@ddalle``: v1.0; from cntl.Cntl
            * 2025-06-02 ``@ddalle``: v1.1; cache in *runner.job*
        """
        # Check for current job
        if self.job is not None:
            return self.job
        # Get options
        rc = self.read_case_json()
        # Environment variable
        envvar = "SLURM_JOB_ID" if rc.get_slurm(0) else "PBS_JOBID"
        # Check value of environment variable
        envid = os.environ.get(envvar, '0')
        # Split into parts by '.'
        parts = envid.split(".", 2)
        # Take first two parts
        jobid = '.'.join(parts[:2])
        # Save it
        self.job = jobid
        # Output
        return jobid

    # Read "STOP-PHASE", if appropriate
    @run_rootdir
    def read_stop_phase(self) -> Tuple[bool, Optional[int]]:
        r"""Read ``CAPE-STOP-PHASE`` file for local stopping criterion

        :Call:
            >>> q, j = runner.read_stop_phase()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: ``True`` | ``False``
                Whether ``CAPE-STOP-PHASE`` file exists
            *j*: ``None`` | :class:`int`
                Phase at which to stop (every phase if ``None``)
        :Versions:
            * 2024-05-26 ``@ddalle``: v1.0
        """
        # Start with negative result (don't run incremental)
        q = False
        j = None
        # Check if file exists
        if os.path.isfile(STOP_PHASE_FILE):
            # Stop phase instructed
            q = True
            # Try to read it
            try:
                j = int(open(STOP_PHASE_FILE).readline.strip())
            except (ValueError, FileNotFoundError):
                # No phase (probably empty file)
                j = None
        # Output
        return q, j

    # Read a jobID.dat file
    @run_rootdir
    def _read_job_id(self, fname: str) -> list:
        # Initialize IDs
        job_ids = []
        # Check if file exists
        if not os.path.isfile(fname):
            # No file to read
            return job_ids
        # Try to read file
        try:
            # Open file
            with open(fname, 'r') as fp:
                # Read max of 20 lines
                for _ in range(MAX_JOB_IDS):
                    # Read line
                    line = fp.readline()
                    # Check for EOF
                    if line == '':
                        break
                    # Remove white space and only use first 'word'
                    job_id = line.strip().split(maxsplit=1)[0]
                    # Check for empty line
                    if job_id == "":
                        continue
                    # Save ID if new
                    if job_id not in job_ids:
                        job_ids.append(job_id)
            # Return job IDs
            return job_ids
        except Exception:
            # Return as many files as we read
            return job_ids

  # *** CNTL/RUN MATRIX ***
   # --- Run matrix ---
    def get_case_index(self) -> Optional[int]:
        r"""Get index of a case in the current run matrix

        :Call:
            >>> i = runner.get_case_index()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *i*: :class:`int` | ``None``
                Index of case with name *frun* in run matrix, if present
        :Versions:
            * 2024-08-15 ``@ddalle``: v1.0
        """
        # Get name of case
        casename = self.get_case_name()
        # Read run matrix control
        cntl = self.read_cntl()
        # Return the index
        return cntl.GetCaseIndex(casename)

    def get_case_name(self) -> str:
        r"""Get name of this case according to CAPE run matrix

        :Call:
            >>> casename = runner.get_case_name()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *casename*: :class:`str`
                Name of case, using ``/`` as path sep
        :Versions:
            * 2024-08-15 ``@ddalle``: v1.0
        """
        # Get run matrix and case root dirs
        cntl_rootdir = self.get_cntl_rootdir()
        case_rootdir = self.root_dir
        # Get relative path
        casename = os.path.relpath(case_rootdir, cntl_rootdir)
        # Replace \ -> / on Windows
        casename = casename.replace(os.sep, '/')
        # Output
        return casename

    def get_cntl_rootdir(self) -> str:
        r"""Get name of this case according to CAPE run matrix

        :Call:
            >>> rootdir = runner.get_cntl_rootdir()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *rootdir*: :class:`str`
                Absolute path to base of run matrix that runs this case
        :Versions:
            * 2024-08-26 ``@ddalle``: v1.0
        """
        # Find folder and JSON files
        root_dir, _ = self._get_cntl_parts()
        # Output
        return root_dir

    def _get_cntl_parts(self) -> tuple:
        # Get module
        mod = self.import_cntlmod()
        # Absolute path to "case.json"
        fjson = os.path.join(self.root_dir, RC_FILE)
        # Default root folder
        root_def = os.path.realpath(
            os.path.join(self.root_dir, '..', '..'))
        root_dir = root_def
        # Default JSON file
        fjson_def = mod.Cntl._fjson_default
        root_json = fjson_def
        # Try to get directly from settings
        if os.path.isfile(fjson):
            # Read case settings
            rc = self.read_case_json()
            # Get root of run matrix
            root_dir = rc.get_RootDir()
            root_dir = root_def if root_dir is None else root_dir
            root_dir = root_dir.replace('/', os.sep)
            # Get JSON file
            root_json = rc.get_JSONFile(vdef=mod.Cntl._fjson_default)
        # Check for such a folder
        if not os.path.isfile(os.path.join(root_dir, root_json)):
            # Use fall-back folder
            root_dir = root_def
        # Output
        return root_dir, root_json

   # --- Cntl ---
    @run_rootdir
    def read_cntl(self) -> CntlBase:
        r"""Read the parent run-matrix control that owns this case

        :Call:
            >>> cntl = runner.read_cntl()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *cntl*: :class:`Cntl` | ``None``
                Run matrix control instance
        :Versions:
            * 2024-08-15 ``@ddalle``: v1.0
            * 2024-08-28 ``@ddalle``: v1.1; can work w/o case.json
        """
        # Check if already read
        if self.cntl is not None:
            return self.cntl
        # Get parts
        cntl_rootdir, cntl_json = self._get_cntl_parts()
        # Check if folder exists
        if not os.path.isdir(cntl_rootdir):
            return
        # Go to root dir (@run_rootdir will return us)
        os.chdir(cntl_rootdir)
        # Check for run-matrix JSON file
        if not os.path.isfile(cntl_json):
            return
        # Get module
        mod = self.import_cntlmod()
        # Read *cntl*
        self.cntl = mod.Cntl(cntl_json)
        # Get case index
        i = self.get_case_index()
        # Save the case runner to avoid re-reads
        self.cntl.caseindex = i
        self.cntl.caserunner = self
        self.cntl.opts.setx_i(i)
        # Copy jobs object
        self.cntl.jobs = self.jobs
        self.cntl.job = self.job
        # Output
        return self.cntl

    # Import appropriate *cntl* module
    def import_cntlmod(self):
        r"""Import appropriate run matrix-level *cntl* module

        :Call:
            >>> mod = runner.import_cntlmod()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *mod*: :class:`module`
                Module imported
        :Versions:
            * 2024-08-14 ``@ddalle``: v1.0
            * 2024-09-07 ``@ddalle``: v1.1; mod for moving cntl to cfdx
        """
        # Get name of *this* case module
        casemodname = self._getmodname()
        # Split into parts, e.g. ["cape", "pyfun", "case"]
        modnameparts = casemodname.split('.')
        # Replace "case" -> "cntl"
        modnameparts[-1] = "cntl"
        cntlmodname = ".".join(modnameparts)
        # Import it
        return importlib.import_module(cntlmodname)

   # --- Cntl tools ---
    def check_mark_pass(self) -> bool:
        r"""Check if this case has been marked PASS in run matrix

        :Call:
            >>> q = runner.check_mark_pass()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                Whether case has been marked PASS or not
        :Versions:
            * 2025-05-01 ``@ddalle``: v1.0
        """
        # Read Cntl
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Check mark
        q = False if i is None else cntl.x.PASS[i]
        return q

    def check_mark_error(self) -> bool:
        r"""Check if this case has been marked ERROR in run matrix

        :Call:
            >>> q = runner.check_mark_error()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                Whether case has been marked ERROR or not
        :Versions:
            * 2025-05-01 ``@ddalle``: v1.0
        """
        # Read Cntl
        cntl = self.read_cntl()
        # Get case index
        i = self.get_case_index()
        # Check mark
        q = False if i is None else cntl.x.ERROR[i]
        return q

    def read_surfconfig(self) -> Optional[SurfConfig]:
        r"""Read surface configuration map file from best source

        :Call:
            >>> conf = runner.read_surfconfig()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *conf*: :class:`SurfConfig` | ``None``
                Surface configuration instance, if possible
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Try to read from cntl
        conf = self.read_surfconfig_cntl()
        # Check if it worked
        if conf is not None:
            return conf
        # Read local config, if able
        return self.read_surfconfig_local()

    def read_surfconfig_local(self) -> Optional[ConfigXML]:
        r"""Read surface configuration map file from case folder

        :Call:
            >>> conf = runner.read_surfconfig()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *conf*: :class:`SurfConfig` | ``None``
                Surface configuration instance, if possible
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Look for a Config.xml file
        fxml = self.search_workdir("*.xml")
        # Check if any found
        if fxml is None:
            return
        # If found, try to read it
        try:
            return ConfigXML(fxml)
        except Exception:
            return None

    def read_surfconfig_cntl(self) -> Optional[SurfConfig]:
        r"""Read surface configuration map file from run matrix control

        :Call:
            >>> conf = runner.read_surfconfig_cntl()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *conf*: :class:`SurfConfig` | ``None``
                Surface configuration instance, if possible
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Try reading Cntl run matrix controller
        cntl = self.read_cntl()
        # Check if it worked
        if cntl is None:
            return
        # Read config
        cntl.ReadConfig()
        # Return it if possible
        return cntl.config

    def expand_compids(self, comp: Union[str, int, list]) -> list:
        r"""Expand component name to list of component ID numbers

        :Call:
            >>> compids = runner.expand_compids(comp)
        :Inputs:
            *comp*: :class:`str` | :class:`int` | :class:`list`
                Component name, index, or list thereof
        :Outputs:
            *compids*: :class:`list`\ [:class:`int`]
                List of component ID numbers
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Convert to list
        comps = comp if isinstance(comp, list) else [comp]
        # Read surface configuration
        conf = self.read_surfconfig()
        # Check if found
        if conf is None:
            return comps
        # Convert
        complist = [conf.GetCompID(subcomp) for subcomp in comps]
        # Avoid duplication
        return list(np.unique(complist))

   # --- Run matrix: other cases ---
    # Run more cases if requested
    @run_rootdir
    def run_more_cases(self) -> int:
        r"""Submit more cases to the queue

        :Call:
            >>> ierr = runner.run_more_cases()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code, ``0`` if no new cases submitted
        :Versions:
            * 2023-12-13 ``@dvicker``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Get "nJob"
        nJob = rc.get_NJob()
        # cd back up and run more cases, but only if nJob is defined
        if nJob > 0:
            # Log
            self.log_both(f"submitting more cases: NJob={nJob}")
            # Default root directory (two levels up from case)
            rootdir = os.path.realpath(os.path.join("..", ".."))
            # Find the root directory
            rootdir = rc.get_RootDir(vdef=rootdir)
            # Name of JSON file
            jsonfile = rc.get_JSONFile()
            # chdir back to the root directory
            os.chdir(rootdir)
            # Get the solver we were using
            solver = self.__class__.__module__.split(".")[-2]
            modname = f"cape.{solver}"
            # Log the call
            self.log_data(
                {
                    "root_dir": rootdir,
                    "json_file": jsonfile,
                    "module": modname,
                    "NJob": nJob,
                })
            # Form the command to run more cases
            cmd = [
                sys.executable,
                "-m", modname,
                "-f", jsonfile,
                "--unmarked",
                "--auto"
            ]
            # Run it
            return self.callf(cmd)
        # Return code
        return IERR_OK

  # *** STATUS ***
   # --- Status: Next action ---
    # Check if case should exit for any reason
    @run_rootdir
    def check_exit(self, ja: int) -> bool:
        r"""Check if a case should exit for any reason

        Reasons a case should exit include

        * The case is finished.
        * An alternate exit criteria was found (e.g. resid converged)
        * A ``CAPE-STOP-PHASE`` file was found.
        * A ``CAPE-STOP-ITER`` file was found.
        * The relevant *StartNextPhase* option is ``False``.
        * The relevant *RestartSamePhase* option is ``False``.

        :Call:
            >>> q = runner.check_exit(ja)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ja*: :class:`int`
                Phase at beginning of run
        :Outputs:
            *q*: ``True`` | ``False``
                Whether case should exit
        :Versions:
            * 2025-05-26 ``@ddalle``: v1.0
        """
        # Read case JSON
        rc = self.read_case_json()
        # Determine current phase at end of run
        jb = self.get_phase_next()
        # Get STOP-PHASE option
        if jb != ja:
            # Log
            self.log_verbose(f"advancing from phase {ja} -> {jb}")
            # Moving to next phase
            if not rc.get_RunControlOpt("StartNextPhase", ja):
                # Exit b/c incremental option in ``case.json``
                self.log_both(
                    f"stopping after phase {ja} b/c StartNextPhase=False")
                return True
            # Read from file
            q, jstop = self.read_stop_phase()
            # Check CAPE-STOP-PHASE settings
            if q and ((jstop is None) or (ja >= jstop)):
                # Log
                self.log_both(
                    f"stopping after phase {ja} due to {STOP_PHASE_FILE}")
                if jstop is not None:
                    self.log_verbose(
                        f"{STOP_PHASE_FILE} stop at phase {jstop}")
                # Delete the STOP file
                os.remove(STOP_PHASE_FILE)
                # Exit
                return True
        else:
            # Restarting same phase
            if not rc.get_RunControlOpt("RestartSamePhase", ja):
                # Exit b/c incremental option in ``case.json``
                self.log_both(
                    f"stopping during phase {ja} b/c RestartSamePhase=False")
                # Exit b/c incremental option on w/i this phase
                return True
        # Fall back to check_complete()
        return self.check_complete()

    # Check for early exit
    def check_alt_exit(self, j: Optional[int] = None) -> bool:
        r"""Check for solver-selected exit, e.g. convergence target hit

        :Call:
            >>> q = runner.check_alt_exit(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
        :Outputs:
            *q*: :class:`bool`
                ``True`` if no listed files have been modified recently
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
        """
        return False

    # Check if case is complete
    def check_complete(self) -> bool:
        r"""Check if a case is complete (DONE)

        :Call:
            >>> q = runner.check_complete()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
            * 2023-07-08 ``@ddalle``: v1.1; support ``STOP``
        """
        # Determine current phase
        j = self.get_phase_next()
        # Final phase and iter
        jb = self.get_last_phase()
        nb = self.get_last_iter()
        # Check if final phase
        if j < jb:
            self.log_verbose(f"case not complete; {j} < {jb}")
            return False
        # Get absolute iter
        n = self.get_iter()
        n = 0 if n is None else n
        # Check for stop iteration
        qstop, nstop = self.get_stop_iter()
        # Check iteration number
        if qstop and (n >= nstop):
            # Stop requested by user
            self.log_verbose(f"case stopped at {n} >= {nstop} iters")
            return True
        elif n < nb:
            # Not enough iterations complete
            self.log_verbose(
                f"case not complete; reached phase {jb} but " +
                f"{n} < {nb} iters")
            return False
        else:
            # All criteria met
            self.log_both(
                f"case complete; phase {j} >= {jb}; iter {n} >= {nb}")
            return True

   # --- Status: Overall ---
    # Check for other errors
    @run_rootdir
    def check_error(self) -> bool:
        r"""Check for other errors; rewrite for each solver

        :Call:
            >>> q = runner.check_error()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                Whether case appears to be an error
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
        """
        # Check for FAIL file
        if os.path.isfile(FAIL_FILE):
            return True
        # Check for other erorr not yet marked
        if self.get_returncode() == IERR_OK:
            # No error
            return False
        else:
            # Other error detected
            return True

    def check_running(self) -> bool:
        r"""Check if a case is currently running

        :Call:
            >>> q = runner.check_running()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                Whether case appears to be running
        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
            * 2025-03-11 ``@ddalle``: v1.1; elimn8 @run_rootdir
        """
        return os.path.isfile(os.path.join(self.root_dir, RUNNING_FILE))

    def check_active(self) -> bool:
        r"""Check if a case is actively running CFD executable

        :Call:
            >>> q = runner.check_active()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                Whether case appears to be running
        :Versions:
            * 2025-03-11 ``@ddalle``: v1.0
        """
        return os.path.isfile(os.path.join(self.root_dir, ACTIVE_FILE))

    # Check error codes
    @run_rootdir
    def get_returncode(self) -> int:
        r"""Check for other errors; rewrite for each solver

        :Call:
            >>> ierr = runner.get_returncode()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
            * 2024-06-17 ``@ddalle``: v1.1; was ``check_error()``
            * 2024-07-16 ``@ddalle``: v1.2; use *self.returncode*
        """
        return getattr(self, "returncode", IERR_OK)

   # --- Status: Category ---
    # Check overall status
    def get_status(self) -> str:
        r"""Calculate status of current job

        :Call:
            >>> sts = runner.get_status()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *sts*: :class:`str`
                One of several possible job statuses

                * ``DONE``: not running and meets finishing criteria
                * ``PASS``: ``DONE`` plus marked in run matrix
                * ``ERROR``: marked as failure in run matrix
                * ``FAIL``: local error detected, not marked in matrix
                * ``RUN``: case is currently running
                * ``INCOMP``: case not running and not finished
                * ``QUEUE``: case not running but in the queu
                * ``PASS*``: marked in run matrix but not ``DONE``
                * ``ZOMBIE``: RUNNING but no recently changed files
        """
        # Check for simple case
        if self.check_mark_error():
            return "ERROR"
        # Get initial status (w/o checking file ages or queue)
        if self.check_error() != IERR_OK:
            # Found FAIL file or other evidence of errors
            sts = "FAIL"
        elif self.check_running():
            # Check for zombie
            if self.check_zombie():
                # RUNNING but no recent updates
                sts = "ZOMBIE"
            else:
                # Found RUNNING file
                sts = "RUN"
        else:
            # Get phase number and iteration required to finish case
            jmax = self.get_last_phase()
            nmax = self.get_last_iter()
            # Get current status
            j = self.get_phase_next()
            n = self.get_iter()
            # Check both requirements
            if (j < jmax) or (n < nmax):
                sts = "INCOMP"
            else:
                # All criteria met
                sts = "DONE"
        # Check for PASS case
        if self.check_mark_pass():
            # Check status
            return "PASS" if (sts == "DONE") else "PASS*"
        # Check INCOMP -> QUEUE
        if sts == "INCOMP":
            # Get status of job
            q = self.check_queue()
            # Status update
            sts = "INCOMP" if q == '.' else "QUEUE"
        # Check for special case w/ auto early exit
        if (sts == "INCOMP") and self.check_alt_exit():
            # Update status -> DONE
            sts = "DONE"
            # Try to update the settings
            self.handle_alt_exit(jmax)
        # Output
        return sts

    # Check if case is a zombie
    @run_rootdir
    def check_zombie(self) -> bool:
        r"""Check a case for ``ZOMBIE`` status

        A running case is declared a zombie if none of the listed files
        (by default ``*.out``) have been modified in the last 30
        minutes.  However, a case cannot be a zombie unless it contains
        a ``RUNNING`` file and returns ``True`` from
        :func:`CheckRunning`.

        :Call:
            >>> q = runner.check_combie()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                ``True`` if no listed files have been modified recently
        :Versions:
            * 2025-06-01 ``@ddalle``: v1.0
        """
        # Get options
        rc = self.read_case_json()
        # List of files to check
        fzomb = rc.get_opt("ZombieFiles", vdef=self._zombie_files)
        # Create list of files matching globs
        fglob = []
        for fg in fzomb:
            fglob += glob.glob(fg)
        # Timeout time (in minutes)
        tmax = rc.get_opt("ZombieTimeout")
        t = tmax
        # Current time
        toc = time.time()
        # Loop through glob files
        for fname in fglob:
            # Get minutes since modification for *fname*
            ti = (toc - os.path.getmtime(fname))/60
            # Running minimum
            t = min(t, ti)
        # Output
        return (t >= tmax)

    # Check if this is being called from a PBS job running this case
    def check_current_job(self) -> bool:
        r"""Check if being called from the PBS job of this case

        :Call:
            >>> q = runner.check_current_job()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *q*: :class:`bool`
                ``True`` if no listed files have been modified recently
        :Versions:
            * 2025-06-01 ``@ddalle``: v1.0
        """
        # Get list of jobs
        jobids = self.get_job_ids()
        # Get current job ID from environment if able
        envjobid = self.get_env_jobid()
        # Check
        return envjobid in jobids

   # --- Status: Log/STDOUT files ---

    # Get CAPE STDOUT files
    @run_rootdir
    def get_cape_stdoutfiles(self) -> list:
        r"""Get list of STDOUT files in order they were run

        :Call:
            >>> runfiles = runner.get_cape_stdoutfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *runfiles*: :class:`list`\ [:class:`str`]
                List of run files, in ascending order
        :Versions:
            * 2024-08-09 ``@ddalle``: v1.0
            * 2025-03-21 ``@ddalle``: v1.1; use search_regex()
        """
        # Find all the runfiles renamed by CAPE
        runfiles = self.search_regex("run.[0-9][0-9]+.[0-9]+")
        # Initialize run files with metadata
        runfile_meta = []
        # Loop through candidates
        for runfile in runfiles:
            # Compare to regex
            re_match = REGEX_RUNFILE.fullmatch(runfile)
            # Save file name, phase, and iter
            runfile_meta.append(
                (runfile, int(re_match.group(1)), int(re_match.group(2))))
        # Check for empty list
        if len(runfile_meta) == 0:
            return []
        # Sort first by iter, then by phase (phase takes priority)
        runfile_meta.sort(key=lambda x: x[2])
        runfile_meta.sort(key=lambda x: x[1])
        # Extract file name for each
        return [x[0] for x in runfile_meta]

    # Get CAPE STDOUT files from a certain phase
    @run_rootdir
    def get_phase_stdoutfiles(self, j: int) -> list:
        r"""Get list of STDOUT files in order they were run

        :Call:
            >>> runfiles = runner.get_cape_stdoutfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *runfiles*: :class:`list`\ [:class:`str`]
                List of run files, in ascending order
        :Versions:
            * 2024-08-09 ``@ddalle``: v1.0
            * 2025-03-21 ``@ddalle``: v1.1; use search_regex()
        """
        # Find all the runfiles renamed by CAPE
        runfiles = self.search_regex(f"run.{j:02d}.[0-9]+")
        # Initialize run files with metadata
        runfile_meta = []
        # Loop through candidates
        for runfile in runfiles:
            # Compare to regex
            re_match = REGEX_RUNFILE.fullmatch(runfile)
            # Save file name, phase, and iter
            runfile_meta.append(
                (runfile, int(re_match.group(2))))
        # Check for empty list
        if len(runfile_meta) == 0:
            return []
        # Sort by iter
        runfile_meta.sort(key=lambda x: x[1])
        # Extract file name for each
        return [x[0] for x in runfile_meta]

   # --- Status: Phase ---
    # Determine phase number
    @run_rootdir
    def get_phase(self, f: bool = True) -> int:
        r"""Determine phase number in present case

        :Call:
            >>> j = runner.get_phase(n, f=True)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``True``} | ``False``
                Force recalculation of phase
        :Outputs:
            *j*: :class:`int`
                Phase number for next restart
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0
            * 2015-10-19 ``@ddalle``: v1.1; FUN3D version
            * 2016-04-14 ``@ddalle``: v1.2; CFDX version
            * 2023-06-16 ``@ddalle``: v2.0; CaseRunner method
        """
        # Check if present
        if not (f or self.j is None):
            # Return precalculated phase
            return self.j
        # Get iteration
        n = self.get_restart_iter()
        # Calculate
        j = self.getx_phase(n)
        # Save and return
        self.j = j
        return j

    # Get phase number by only checking output files
    @run_rootdir
    def get_phase_simple(self, f: bool = True) -> int:
        r"""Determine phase number, only checking output files

        :Call:
            >>> j, jlast = runner.get_phase_simple(f=True)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``True``} | ``False``
                Force recalculation of phase
        :Outputs:
            *j*: :class:`int`
                Phase number for current or next restart
            *jlast*: :class:`int`
                Last phase expected
        :Versions:
            * 2025-03-02 ``@ddalle``: v1.0
        """
        # Get list of phases
        phases = self.get_phase_sequence()
        # Loop through them in reverse
        for j in reversed(phases):
            # Check if any output files exists
            if len(self.search_regex(f"run.{j:02d}.[0-9]+")) > 0:
                # Found a phase that has been run
                break
        # Output phase
        return j, phases[-1]

    # Get next phase to run
    def get_phase_next(self) -> int:
        r"""Get phasen number to use for restart

        :Call:
            >>> j = runner.get_phase_next()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *j*: :class:`int`
                Phase number for next restart
        :Versions:
            * 2025-03-30 ``@ddalle``: v1.0
            * 2025-04-05 ``@ddalle``: v2.0; use `check_phase()`
            * 2025-04-06 ``@ddalle``: v2.1; include `get_phase_recent()`
        """
        # Check recently run phases
        ja = self.get_phase_recent()
        # Index thereof
        ia = self.get_phase_index(ja)
        ia = 0 if ia is None else ia
        # Loop through phase sequence
        for i, j in enumerate(self.get_phase_sequence()):
            # Don't check prior phases
            if i < ia:
                continue
            # Check *ja* and later
            if not self.check_phase(j):
                return j
        # All phases complete; return last phase
        return j

    # Determine which phase was most recently completed
    @run_rootdir
    def get_phase_recent(self) -> Optional[int]:
        r"""Get the number of the last phase successfully completed

        :Call:
            >>> j = runner.get_phase_recent()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *j*: :class:`int`
                Phase number last completed
        :Versions:
            * 2025-03-21 ``@ddalle``: v1.0
        """
        # Get list of STDOUT files
        logfiles = self.get_cape_stdoutfiles()
        # Return None if no phase has been completed
        if len(logfiles) == 0:
            return None
        # Check last file (pre-sorted)
        return int(logfiles[-1].split('.')[1])

    # Check if a phase is completed
    def check_phase(self, j: int) -> bool:
        r"""Check if phase *j* has been completed

        :Call:
            >>> q = runner.check_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *q*: :class:`bool`
                Whether phase *j* looks complete
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
            * 2025-08-14 ``@ddalle``: v1.1; ``check_alt_exit()``
        """
        # Check for early termination
        if self.check_alt_exit(j):
            # Reduce iter cutoff if appropriate
            self.handle_alt_exit(j)
            return True
        # Check iteration requirements
        if not self.check_phase_iters(j):
            return False
        # Apply special checks
        return self.checkx_phase(j)

    # Check iteration requirements for phase *j*
    def check_phase_iters(self, j: int) -> bool:
        r"""Check if phase *j* has run the appropriate iterations

        :Call:
            >>> q = runner.check_phase_iters(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *q*: :class:`bool`
                Whether phase *j* looks complete
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
        """
        # Get iteration run by phase *j*
        n = self.get_phase_n(j)
        # Exit if none reported
        if n is None:
            return False
        # Get number of iterations required for this phase
        nj = self.get_phase_iters(j)
        nj = 0 if nj is None else nj
        # Check if iteration reached
        if nj > n:
            return False
        # Number of iters actually run and min req'd
        nrun = self.get_phase_nrun(j)
        nreq = self.get_phase_nreq(j)
        # Check if actually run
        return nrun >= nreq

    def handle_alt_exit(self, j: int):
        r"""Reduce *PhaseIters* if an early exit was detected

        :Call:
            >>> runner.handle_alt_exit(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Versions:
            * 2025-08-15 ``@ddalle``: v1.0
        """
        # Get current iter
        n = self.get_iter()
        # Get cutoff for phase *j*
        nmax = self.get_phase_iters(j)
        # Try to update the settings
        try:
            # Perform action
            self.set_phase_iters(min(n, nmax), j)
            # Log action
            if n < nmax:
                self.log_both(f"Reducing phase {j} -> {n} iters")
        except PermissionError:
            pass

    # Check if a phase is rerun-able
    def check_phase_rerunable(self, j: int) -> bool:
        r"""Check if a phase can be rerun, else it must progress to next

        :Call:
            >>> q = runner.check_phase_rerunable(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *q*: :class:`bool`
                Whether phase *j* can be restarted
        :Versions:
            * 2025-06-24 ``@ddalle``: v1.0
        """
        return True

    # Solver-specifc phase
    def checkx_phase(self, j: int) -> bool:
        r"""Apply solver-specific checks for phase *j*

        This generic version always returns ``True``

        :Call:
            >>> q = runner.checkx_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *q*: :class:`bool`
                Whether phase *j* looks complete
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
        """
        return True

    # Iter run by phase
    def get_phase_n(self, j: int) -> Optional[int]:
        r"""Get last reported iteration run in phase *j*, if any

        :Call:
            >>> n = runner.get_phase_n(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *n*: ``None`` | :class:`int`
                Overall iteration reported run for phase *j*
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
        """
        # Search for log files
        runfiles = self.get_phase_stdoutfiles(j)
        # Exit if empty
        if len(runfiles) == 0:
            return None
        # Get last iteration
        n = int(runfiles[-1].split('.')[2])
        # Output
        return n

    # Iters run by phase
    def get_phase_nrun(self, j: int) -> Optional[int]:
        r"""Get number of iterations reported run by phase *j*, if any

        :Call:
            >>> nrun = runner.get_phase_nrun(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *nrun*: ``None`` | :class:`int`
                Number of iterations reported run by phase *j*
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
        """
        # Get iteration reported for phase *j*
        n = self.get_phase_n(j)
        # Check for valid run
        if n is None:
            return None
        # Identify previous phase
        i = self.get_phase_index(j)
        # Check for no-previous-phase
        if i == 0:
            return n
        # Get number of previous phae
        ja = self.get_phase_sequence()[i - 1]
        # Get iteratiosn reported by *that* phase
        na = self.get_phase_n(ja)
        na = 0 if na is None else na
        # Return the difference
        return n - na

    # Iteratiosn required by phase
    def get_phase_nreq(self, j: int) -> int:
        r"""Get number of iterations required to be run by phase *j*

        :Call:
            >>> nreq = runner.get_phase_nreq(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number last completed
        :Outputs:
            *nreq*: :class:`int`
                Min iterations req'd by phase *j*, from ``"nIter"``
        :Versions:
            * 2025-04-05 ``@ddalle``: v1.0
        """
        # Read options
        rc = self.read_case_json()
        # Check "nIter" setting
        nreq = rc.get_opt("nIter", j)
        nreq = 0 if nreq is None else nreq
        return nreq

    # Determine phase number
    def getx_phase(self, n: int):
        r"""Calculate phase based on present files

        The ``x`` in the title implies this case might be rewritten for
        each module.

        :Call:
            >>> j = runner.getx_phase(n)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Phase number for next restart
        :Versions:
            * 2023-06-16 ``@ddalle``: v1.0
            * 2023-07-06 ``@ddalle``: v1.1; *PhaseSequence* repeats ok
            * 2024-08-12 ``@ddalle``: v1.2; refine file names slightly
        """
        # Get case options
        rc = self.read_case_json()
        # Get list of of STDOUT files, run.00.*, run.01.*, etc.
        logfiles = self.get_cape_stdoutfiles()
        # Get phase sequence
        phases = self.get_phase_sequence()
        # Loop through possible phases
        for j in phases:
            # Check for output files
            if len(fnmatch.filter(logfiles, f"run.{j:02d}.*")) == 0:
                # This run has not been completed yet
                return j
            # Check the iteration number
            if n < rc.get_PhaseIters(j):
                # Phase has been run but not reached phase iter target
                return j
        # Case completed; just return the last value.
        return j

    # Get last phase
    def get_last_phase(self) -> int:
        r"""Get min phase required for a given case

        :Call:
            >>> jmax = runner.get_last_phase()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *jmax*: :class:`int`
                Min phase required for job
        :Versions:
            * 2024-06-17 ``@ddalle``: v1.0
            * 2024-07-18 ``@ddalle``: v1.1; force reread, handle errors
        """
        # Read phase sequence
        phases = self.get_phase_sequence()
        # Return the last one
        return phases[-1]

    # Get next phase
    def get_next_phase(self, j: int) -> Optional[int]:
        r"""Get the number of the phase that follows *j*

        :Call:
            >>> jnext = runner.get_next_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Current phase number
        :Outputs:
            *jnext*: :class:`int` | ``None``
                Next phase number, if applicable
        :Versions:
            * 2024-08-11 ``@ddalle``: v1.0
        """
        # Get phase sequence
        phase_sequence = self.get_phase_sequence()
        # Get index
        k = self.get_phase_index(j)
        # Check if *j* is not prescribed or is last phase
        if k is None:
            # Phase not found
            return None
        elif k + 1 >= len(phase_sequence):
            # No next phase
            return j
        else:
            # Return following phase
            return phase_sequence[k + 1]

    # Get index of phase (usually same as phase)
    def get_phase_index(self, j: int) -> Optional[int]:
        r"""Get index of phase in ``"PhaseSequence"``

        :Call:
            >>> k = runner.get_phase_index(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *k*: ``None`` | :class:`int`
                Index of *j* in *PhaseSequence*; ``None`` if *j* is not
                one of the prescribed phases
        :Versions:
            * 2024-08-11 ``@ddalle``: v1.0
        """
        # Get phase sequence
        phase_sequence = self.get_phase_sequence()
        # Check if *j* is in it
        if j in phase_sequence:
            # Find where it occurs in *phase_sequence*
            return phase_sequence.index(j)
        else:
            # No match
            return None

    # Get phase sequence
    def get_phase_sequence(self) -> list:
        r"""Get list of prescribed phases for a case

        :Call:
            >>> phases = runner.get_phase_sequence()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *phases*: :class:`int`\ [:class:`int`]
                Min phase required for job
        :Versions:
            * 2024-07-18 ``@ddalle``: v1.0
        """
        # (Re)read local cas.json
        rc = self.read_case_json()
        # Check for null file
        if rc is None:
            return [0]
        # Get phase sequence
        phases = rc.get_PhaseSequence()
        # Check for None
        phases = [0] if phases is None else phases
        # Output
        return phases

   # --- Status: Iteration ---
    # Get most recent observable iteration
    @run_rootdir
    def get_iter(self, f: bool = True) -> int:
        r"""Detect most recent iteration

        :Call:
            >>> n = runner.get_iter(f=True)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``True``} | ``False``
                Force recalculation of phase
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
        """
        return self.get_iter_simple(f)

    # Get iteration using simpler methods
    @run_rootdir
    def get_iter_simple(self, f: bool = True) -> int:
        r"""Detect most recent iteration

        :Call:
            >>> n = runner.get_iter_simple(f=True)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``True``} | ``False``
                Force recalculation of phase
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2025-03-21 ``@ddalle``: v1.0
        """
        # Check if present
        if not (f or self.n is None):
            # Return existing calculation
            return self.n
        # Get iterations previously completed
        na = self.get_iter_completed()
        # Get iterations run since then (currently active)
        nb = self.get_iter_active()
        # Add them up
        self.n = na + nb
        return self.n

    # Get most recent iteration of completed run
    @run_rootdir
    def get_iter_completed(self) -> int:
        r"""Detect most recent iteration from completed runs

        :Call:
            >>> n = runner.get_iter_completed()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2025-03-21 ``@ddalle``: v1.0
        """
        # Get log files
        logfiles = self.get_cape_stdoutfiles()
        # Use last file
        return 0 if len(logfiles) == 0 else int(logfiles[-1].split('.')[2])

    # Get iterations of current running since last completion
    @run_rootdir
    def get_iter_active(self) -> int:
        r"""Detect any iterations run since last completed phase run

        :Call:
            >>> n = runner.get_iter_active()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2025-03-21 ``@ddalle``: v1.0
            * 2025-04-01 ``@ddalle``: v1.1; use getx_iter() for default
        """
        # Default: overall minus completed
        nc = self.get_iter_completed()
        nt = self.getx_iter()
        nt = 0 if nt is None else nt
        return max(0, nt-nc)

    # Get most recent observable iteration
    def getx_iter(self) -> int:
        r"""Calculate most recent iteration

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
        """
        # CFD{X} version
        return 0

    # Get last iteration
    def get_last_iter(self) -> int:
        r"""Get min iteration required for a given case

        :Call:
            >>> nmax = runner.get_last_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nmax*: :class:`int`
                Number of iterations
        :Versions:
            * 2024-06-17 ``@ddalle``: v1.0
            * 2024-07-18 ``@ddalle``: v1.1; better exception handling
        """
        # Get last phase
        jmax = self.get_last_phase()
        # Get iteration for that phase
        return self.get_phase_iters(jmax)

    # Get phase iters
    def get_phase_iters(self, j: int) -> int:
        r"""Get min iteration required for completion of phase *j*

        :Call:
            >>> n = runner.get_phase_iters(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase index
        :Outputs:
            *n*: :class:`int`
                Number of iterations
        :Versions:
            * 2024-07-18 ``@ddalle``: v1.0
        """
        # (Re)read the local case.json file
        rc = self.read_case_json()
        # Check for null file
        if rc is None:
            return 0
        # Ensure *PhaseIters*
        if "PhaseIters" not in rc:
            rc.set_opt("PhaseIters", [0])
        # Get iterations for requested phase
        return rc.get_PhaseIters(j)

    # Set phase iters
    def set_phase_iters(self, n: int, j: int):
        r"""Set min iteration required for completion of phase *j*

        :Call:
            >>> runner.set_phase_iters(n, j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: :class:`int`
                Number of iterations
            *j*: :class:`int`
                Phase index
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
        """
        # (Re)read the local case.json file
        rc = self.read_case_json()
        # Check for null file
        if rc is None:
            return
        # Set iterations for requested phase
        rc.set_PhaseIters(n, j=j)
        # Write modified settings
        self.write_case_json(rc)

    # Get iteration at which to stop requested by user
    def get_stop_iter(self):
        r"""Read iteration at which to stop

        :Call:
            >>> nstop = runner.get_stop_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nstop*: :class:`int` | ``None``
                Iteration at which to stop, if any
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0
        """
        # No general case
        return False, None

    # Get suspected restart iteration
    @run_rootdir
    def get_restart_iter(self, f=True):
        r"""Get number of iteration if case should restart

        :Call:
            >>> nr = runner.get_restart_iter(f=True)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: {``True``} | ``False``
                Force recalculation of phase
        :Outputs:
            *nr*: :class:`int`
                Restart iteration number
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0; ``cfdx`` abstract version
        """
        # Check if present
        if not (f or self.nr is None):
            # Return existing calculation
            return self.nr
        # Otherwise, calculate
        nr = self.getx_restart_iter()
        self.nr = 0 if nr is None else nr
        # Output
        return self.nr

    # Calculate suspected restart iteration
    def getx_restart_iter(self):
        r"""Calculate number of iteration if case should restart

        :Call:
            >>> nr = runner.gets_restart_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nr*: :class:`int`
                Restart iteration number
        :Versions:
            * 2023-06-20 ``@ddalle``: v1.0; ``cfdx`` abstract version
        """
        # CFD{X} version
        return 0

    # Get run log iteration history
    @run_rootdir
    def get_runlog(self) -> np.ndarray:
        r"""Create a 2D array of CAPE exit phases and iters

        Each row of the output is the phase number and iteration at
        which CAPE exited. The array is sorted by ascending phase then
        iteration.

        :Call:
            >>> runlog = runner.get_runlog()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *runlog*: :class:`np.ndarray`\ [:class:`int`]
                2D array of all CAPE exit phase and iteration numbers
        :Versions:
            * 20254-03-23 ``@ddalle``: v1.0
        """
        # Check for certain files run.NN.N+
        filelist = glob.glob("run.[0-9][0-9]*.[1-9]*")
        # Initialize outputs
        phases = []
        iters = []
        # Loop through files
        for filename in filelist:
            # Process agaisnt regex; ignore "run.01b.1c", etc.
            match = REGEX_RUNFILE.fullmatch(filename)
            # Check for mismatch
            if match is None:
                continue
            # Process phase and iter
            phasetxt, itertxt = match.groups()
            # Save to list
            phases.append(int(phasetxt))
            iters.append(int(itertxt))
        # Sort
        iord = np.lexsort((phases, iters))
        # Convert to 2D array
        runlog = np.stack((phases, iters), axis=1)[iord, :]
        # Output
        return runlog

    # Get iteration from run.[0-9]{2}.[0-9]+ files
    @run_rootdir
    def get_runlog_iter(self):
        r"""Get phase and iteration from most recent CAPE log file name

        :Call:
            >>> phase, iter = runner.get_runlog_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *phase*: :class:`int`
                Phase number reported by CAPE
            *iter*: :class:`int`
                Iteration number reported by CAPE
        :Versions:
            * 2024-03-22 ``@ddalle``: v1.0
        """
        # Check for certain files run.NN.N+
        filelist = glob.glob("run.[0-9][0-9]*.[1-9]*")
        # Initialize
        phase = 0
        iter = 0
        # Loop through files
        for filename in filelist:
            # Process agaisnt regex; ignore "run.01b.1c", etc.
            match = REGEX_RUNFILE.fullmatch(filename)
            # Check for mismatch
            if match is None:
                continue
            # Process phase and iter
            phasetxt, itertxt = match.groups()
            # Convert to integers
            phasej = int(phasetxt)
            iterj = int(itertxt)
            # Check if it's an increase
            if (phasej >= phase) and (iterj > iter):
                # This is the new "latest"
                phase, iter = phasej, iterj
        # Output
        return phase, iter

   # --- Status: PBS/Slurm ---
    def check_queue(self) -> str:
        r"""Get queue status of current job

        :Call:
            >>> s = runner.check_queue()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *s*: :class:`str`
                Job queue status

                * ``.``: not in queue
                * ``Q``: job queued (not running)
                * ``R``: job running
                * ``H``: job held
                * ``E``: job error status
        :Versions:
            * 2025-05-04 ``@ddalle``: v1.0
        """
        # Get job stats collector
        jobs = self._get_qstat()
        # Get this job
        jobid = self.get_job_id()
        # Get owner of this case
        uname = self.get_user()
        # Get stats for this job
        stats = jobs.check_job(jobid, u=uname)
        # Map None -> {}
        stats = {} if stats is None else stats
        # Output
        return stats.get("R", '.')

    def _get_qstat(self) -> queue.QStat:
        # Check current attribute
        if isinstance(self.jobs, queue.QStat):
            return self.jobs
        # Create one
        self.jobs = queue.QStat()
        # Set Slurm vs PBS
        rc = self.read_case_json()
        sched = "slurm" if rc.get_slurm(0) else "pbs"
        self.jobs.scheduler = sched
        # Output
        return self.jobs

  # *** ARCHIVING ***
   # --- Archive: actions ---
    def clean(self, test: bool = False):
        r"""Run the ``--clean`` archiving action

        :Call:
            >>> runner.clean(test=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *test*: ``True`` | {``False``}
                Option to log all actions but not actually copy/delete
        :Versions:
            * 2024-09-18 ``@ddalle`: v1.0
        """
        # Get archivist
        a = self.get_archivist()
        # Save restart files
        self.save_restartfiles()
        # Clean
        a.clean(test)

    def archive(self, test: bool = False):
        r"""Run the ``--archive`` archiving action

        :Call:
            >>> runner.archive(test=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *test*: ``True`` | {``False``}
                Option to log all actions but not actually copy/delete
        :Versions:
            * 2024-09-18 ``@ddalle`: v1.0
        """
        # Get archivist
        a = self.get_archivist()
        # Save report files
        self.save_reportfiles()
        # Clean
        a.archive(test)

    def skeleton(self, test: bool = False):
        r"""Run the ``--skeleton`` archiving action

        :Call:
            >>> runner.skeleton(test=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *test*: ``True`` | {``False``}
                Option to log all actions but not actually copy/delete
        :Versions:
            * 2024-09-18 ``@ddalle`: v1.0
        """
        # Get archivist
        a = self.get_archivist()
        # Clean
        a.archive(test)
        a.skeleton(test)

    def unarchive(self, test: bool = False):
        r"""Run the ``--unarchive`` archiving action

        :Call:
            >>> runner.unarchive(test=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *test*: ``True`` | {``False``}
                Option to log all actions but not actually copy/delete
        :Versions:
            * 2024-09-20 ``@ddalle`: v1.0
        """
        # Get archivist
        a = self.get_archivist()
        # Clean
        a.unarchive(test)

   # --- Archive: interface ---
    def get_archivist(self) -> CaseArchivist:
        r"""Get or read archivist instance

        :Call:
            >>> a = runner.get_archivist()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *a*: :class:`CaseArchivist`
                Archive controller for one case
        :Versions:
            * 2024-09-13 ``@ddalle``: v1.0
            * 2024-12-09 ``@ddalle``: v1.1; prefer *cntl* opts over case
        """
        # Check if already exists
        if self.archivist is not None:
            return self.archivist
        # Isolate "Archive" section
        opts = self.read_archive_opts()
        # Check for relative ArchiveDir
        archivedir = opts.get_ArchiveFolder()
        if not os.path.isabs(archivedir):
            # Absolutize
            opts.set_ArchiveFolder(os.path.join(self.root_dir, archivedir))
        # Get case name
        casename = self.get_case_name()
        # Initialize archivist
        a = self._archivist_cls(opts, self.root_dir, casename)
        # Save it
        self.archivist = a
        # Return it
        return a

   # --- Archive: protected files
    def save_reportfiles(self):
        r"""Update list of protected files for generating reports

        :Call:
            >>> runner.save_reportfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2024-09-14 ``@ddalle``: v1.0
        """
        # Get list of files needed for restart
        filelist = self.get_reportfiles()
        # Get archivist
        a = self.get_archivist()
        # Save them
        a.save_reportfiles(filelist)

    def save_restartfiles(self):
        r"""Update list of protected files for restarting case

        :Call:
            >>> runner.save_restartfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2024-09-14 ``@ddalle``: v1.0
        """
        # Get list of files needed for restart
        filelist = self.get_restartfiles()
        # Get archivist
        a = self.get_archivist()
        # Save them
        a.save_restartfiles(filelist)

    def get_reportfiles(self) -> list:
        r"""Generate list of report files

        :Call:
            >>> filelist = runner.get_reportfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to protect
        :Verions:
            * 2024-09-14 ``@ddalle``: v1.0
        """
        return []

    def get_restartfiles(self) -> list:
        r"""Generate list of restart files

        :Call:
            >>> filelist = runner.get_restartfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to protect
        :Verions:
            * 2024-09-14 ``@ddalle``: v1.0
        """
        return []

  # *** LOGGING ***
   # --- Logging: actions ---
    def log_main(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to primary log

        :Call:
            >>> runner.log_main(msg, title, parent=0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *msg*: :class:`str`
                Primary content of message
            *title*: {``None``} | :class:`str`
                Manual title (default is name of calling function)
            *parent*: {``0``} | :class:`int`
                Extra levels to use for calling function name
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)

    def log_both(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to both primary and verbose logs

        :Call:
            >>> runner.log_both(title, msg)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *msg*: :class:`str`
                Primary content of message
            *title*: {``None``} | :class:`str`
                Manual title (default is name of calling function)
            *parent*: {``0``} | :class:`int`
                Extra levels to use for calling function name
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)
        logger.log_verbose(title, msg)

    def log_verbose(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to verbose log

        :Call:
            >>> runner.log_verbose(title, msg)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *msg*: :class:`str`
                Primary content of message
            *title*: {``None``} | :class:`str`
                Manual title (default is name of calling function)
            *parent*: {``0``} | :class:`int`
                Extra levels to use for calling function name
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_verbose(title, msg)

    def log_data(
            self,
            data: dict,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write :class:`dict` to verbose log as JSON

        :Call:
            >>> runner.log_data(title, data)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *data*: :class:`dict`
                Parameters to write to verbose log as JSON
            *msg*: :class:`str`
                Primary content of message
            *title*: {``None``} | :class:`str`
                Manual title (default is name of calling function)
            *parent*: {``0``} | :class:`int`
                Extra levels to use for calling function name
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get looger
        logger = self.get_logger()
        # Log parameters in the dict
        logger.logdict_verbose(title, data)

    def get_logger(self) -> CaseLogger:
        r"""Get or create logger instance

        :Call:
            >>> logger = runner.get_logger()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *logger*: :class:`CaseLogger`
                Logger instance
        :Versions:
            * 2024-08-16 ``@ddalle``: v1.0
        """
        # Initialize if it's None
        if self.logger is None:
            self.logger = CaseLogger(self.root_dir)
        # Output
        return self.logger

   # --- Function name ---
    def get_funcname(self, frame: int = 1) -> str:
        r"""Get name of calling function, mostly for log messages

        :Call:
            >>> funcname = runner.get_funcname(frame=1)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *frame*: {``1``} | :class:`int`
                Depth of function to seek title of
        :Outputs:
            *funcname*: :class:`str`
                Name of calling function
        :Versions:
            * 2024-08-16 ``@ddalle``
        """
        # Get frame of function calling this one
        func = sys._getframe(frame).f_code
        # Get name
        return func.co_name

   # --- Timing ---
    # Initialize running case
    def init_timer(self):
        r"""Mark a case as ``RUNNING`` and initialize a timer

        :Call:
            >>> tic = runner.init_timer()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *tic*: :class:`datetime.datetime`
                Time at which case was started
        :Versions:
            * 2021-10-21 ``@ddalle``: v1.0; from :func:`run_fun3d`
            * 2023-06-20 ``@ddalle``: v1.1; instance method, no mark()
        """
        # Start timer
        self.tic = datetime.now()
        # Output
        return self.tic

    # Read total time
    def get_cpu_time(self) -> Optional[float]:
        r"""Read most appropriate total CPU usage for current case

        :Call:
            >>> corehrs = runner.get_cpu_time()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *corehrs*: ``None`` | :class:`float`
                Core hours since last start or ``None`` if not running
        :Versions:
            * 2015-12-22 ``@ddalle``: v1.0 (``Cntl.GetCPUTimeBoth``)
            * 2016-08-30 ``@ddalle``: v1.1; check for ``RUNNING``
            * 2023-07-09 ``@ddalle``: v2.0; rename, ``CaseRunner``
        """
        # Get both times
        cpu_start = self.get_cpu_time_start()
        cpu_phase = self.get_cpu_time_user()
        # Combine times as appropriate
        if cpu_start is None:
            # Case not running; use data from completed
            return cpu_phase
        # Check for case w/ no previous completed cycles
        if cpu_phase is None:
            # Return only running time
            return cpu_start
        # Otherwise add both together
        return cpu_phase + cpu_start

    # Read core hours from previous start
    @run_rootdir
    def get_cpu_time_start(self):
        r"""Read total core hours since start of current running phase

        :Call:
            >>> corehrs = runner.get_cpu_time_start()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *corehrs*: ``None`` | :class:`float`
                Core hours since last start or ``None`` if not running
        :Versions:
            * 2015-08-30 ``@ddalle``: v1.0 (Cntl.GetCPUTimeFromStart)
            * 2023-07-09 ``@ddalle``: v2.0
        """
        # Check if running
        if not os.path.isfile(RUNNING_FILE):
            # Case not running
            return
        # Get class's name options
        pymod = self._modname
        # Form both file names
        fstart = f"{pymod}_start.dat"
        # Check for file
        if not os.path.isfile(fstart):
            # No log file found
            return
        # Read file
        ncpus, tic = self.read_start_time()
        # Check for invalid output
        if ncpus is None or tic is None:
            # Unreadable
            return 0.0
        # Get current time
        toc = datetime.now()
        # Subtract time
        dt = toc - tic
        # Calculate CPU hours
        corehrs = ncpus * (dt.days*24 + dt.seconds/3600.0)
        # Output
        return corehrs

    # Read total core hours from py{x}_time.dat
    @run_rootdir
    def get_cpu_time_user(self):
        r"""Read total core hours from completed phase cycles

        :Call:
            >>> corehrs = runner.get_cpu_time_user()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *corehrs*: ``None`` | :class:`float`
                Total core hours used or ``None`` if no log file found
        :Versions:
            * 2015-12-22 ``@ddalle``: v1.0 (Cntl.GetCPUTimeFromFile)
            * 2023-07-09 ``@ddalle``: v2.0
        """
        # Get class's name options
        pymod = self._modname
        # Form both file names
        fname = f"{pymod}_time.dat"
        # Check for no file
        if not os.path.isfile(fname):
            return None
        # Try to read first column
        with open(fname, 'r') as fp:
            # Initialize total
            corehrs = 0.0
            # Loop through file
            while True:
                # Read next line
                line = fp.readline()
                # Check for EOF
                if line == "":
                    break
                # Check for comment or empty line
                if line.startswith("#") or line.strip() == "":
                    continue
                # Attempt to parse line
                try:
                    # Get first column value, comma-delimited
                    parts = line.split(',', maxsplit=1)
                    # Convert to float
                    corehrs += float(parts[0].strip())
                except Exception:
                    # Invalid line
                    print(
                        f"    Invalid CPUhours in '{fname}' from this line:" +
                        f"\n        {line}")
        # Return total
        return corehrs

    # Read *tic* from start_time file
    @run_rootdir
    def read_start_time(self):
        r"""Read the most recent start time to file

        :Call:
            >>> nProc, tic = runner.read_start_time()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file containing CPU usage history
        :Outputs:
            *nProc*: ``None`` | :class:`int`
                Number of cores
            *tic*: ``None`` | :class:`datetime.datetime`
                Time at which most recent run was started
        :Versions:
            * 2016-08-30 ``@ddalle``: v1.0 (stand-alone)
            * 2023-06-17 ``@ddalle``: v2.0; ``CaseRunner`` method
        """
        # Get class's name options
        pymod = self._modname
        # Form a file name
        fname = f"{pymod}_start.dat"
        # Try to read it
        try:
            return self._read_start_time(fname)
        except Exception:
            # No start times found
            return None, None

    # Write *tic* to a file
    @run_rootdir
    def write_start_time(self, j: int):
        r"""Write current start time, *runner.tic*, to file

        :Call:
            >>> runner.write_start_time(tic, j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2015-12-09 ``@ddalle``: v1.0 (pycart)
            * 2015-12-22 ``@ddalle``: v1.0; module function
            * 2023-06-16 ``@ddalle``: v2.0; ``CaseRunner`` method
        """
        # Get class's name options
        pymod = self._modname
        # Form a file name
        fname = f"{pymod}_start.dat"
        # Check if the file exists
        qnew = not os.path.isfile(fname)
        # Open file
        with open(fname, 'a') as fp:
            # Write header if new
            if qnew:
                fp.write("# nProc, program, date, jobID\n")
            # Write remainder of file
            self._write_start_time(fp, j)

    # Write execution time to file
    @run_rootdir
    def write_user_time(self, j: int):
        r"""Write time usage since time *tic* to file

        :Call:
            >>> runner.write_user_time(tic, j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2015-12-09 ``@ddalle``: v1.0 (pycart)
            * 2015-12-22 ``@ddalle``: v1.0; module function
            * 2023-06-16 ``@ddalle``: v2.0; ``CaseRunner`` method
        """
        # Get class's name options
        pymod = self._modname
        # Form a file name
        fname = f"{pymod}_time.dat"
        # Check if the file exists
        qnew = not os.path.isfile(fname)
        # Open file
        with open(fname, 'a') as fp:
            # Write header if new
            if qnew:
                fp.write("# TotalCPUHours, nProc, program, date, jobID\n")
            # Write remainder of file
            self._write_user_time(fp, j)

    # Read time from file handle
    def _read_start_time(self, fname: str):
        r"""Read most recent start time

        :Call:
            >>> nProc, tic = runner._read_start_time(fname)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file containing CPU usage history
        :Outputs:
            *nProc*: :class:`int`
                Number of cores
            *tic*: :class:`datetime.datetime`
                Time at which most recent run was started
        :Versions:
            * 2016-08-30 ``@ddalle``: v1.0 (stand-alone)
            * 2023-06-17 ``@ddalle``: v2.0; ``CaseRunner`` method
        """
        # Read the last line and split on commas
        vals = fileutils.tail(fname).split(',')
        # Get the number of processors
        nProc = int(vals[0])
        # Split date and time
        dtxt, ttxt = vals[2].strip().split()
        # Get year, month, day
        year, month, day = [int(v) for v in dtxt.split('-')]
        # Get hour, minute, second
        hour, minute, sec = [int(v) for v in ttxt.split(':')]
        # Construct date
        tic = datetime(year, month, day, hour, minute, sec)
        # Output
        return nProc, tic

    # Write start time
    def _write_start_time(self, fp, j: int):
        # Get options
        rc = self.read_case_json()
        # Initialize job ID
        jobID = self.get_job_id()
        # Program name
        prog = self._progname
        # Number of processors
        nproc = cmdgen.get_nproc(rc, j)
        # Set to one if `None`
        nproc = 1 if nproc is None else nproc
        # Format time
        t_text = self.tic.strftime('%Y-%m-%d %H:%M:%S %Z')
        # Write the data
        fp.write('%4i, %-20s, %s, %s\n' % (nproc, prog, t_text, jobID))

    # Write time since
    def _write_user_time(self, fp, j: int):
        # Get options
        rc = self.read_case_json()
        # Initialize job ID
        jobID = self.get_job_id()
        # Program name
        prog = self._progname
        # Get the time
        toc = datetime.now()
        # Number of processors
        nProc = cmdgen.get_nproc(rc, j)
        # Set to one if `None`
        nProc = 1 if nProc is None else nProc
        # Time difference
        t = toc - self.tic
        # Calculate CPU hours
        CPU = nProc * (t.days*24 + t.seconds/3600.0)
        # Format time
        t_text = toc.strftime('%Y-%m-%d %H:%M:%S %Z')
        # Write the data
        fp.write(
            '%8.2f, %4i, %-20s, %s, %s\n'
            % (CPU, nProc, prog, t_text, jobID))

   # --- Properties ---
    def _cls(self) -> str:
        r"""Get the full class name, e.g. ``cape.cfdx.casecntl.CaseRunner``

        :Call:
            >>> clsname = runner._clsname()
        :Outputs:
            *clsname*: :class:`str`
                Name of class w/o module included
        """
        # Get class
        cls = self.__class__
        # Get module and name
        return f"{cls.__module__}.{cls.__name__}"

    def _getmodname(self) -> str:
        r"""Get the module name for the class of *runner*

        :Call:
            >>> modname = runner._modname()
        :Outputs:
            *modname*: :class:`str`
                Name of module of *runner.__class__*
        """
        return self.__class__.__module__

    def _clsname(self) -> str:
        r"""Get the name of the class

        :Call:
            >>> clsname = runner._clsname()
        :Outputs:
            *clsname*: :class:`str`
                Name of class w/o module included
        """
        return self.__class__.__name__

  # *** WORKERS ***
   # --- Driver ---
    # Start concurrent workers and then run phase
    def run_phase_main(self, j: int) -> int:
        r"""Run one instance of one phase, including running any hooks

        :Call:
            >>> runner.run_phase_main(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Mark case as "active"
        self.mark_active()
        # Run workers
        self.run_worker_cmds(j)
        self.run_worker_pyfuncs(j)
        # Run command
        ierr = self.run_phase(j)
        # Mark case as not active (notify workers)
        self.mark_inactive()
        # Kill workers, if any
        self.kill_workers(j)
        # Output
        return ierr

   # --- Examples ---
    def archive_worker(self, host: Optional[str] = None):
        r"""Archive some files while running

        """
        # Get archivist
        a = self.get_archivist()
        # Save report files
        self.save_reportfiles()
        # Clean
        a.archive()

   # --- Runners ---
    def run_worker_cmds(self, j: int):
        r"""Run shell commands in parallel while main solver runs

        This will fork off one additional process for each entry of
        *WorkerShellCmds*, which will repeat every *WorkerSleepTime*
        seconds.

        :Call:
            >>> runner.run_worker_cmds(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Get shells
        shellcmds = rc.get_opt("WorkerShellCmds", j=j)
        sleeptime = rc.get_opt("WorkerSleepTime", j=j)
        # Exit if None
        if shellcmds is None or len(shellcmds) == 0:
            return
        # Log
        self.log_both(f"starting {len(shellcmds)} *WorkerShellCmds*", parent=1)
        self.log_verbose(f"using WorkerSleepTime={sleeptime} s", parent=1)
        # Loop through commands
        for shellcmd in shellcmds:
            self.fork_worker_shell(shellcmd, sleeptime)

    def run_worker_pyfuncs(self, j: int):
        r"""Run Python functions in parallel while main solver runs

        This will fork off one additional process for each entry of
        *WorkerPythonFuncs*, which will repeat every *WorkerSleepTime*
        seconds.

        :Call:
            >>> runner.run_worker_pyfuncs(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Read settings
        rc = self.read_case_json()
        # Get shells
        funclist = rc.get_opt("WorkerPythonFuncs", j=j)
        sleeptime = rc.get_opt("WorkerSleepTime", j=j)
        # Exit if None
        if funclist is None or len(funclist) == 0:
            return
        # Log
        self.log_both(
            f"starting {len(funclist)} *WorkerPythonFuncs*", parent=1)
        self.log_verbose(f"using WorkerSleepTime={sleeptime} s", parent=1)
        # Loop through commands
        for funcspec in funclist:
            self.fork_worker_func(funcspec, sleeptime)

    def fork_worker_shell(
            self,
            shellcmd: str,
            sleeptime: Optional[Union[float, int]] = None):
        r"""Fork one process to run a shell command while case runs

        :Call:
            >>> runner.fork_worker_shell(shellcmd, sleeptime=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *shellcmd*: :class:`str`
                Shell command to run
            *sleeptime*: {``None``} | :class:`float` | :class:`int`
                Time to sleep before restarting *shellcmd*
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Call the fork
        pid = os.fork()
        # Default sleep time
        sleeptime = DEFAULT_SLEEPTIME if sleeptime is None else sleeptime
        # Check parent/child
        if pid != 0:
            # Save the PID
            self.workers.append(pid)
            return
        # Loop until case is completed
        while True:
            # Sleep
            time.sleep(sleeptime)
            # Run command
            os.system(shellcmd)
            # Check if case is running
            if not self.check_active():
                os._exit(0)

    def fork_worker_func(
            self,
            funcspec: Union[str, dict],
            sleeptime: Optional[Union[float, int]] = None):
        r"""Fork one process to run a Python function while case runs

        :Call:
            >>> runner.fork_worker_func(funcspec, sleeptime=None)
            >>> runner.fork_worker_func(funcname, sleeptime=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *funcname*: :class:`str`
                Simple Python command name to execute
            *funcspec*: :class:`dict`
                Python function spec, see :class:`UserFuncOpts`
            *sleeptime*: {``None``} | :class:`float` | :class:`int`
                Time to sleep before restarting *shellcmd*
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Call the fork
        pid = os.fork()
        # Default sleep time
        sleeptime = DEFAULT_SLEEPTIME if sleeptime is None else sleeptime
        # Check parent/child
        if pid != 0:
            # Save the PID
            self.workers.append(pid)
            return
        # Loop until case is completed
        while True:
            # Sleep
            time.sleep(sleeptime)
            # Run command
            self.exec_modfunc(funcspec)
            # Check if case is running
            if not self.check_active():
                os._exit(0)

   # --- Cleanup ---
    def kill_workers(self, j: int = 0):
        r"""Kill any worker PIDs that may be running after phase

        This will wait up to *WorkerTimeout* seconds after the main
        solver is no longer active before killing subprocesses.

        :Call:
            >>> runner.kill_workers(j=0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``0``} | :class:`int`
                Phase number
        :Versions:
            * 2025-03-07 ``@ddalle``: v1.0
        """
        # Get current list of workers
        workers = self.workers
        # Check if workers are present
        if (workers is None) or (len(workers) == 0):
            # No action
            return
        # Read options
        rc = self.read_case_json()
        # Get wait time
        maxwait = rc.get_opt("WorkerTimeout", j=j, vdef=600.0)
        # Number of intervals to check on workers
        maxtries = 100
        # Get current time
        tic = time.time()
        # Wait time between checks
        dt = max(0.5, maxwait / maxtries)
        # Loop through them
        for _ in range(maxtries):
            # Initialize list of workers that are still running
            current_workers = []
            # Loop through all workers
            for pid in workers:
                try:
                    # Check on the requested process
                    outpid, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    continue
                # Check if it's running
                if outpid == 0:
                    # Still running
                    current_workers.append(outpid)
                else:
                    # Already done
                    self.log_verbose(f"worker PID {pid} already complete")
            # Check for current workers
            if len(current_workers) == 0:
                # All workers completed
                self.log_verbose("all workers completed")
                return
            # Wait
            time.sleep(dt)
            # Check for timeout
            if time.time() - tic >= maxwait:
                break
        # Kill remaining workers
        for pid in current_workers:
            # Check on the requested process
            outpid, _ = os.waitpid(pid, os.WNOHANG)
            # Check if it's running
            if outpid == 0:
                # Kill it
                os.kill(pid, signal.SIGTERM)
                # Log action after to avoid *pid* dieing early
                self.log_verbose(f"killed worker {pid} early")
            else:
                # Already done
                self.log_verbose(f"worker PID {pid} already complete")


# Function to call script or submit.
def StartCase():
    r"""Empty template for starting a case

    The function is empty but does not raise an error

    :Call:
        >>> cape.casecntl.StartCase()
    :See also:
        * :func:`cape.pycart.casecntl.StartCase`
        * :func:`cape.pyfun.casecntl.StartCase`
        * :func:`cape.pyover.casecntl.StartCase`
    :Versions:
        * 2015-09-27 ``@ddalle``: v1.0
        * 2023-06-02 ``@ddalle``: v2.0; empty
    """
    pass


# Set resource limit
def set_rlimit(
        r: int,
        ulim: ulimitopts.ULimitOpts,
        u: str,
        i: int = 0,
        unit: int = 1024):
    r"""Set resource limit for one variable

    :Call:
        >>> set_rlimit(r, ulim, u, i=0, unit=1024)
    :Inputs:
        *r*: :class:`int`
            Integer code of particular limit, from :mod:`resource`
        *ulim*: :class:`ULimitOpts`
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
        * 2016-03-13 ``@ddalle``: v1.0
        * 2021-10-21 ``@ddalle``: v1.1; check if Windows
        * 2023-06-20 ``@ddalle``: v1.2; was ``SetResourceLimit()``
    """
    # Check if limit not known or not applicable
    if u not in ulim or resource is None:
        return
    # Get the value of the limit
    l = ulim.get_ulimit(u, i)
    # Check the type
    if isinstance(l, (int, float)) and (l > 0):
        # Set the value numerically
        resource.setrlimit(r, (unit*l, unit*l))
    else:
        # Set unlimited
        resource.setrlimit(r, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


# Submit a job using PBS, Slurm, (something else,) or nothing
def _submit_job(rc, fpbs: str, j: int):
    r"""Submit a case to PBS, Slurm, or nothing

    :Call:
        >>> job_id = _submit_job(fpbs, j)
    :Inputs:
        *fpbs*: :class:`str`
            File name of PBS script
        *j*: :class:`int`
            Phase number
    :Outputs:
        *job_id*: ``None`` | :class:`str`
            Job ID number of new job, if appropriate
    :Versions:
        * 2023-06-02 ``@ddalle``: v1.0
        * 2023-11-07 ``@ddalle``: v1.1; switch test order
    """
    # Check submission type
    if rc.get_slurm(j):
        # Submit slurm job
        return queue.psbatch(fpbs)
    elif rc.get_qsub(j):
        # Submit PBS job
        return queue.pqsub(fpbs)


# Print current time
def _strftime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _shjoin(cmdi: Union[list, str]) -> str:
    # Check input type
    if isinstance(cmdi, str):
        # Already a string
        return cmdi
    else:
        # combine args
        return ' '.join([shlex.quote(arg) for arg in cmdi])


# Function to determine newest triangulation file
def GetTriqFile(proj='Components'):
    r"""Get most recent ``triq`` file and its associated iterations

    This is a template version with specific implementations for each
    solver. The :mod:`cape.cfdx` version simply returns the most recent
    ``triq`` file in the  folder with no iteration information.

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
        * 2016-12-19 ``@ddalle``: v1.0
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


# Convert scalar to list if necessary
def _listify(str_or_list: Union[str, list, tuple]) -> list:
    # Check if list
    if isinstance(str_or_list, (list, tuple)):
        return list(str_or_list)
    else:
        return [str_or_list]
