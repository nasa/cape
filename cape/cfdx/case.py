r"""
:mod:`cape.cfdx.case`: Case Control Module
==========================================
This module contains templates for interacting with and executing
individual cases. Since this is one of the most highly customized
modules of the CAPE system, there are few functions here, and the
functions that are present are mostly templates.

In general, the :mod:`case` module is used for actually running the CFD
solver (and any additional binaries that may be required as part of the
run process), and it contains other capabilities for renaming files and
determining the settings for a particular case. CAPE saves many settings
for the CFD solver and archiving in a file called ``case.json`` within
each case folder, which allows for the settings of one case to diverge
from the other cases in the same run matrix.

Actual functionality is left to individual modules listed below.

    * :mod:`cape.pycart.case`
    * :mod:`cape.pyfun.case`
    * :mod:`cape.pyover.case`
"""

# Standard library modules
import functools
import glob
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime
from io import IOBase, StringIO
from typing import Optional, Tuple

# System-dependent standard library
if os.name == "nt":
    resource = None
else:
    import resource

# Third-party
import numpy as np

# Local imports
from . import queue
from . import cmdrun
from .. import argread
from .. import fileutils
from .. import text as textutils
from .options import RunControlOpts
from ..errors import CapeRuntimeError
from ..tri import Tri


# Constants:
# Name of file that marks a case as currently running
RUNNING_FILE = "RUNNING"
# Name of file marking a case as in a failure status
FAIL_FILE = "FAIL"
# Name of file to stop at end of phase
STOP_PHASE_FILE = "CAPE-STOP-PHASE"
# Case settings
RC_FILE = "case.json"
# Run matrix conditions
CONDITIONS_FILE = "conditions.json"
# Logger files
LOGDIR = "cape"
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

# Return codes
IERR_OK = 0
IERR_CALL_RETURNCODE = 1
IERR_BOMB = 2
IERR_PERMISSION = 13
IERR_UNKNOWN = 14
IERR_NANS = 32
IERR_INCOMPLETE_ITER = 65
IERR_RUN_PHASE = 128

# Regular expression for run log files written by CAPE
REGEX_RUNFILE = re.compile("run.([0-9][0-9]+).([0-9]+)")


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


# Decorator for moving directories
def run_rootdir(func):
    r"""Decorator to run a function within a specified folder

    :Call:
        >>> func = run_rootdir(func)
    :Wrapper Signature:
        >>> v = runner.func(*a, **kw)
    :Inputs:
        *func*: :class:`func`
            Name of function
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
        *a*: :class:`tuple`
            Positional args to :func:`cntl.func`
        *kw*: :class:`dict`
            Keyword args to :func:`cntl.func`
    :Versions:
        * 2020-02-25 ``@ddalle``: v1.1 (:mod:`cape.cntl`)
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Declare wrapper function to change directory
    @functools.wraps(func)
    def wrapper_func(self, *args, **kwargs):
        # Recall current directory
        fpwd = os.getcwd()
        # Go to specified directory
        os.chdir(self.root_dir)
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
            # Go back to original folder
            os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func


# Logger for actions in a case
class CaseLogger(object):
    r"""Logger for an individual CAPE case

    :Call:
        >>> logger = CaseLogger(rootdir)
    :Inputs:
        *rootdir*: {``None``} | :class:`str`
            Absolute path to root folder of case
    :Outputs:
        *logger*: :class:`CaseLogger`
            Looger instance for one case
    """
   # --- Class attributes ---
    # Instance attributes
    __slots__ = (
        "root_dir",
        "fp",
    )

   # --- __dunder__ ---
    # Initialization
    def __init__(self, rootdir: Optional[str] = None):
        r"""Initialization method

        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Check for default
        if rootdir is None:
            # Save current path
            self.root_dir = os.getcwd()
        elif isinstance(rootdir, str):
            # Save absolute path
            self.root_dir = os.path.abspath(rootdir)
        else:
            # Bad type
            raise TypeError(
                "Logger *rootdir*: expected 'str' " +
                f"but got '{type(rootdir).__name}'")
        # Initialize file handles
        self.fp = {}

   # --- Actions ---
    def log_main(self, title: str, msg: str):
        r"""Write a message to primary case log

        :Call:
            >>> logger.log_main(title, msg)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *title*: :class:`str`
                Short string to use as classifier for log message
            *msg*: :class:`str`
                Main content of log message
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Remove newline
        msg = msg.rstrip('\n')
        # Create overall message
        line = f"{title},{_strftime()},{msg}\n"
        # Write it
        self.rawlog_main(line)

    def log_verbose(self, title: str, msg: str):
        r"""Write a message to verbose case log

        :Call:
            >>> logger.log_verbose(title, msg)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *title*: :class:`str`
                Short string to use as classifier for log message
            *msg*: :class:`str`
                Main content of log message
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Remove newline
        msg = msg.rstrip('\n')
        # Create overall message
        line = f"{title},{_strftime()},{msg}\n"
        # Write it
        self.rawlog_verbose(line)

    def logdict_verbose(self, title: str, data: dict):
        r"""Write a :class:`dict` to the verbose log as JSON content

        :Call:
            >>> logger.logdict_verbose(title, data)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *title*: :class:`str`
                Short string to use as classifier for log message
            *data*: :class:`dict`
                Information to write as JSON log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Convert *data* to string
        msg = json.dumps(data, indent=4, cls=_NPEncoder)
        # Create overall message
        txt = f"{title},{_strftime()}\n{msg}\n"
        # Write it
        self.rawlog_verbose(txt)

    def rawlog_main(self, msg: str):
        r"""Write a raw message to primary case log

        :Call:
            >>> logger.rawlog_main(msg)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *msg*: :class:`str`
                Content of log message
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Get file handle
        fp = self.open_main()
        # Write message
        fp.write(msg)

    def rawlog_verbose(self, msg: str):
        r"""Write a raw message to verbose case log

        :Call:
            >>> logger.rawlog_verbose(msg)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *msg*: :class:`str`
                Content of log message
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Get file handle
        fp = self.open_verbose()
        # Write message
        fp.write(msg)

   # --- File handles ---
    # Get main log file
    def open_main(self) -> IOBase:
        r"""Open and return the main log file handle

        :Call:
            >>> fp = logger.open_main()
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
        :Outputs:
            *fp*: :class:`IOBase`
                File handle or string stream for main log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        return self.open_logfile("main", LOGFILE_MAIN)

    # Get verbose log file
    def open_verbose(self) -> IOBase:
        r"""Open and return the verbose log file handle

        :Call:
            >>> fp = logger.open_main()
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
        :Outputs:
            *fp*: :class:`IOBase`
                File handle or string stream for verbose log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        return self.open_logfile("verbose", LOGFILE_VERBOSE)

    # Get file handle
    def open_logfile(self, name: str, fname: str) -> IOBase:
        r"""Open a log file, or get already open handle

        :Call:
            >>> fp = logger.open_logfile(name, fname)
        :Inputs:
            *logger*: :class:`CaseLogger`
                Looger instance for one case
            *name*: :class:`str`
                Name of logger, used as key in *logger.fp*
            *fname*: :class:`str`
                Name of log file relative to case's log dir
        :Outputs:
            *fp*: :class:`IOBase`
                File handle or string stream for verbose log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Get existing handle, if able
        fp = self.fp.get(name)
        # Check if it exists
        if fp is not None:
            # Use it
            return fp
        # Otherwise, open it
        fp = self._open_logfile(fname)
        # Save it and return it
        self.fp[name] = fp
        return fp

    # Open a file
    def _open_logfile(self, fname: str) -> IOBase:
        # Create log folder
        ierr = self._make_logdir()
        # If no folder made, use a text stream
        if ierr != IERR_OK:
            return StringIO()
        # Path to log file
        fabs = os.path.join(self.root_dir, LOGDIR, fname)
        # Try to open the file
        try:
            # Create the folder (if able)
            return open(fabs, 'a')
        except PermissionError:
            # Could not open file for writing; use text stream
            return StringIO()

    # Create log folder
    def _make_logdir(self) -> int:
        # Path to log folder
        fabs = os.path.join(self.root_dir, LOGDIR)
        # Check if it exists
        if os.path.isdir(fabs):
            # Already exists
            return IERR_OK
        # Try to make the folder
        try:
            # Create the folder (if able)
            os.mkdir(fabs)
            # Return code
            return IERR_OK
        except PermissionError:
            # Nonzero return code
            return IERR_PERMISSION


# Case runner class
class CaseRunner(object):
    r"""Class to handle running of individual CAPE cases

    :Call:
        >>> runner = CaseRunner(fdir=None)
    :Inputs:
        *fdir*: {``None``} | :class:`str`
            Optional case folder (by default ``os.getcwd()``)
    :Outputs:
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
    """
  # === Class attributes ===
    # Attributes
    __slots__ = (
        "j",
        "logger",
        "n",
        "nr",
        "rc",
        "returncode",
        "root_dir",
        "tic",
        "xi",
        "_mtime_case_json",
    )

    # Maximum number of starts
    _nstart_max = 100

    # Help message
    _help_msg = HELP_RUN_CFDX

    # Names
    _modname = "cfdx"
    _progname = "cfdx"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts

  # === __dunder__ ===
    def __init__(self, fdir=None):
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
        # Initialize slots
        self.j = None
        self.logger = None
        self.n = None
        self.nr = None
        self.rc = None
        self.tic = None
        self.xi = None
        self.returncode = IERR_OK
        self._mtime_case_json = 0.0
        # Other inits
        self.init_post()

   # --- Config ---
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

  # === Runners ===
   # --- Main runner methods ---
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
        j = self.get_phase()
        # Log progress
        self.log_verbose("start" f"phase={j}")
        # Get script name
        fpbs = self.get_pbs_script(j)
        # Check submission options
        if rc.get_slurm(j):
            # Verbose log
            self.log_verbose("start", "submitting slurm job")
            # Submit case
            job_id = queue.psbatch(fpbs)
            # Log
            self.log_both("start", f"submitted slurm job {job_id}")
            # Output
            return IERR_OK, job_id
        elif rc.get_qsub(j):
            # Verbose log
            self.log_verbose("start", "submitting PBS job")
            # Submit case
            job_id = queue.pqsub(fpbs)
            # Log
            self.log_both("start", f"submitted slurm job {job_id}")
            # Output
            return IERR_OK, job_id
        else:
            # Log
            self.log_both("start", "running in same process")
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
        """
        # Parse arguments
        a, kw = argread.readkeys(sys.argv)
        # Check for help argument.
        if kw.get('h') or kw.get('help'):
            # Display help and exit
            print(textutils.markdown(self._help_msg))
            # Stop execution
            return IERR_OK
        # Log startup
        self.log_verbose("run", f"start f{self._cls()}.run()")
        # Check if case is already running
        self.assert_not_running()
        # Mark case running
        self.mark_running()
        # Start a timer
        self.init_timer()
        # Log beginning
        self.log_main("run", f"{self._cls()}.run()")
        self.log_verbose("run", f"{self._cls()}.run() phase loop")
        # Initialize start counter
        nstart = 0
        # Loop until case exits, fails, or reaches start count limit
        while nstart < self._nstart_max:
            # Determine the phase
            j = self.get_phase()
            # Write start time
            self.write_start_time(j)
            # Prepare files as needed
            self.prepare_files(j)
            # Prepare environment variables
            self.prepare_env(j)
            # Run appropriate commands
            try:
                # Log
                self.log_both("run", f"run_phase({j})")
                # Run primary
                self.run_phase(j)
            except Exception:
                # Log failure encounter
                self.log_both("run", f"error during run_phase({j})")
                # Failure
                self.mark_failure("run_phase")
                # Stop running marker
                self.mark_stopped()
                # Return code
                return IERR_RUN_PHASE
            # Run *PostShellCmds* hook
            self.run_post_shell_cmds(j)
            # Clean up files
            self.finalize_files(j)
            # Save time usage
            self.write_user_time(j)
            # Check for other errors
            ierr = self.get_returncode()
            # If nonzero
            if ierr != IERR_OK:
                # Log return code
                self.log_both("run", "unsuccessful exit")
                self.log_both("run", f"retruncode={ierr}")
                # Stop running case
                self.mark_stopped()
                # Return code
                return ierr
            # Update start counter
            nstart += 1
            # Check for explicit exit
            if self.check_exit(j):
                # Log
                self.log_verbose("run", "explicit exit detected")
                break
            # Submit new PBS/Slurm job if appropriate
            if self.resubmit_case(j):
                # If new job started, this one should stop
                self.log_verbose(
                    "run", "exiting phase loop b/c new job submitted")
                break
        # Remove the RUNNING file
        self.mark_stopped()
        # Check for completion
        if self.check_complete():
            # Log
            self.log_both("run", "case completed")
            # Submit additional jobs if appropriate
            self.run_more_cases()
        # Return code
        return IERR_OK

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
            self.log_both("start", f"submitting more cases: NJob={nJob}")
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
                "start",
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
        """
        # Read settings
        rc = self.read_case_json()
        # Get "PostCmds"
        post_cmdlist = rc.get_RunControlOpt("PostShellCmds", j=j)
        # De-None it
        if post_cmdlist is None:
            post_cmdlist = []
        # Get new status
        j1 = self.get_phase()
        n1 = self.get_iter()
        # Run post commands
        for cmdj, cmdv in enumerate(post_cmdlist):
            # Create log file name
            flogbase = "postcmd%i.%02i.%i" % (cmdj, j1, n1)
            fout = flogbase + "out"
            ferr = flogbase + "err"
            # Check if we were given a string
            is_str = isinstance(cmdv, str)
            # Execute command
            cmdrun.callf(
                cmdv, f=fout, e=ferr, check=False, shell=is_str)

   # --- Runners: other executables ---
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
        ``aflr3bc`` files using :class:`cape.tri.Tri`.  The function
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
        """
        # Get iteration
        n = self.get_iter()
        # Check for initial run
        if (n is not None) or j:
            # Don't run AFLR3 if >0 iterations already complete
            return
        # Read settings
        rc = self.read_case_json()
        # Check for option to run AFLR3
        if not rc.get_aflr3_run(j=0):
            # AFLR3 not requested for this run
            return
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
                raise ValueError(
                    "User has requested AFLR3 volume mesh.\n" +
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
        cmdrun.aflr3(opts=rc)
        # Check for failure; aflr3 returns 0 status even on failure
        if os.path.isfile(ffail):
            # Remove RUNNING file
            self.mark_stopped()
            # Create failure file
            self.mark_failure("aflr3")
            # Error message
            raise RuntimeError(
                "Failure during AFLR3 run:\n" +
                ("File '%s' exists." % ffail))

    # Function to intersect geometry if appropriate
    def run_intersect(self, j: int, proj="Components"):
        r"""Run ``intersect`` to combine geometries if appropriate

        This is a multistep process in order to preserve all the component
        IDs of the input triangulations. Normally ``intersect`` requires
        each intersecting component to have a single component ID, and each
        component must be a water-tight surface.

        Cape utilizes two input files, ``Components.c.tri``, which is the
        original triangulation file with intersections and the original
        component IDs, and ``Components.tri``, which maps each individual
        original ``tri`` file to a single component. The files involved are
        tabulated below.

        * ``Components.tri``: Intersecting components, each with own compID
        * ``Components.c.tri``: Intersecting triangulation, original compIDs
        * ``Components.o.tri``: Output of ``intersect``, only a few compIDs
        * ``Components.i.tri``: Original compIDs mapped to intersected tris

        More specifically, these files are ``"%s.i.tri" % proj``, etc.; the
        default project name is ``"Components"``.  This function also calls
        the Chimera Grid Tools program ``triged`` to remove unused nodes from
        the intersected triangulation and optionally remove small triangles.

        :Call:
            >>> runner.run_intersect(j, proj)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
            *proj*: {``'Components'``} | :class:`str`
                Project root name
        :See also:
            * :class:`cape.tri.Tri`
            * :func:`cape.bin.intersect`
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0 (``CaseIntersect``)
            * 2016-04-05 ``@ddalle``: v1.1; generalize to ``cfdx``
            * 2023-06-21 ``@ddalle``: v1.2; update name; instance method
        """
        # Exit if not phase zero
        if j > 0:
            return
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
            print("File '%s' already exists; aborting intersect." % fitri)
            return
        # Set file names
        rc.set_intersect_i(ftri)
        rc.set_intersect_o(fotri)
        # Run intersect
        if not os.path.isfile(fotri):
            cmdrun.intersect(opts=rc)
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
        # Check if we can use ``triged`` to remove unused triangles
        if o_triged:
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
        else:
            # Trim unused trianlges (internal)
            trii.RemoveUnusedNodes(v=True)
            # Write trimmed triangulation
            trii.Write(futri)
        # Check if we should remove small triangles
        if o_rm and o_triged:
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
            print(" > triged < triged.%s.i > triged.%s.o" % (infix, infix))
            os.system("triged < triged.%s.i > triged.%s.o" % (infix, infix))
        elif o_rm:
            # Remove small triangles (internally)
            trii.RemoveSmallTris(o_smalltri, v=True)
            # Write final triangulation file
            trii.Write(fitri)
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
            *rc*: :class:`RunControlOpts`
                Case options interface from ``case.json``
            *proj*: {``'Components'``} | :class:`str`
                Project root name
            *n*: :class:`int`
                Iteration number
            *fpre*: {``'run'``} | :class:`str`
                Standard output file name prefix
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0; from :func:`run_flowCart`
            * 2016-04-05 ``@ddalle``: v1.1; generalize to :mod:`cape`
            * 2023-06-21 ``@ddalle``: v2.0; instance method
        """
        # Exit if not phase zero
        if j > 0:
            return
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
        # Set file name
        rc.set_verify_i('%s.i.tri' % proj)
        # Run it.
        cmdrun.verify(opts=rc)

   # --- System ---
    # Run a function
    def callf(
            self,
            cmdi: list,
            f: Optional[str] = None,
            e: Optional[str] = None) -> int:
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
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2024-07-16 ``@ddalle``: v1.0
            * 2024-08-03 ``@ddalle``: v1.1; add log messages
        """
        # Log command
        self.log_main("run", "> " + _shjoin(cmdi))
        self.log_data(
            "run", {
                "cmd": _shjoin(cmdi),
                "stdout": f,
                "stderr": e,
                "cwd": os.getcwd()
            })
        # Run command
        ierr = cmdrun.callf(cmdi, f=f, e=e, check=False)
        # Save return code
        self.log_both("run", f"returncode={ierr}")
        # Save return code
        self.returncode = ierr
        # Output
        return ierr

  # === Readers ---
   # --- Local info ---
    # Read ``case.json``
    def read_case_json(self):
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
                # Create default
                self.rc = self._rc_cls()
            # Save modification time
            self._mtime_case_json = mtime
        # Output
        return self.rc

    # Read ``conditions.json``
    def read_conditions(self, f=False):
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
    def read_condition(self, key: str, f=False):
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
        j = self.get_phase(f=False)
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

  # === Status ===
   # --- Next action ---
    # Check if case should exit for any reason
    @run_rootdir
    def check_exit(self, ja: int) -> bool:
        r"""Check if a case should exit for any reason

        Reasons a case should exit include

        * The case is finished.
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
        jb = self.get_phase(rc)
        # Get STOP-PHASE option
        if jb != ja:
            # Moving to next phase
            if not rc.get_RunControlOpt("StartNextPhase", ja):
                # Exit b/c incremental option in ``case.json``
                return True
            # Read from file
            q, jstop = self.read_stop_phase()
            # Check CAPE-STOP-PHASE settings
            if q and ((jstop is None) or (ja >= jstop)):
                # Delete the STOP file
                os.remove(STOP_PHASE_FILE)
                # Exit
                return True
        else:
            # Restarting same phase
            if not rc.get_RunControlOpt("RestartSamePhase", ja):
                # Exit b/c incremental option on w/i this phase
                return True
        # Fall back to check_complete()
        return self.check_complete()

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
        # Read case JSON
        rc = self.read_case_json()
        # Determine current phase
        j = self.get_phase(rc)
        # Check if final phase
        if j < self.get_last_phase():
            return False
        # Get absolute iter
        n = self.get_iter()
        # Get restart iter (calculated above)
        nr = self.nr
        # Check for stop iteration
        qstop, nstop = self.get_stop_iter()
        # Check iteration number
        if nr is None:
            # No iterations complete
            return False
        elif qstop and (n >= nstop):
            # Stop requested by user
            return True
        elif nr < rc.get_LastIter():
            # Not enough iterations complete
            return False
        else:
            # All criteria met
            return True

   # --- Overall status ---
    # Check overall status
    @run_rootdir
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
                * ``ERROR``: error detected
                * ``RUNNING``: case is currently running
                * ``INCOMP``: case not running and not finished
        """
        # Get initial status (w/o checking file ages or queue)
        if self.check_error() != IERR_OK:
            # Found FAIL file or other evidence of errors
            sts = "ERROR"
        elif self.check_running():
            # Found RUNNING file
            sts = "RUNNING"
        else:
            # Get phase number and iteration required to finish case
            jmax = self.get_last_phase()
            nmax = self.get_last_iter()
            # Get current status
            j = self.get_phase()
            n = self.get_iter()
            # Check both requirements
            if (j < jmax) or (n < nmax):
                sts = "INCOMP"
            else:
                # All criteria met
                sts = "DONE"
        # Output
        return sts

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

    @run_rootdir
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
        """
        return os.path.isfile(RUNNING_FILE)

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

   # --- Phase ---
    # Determine phase number
    @run_rootdir
    def get_phase(self, f=True) -> int:
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
        """
        # Get case options
        rc = self.read_case_json()
        # Get prefix
        fpre = self._logprefix
        # Get phase sequence
        phases = self.get_phase_sequence()
        # Initialize
        j = 0
        # Loop through possible phases
        for i, j in enumerate(phases):
            # Check for output files
            if len(glob.glob('%s.%02i.*' % (fpre, j))) == 0:
                # This run has not been completed yet
                return j
            # Check the iteration number
            if n < rc.get_PhaseIters(i):
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

   # --- Iteration ---
    # Get most recent observable iteration
    @run_rootdir
    def get_iter(self, f=True):
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
        # Check if present
        if not (f or self.n is None):
            # Return existing calculation
            return self.n
        # Otherwise, calculate
        self.n = self.getx_iter()
        # Output
        return self.n

    # Get most recent observable iteration
    def getx_iter(self):
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
            >>> nmax = runner.get_phase_iters(j)
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
        self.nr = self.getx_restart_iter()
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

   # --- File names ---
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
        j1 = self.get_phase()
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
                self.log_verbose("run", f"continuing phase {j0} in same job")
                return False
        # Now we know we're going to new phase; check the *Resubmit* opt
        if rc.get_RunControlOpt("ResubmitNextPhase", j0):
            # Submit phase *j1* as new job
            _submit_job(rc, fpbs, j1)
            return True
        else:
            # Continue to next phase in same job
            self.log_verbose("run", f"continuing to phase {j1} in same job")
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
            self.log_verbose("run", f"qdel {jobID}")
            # Delete Slurm job
            queue.scancel(jobID)
        elif rc.get_qsub(self.j):
            # Log message
            self.log_verbose("run", f"scancel {jobID}")
            # Delete PBS job
            queue.qdel(jobID)

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
        self.log_verbose("run", "case running")
        # Create RUNNING file
        fileutils.touch(RUNNING_FILE)

    # General function to mark failures
    @run_rootdir
    def mark_failure(self, msg="no details"):
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
        self.log_both("run", f"error, {txt}")
        # Append message to failure file
        open(FAIL_FILE, "a+").write(txt)

    # Delete running file if appropriate
    @run_rootdir
    def mark_stopped(self):
        r"""Delete the ``RUNNING`` file if it exists

        :Call:
            >>> mark_stopped()
        :Versions:
            * 2023-06-02 ``@ddalle``: v1.0
            * 2024-08-03 ``@ddalle``: v1.1; add log message
        """
        # Log
        self.log_verbose("run", "case stopped")
        # Check if file exists
        if os.path.isfile(RUNNING_FILE):
            # Delete it
            os.remove(RUNNING_FILE)

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
            self.log_verbose("run", "case already running")
            # Case already running
            raise CapeRuntimeError('Case already running!')

   # --- Configuration ---
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
            # Set the environment variable from scratch
            os.environ[key] = val
        # Get ulimit parameters
        ulim = rc['ulimit']
        # Block size
        block = resource.getpagesize()
        # Set the stack size
        set_rlimit(resource.RLIMIT_STACK,   ulim, 's', j, 1024)
        set_rlimit(resource.RLIMIT_CORE,    ulim, 'c', j, block)
        set_rlimit(resource.RLIMIT_DATA,    ulim, 'd', j, 1024)
        set_rlimit(resource.RLIMIT_FSIZE,   ulim, 'f', j, block)
        set_rlimit(resource.RLIMIT_MEMLOCK, ulim, 'l', j, 1024)
        set_rlimit(resource.RLIMIT_NOFILE,  ulim, 'n', j, 1)
        set_rlimit(resource.RLIMIT_CPU,     ulim, 't', j, 1)
        set_rlimit(resource.RLIMIT_NPROC,   ulim, 'u', j, 1)

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
            * 2023-06-20 ``@ddalle``: ``cfdx`` abstract method
        """
        pass

  # === Logging ===
   # --- Logging ---
    def log_main(self, title: str, msg: str):
        r"""Write a message to primary log

        :Call:
            >>> runner.log_main(title, msg)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *title*: :class:`str`
                Title/classifier for log message
            *msg*: :class:`str`
                Primary content of message
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)

    def log_both(self, title: str, msg: str):
        r"""Write a message to both primary and verbose logs

        :Call:
            >>> runner.log_both(title, msg)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *title*: :class:`str`
                Title/classifier for log message
            *msg*: :class:`str`
                Primary content of message
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)
        logger.log_verbose(title, msg)

    def log_verbose(self, title: str, msg: str):
        r"""Write a message to verbose log

        :Call:
            >>> runner.log_verbose(title, msg)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *title*: :class:`str`
                Title/classifier for log message
            *msg*: :class:`str`
                Primary content of message
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_verbose(title, msg)

    def log_data(self, title: str, data: dict):
        r"""Write :class:`dict` to verbose log as JSON

        :Call:
            >>> runner.log_data(title, data)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *title*: :class:`str`
                Title/classifier for log message
            *data*: :class:`dict`
                Parameters to write to verbose log as JSON
        :Versions:
            * 2024-08-01 ``@ddalle``: v1.0
        """
        # Get looger
        logger = self.get_logger()
        # Log parameters in the dict
        logger.logdict_verbose(title, data)

    def get_logger(self) -> CaseLogger:
        # Initialize if it's None
        if self.logger is None:
            self.logger = CaseLogger(self.root_dir)
        # Output
        return self.logger

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
    def get_cpu_time(self):
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
        nProc = rc.get_nProc(j)
        # Set to one if `None`
        nProc = 1 if nProc is None else nProc
        # Format time
        t_text = self.tic.strftime('%Y-%m-%d %H:%M:%S %Z')
        # Write the data
        fp.write('%4i, %-20s, %s, %s\n' % (nProc, prog, t_text, jobID))

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
        nProc = rc.get_nProc(j)
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
        r"""Get the full class name, e.g. ``cape.cfdx.case.CaseRunner``

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


# Function to call script or submit.
def StartCase():
    r"""Empty template for starting a case

    The function is empty but does not raise an error

    :Call:
        >>> cape.case.StartCase()
    :See also:
        * :func:`cape.pycart.case.StartCase`
        * :func:`cape.pyfun.case.StartCase`
        * :func:`cape.pyover.case.StartCase`
    :Versions:
        * 2015-09-27 ``@ddalle``: v1.0
        * 2023-06-02 ``@ddalle``: v2.0; empty
    """
    pass


# Set resource limit
def set_rlimit(r, ulim, u, i=0, unit=1024):
    r"""Set resource limit for one variable

    :Call:
        >>> set_rlimit(r, ulim, u, i=0, unit=1024)
    :Inputs:
        *r*: :class:`int`
            Integer code of particular limit, from :mod:`resource`
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


def _shjoin(cmdi: list) -> str:
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


# Customize JSON serializer
class _NPEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check for array
        if isinstance(obj, np.ndarray):
            # Check for scalar
            if obj.ndim > 0:
                # Convert to list
                return list(obj)
            elif np.issubdtype(obj.dtype, np.integer):
                # Convert to integer
                return int(obj)
            else:
                # Convert to float
                return float(obj)
        elif isinstance(obj, np.integer):
            # Convert to integer
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert to float
            return float(obj)
        # Otherwise use the default
        return super().default(obj)

