r"""
:mod:`cape.cfdx.casecntlbase`: Abstract base classes for case interface
=======================================================================

This module provides an abstract base class for the
:class:`cape.cfdx.casecntl.CaseRunner` class that controls the CAPE
interface to individual CFD cases. The base class is
:mod:`CaseRunnerBase`.
"""

# Standard library
import re
from abc import ABC, abstractmethod
from typing import Optional

# Local imports
from .archivist import CaseArchivist
from .caseutils import run_rootdir
from .options import RunControlOpts

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

# Regular expression for run log files written by CAPE
REGEX_RUNFILE = re.compile("run.([0-9][0-9]+).([0-9]+)")


# Definition
class CaseRunnerBase(ABC):
    r"""Abstract base class for :class:`cape.cfdx.casecntl.CaseRunner`

    The main purpose for this class is to provide useful type
    annotations for :mod:`cape.cfdx.cntl` without circular imports.

    :Call:
        >>> runner = CaseRunnerBase()
    :Outputs:
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
    :Class attributes:
        * :attr:`_modname`
        * :attr:`_progname`
        * :attr:`_logprefix`
        * :attr:`_rc_cls`
        * :attr:`_archivist_cls`
    """
    # Maximum number of starts
    _nstart_max = 100

    # Names
    #: :class:`str`, Name of module
    _modname = "cfdx"
    #: :class:`str`, Name of main program controlled
    _progname = "cfdx"
    #: :class:`str`, Prefix for log files
    _logprefix = "run"

    # Specific classes
    #: Class for interpreting *RunControl* options from ``case.json``
    _rc_cls = RunControlOpts
    #: Class for case archiving instances
    _archivist_cls = CaseArchivist

    # Read ``case.json``
    @abstractmethod
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
        """
        pass

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

    # Get most recent observable iteration
    @abstractmethod
    def get_iter(self, f: bool = True):
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
        """
        pass

    # Determine phase number
    @abstractmethod
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
        """
        pass

    # Read total time
    @abstractmethod
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
        """
        pass

    # Get PBS/Slurm job ID
    @abstractmethod
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
        """
        pass

    # Write case settings to ``case.json``
    @abstractmethod
    def write_case_json(self, rc: RunControlOpts):
        r"""Write the current settinsg to ``case.json``

        :Call:
            >>> runner.write_case_json(rc)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *rc*: :class:`RunControlOpts`
                Options interface from ``case.json``
        """
        pass

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

    # Get iteration using simpler methods
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

