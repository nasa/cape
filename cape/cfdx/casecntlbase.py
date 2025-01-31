r"""
:mod:`cape.cfdx.casecntlbase`: Abstract base classes for case interface
=======================================================================

This module provides an abstract base class for the
:class:`cape.cfdx.casecntl.CaseRunner` class that controls the CAPE
interface to individual CFD cases. The base class is
:mod:`CaseRunnerBase`.
"""

# Standard library
from abc import ABC, abstractmethod
from typing import Optional

# Local imports
from .archivist import CaseArchivist
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


# Definition
class CaseRunnerBase(ABC):

    # Maximum number of starts
    _nstart_max = 100

    # Names
    _modname = "cfdx"
    _progname = "cfdx"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts
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

