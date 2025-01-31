r"""
:mod:`cape.cfdx.casecntlbase`: Abstract base module for case control
=======================================================================

This module provides an abstract base class :class:`CaseRunnerBase`,
which contains much of the functionality for interacting with a single
case of one CFD solver

:See also:
    * :mod:`cape.cfdx.casecntl`
"""

# Standard library modules
import os
from abc import ABC, abstractmethod

# Local imports
from .archivist import CaseArchivist
from .options import RunControlOpts


# Return codes
IERR_OK = 0
IERR_CALL_RETURNCODE = 1
IERR_BOMB = 2
IERR_PERMISSION = 13
IERR_UNKNOWN = 14
IERR_NANS = 32
IERR_INCOMPLETE_ITER = 65
IERR_RUN_PHASE = 128


# Case runner class
class CaseRunnerBase(ABC):
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
  # === Config ===
   # --- Class attributes ---
    # Attributes
    __slots__ = (
        "cntl",
        "j",
        "logger",
        "archivist",
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

    # Names
    _modname = "cfdx"
    _progname = "cfdx"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts
    _archivist_cls = CaseArchivist

   # --- __dunder__ ---
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
        self.cntl = None
        self.j = None
        self.logger = None
        self.archivist = None
        self.n = None
        self.nr = None
        self.rc = None
        self.tic = None
        self.xi = None
        self.returncode = IERR_OK
        self._mtime_case_json = 0.0
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

