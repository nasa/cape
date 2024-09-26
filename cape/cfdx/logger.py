r"""
:mod:`cape.cfdx.logger`: Logging utilities for CAPE
=====================================================

This module provides the class :class:`CaseLogger` that is used to
record log messages for individual cases.
"""

# Standard library modules
import json
import os
import time
from io import IOBase, StringIO
from typing import Optional

# Third-party

# Local imports
from ..optdict import _NPEncoder


# Constants:
# Logger files
LOGDIR = "cape"
LOGFILE_MAIN = "cape-main.log"
LOGFILE_VERBOSE = "cape-verbose.log"
LOGFILE_ARCHIVE = "archive.log"
LOGFILE_ARCHIVE_WARNINGS = "archive-warnings.log"

# Return codes
IERR_OK = 0
IERR_CALL_RETURNCODE = 1
IERR_BOMB = 2
IERR_PERMISSION = 13
IERR_UNKNOWN = 14
IERR_NANS = 32
IERR_INCOMPLETE_ITER = 65
IERR_RUN_PHASE = 128


# Base logger class
class BaseLogger(object):
    r"""Template logger class

    :Call:
        >>> logger = BaseLogger()
    :Inputs:
        *rootdir*: {``None``} | :class:`str`
            Absolute path to root folder of case/run matrix
    :Outputs:
        *logger*: :class:`BaseLogger`
            Looger instance for one case
    """
   # --- Class attributes ---
    # Instance attributes
    __slots__ = (
        "root_dir",
        "fp",
    )

    # Class attributes
    _logdir = LOGDIR

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
        fabs = os.path.join(self.root_dir, self.__class__._logdir, fname)
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
        fabs = os.path.join(self.root_dir,  self.__class__._logdir)
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


# Logger for actions in a case
class CaseLogger(BaseLogger):
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
    __slots__ = ()

   # --- __dunder__ ---

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
        fp.flush()

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
        fp.flush()

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
            >>> fp = logger.open_verbose()
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


# Logger for actions in a case
class ArchivistLogger(BaseLogger):
    r"""Logger for archiving of individual case

    :Call:
        >>> logger = ArchivistLogger(rootdir)
    :Inputs:
        *rootdir*: {``None``} | :class:`str`
            Absolute path to root folder of case
    :Outputs:
        *logger*: :class:`CaseLogger`
            Looger instance for one case
    """
   # --- Class attributes ---
    # Instance attributes
    __slots__ = ()

   # --- Logging ---
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

    def log_warning(self, title: str, msg: str):
        r"""Write a message to verbose case log

        :Call:
            >>> logger.log_warning(title, msg)
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
        self.rawlog_warning(line)

    def rawlog_main(self, msg: str):
        r"""Write a raw message to primary case archiving log

        :Call:
            >>> logger.rawlog_main(msg)
        :Inputs:
            *logger*: :class:`ArchivistLogger`
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
        fp.flush()

    def rawlog_warning(self, msg: str):
        r"""Write a raw message to archiving warning log

        :Call:
            >>> logger.rawlog_warning(msg)
        :Inputs:
            *logger*: :class:`ArchivistLogger`
                Looger instance for one case
            *msg*: :class:`str`
                Content of log message
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        # Get file handle
        fp = self.open_warnings()
        # Write message
        fp.write(msg)
        fp.flush()

   # --- File handles ---
    # Get main log file
    def open_main(self) -> IOBase:
        r"""Open and return the main log file handle

        :Call:
            >>> fp = logger.open_main()
        :Inputs:
            *logger*: :class:`ArchivistLogger`
                Looger instance for one case
        :Outputs:
            *fp*: :class:`IOBase`
                File handle or string stream for main log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        return self.open_logfile("main", LOGFILE_ARCHIVE)

    # Get verbose log file
    def open_warnings(self) -> IOBase:
        r"""Open and return the warning log file handle

        :Call:
            >>> fp = logger.open_warnings()
        :Inputs:
            *logger*: :class:`ArchivistLogger`
                Looger instance for one case
        :Outputs:
            *fp*: :class:`IOBase`
                File handle or string stream for verbose log
        :Versions:
            * 2024-07-31 ``@ddalle``: v1.0
        """
        return self.open_logfile("verbose", LOGFILE_ARCHIVE_WARNINGS)


# Print current time
def _strftime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

