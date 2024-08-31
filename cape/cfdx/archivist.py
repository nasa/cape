r"""
:mod:`cape.cfdx.archivist`: File archiving and clean-up for cases
===================================================================

This module provides the class :mod:`CaseArchivist`, which conducts the
operations of commands such as

.. code-block:: bash

    pyfun --archive
    pycart --clean
    pyover --skeleton
"""

# Standard library
import os
import re
import sys
from typing import Optional

# Local imports
from .logger import ArchivistLogger
from .options.archiveopts import ArchiveOpts


# Class definition
class CaseArchivist(object):
   # --- Class attributes ---
    # Class attributes
    __slots__ = (
        "archivedir",
        "rootdir",
        "logger",
        "opts",
        "_restart_files",
    )

   # --- __dunder__ ---
    def __init__(self, opts: ArchiveOpts, where: Optional[str] = None):
        # Save root dir
        if where is None:
            # Use current dir
            self.rootdir = os.getcwd()
        else:
            # User-specified
            self.rootdir = where
        # Save p[topms
        self.opts = opts
        # Get archive dir (absolute)
        self.archivedir = os.path.abspath(opts.get_ArchiveFolder())

   # --- Logging ---
    def log(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to primary log

        :Call:
            >>> runner.log(msg, title, parent=0)
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
            * 2024-08-29 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)

    def warn(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to verbose log

        :Call:
            >>> runner.warn(title, msg)
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
            * 2024-08-29 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_warning(title, msg)

    def get_logger(self) -> ArchivistLogger:
        r"""Get or create logger instance

        :Call:
            >>> logger = runner.get_logger()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *logger*: :class:`ArchivistLogger`
                Logger instance
        :Versions:
            * 2024-08-29 ``@ddalle``: v1.0
        """
        # Initialize if it's None
        if self.logger is None:
            self.logger = ArchivistLogger(self.root_dir)
        # Output
        return self.logger

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


def ls_regex(pat: str) -> list:
    # Get folder name
    dirname, filepat = os.path.split(pat)
    # Folder; empty *dirname* -> "."
    listdir = dirname if dirname else "."
    # Get contents of folder
    allfiles = os.listdir(listdir)
    # Compile regex
    regex = re.compile(filepat)
    # Initialize outputs
    matches = []
    # Loop through files
    for fname in allfiles:
        # Compare against regex
        re_match = regex.fullmatch(fname)
        # Check for match
        if re_match is None:
            continue
        # Initialize group keys
        groupkeys = []
        # Full groups
        fullgroup = re_match.group()
        # Get groups with names
        groups = re_match.groupdict()
        # Loop through groups
        for j, group in enumerate(re_match.groups()):
            # Check for named group
            for k, v in groups.items():
                if v == group:
                    key = k
                    break
            else:
                # No named group; use index for key
                key = j
            # Save value
            groupkeys.append((key, group))
        # Save the match
        matches.append((tuple(groupkeys), os.path.join(dirname, fullgroup)))
    # Output
    return matches
