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
from collections import defaultdict
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


def ls_regex(pat: str) -> dict:
    r"""List files that match regex, grouping matches by regex groups

    That is, all files that match the full pattern will be returned, but
    if *pat* has any regex groups in it, each matching file will be
    identified by the values of those groups.

    This is useful for archiving because it can find the most recent
    file for many file groups simultaneously with a properly constructed
    regular expression.

    The *pat* can also refer to subfolders, with two caveats:

    *   the folder names are literal (not regular expressions), and
    *   *pat* should use ``/`` as the path sep, not ``os.sep``.

    :Call:
        >>> matchdict = ls_regex(pat)
    :Inputs:
        *pat*: :class:`str`
            Regular expression pattern
    :Outputs:
        *matchdict*: :class:`dict`\ [:class:`list`]
            Mapping of files matching *pat* keyed by identifier for the
            groups in *pat*
        *matchdict[lbl]*: :class:`list`\ [:class:`str`]
            List of files matching *pat* with group values identified in
            *lbl*
    :Versions:
        * 2024-09-01 ``@ddalle``: v1.0
    """
    # Get folder name
    dirname, filepat = os.path.split(pat)
    # Folder; empty *dirname* -> "."
    listdir = dirname if dirname else "."
    # Get contents of folder
    allfiles = os.listdir(listdir)
    # Compile regex
    regex = re.compile(filepat)
    # Initialize outputs
    matchdict = defaultdict(list)
    # Loop through files
    for fname in allfiles:
        # Compare against regex
        re_match = regex.fullmatch(fname)
        # Check for match
        if re_match is None:
            continue
        # Generate label
        lbl = _match2str(re_match)
        # Full path to file
        fullfname = os.path.join(dirname, fname)
        # Append to list for that group identifier
        matchdict[lbl].append(fullfname)
    # Output
    return dict(matchdict)


# Convert match groups to string
def _match2str(re_match) -> str:
    r"""Create a tag describing the groups in a regex match object

    :Call:
        >>> lbl = _match2str(re_match)
    :Inputs:
        re_match: :mod:`re.Match`
            Regex match instance
    :Outputs:
        *lbl*: :class:`str`
            String describing contents of groups in *re_match*
    """
    # Initialize string
    lbl = ""
    # Get named groups
    groups = re_match.groupdict()
    # Loop through groups
    for j, group in enumerate(re_match.groups()):
        # Check for named group
        for k, v in groups.items():
            # Check this named group
            if v == group:
                # Found match
                lblj = f"{k}='{v}'"
                break
        else:
            # No named group; use index for key
            lblj = f"{j}='{group}'"
        # Add a space if necessary
        if lbl:
            lbl += " " + lblj
        else:
            lbl += lblj
    # Output
    return lbl
