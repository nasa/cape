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
from typing import Optional, Union

# Local imports
from .logger import ArchivistLogger
from .options.archiveopts import ArchiveOpts
from ..optdict import INT_TYPES


# Class definition
class CaseArchivist(object):
   # --- Class attributes ---
    # Class attributes
    __slots__ = (
        "archivedir",
        "casename",
        "logger",
        "opts",
        "root_dir",
        "_restart_files",
    )

   # --- __dunder__ ---
    def __init__(
            self,
            opts: ArchiveOpts,
            where: Optional[str] = None,
            casename: Optional[str] = None):
        # Save root dir
        if where is None:
            # Use current dir
            self.root_dir = os.getcwd()
        else:
            # User-specified
            self.root_dir = where
        # Default casename
        if casename is None:
            # Use two-levels of parent
            frun = os.path.basename(self.root_dir)
            fgrp = os.path.basename(frun)
            casename = f"{fgrp}/{frun}"
        # Save case name
        self.casename = casename
        # Save p[topms
        self.opts = opts
        # Get archive dir (absolute)
        self.archivedir = os.path.abspath(opts.get_ArchiveFolder())

   # --- Archive home ---
    # Ensure root of target archive exists
    def assert_archive(self):
        # Make sure archive root folder exists:
        if not os.path.isdir(self.archivedir):
            raise FileNotFoundError(
                "Cannot archive because archive\n" +
                f"  '{self.archivedir}' not found")

    # Make folders as needed for case
    def make_archive(self):
        # Test if archive exists
        self.assert_archive()
        # Get full/partial type
        atype = self.opts.get_ArchiveType()
        # Split case name into parts
        caseparts = self.casename.split('/')
        # If full archive, don't create last level
        if atype == "full":
            caseparts.pop(-1)
        # Loop through group folder(s)
        for part in caseparts:
            ...

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


# Search for file/folders matching regex, sorting by group
def rematch(pat: str) -> dict:
    r"""Search for file and folder names matching regular expression

    If the regex contains groups (parentheses), the results are grouped
    by the values of those groups. If the regex does not contain groups,
    the results are all in a single group called ``''``.

    :Call:
        >>> matchdict = rematch(pat)
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
        * 2024-09-02 ``@ddalle``: v1.0
    """
    # Split into parts
    pats = pat.split(os.sep)
    # Compile full regex
    regex = re.compile(pat)
    # Construct cumulative patterns (by folder depth level)
    fullpat = ""
    regexs = []
    cumpats = []
    for subpat in pats:
        # Combine path so far
        fullpat = os.path.join(fullpat, subpat)
        # Save it
        regexs.append(re.compile(subpat))
        cumpats.append(fullpat)
    # Get depth
    maxdepth = len(pats) - 1
    # Initialize matches
    matchdict = defaultdict(list)
    # Walk through file tree
    for root, dirnames, filenames in os.walk('.'):
        # Get depth
        depth = root.count(os.sep)
        # Check if final level
        if depth == maxdepth:
            # Final level; check folders and files
            for name in dirnames + filenames:
                # Full path
                fullpath = os.path.relpath(os.path.join(root, name), '.')
                # Check against full regex
                re_match = regex.fullmatch(fullpath)
                # Skip if no match
                if re_match is None:
                    continue
                # Compile label
                lbl = _match2str(re_match)
                # Add this match
                matchdict[lbl].append(fullpath)
            # Do not continue search deeper
            dirnames.clear()
        # Get regex for sub-level
        regexj = regexs[depth]
        # Get matches
        matchesj = _refilter(dirnames, regexj)
        # Replace full list with matches
        dirnames.clear()
        dirnames.extend(matchesj)
    # Output
    return dict(matchdict)


# Search for file/folders matching regex, sorting by group
def reglob(pat: str) -> list:
    r"""Search for file and folder names matching regular expression

    This function is constructed as a regular-expression version of
    :func:`glob.glob`, but it does not work with absolute paths.

    :Call:
        >>> matchlist = reglob(pat)
    :Inputs:
        *pat*: :class:`str`
            Regular expression pattern
    :Outputs:
        *matchlist*: :class:`list`\ [:class:`str`]
            Files and folders matching regular expression *pat* relative
            to current working directory
    :Versions:
        * 2024-09-02 ``@ddalle``: v1.0
    """
    # Split into parts
    pats = pat.split(os.sep)
    # Construct cumulative patterns (by folder depth level)
    fullpat = ""
    regexs = []
    cumpats = []
    for subpat in pats:
        # Combine path so far
        fullpat = os.path.join(fullpat, subpat)
        # Save it
        regexs.append(re.compile(subpat))
        cumpats.append(fullpat)
    # Get depth
    maxdepth = len(pats) - 1
    # Initialize matches
    matches = []
    # Walk through file tree
    for root, dirnames, filenames in os.walk('.'):
        # Get depth
        depth = root.count(os.sep)
        # Get regex for sub-level
        regexj = regexs[depth]
        # Check if final level
        if depth == maxdepth:
            # Final level; check folders and files
            submatches = _refilter(dirnames + filenames, regexj)
            # Construct complete paths
            for name in submatches:
                # Full path
                fullpath = os.path.relpath(os.path.join(root, name), '.')
                matches.append(fullpath)
            # Do not continue search deeper
            dirnames.clear()
        # Get matches
        matchesj = _refilter(dirnames, regexj)
        # Replace full list with matches
        dirnames.clear()
        dirnames.extend(matchesj)
    # Output
    return matches


def expand_fileopt(rawval: Union[list, dict, str], vdef: int = 0) -> dict:
    r"""Expand *Archive* file name/list/dict to common format

    The output is a :class:`dict` where the key is the pattern of file
    names to process and the value is an :class:`int` that represents
    the number of most recent files matching that pattern to keep.

    :Call:
        >>> patdict = expand_fileopt(rawstr, vdef=0)
        >>> patdict = expand_fileopt(rawlist, vdef=0)
        >>> patdict = expand_fileopt(rawdict, vdef=0)
    :Inputs:
        *rawstr*: :class:`str`
            Pattern of file names to process
        *rawlist*: :class:`list`\ [:class:`str`]
            List of filee name patterns to process
        *rawdict*: :class:`dict`\ [:class:`int`]
            Dictionary of file name patterns to process and *n* to keep
    :Outputs:
        *patdict*: :class:`dict`\ [:class:`int`]
            File name patterns as desribed above
    :Versions:
        * 2024-09-02 ``@ddalle``: v1.0
    """
    # Check for dict
    if isinstance(rawval, dict):
        # Copy it
        optval = dict(rawval)
        # Remove any non-int
        for k, v in rawval.items():
            # Check type
            if not isinstance(v, INT_TYPES):
                optval.pop(k)
        # Output
        return optval
    # Check for string
    if isinstance(rawval, str):
        return {rawval: vdef}
    # Initialize output for list
    optval = {}
    # Loop through items of list
    for rawj in rawval:
        # Check type
        if not isinstance(rawj, (dict, str, list, tuple)):
            continue
        # Recurse
        valj = expand_fileopt(rawj)
        # Save to total dict
        optval.update(valj)
    # Output
    return optval


# Filter list by regex
def _refilter(names: list, regex) -> list:
    # Initialize matches
    matches = []
    # Loop through candidates
    for name in names:
        if regex.fullmatch(name):
            matches.append(name)
    # Output
    return matches


# Match with groups
def _regroup(regex, name: str) -> tuple:
    # Match
    re_match = regex.fullmatch(name)
    # Check match
    if re_match is None:
        return None
    # Convert group info to string
    return _match2str(re_match)


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
