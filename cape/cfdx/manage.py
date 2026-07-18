r"""
:mod:`cape.cfdx.manage`: Manage file counts and quotas
=========================================================

This module provides a variety of CAPE-related file management tools,
including

    * :func:`find_json` to find apparent main CAPE JSON files
    * :func:`find_large_cases` to find large case folders

and more.
"""

# Standard library
import fnmatch
import os
from typing import Optional, Union

# Local imports
from .cntl import Cntl
from ..argread import clitext
from ..fileutils import grep
from ..gitutils import GitRepo
from ..optdict import OptionsDict


# Default warning mode
_WARNMODE = Cntl._warnmode_default


# Find JSON files
def find_json(pat: Optional[str] = None) -> list:
    r"""Find all tracked CAPE JSON files in a repository

    The test is not perfect and consists of the following three fairly
    reliable criteria:

    1.  The JSON file is tracked by ``git``
    2.  The file name ends with ``.json``
    3.  The file contains ``"RunControl"``

    Obviously from criterion #1, this function only works in git
    repositories.

    :Call:
        >>> cape_json_files = find_json(pat=None)
    :Inputs:
        *pat*: {``None``} | :class:`str`
            Pattern to search for, defaults to ``"*.json"``
    :Outputs:
        *cape_json_files*: :class:`list`\ [:class:`str`]
            List of apparent CAPE JSON files
    :Versions:
        * 2025-09-25 ``@ddalle``: v1.0
        * 2026-07-18 ``@ddalle``: v1.1; search ``"RunControl"``
    """
    # Read a git repository
    repo = GitRepo()
    # Get list of tracked files
    fnames = repo.ls_tree()
    # Default pattern
    pat = "*.json" if pat is None else pat
    # Filter them to JSON files
    json_files = fnmatch.filter(fnames, pat)
    # Initialize list
    cape_json_files = []
    # Loop through candidates
    for candidate in json_files:
        # Check for "RunMatrix"
        if len(grep('"RunControl"', candidate)) > 0:
            # Append to list
            cape_json_files.append(candidate)
    # Output
    return cape_json_files


# Find JSON files and identify solver
def find_json_solver(pat: Optional[str] = None) -> list:
    r"""Find tracked CAPE JSON files in repo and report which solver

    The results will be returned in order from most recently modified to
    least recently modified.

    The test is not perfect and consists of the following three fairly
    reliable criteria:

    1.  The JSON file is tracked by ``git``
    2.  The file name ends with ``.json``
    3.  The file contains ``"RunControl"``

    Obviously from criterion #1, this function only works in git
    repositories.

    :Call:
        >>> json_files = find_json_solver()
    :Inputs:
        *pat*: {``None``} | :class:`str`
            Pattern to search for, defaults to ``"*.json"``
    :Outputs:
        *json_files*: :class:`list`\ [:class:`str`, :class:`str`]
            List of apparent CAPE JSON files
    :Versions:
        * 2026-07-18 ``@ddalle``: v1.0
    """
    # Read a git repository
    repo = GitRepo()
    # Get list of tracked files
    fnames = repo.ls_tree()
    # Default pattern
    pat = "*.json" if pat is None else pat
    # Filter them to JSON files
    json_files = fnmatch.filter(fnames, pat)
    # Initialize list
    cape_json_files = []
    # Loop through candidates
    for candidate in json_files:
        # Identify the solver
        solver = identify_solver(candidate)
        # Check result
        if solver is None:
            continue
        # Get modification time
        mtime = os.path.getmtime(candidate)
        # Append to list
        cape_json_files.append((solver, candidate, mtime))
    # Sort by modification time
    cape_json_files.sort(key=lambda x: x[2], reverse=True)
    # Eliminate mod times
    json_files = [mtch[:2] for mtch in cape_json_files]
    # Output
    return cape_json_files


# Identify solver
def identify_solver(fjson: str) -> Optional[str]:
    # Check for "RunControl"
    if len(grep('"RunControl"', fjson)) == 0:
        return
    # Read the file
    try:
        opts = OptionsDict(fjson)
    except Exception:
        return
    # Confirm *RunControl* is in the right place
    if "RunControl" not in opts:
        return
    # Select the RunControl section
    rc = opts["RunControl"]
    if not isinstance(rc, dict):
        return
    # Select
    # Check for identifying sections
    if "LAVASolver" in rc:
        solver = "pylava"
    elif "Namelist" in opts:
        solver = "pyfun"
    elif "InputCntl" in opts:
        solver = "pycart"
    elif "OverNamelist" in opts:
        solver = "pyover"
    elif "JobXML" in opts:
        solver = "pykes"
    elif "Overflow" in opts and isinstance(opts["Overflow"], dict):
        solver = "pyover"
    elif "RunInputs" in opts and isinstance(opts["RunInputs"], dict):
        solver = "pylava"
    elif "Fun3D" in opts and isinstance(opts["Fun3D"], dict):
        solver = "pyfun"
    elif "AeroCsh" in opts:
        solver = "pycart"
    else:
        solver = "cfdx"
    # Output
    return solver


# Find all large cases in repo
def search_repo_large(
        pat: Optional[str] = None,
        cutoff: Union[str, float, int] = "100MB", **kw) -> dict:
    # Initialize results
    configs = {}
    # Find JSON files
    json_files = find_json(pat)
    # Turn off warnings
    Cntl._warnmode_default = 0
    # Loop through
    for json_file in json_files:
        # Print name of JSON file
        print(clitext.bold(json_file))
        # Read JSON file
        try:
            cntl = Cntl(json_file)
        except Exception:
            continue
        # Find large files, w/ STDOUT
        large_cases = cntl.find_large_cases(cutoff, **kw)
        # Append to list
        configs[json_file] = large_cases
    # Reset warning mode
    Cntl._warnmode_default = _WARNMODE
    # Output
    return configs
