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
import glob
import os
from os.path import isfile
from typing import Optional, Union

# Local imports
from .cntl import Cntl
from ..argread import clitext
from ..errors import CapeFileNotFoundError
from ..fileutils import grep
from ..gitutils import GitRepo
from ..optdict import OptionsDict


# Default warning mode
_WARNMODE = Cntl._warnmode_default


# List of default JSON file names
DEFAULT_JSON_FILES = (
    "pyCart.json",
    "pyFun.json",
    "pyKes.json",
    "pyLCH.json",
    "pyLava.json",
    "pyOver.json",
    "cape.json",
)


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
    # Read a git repository, if possible
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
        * 2026-07-20 ``@ddalle``: v1.1; special rules for `py{X}.json``
        * 2026-07-29 ``@ddalle``: v1.2; work outside of git repo
    """
    # Default pattern
    pat = "*.json" if pat is None else pat
    # Read a git repository, if possible
    try:
        repo = GitRepo()
        # Get list of tracked files
        fnames = repo.ls_tree()
        # Filter them to JSON files
        raw_json_files = fnmatch.filter(fnames, pat)
    except SystemError:
        # No git repo to search for candidate files
        raw_json_files = (
            glob.glob(pat) +
            glob.glob(os.path.join("run", pat)))
    # Append pyCart.json, etc., if found (usually not tracked)
    for fname in DEFAULT_JSON_FILES:
        if isfile(fname) and fname not in raw_json_files:
            raw_json_files.append(fname)
    # Initialize list
    cape_json_files = []
    # Loop through candidates
    for candidate in raw_json_files:
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
    # Re-sort so that py{X}.json links are at the top
    for fname in DEFAULT_JSON_FILES:
        if not os.path.islink(fname):
            continue
        # Find the entry
        fname_list = [v[1] for v in json_files]
        i = fname_list.index(fname)
        # Remove that entry and move it to the top
        entry = json_files.pop(i)
        json_files.insert(0, entry)
    # Output
    return json_files


# Identify solver
def identify_solver(fjson: str) -> Optional[str]:
    r"""Determine the intended solver for a CAPE JSON file

    :Call:
        >>> solver = identify_solver(fjson)
    :Inputs:
        *fjson*: :class:`str`
            Name of JSON file to investigate
    :Outputs:
        *solver*: :class:`str` | ``None``
            Intended solver ``"pycart"``, ``"pyfun"``, etc., if one
            could be determined. If no `"RunControl"` section is found,
            returns ``None``. If  `"RunControl"` section is present but
            no other identifying features were found for a specific
            solver, returns ``"cfdx"``
    :Versions:
        * 2026-07-18 ``@ddalle``: v1.0
    """
    # Check for file
    if not os.path.isfile(fjson):
        raise CapeFileNotFoundError(f"No such file: '{fjson}'")
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
    elif "VarsFile" in opts:
        solver = "pylch"
    elif "Overflow" in opts and isinstance(opts["Overflow"], dict):
        solver = "pyover"
    elif "RunInputs" in opts and isinstance(opts["RunInputs"], dict):
        solver = "pylava"
    elif "Fun3D" in opts and isinstance(opts["Fun3D"], dict):
        solver = "pyfun"
    elif "AeroCsh" in opts:
        solver = "pycart"
    elif "flowCart" in rc and isinstance(rc["flowCart"], dict):
        solver = "pycart"
    elif "Vars" in opts and isinstance(opts["Vars"], dict):
        solver = "pylch"
    else:
        solver = "cfdx"
    # Output
    return solver


def identify_case_solver() -> Optional[str]:
    r"""Determine the intended solver of the current case folder

    :Call:
        >>> solver = identify_case_solver()
    :Outputs:
        *solver*: :class:`str` | ``None``
            Intended solver ``"pycart"``, ``"pyfun"``, etc., if one
            could be determined. If no ``case.json`` file is found,
            returns ``None``. If  ``case.json`` is present but
            no other identifying features were found for a specific
            solver, returns ``"cfdx"``
    :Versions:
        * 2026-07-20 ``@ddalle``: v1.0
    """
    # Check for main file
    if not isfile("case.json"):
        return
    # Check for identifying conditions
    if isfile("run_fun3d.pbs"):
        solver = "pyfun"
    elif isfile("run_cart3d.pbs"):
        solver = "pycart"
    elif isfile("run_chem.pbs"):
        solver = "pylch"
    elif isfile("run_overflow.pbs"):
        solver = "pyover"
    elif isfile("run_lava.pbs"):
        solver = "pylava"
    elif isfile("run_kestrel.pbs"):
        solver = "pykes"
    elif isfile("run_fun3d.00.pbs"):
        solver = "pyfun"
    elif isfile("run_cart3d.00.pbs"):
        solver = "pycart"
    elif isfile("run_chem.00.pbs"):
        solver = "pylch"
    elif isfile("run_overflow.00.pbs"):
        solver = "pyover"
    elif isfile("run_lava.00.pbs"):
        solver = "pylava"
    elif isfile("run_kestrel.00.pbs"):
        solver = "pykes"
    elif isfile("fun3d.nml") or isfile("fun3d.00.nml"):
        solver = "pyfun"
    elif isfile("input.cntl") or isfile("input.00.cntl"):
        solver = "pycart"
    elif isfile("run.00.inputs"):
        solver = "pylava"
    else:
        # Default
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
