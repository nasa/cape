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
from ..argread import clitext
from ..errors import CapeFileNotFoundError
from ..fileutils import grep
from ..gitutils import GitRepo
from ..optdict import OptionsDict


# List of default JSON file names
DEFAULT_JSON_FILES = (
    "pyCart.json",
    "pyFun.json",
    "pyKes.json",
    "pyLCH.json",
    "pyLava.json",
    "pyOver.json",
    "pyVul.json",
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
    r"""Find CAPE JSON files and report which solver

    The results will be returned in order from most recently modified to
    least recently modified.

    Candidate files are those matching *pat* that are either tracked by
    ``git`` (if in a git repo) or found in the current folder or one of
    its immediate subfolders, plus the standard ``py{X}.json`` names.
    The search is deliberately not recursive beyond one level because
    run folders can contain huge numbers of untracked files. Candidates
    are kept only if :func:`identify_solver` recognizes them, which
    requires a ``"RunControl"`` section.

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
        * 2026-10-03 ``@ddalle``: v1.3; add shallow glob search
    """
    # Default pattern
    pat = "*.json" if pat is None else pat
    # Read a git repository, if possible
    try:
        repo = GitRepo()
        # Get list of tracked files
        fnames = repo.ls_tree()
        # Filter them to JSON files
        git_json_files = fnmatch.filter(fnames, pat)
    except SystemError:
        # No git repo to search for candidate files
        git_json_files = []
    # Shallow search of current folder and immediate subfolders; avoid
    # recursive glob b/c this may be run from a huge run folder
    glob_json_files = sorted(glob.glob(pat))
    glob_json_files += sorted(glob.glob(os.path.join("*", pat)))
    # Combine candidates, including pyCart.json, etc. (usually untracked)
    raw_json_files = []
    found = set()
    for fname in git_json_files + glob_json_files + list(DEFAULT_JSON_FILES):
        # Normalize so git and glob results can be compared
        fname = os.path.normpath(fname)
        # Skip duplicates and missing files (e.g. deleted but tracked)
        if fname in found or not isfile(fname):
            continue
        found.add(fname)
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
        # Skip links to files not recognized as CAPE JSON files
        if fname not in fname_list:
            continue
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
    elif "VulcanInputFile" in opts:
        solver = "pyvul"
    elif "VulcanInpFile" in opts:
        solver = "pyvul"
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
    elif "Vulcan" in opts and isinstance(opts["Vulcan"], dict):
        solver = "pyvul"
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
    # Import Cntl here to avoid excessive overhead for calls such as
    # ``cape -h`` that import :mod:`cape.cfdx.cli` and thus this module
    from .cntl import Cntl
    # Initialize results
    configs = {}
    # Find JSON files
    json_files = find_json(pat)
    # Get current warning mode
    warnmode = Cntl._warnmode_default
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
    Cntl._warnmode_default = warnmode
    # Output
    return configs
