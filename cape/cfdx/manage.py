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
from typing import Optional

# Local imports
from .cntl import Cntl
from ..fileutils import grep
from ..gitutils import GitRepo


# Find JSON files
def find_json(pat: Optional[str] = None) -> list:
    r"""Find all apparent CAPE JSON files in a repository

    The test is not perfect and consists of the following three fairly
    reliable criteria:

    1.  The JSON file is tracked by ``git``
    2.  The file name ends with ``.json``
    3.  The file contains ``"RunMatrix"``

    Obviously from criterion #1, this function only works in git
    repositories.

    :Call:
        >>> cape_json_files = find_json()
    :Outputs:
        *cape_json_files*: :class:`list`\ [:class:`str`]
            List of apparent CAPE JSON files
    :Versions:
        * 2025-09-25 ``@ddalle``: v1.0
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
        if len(grep('"RunMatrix"', candidate)):
            # Append to list
            cape_json_files.append(candidate)
    # Output
    return cape_json_files
