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

# Local imports
from ..gitutils import GitRepo


# Find JSON files
def find_json() -> list:
    # Read a git repository
    repo = GitRepo()
    # Get list of tracked files
    fnames = repo.ls_tree()
    # Filter them to JSON files
    json_files = fnmatch.filter(fnames, "*.json")
    # Output
    return json_files
