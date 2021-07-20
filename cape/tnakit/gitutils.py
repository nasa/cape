#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`gitutils`: Basic utilities for git repos
================================================

This module provides one or more functions that provide information
about or execute actions in a git repository.

Commands can be executed on a remote host using SSH commands without any
third-party dependencies.

"""

# Standard library modules
import os

# Local modules
from . import shellutils


def get_gitdir(path=None, host=None, **kw):
    r"""Low-level interface to get working repo from ``git``

    If a bare repo, this will return the *GITDIR*, which is a folder
    ending with ``.git``.  Otherwise it will return the "working
    directory", which contains the ``.git/`` folder.

    :Call:
        >>> fgit = repo.get_gitdir(path, host=None, **kw)
    :Inputs:
        *repo*: :class:`aerohub.gitrepo.GitRepo`
            Interface to an AeroHub git repository
        *path*: :class:`str`
            Absolute path from which to get ``git`` repo
        *host*: {``None``} | :class:`str`
            Name of remote SSH host, if any
    :Outputs:
        *fgit*: ``None`` | :class:`str`
            Absolute path to top-level git directory
    :Versions:
        * 2021-07-19 ``@ddalle``: Version 1.0
    """
    # Default path
    if path is None:
        path = os.getcwd()
    # Combined options
    kw_call = dict(kw, cwd=path, host=host)
    # Check if in an obvious bare repo
    if path.endswith(".git"):
        # Assume a bare repo
        gitdir, _, ierr = shellutils.call_oe(
            ["git", "rev-parse", "--absolute-git-dir"], **kw_call)
    else:
        # Guess that it's a working repo and then double-check
        gitdir, _, ierr = shellutils.call_oe(
            ["git", "rev-parse", "--show-toplevel"], **kw_call)
        # If it's bare, *gitdir* will be empty
        if (ierr == 0) and (gitdir == ""):
            # It's a bare repo
            gitdir, _, ierr = shellutils.call_oe(
                ["git", "rev-parse", "--absolute-git-dir"], **kw_call)
    # Check for errors
    if ierr:
        return None
    # Output
    return gitdir.strip()

