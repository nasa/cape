#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.fileutils`: File system utilities
=======================================================

This small module contains utilities for the CAPE testing system to
perform tasks on file trees.  This includes expanding a list of file
wildcard patterns into lists of extant files/folders.

"""

# Standard library modules
import os
import glob
import shutil


# Process list of files
def expand_file_list(fglobs, typ="f", error=True):
    """Expand a list of file globs
    
    :Call:
        >>> fnames = expand_file_list(fglobs, typ="f", error=True)
        >>> fnames = expand_file_list(fglob, typ="f", error=True)
    :Inputs:
        *fglob*: :class:`str` | ``None``
            Single file pattern
        *fglobs*: :class:`list`\ [:class:`str`]
            List of zero or more file patterns
        *typ*: ``"d"`` | {``"f"``} | ``"l"``
            File type; directory, file, or link
        *error*: {``True``} | ``False``
            Whether or not to raise an exception if a glob has not
            matches
    :Outputs:
        *fnames*: :class:`list`\ [:class:`str`]
            List of matches to *fglob*, without duplicates
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    # Ensure lists
    if not fglobs:
        fglobs = []
    elif not isinstance(fglobs, (tuple, list)):
        fglobs = [fglobs]
    # Initialize outputs
    fnames = []
    # Loop through entries
    for pattern in fglobs:
        # Get the matches
        fglob = glob.glob(pattern)
        # Check for empty glob
        if len(fglob) == 0:
            # Check for error
            if error:
                raise ValueError("No matches for pattern '%s'" % pattern)
            # Don't process further
            continue
        # Sort these matches
        fglob.sort()
        # Loop through matches to add them to overall list
        for fname in fglob:
            # Filter by type
            if typ in ["file", "f"]:
                # File: no links and no dirs
                if os.path.islink(fname):
                    continue
                elif os.path.isdir(fname):
                    continue
            elif typ in ["dir", "d"]:
                # Directory: no links and only dirs
                if os.path.islink(fname):
                    continue
                elif not os.path.isdir(fname):
                    continue
            elif typ in ["link", "l"]:
                # Link: link only
                if not os.path.islink(fname):
                    continue
            else:
                # Unrecognized type
                raise ValueError(
                    ("File type must be 'f', 'd', or 'l'; ") +
                    ("received '%s'" % typ))
            # Check if already in list
            if fname not in fnames:
                # Append to list
                fnames.append(fname)
    # Output
    return fnames
