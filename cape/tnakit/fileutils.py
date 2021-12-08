# -*- coding: utf-8 -*-
r"""
:mod:`cape.tnakit.fileutils`: Platform-independent file read tools
===================================================================

This module provides Python versions of tools familiar to the Linux
command line. For example :func:`tail` to read the last one or more
lines of a file.

"""

# Standard library modules


# Local imports
from . import typeutils


# Read last *n* lines of file
def tail(fname, n=1):
    r"""Read last *n* lines of a file

    :Call:
        >>> lines = tail(fname, n=1)
        >>> lines = tail(fp, n=1)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *fp*: :class:`file`
            File open for reading
        *n*: {``1``} | :class:`int`
            Number of lines to read from end
    :Outputs:
        *lines*: :class:`list`\ [:class:`str`]
            Last *n* lines of file, or fewer if file is shorter
    :Versions:
        * 2021-11-05 ``@ddalle``: Version 1.0
    """
    # Check type
    if typeutils.isstr(fname):
        # Open the file by name
        with open(fname, 'r') as fp:
            return _tail(fp, n=n)
    elif typeutils.isfile(fname):
        # File already opened
        return _tail(fname, n=n)
    else:
        raise TypeError(
            "Expected file name or handle, got '%s'" % type(fname).__name__)


# Move backwards by one line
def readline_reverse(fp):
    r"""Read backwards from current position until start of line

    :Call:
        >>> line = readline_reverse(fp)
    :Inputs:
        *fp*: :class:`file` | :class:`io.IOBase`
            Open file handle
    :Outputs:
        *line*: :class:`str`
            Contents of *fp* from start of line to current position
    :Versions:
        * 2021-11-05 ``@ddalle``: Version 1.0
    """
    # Save current position
    pos0 = fp.tell()
    pos = pos0 - 1
    # Search backwards
    while pos > 0:
        # Move backwards
        fp.seek(pos - 1)
        # Read character
        c = fp.read(1)
        # Check for newline
        if c == "\n":
            break
        # Move back one character
        pos -= 1
    # Don't go before position 0
    pos = max(0, pos)
    # Read from *pos* to initial position
    line = fp.read(pos0 - pos)
    # Reset position
    fp.seek(pos)
    # Output
    return line


def _tail(fp, n=1):
    # Set current position
    pos = fp.tell()
    # Go to end of file
    fp.seek(0, 2)
    # Initialize outputs
    lines = []
    # Read backwards
    for j in range(n):
        # Read the line
        line = readline_reverse(fp)
        # Check for empty line (if *n* > lines in file)
        if line:
            lines.insert(0, line)
    # Reset position
    fp.seek(pos)
    # Output
    return lines

        