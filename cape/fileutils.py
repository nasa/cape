r"""
:mod:`cape.fileutils`: Pure-Python file information utilities
=================================================================

This module provides several file utilities that mimic common BASH
commands or programs, but written in pure Python to remove operating
system dependencies.
"""

# Standard library
import os
import time


# Default encoding
DEFAULT_ENCODING = "utf-8"


# Return first few lines
def head(fname: str, n=1, encoding=DEFAULT_ENCODING):
    r"""Get first *n* lines of a file

    :Call:
        >>> txt = head(fname, n=1)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *n*: {``1``} | :class:`int`
            Number of lines to read from beginning of file
        *encoding*: {``"utf-8"``} | :class:`str`
            Encoding for text file
    :Versions:
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Initialize text
    txt = ""
    # Open file
    with open(fname, 'r', encoding=encoding) as fp:
        # Loop through lines
        for _ in range(n):
            # Read next line
            line = fp.readline()
            # Check for EOF
            if line == "":
                break
            # Append to text
            txt += line
    # Output
    return txt


# Pure Python to tail a file
def tail(fname: str, n=1, encoding=DEFAULT_ENCODING):
    r"""Get last *n* lines of a file

    :Call:
        >>> txt = tail(fname, n=1)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *n*: {``1``} | :class:`int`
            Number of lines to read from end of file
        *encoding*: {``"utf-8"``} | :class:`str`
            Encoding for text file
    :Versions:
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Number of lines read
    m = 0
    # Open file binary to enable relative search
    with open(fname, 'rb') as fb:
        # Go to end of file
        pos = fb.seek(0, 2)
        # Special case for empty file
        if pos < 2:
            # Return whole file if 0 or 1 chars
            fb.seek(0)
            return fb.read().decode(encoding)
        # Loop backwards through file until *n* newline chars
        while m < n:
            # Go back two chars so we can read previous one
            # Note special case:
            #    We don't actually check the final char!
            #    This avoids checking if file ends with \n
            pos = fb.seek(-2, 1)
            # Check for beginning of file
            if pos == 0:
                break
            # Read that character
            c = fb.read(1)
            # Check for newline
            if c == b"\n":
                m += 1
        # File is no after last \n; read to EOF
        return fb.read().decode(encoding)


# Create a file
def touch(fname: str):
    r"""Replicate ``touch``; creating new file or updating mod time

    :Call:
        >>> touch(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to create/modify
    :Versions:
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Test if file exists
    if os.path.isfile(fname):
        # Get current time
        tic = time.time()
        # Set access time and mod time
        os.utime(fname, (tic, tic))
    else:
        # Create the file and close it
        open(fname).close()

