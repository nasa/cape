r"""
:mod:`cape.fileutils`: Pure-Python file information utilities
=================================================================

This module provides several file utilities that mimic common BASH
commands or programs, but written in pure Python to remove operating
system dependencies.
"""

# Standard library
import glob
import os
import re
import time
from io import IOBase
from typing import Optional


# Default encoding
DEFAULT_ENCODING = "utf-8"


# Return each line w/ a regular expression
def grep(pat: str, fname: str, encoding=DEFAULT_ENCODING) -> list:
    r"""Find lines of a file containing a regular expressoin

    :Call:
        >>> lines = grep(pat, fname, encoding="utf-8")
    :Inputs:
        *pat*: :class:`str`
            String of regular expression pattern
        *fname*: :class:`str`
            Name of file to search
        *encoding*: {``"utf-8"``} | :class:`str`
            Encoding for file
    :Outputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines containing a match of *pat*
    :Versions:
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Initialize output
    lines = []
    # Compile regular expression
    regex = re.compile(pat)
    # Open file
    with open(fname, encoding=encoding) as fp:
        # Loop through lines of file
        while True:
            # Read next line
            line = fp.readline()
            # Check for EOF
            if line == "":
                break
            # Check regular expression
            if regex.search(line):
                # Append to lines of matches
                lines.append(line)
    # Output
    return lines


# Return first few lines
def head(fname: str, n=1, encoding=DEFAULT_ENCODING) -> str:
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
    :Outputs:
        *txt*: :class:`str`
            Text of last *n* lines of *fname*
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
        # File is now after last \n; read to EOF
        return fb.read().decode(encoding)


# Read line backwards
def readline_reverse(fb: IOBase) -> bytes:
    r"""Read line ending at current position

    :Call:
        >>> txt = readline_reverse(fb)
    :Inputs:
        *fb*: :class:`IOBase`
            File handle open for reading in binary mode
    :Outputs:
        *txt*: :class:`bytes`
            Encoded text of last line
    :Versions:
        * 2024-07-26 ``@ddalle``: v1.0
    """
    # Check for start of file
    if fb.tell() == 0:
        return b''
    # Loop backwards
    while True:
        # Go back two chars so we can read previous one
        # Note special case:
        #    We don't actually check the final char!
        #    This avoids checking if file ends with \n
        pos = fb.seek(-2, 1)
        # Read that character
        c = fb.read(1)
        # Check for newline
        if (c == b"\n"):
            # Found newline, read line after *c*
            line = fb.readline()
            # Use position after *c*
            pos += 1
            break
        # Check for beginning of file
        # (This check comes second in case file starts with blank line)
        if pos == 0:
            # Go back before *c*
            fb.seek(-1, 1)
            # Read line
            line = fb.readline()
            break
        # Set current position to end of previous line
        fb.seek(pos)
        # Output
        return line


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
        open(fname, 'w').close()


# Count lines
def count_lines(fname: str) -> int:
    r"""Count lines in a file

    Meant to mimic results of

    .. code-block:: console

        $ wc -l $FNAME

    but written in pure Python.

    :Call:
        >>> n = count_lines(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`int`
            Number of lines in file
    :Versions:
        * 2024-07-15 ``@ddalle``: v1.0
    """
    # Number of bytes to read in one block
    m = 10000
    # Initialize count
    n = 0
    # Open file
    with open(fname, 'rb') as fp:
        # Read file in blocks of size *m* (avoids memory issues)
        while True:
            # Read block
            txt = fp.read(m)
            # Count newlines
            n += txt.count(b'\n')
            # Check for EOF
            if len(txt) < m:
                break
    # Check for missing \n at end of file
    if not txt.endswith(b'\n'):
        # This is definitely a line
        n += 1
    # Output
    return n


# Get latest file matching a regex
def get_latest_regex(pat: str, baseglob=None):
    r"""Get the latest modified file matching a regular expression

    :Call:
        >>> fname, regmatch = get_latest_regex(pat, baseglob=None)
    :Inputs:
        *pat*: :class:`str`
            Regular expression string
        *baseglob*: {``None``} | :class:`str`
            Optional glob pattern to narrow candidates in current dir
    :Outputs:
        *fname*: :class:`str`
            Name of latest modified file in list
        *regmatch*: :class:`re.Match`
            Regular expression groups, etc. for *fname*
    :Version:
        * 2024-03-24 ``@ddalle``: v1.0
    """
    # Get initial candidates
    if baseglob is None:
        # Use all files if not given an initial filter
        filelist = os.listdir('.')
    else:
        # Use files matching specified blog
        filelist = glob.glob(baseglob)
    # Initialize
    fname = None
    mtime = -1
    regmatch = None
    # Compile regular expression
    regex = re.compile(pat)
    # Loop through list
    for fnamej in filelist:
        # Check if file matches pattern
        matchj = regex.fullmatch(fnamej)
        # Skip if not a match
        if matchj is None:
            continue
        # Skip if a linke
        if not os.path.isfile(fnamej) or os.path.islink(fnamej):
            continue
        # Get modtime
        mtimej = os.path.getmtime(fnamej)
        # Check if it's the latest so far
        if mtimej > mtime:
            # Reassign
            fname, mtime, regmatch = fnamej, mtimej, matchj
    # Output
    return fname, regmatch


# Get latest file
def get_latest_file(filelist: list) -> str:
    r"""Get the latest modified file from a list of files

    :Call:
        >>> fname = get_latest_file(filelist)
    :Inputs:
        *filelist*: :class:`list`\ [:class:`str`]
            List of file names
    :Outputs:
        *fname*: :class:`str`
            Name of latest modified file in list
    :Version:
        * 2024-01-21 ``@ddalle``: v1.0
    """
    # Initialize
    fname = None
    mtime = -1.0
    # Loop through list
    for fnamej in filelist:
        # Check if file exists
        if not os.path.isfile(fnamej):
            continue
        # Get modtime
        mtimej = os.path.getmtime(fnamej)
        # Check if latest
        if mtimej > mtime:
            fname, mtime = fnamej, mtimej
    # Output
    return fname


# Sort file list by mod time
def sort_by_mtime(filelist: list) -> list:
    r"""Sort list of files by modification time (ascending)

    :Call:
        >>> tfiles = sort_by_mtime(filelist)
    :Inputs:
        *filelist*: :class:`list`\ [:class:`str`]
            List of file names
    :Outputs:
        *tfiles*: :class:`list`\ [:class:`str`]
            Files of *filelist* in order of ascending modification time
    :Version:
        * 2024-01-21 ``@ddalle``: v1.0
    """
    # Sort by mod time
    return sorted(filelist, key=os.path.getmtime)
