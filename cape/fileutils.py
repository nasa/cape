r"""
:mod:`cape.fileutils`: Pure-Python file information utilities
=================================================================

This module provides several file utilities that mimic common BASH
commands or programs, but written in pure Python to remove operating
system dependencies.
"""


# Default encoding
DEFAULT_ENCODING = "utf-8"


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

