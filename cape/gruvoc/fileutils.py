r"""
:mod:`gruvoc.fileutils`: Common file analysis tools
======================================================

This module provides functions for opening or analyzing files.

"""

# Standard library
import functools
from io import IOBase
from typing import Union

# Local imports
from .errors import assert_isfile, assert_isinstance


# Default encoding
DEFAULT_ENCODING = "utf-8"


# Open a file
def openfile(fname_or_fp: Union[str, IOBase], mode: str = "rb"):
    r"""Process a file handle or name and save attributes

    :Call:
        >>> fb = openfile(fname, mode="rb")
        >>> fb = openfile(fp, mode="rb")
        >>> fb = openfile(fb, mode="rb")
        >>> fb = openfile(None, mode="rb")
    :Inputs:
        *fname*: :class:`str`
            Name of a file to read
        *fp*: :class:`file`
            Existing file handle open in 'r' mode
        *fb*: :class:`file`
            Existing file handle open in 'rb' mode
        *mode*: {``"rb"``} | :class:`str`
            File read/write mode
    :Outputs:
        *fp*: :class:`io.IOBase`
            File handle open in *mode* mode
    """
    # Check other types
    if isfile(fname_or_fp):
        # Already a handle
        fp = fname_or_fp
        # Check if open
        if fp.closed:
            # Reopen
            return open(fp.name, mode)
        # Check mode
        if fp.mode == mode:
            # Already good
            return fp
        else:
            # Reopen it in suggested mode
            return open(fp.name, mode)
    elif isinstance(fname_or_fp, str):
        # Open the file
        return open(fname_or_fp, mode)
    # Bad type
    assert_isinstance(fname_or_fp, (str, IOBase), "file name or handle")


def isfile(fp) -> bool:
    r"""Check if a variable is a file handle

    :Class:
        >>> q = isfile(fp)
    :Inputs:
        *fp*: :class:`file` | **any**
            Any variable, usually file handle or file name
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *fp* is a file handle
    """
    return isinstance(fp, IOBase)


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


def keep_pos(func):
    @functools.wraps(func)
    def wrapper(fp, *args, **kwargs):
        # Run with exception handling
        try:
            # Get current position
            pos = fp.tell()
            # Attempt to run the function
            v = func(fp, *args, **kwargs)
        except Exception:
            # Raise the original error
            raise
        finally:
            # Return file to orginal position
            fp.seek(pos)
        # Output
        return v
    # Apply the wrapper
    return wrapper

