r"""
:mod:`cape.pylava.runinpfile`: Interface to LAVA-Cartesian input files
=======================================================================

This module provides the class :class:`CartInputFile`, which is used to
parse and modify the C-like input file to LAVA-Cartesian
"""

# Standard library
import os
import re
from io import IOBase
from typing import Optional

# Third-party

# Local imports


# Other constants
SPECIAL_CHARS = "{}()[]:=,;"

# Regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.?[0-9]*([EDed][+-]?[0-9]+)?")
RE_INT = re.compile(r"[+-]?[0-9]+")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# Base class
class CartInputFile(dict):
    r"""Interface to LAVA-Cartesian input files

    :Call:
        >>> inp = CartInputFile(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of input file
    :Outputs:
        *inp*: :class:`CartInputFile`
            Interface to one LAVA-Cartesian input file
    """
   # --- Class attributes ---
    __slots__ = (
        "fdir",
        "fname",
        "tab",
        "_target",
        "_terminated",
        "_word",
    )

   # --- __dunder__ ---
    def __init__(self, fname: Optional[str] = None):
        self.fdir = None
        self.fname = None
        self.tab = "    "
        # Initialize hidden attributes
        self._target = 1
        self._terminated = True
        self._word = ""
        # Process up to one arg
        if isinstance(fname, str):
            # Read namelist
            self.read_inpfile(fname)

   # --- Readers ---
    def read_inpfile(self, fname: str):
        # Absolutize file name
        if not os.path.isabs(fname):
            fname = os.path.realpath(fname)
        # Save folder and base file name
        self.fdir, self.fname = os.path.split(fname)

    # Read next thing
    def _read_next(self, fp: IOBase) -> int:
        r"""Read next unit of ``.vars`` file

        :Call:
            >>> opts._read_next(fp)
        """
        # Read next chunk
        chunk = _next_chunk(fp)
        # Check for special cases
        if chunk == "":
            # EOF
            return 0
        if chunk == "{":
            # Start of main section
            self._target = 0
            self._terminated = False
            return 1
        elif chunk == "}":
            # Back to "preamble" (??)
            self._target = 1
            self._terminated = True
            return 1
        # This should be an option name
        opt = chunk
        # We must have a "word" at this point
        assert_regex(opt, RE_WORD)
        # Read next chunk; must be ':'
        chunk = _next_chunk(fp)
        assert_nextstr(chunk, ':')
        # Read value
        val = _next_value(fp, opt)
        # Save it
        if self._target == 1:
            # Save to preamble
            self.preamble[opt] = val
        else:
            # Save to main body
            self[opt] = val
        # Read something if reaching this point
        return 1


# Check a character against an exact target
def assert_nextstr(c: str, target: str, desc=None):
    r"""Check if a string matches a specified target

    :Call:
        >>> assert_nextstr(c, target, desc=None)
    :Inputs:
        *c*: :class:`str`
            Any string
        *target*: :class:`str`
            Target value for *c*
        *desc*: {``None``} | :class:`str`
            Optional description for what is being tested
    :Raises:
        * :class:`ValueError` if *c* does not match *target*
    :Versions:
        * 2024-02-23 ``@ddalle``: v1.0
    """
    # Check if *c* is allowed
    if c == target:
        return
    # Create error message
    if desc is None:
        # Generic message
        msg1 = "Expected next char(s): '"
    else:
        # User-provided message
        msg1 = f"After {desc} expected: '"
    # Show what we got
    msg3 = f"'; got {repr(c)}"
    # Raise an exception
    raise ValueError(msg1 + target + msg3)


# Check that a string matches a compiled regular expression
def assert_regex(c: str, regex, desc=None):
    r"""Check if a string matches a compiled regular expression

    :Call:
        >>> assert_regex(c, regex, desc=None)
    :Inputs:
        *c*: :class:`str`
            Any string
        *regex*: :class:`re.Pattern`
            A compiled regular expression
        *desc*: {``None``} | :class:`str`
            Optional description for what is being tested
    :Raises:
        * :class:`ValueError` if *c* does not match *regex*
    :Versions:
        * 2024-02-23 ``@ddalle``: v1.0
    """
    # Check if *c* is allowed
    if regex.fullmatch(c):
        return
    # Combine
    msg2 = f"'{regex.pattern}'"
    # Create error message
    if desc is None:
        # Generic message
        msg1 = "Regex for next char: "
    else:
        # User-provided message
        msg1 = f"After {desc} expected to match: "
    # Show what we got
    msg3 = f"; got '{c}'"
    # Raise an exception
    raise ValueError(msg1 + msg2 + msg3)


# Read the next chunk
def _next_chunk(fp: IOBase) -> str:
    r"""Read the next "chunk", word, number, or special character

    Examples of chunks are:
        * Special single characters: ``{}[]<>():=,``
        * Otherwise the next sequence until a white space
        * An empty string if the end of the file

    :Call:
        >>> chunk = _next_chunk(fp)
    "Inputs:"
        *fp*: :class:`file`
            File handle open for reading
    :Outputs:
        *chunk*: :class:`str`
            Next chunk, zero or more characters, from one line
    """
    # Read first character
    c = _next_char(fp, newline=False)
    # Some single characters are automatically chunks
    if c in SPECIAL_CHARS:
        return c
    # Initialize
    chunk = c
    # Loop until white space encountered
    while True:
        # Next character
        c = fp.read(1)
        # Check for white space
        if c == ' ':
            # Single space is a special case
            if chunk and RE_FLOAT.fullmatch(chunk):
                # Check for units
                c = _next_char(fp, newline=True)
                # Check if it looks like units
                if RE_WORD.match(c):
                    # Looks like we found "units" for a float
                    fp.seek(fp.tell() - 1)
                    # Read the next chunk, which is units
                    units = _next_chunk(fp)
                    # Include the units with the value
                    return chunk + ' ' + units
                # Go back on char if we didn't find units
                if c != '':
                    fp.seek(fp.tell() - 1)
            # Still the end of the chunk
            return chunk
        elif c in ' \r\t\n':
            # White space encountered
            return chunk
        elif c == '/':
            # Read next char to check for comment
            c1 = fp.read(1)
            if c1 == '/':
                # Comment encountered
                fp.readline()
                return chunk
            elif c1 != '':
                # Single-slash; go back to prev char
                fp.seek(fp.tell() - 1)
        elif c in SPECIAL_CHARS:
            # Revert one character
            fp.seek(fp.tell() - 1)
            # Output
            return chunk
        # Otherwise, append to chunk
        chunk = chunk + c


# Read the next non-space character
def _next_char(fp: IOBase, newline: bool = True) -> str:
    r"""Read the next character of a ``.vars`` file

    This will skip white space and read to EOL if a comment

    :Call:
        >>> c = _next_char(fp, newline=True)
    :Inputs:
        *fp*: :class:`io.TextIOBase`
            File handle open for reading
        *newline*: {``True``} | ``False``
            Whether *c* can be ``"\n"``
    :Outputs:
        *c*: :class:`str`
            Single character, ``''`` for EOF
    """
    # Loop until we get a good character
    while True:
        # Read character
        c = fp.read(1)
        # Check it
        if c == '':
            # End of file
            return ''
        elif c in ' \t\r':
            # White space, try again
            continue
        elif c == '\n':
            # EOL
            if newline:
                # Return \n as char
                return c
            else:
                # Treat \n as white space
                continue
        elif c == '#':
            # Comment; read to end of line
            fp.readline()
            if newline:
                # End of line is next char
                return '\n'
            else:
                # End of line, but look for next char
                continue
        elif c == '/':
            # Check for another one
            c1 = fp.read(1)
            # If both are `/`, it's a comment
            if c1 == '/':
                # Comment, discard line
                fp.readline()
                # Read to end of line
                if newline:
                    # End of line is next char
                    return '\n'
                else:
                    # End of line, but look for next char
                    continue
            elif c1 != '':
                # Move back
                fp.seek(fp.tell() - 1)
        # Something else... return it
        return c
