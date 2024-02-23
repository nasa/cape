r"""
:mod:`cape.pylch.varsfile`: Interface for Loci/CHEM ``.vars`` files
====================================================================

This module provides the class :class:`VarsFile` that reads, modifies,
and writes instances of the Loci/CHEM primary input file with the
extension ``.vars``.
"""

# Standard library
import os

# Third-party


# Other constants
SPECIAL_CHARS = "{}<>()[]:=,"


# Class for overall file
class VarsFile(dict):
    # Attributes
    __slots__ = (
        "fdir",
        "fname",
        "tab",
    )

   # --- __dunder__ ---
    # Initialization method
    def __init__(self, *args, **kw):
        # Number of positional args
        narg = len(args)
        # Initialize slots
        self.fdir = None
        self.fname = None
        self.section_char = None
        self.tab = "    "
        # Initialize section list
        self.section_order = []
        # Process up to one arg
        if narg == 0:
            # No arg
            a = None
        elif narg == 1:
            # One arg
            a = args[0]
        else:
            # Too many args
            raise TypeError(
                "%s() takes 0 to 1 arguments, but %i were given" %
                (type(self).__name__, narg))
        # Check sequential arg type
        if isinstance(a, str):
            # Read namelist
            self.read_varsfile(a)

   # --- Read ---
    # Read a file
    def read_varsfile(self, fname: str):
        r"""Read a Chem ``.vars`` file

        :Call:
            >>> opts.read_varsfile(fname)
        :Inputs:
            *fname*: :class:`str`
                Name of file to read
        :Versiosn:
            * 2024-03-12 ``@ddalle``: v1.0
        """
        # Absolutize file name
        if not os.path.isabs(fname):
            fname = os.path.realpath(fname)
        # Save folder and base file name
        self.fdir, self.fname = os.path.split(fname)
        # Open the file
        with open(fname, 'r') as fp:
            ...


# Read the next chunk
def _next_chunk(fp) -> str:
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
        if c in ' \r\t\n':
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
def _next_char(fp, newline=True) -> str:
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
