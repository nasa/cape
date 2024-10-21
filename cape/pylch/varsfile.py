r"""
:mod:`cape.pylch.varsfile`: Interface for Loci/CHEM ``.vars`` files
====================================================================

This module provides the class :class:`VarsFile` that reads, modifies,
and writes instances of the Loci/CHEM primary input file with the
extension ``.vars``.
"""

# Standard library
import os
import re
from io import IOBase
from typing import Any, Optional

# Third-party
import numpy as np

# Local imports
from ..units import mks


# Other constants
SPECIAL_CHARS = "{}<>()[]:=,"
DEG = np.pi / 180.0

# Regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.?[0-9]*([EDed][+-]?[0-9]+)?")
RE_INT = re.compile(r"[+-]?[0-9]+")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# Class for overall file
class VarsFile(dict):
    # Attributes
    __slots__ = (
        "fdir",
        "fname",
        "preamble",
        "tab",
        "_target",
        "_terminated",
    )

   # --- __dunder__ ---
    # Initialization method
    def __init__(self, *args, **kw):
        # Number of positional args
        narg = len(args)
        # Initialize slots
        self.fdir = None
        self.fname = None
        self.tab = "    "
        self.preamble = {}
        # Initialize hidden attributes
        self._target = 1
        self._terminated = True
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
            # Loop until end of file
            while True:
                # Read the next entity
                n = self._read_next(fp)
                # Exit if nothing was read
                if n == 0:
                    break

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

   # --- Write ---
    # Write to file
    def write(self, fname: Optional[str] = None):
        r"""Write contents to ``.vars`` file

        :Call:
            >>> opts.write(fname=None)
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *fname*: {``None``} | :class:`str`
                File name to write
        :Versions:
            * 2024-02-24 ``@ddalle``: v1.0
        """
        # Default file name
        if fname is None:
            fname = os.path.join(self.fdir, self.fname)
        # Open file
        with open(fname, 'w') as fp:
            self._write(fp)

    def _write(self, fp: IOBase):
        r"""Write to ``.vars`` file handle

        :Call:
            >>> opts._write(fp)
        :Inputs:
            *fp*: :class:`IOBase`
                File handle, open for writing
        """
        # Write blank line
        fp.write("\n")
        # Loop through preamble entries
        for opt, val in self.preamble.items():
            # Write the statement
            fp.write(f"{opt}: {to_text(val)}\n")
        # Add another blank line, then open main dict
        fp.write("\n{\n")
        # Loop through main values
        for opt, val in self.items():
            # Write name and value, to_text() may recurse
            fp.write(f"{opt}: {to_text(val)}\n")
        # Last blank line and closing brace
        fp.write("}\n\n")

   # --- Data ---
    # Get Mach number
    def get_mach(
            self,
            name: str = "farfield",
            comp: Optional[str] = None) -> Optional[float]:
        r"""Get current Mach number, usually from "farifled" BC

        :Call:
            >>> m = opts.get_mach(name="farfield", comp=None)
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to query
        :Outputs:
            *m*: :class:`float` | ``None``
                Mach number from first farfield() function, if any
        :Versions:
            * 2024-02-26 ``@ddalle``: v1.0
            * 2024-10-21 ``@ddalle``: v2.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Check if we found any
        if len(funcs) == 0:
            # No Mach number
            return
        # Get the first function
        for func in funcs.values():
            break
        # Search the keyword args
        kwargs = func.get("kwargs", {})
        # Try to find "Mach" (*M*) input
        m = kwargs.get("M", kwargs.get("m"))
        # Check for Mach
        if m is not None:
            # Get value from magnitude or polar
            return get_magnitude(m)

    # Get angle of attack
    def get_alpha(
            self,
            name: str = "farfield",
            comp: Optional[str] = None) -> Optional[float]:
        r"""Get current angle of attack for one BC, in degrees

        :Call:
            >>> m = opts.get_alpha(name="farfield", comp=None)
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to query
        :Outputs:
            *a*: :class:`float` | ``None``
                Angle of attack [deg]
        :Versions:
            * 2024-10-21 ``@ddalle``: v1.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Check if we found any
        if len(funcs) == 0:
            # No Mach number
            return
        # Get the first function
        for func in funcs.values():
            break
        # Search the keyword args
        kwargs = func.get("kwargs", {})
        # Try to find "Mach" (*M*) input
        m = kwargs.get("M", kwargs.get("m"))
        # Check for Mach
        if isinstance(m, dict) and m.get("@function") == "polar":
            # Get arguments
            args = m.get("args", [])
            # Check length
            if len(args) > 1:
                return args[1] / DEG

    # Get angle of attack
    def get_beta(
            self,
            name: str = "farfield",
            comp: Optional[str] = None) -> Optional[float]:
        r"""Get current angle of attack for one BC, in degrees

        :Call:
            >>> m = opts.get_beta(name="farfield", comp=None)
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to query
        :Outputs:
            *b*: :class:`float` | ``None``
                Sideslip angle [deg]
        :Versions:
            * 2024-10-21 ``@ddalle``: v1.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Check if we found any
        if len(funcs) == 0:
            # No Mach number
            return
        # Get the first function
        for func in funcs.values():
            break
        # Search the keyword args
        kwargs = func.get("kwargs", {})
        # Try to find "Mach" (*M*) input
        m = kwargs.get("M", kwargs.get("m"))
        # Check for Mach
        if isinstance(m, dict) and m.get("@function") == "polar":
            # Get arguments
            args = m.get("args", [])
            # Check length
            if len(args) > 2:
                return args[2] / DEG

    # Set Mach number for farfield
    def set_mach(
            self,
            m: float,
            name: str = "farfield",
            comp: Optional[str] = None):
        r"""Set the Mach number for one or more farfield condition

        :Call:
            >>> opts.set_mach(m, name="farfield")
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *m*: :class:`float` | ``None``
                Mach number from first farfield() function, if any
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to which to apply BC
        :Versions:
            * 2024-02-26 ``@ddalle``: v1.0
            * 2024-10-21 ``@ddalle``: v2.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Set them all
        for func in funcs.values():
            # Get keyword args
            kwargs = func.setdefault("kwargs", {})
            # Get value of Mach number
            machval = kwargs.get("M", kwargs.pop("m", None))
            # Set parameter
            machfunc = set_polar_arg(machval, 0, m, np.zeros(3))
            # Save function
            kwargs["M"] = machfunc

    # Set angle of attack for farfield
    def set_alpha(
            self,
            a: float,
            name: str = "farfield",
            comp: Optional[str] = None):
        r"""Set the angle of attack for one or more farfield condition

        :Call:
            >>> opts.set_alpha(a, name="farfield")
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *a*: :class:`float` | ``None``
                Angle of attack [deg]
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to which to apply BC
        :Versions:
            * 2024-02-26 ``@ddalle``: v1.0
            * 2024-10-21 ``@ddalle``: v2.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Set them all
        for func in funcs.values():
            # Get keyword args
            kwargs = func.setdefault("kwargs", {})
            # Get value of Mach number
            machval = kwargs.get("M", kwargs.pop("m", None))
            # Set parameter
            machfunc = set_polar_arg(machval, 1, a*DEG, np.zeros(3))
            # Save function
            kwargs["M"] = machfunc

    # Set angle of sideslip for farfield
    def set_beta(
            self,
            b: float,
            name: str = "farfield",
            comp: Optional[str] = None):
        r"""Set the sideslip angle for one or more farfield condition

        :Call:
            >>> opts.set_alpha(a, name="farfield")
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *b*: :class:`float` | ``None``
                Sideslip angle [deg]
            *name*: {``"farfield"``} | :class:`str`
                Name of function to find
            *comp*: {``None``} | :class:`str`
                Optional name of component to which to apply BC
        :Versions:
            * 2024-10-21 ``@ddalle``: v1.0
        """
        # Find "boundary_conditions" functions of type *name*
        funcs = self.find_subfunctions("boundary_conditions", name, comp)
        # Set them all
        for func in funcs.values():
            # Get keyword args
            kwargs = func.setdefault("kwargs", {})
            # Get value of Mach number
            machval = kwargs.get("M", kwargs.pop("m", None))
            # Set parameter
            machfunc = set_polar_arg(machval, 2, b*DEG, np.zeros(3))
            # Save function
            kwargs["M"] = machfunc

   # --- General Data ---
    # Find or create function in subsection
    def find_subfunctions(
            self,
            sec: str,
            name: str,
            comp: Optional[str] = None,
            nmax: Optional[int] = None) -> dict:
        # Intialize subsection
        if sec not in self:
            self[sec] = VFileSubsec()
        # Get the subsection
        secdata = self[sec]
        # Search it
        funcs = _find_function(secdata, name, comp=comp, nmax=nmax)
        # Check for more than 0 matches OR no *comp* to initialize
        if (comp is None) or len(funcs):
            return funcs
        # Initialize a funciton
        func = {
            "@function": name,
            "args": [],
        }
        # Save it
        secdata[comp] = func
        # Return it
        return {comp: func}

    # Find a function of given name
    def find_function(self, name: str, nmax: Optional[int] = None) -> dict:
        r"""Find function(s) by name within ``.vars`` file

        :Call:
            >>> funcs = opts.find_function(name, nmax=None)
        :Inputs:
            *opts*: :class:`VarsFile`
                Chem ``.vars`` file interface
            *name*: :class:`str`
                Name of function to find
            *nmax*: {``None``} | :class:`int`
                Maximum number of times to find *name* functions
        :Outputs:
            *funcs*: :class:`dict`
                Each instance of *name* function found; key of each item
                says where the function was located w/i *data*
        :Versions:
            * 2024-02-24 ``@ddalle``: v1.0
        """
        # Pass to module-level functions
        return _find_function(self, name, nmax=nmax)


# Class for <> subsections
class VFileSubsec(dict):
    r"""Section for reading subsections marked by angle-brackets
    """
    # Attributes
    __slots__ = (
        "_terminated"
    )

    def __init__(self, a=None):
        # Set flag that the section was properly terminated
        self._terminated = True
        # Check type
        if isinstance(a, dict):
            # Just save the entities
            self.update(a)
        elif isinstance(a, IOBase):
            # Read it
            self._read(a)

    # Read
    def _read(self, fp: IOBase):
        # Read next chunk
        chunk = _next_chunk(fp)
        # This should be '<'
        assert_nextstr(chunk, '<')
        # Begin reading values
        while True:
            # Read next chunk
            chunk = _next_chunk(fp)
            # Check for end of section
            if chunk == '>':
                # End of section
                return
            elif chunk == '':
                # Reached EOF in middle of section
                self._terminated = False
                return
            # Otherwise we should have a "word"
            assert_regex(chunk, RE_WORD)
            # Got name of next value
            name = chunk
            # Now search for '='
            chunk = _next_chunk(fp)
            assert_nextstr(chunk, '=', f"delim after subsec param '{name}'")
            # Now get the next value (can be recursive)
            val = _next_value(fp, name)
            # Save
            self[name] = val
            # Read next chunk
            chunk = _next_chunk(fp)
            # Test it
            if chunk == ',':
                # Move on to next section
                continue
            elif chunk == '>':
                # End of subsection
                self._terminated = True
                return
            # Run-on
            raise ValueError(
                f"After subsection param '{name}', " +
                f"expected ',' or '>'; got '{chunk}'")


# Class for [] lists
class VFileList(list):
    r"""Section for reading subsections marked by angle-brackets
    """
    # Attributes
    __slots__ = (
        "_terminated"
    )

    def __init__(self, a=None):
        # Set flag that the section was properly terminated
        self._terminated = True
        # Check type
        if isinstance(a, list):
            # Just save the entities
            self.extend(a)
        elif isinstance(a, IOBase):
            # Read it
            self._read(a)

    # Read
    def _read(self, fp: IOBase):
        # Read next chunk
        chunk = _next_chunk(fp)
        # This should be '['
        assert_nextstr(chunk, '[')
        # Initialize "unterminated"
        self._terminated = False
        # Begin reading values
        while True:
            # Check current list
            name = f"list entry {len(self) + 1}"
            # Get the next value (could be recursive)
            val = _next_value(fp, name)
            # Save
            self.append(val)
            # Read next chunk
            chunk = _next_chunk(fp)
            # Test it
            if chunk == ',':
                # Move on to next section
                continue
            elif chunk == ']':
                # End of list
                self._terminated = True
                return
            # Run-on
            raise ValueError(
                f"After list entry {len(self)}; " +
                f"expected ',' or ']'; got '{chunk}'")


# Class for <> subsections
class VFileFunction(dict):
    r"""Section for reading "functions" marked by regular parentheses
    """
    # Attributes
    __slots__ = (
        "_terminated"
    )

    def __init__(self, a, name: str):
        # Set flag that the section was properly terminated
        self._terminated = True
        # Initialize name
        self["@function"] = name
        self["args"] = []
        self["kwargs"] = {}
        # Check type
        if isinstance(a, dict):
            # Just save the entities
            self["data"].update(a)
        elif isinstance(a, IOBase):
            # Read it
            self._read(a)

    # Read
    def _read(self, fp: IOBase):
        # Get handle to data portion
        args = self["args"]
        kwargs = self["kwargs"]
        # Function name
        funcname = self["@function"]
        # Read next chunk
        chunk = _next_chunk(fp)
        # This should be '('
        assert_nextstr(chunk, '(')
        # Begin reading values
        while True:
            # Read next chunk
            chunk = _next_chunk(fp)
            # Check for end of section
            if chunk == ')':
                # End of section
                return
            elif chunk == '':
                # Reached EOF in middle of section
                self._terminated = False
                return
            # Read next char to check for , or =
            delim = _next_chunk(fp)
            # Check if it's a comma
            if delim == ',':
                # Found an arg
                args.append(to_val(chunk))
                continue
            elif delim == ")":
                # Found an arg and end of func
                args.append(to_val(chunk))
                self._terminated = True
                return
            elif delim != '=':
                # Should be one of these three
                raise ValueError(
                    f"After '{chunk}' in function {funcname}(), " +
                    f"expected ',', '=', or ')'; got '{delim}'")
            # Otherwise we should have a "keyword"
            assert_regex(chunk, RE_WORD)
            # Got name of next value
            name = chunk
            # Now get the next value (can be recursive)
            val = _next_value(fp, name)
            # Save
            kwargs[name] = val
            # Read next chunk
            chunk = _next_chunk(fp)
            # Test it
            if chunk == ',':
                # Move on to next section
                continue
            elif chunk == ')':
                # End of subsection
                self._terminated = True
                return
            # Run-on
            raise ValueError(
                f"After kwarg '{name}' in function {funcname}(), " +
                f"expected ',' or ')'; got '{chunk}'")


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


# Convert Python value to text
def to_text(val: object) -> str:
    r"""Convert appropriate Python value to ``.vars`` file text

    :Call:
        >>> txt = to_text(val)
    :Inputs:
        *val*: :class:`object`
            One of several appropriate values for ``.vars`` files
    :Outputs:
        *txt*: :class:`str`
            Converted text
    :Versions:
        * 2024-02-24 ``@ddalle``: v1.0
    """
    # Check type
    if isinstance(val, list):
        # Convert each element of a list
        txts = [to_text(valj) for valj in val]
        # Join them
        return '[' + ', '.join(txts) + ']'
    elif isinstance(val, dict):
        # If it's a dict, check if it's a "function"
        if "@function" in val:
            # Write a "function"
            txt = val["@function"] + "("
            # Get args and kwargs
            args = val.get("args", [])
            kwargs = val.get("kwargs", {})
            # Convert args
            argtxts = [to_text(aj) for aj in args]
            # Add args
            txt += ", ".join(argtxts)
            # Check for kwargs
            if len(args) and len(kwargs):
                # Add another comma to separate args and kwargs
                txt += ", "
            # Convert kwargs to text
            kwargtxts = [f"{k}={to_text(v)}" for k, v in kwargs.items()]
            # Add the kwargs to text
            txt += ", ".join(kwargtxts)
            # Close the function
            return txt + ")"
        else:
            # Loop through values of "subsection" <angle brackets>
            lines = [f"    {k}: {to_text(v)}" for k, v in val.items()]
            # Combine lines
            return '<\n' + '\n'.join(lines) + '\n>\n'
    # Otherwise convert to string directly (no quotes on strings)
    return str(val)


# Convert text to Python value
def to_val(txt: str):
    r"""Convert ``.vars`` file text to a Python value

    This only applies to single entries and does not parse lists, etc.

    :Call:
        >>> v = to_val(txt)
    :Inputs:
        *txt*: :class:`str`
            Any valid text for a single value in a ``.vars`` file
    :Outputs:
        *v*: :class:`str` | :class:`int` | :class:`float`
            Interpreted value
    :Versions:
        * 2024-02-23 ``@ddalle``: v1.0
    """
    # Check if it could be an integer
    if RE_INT.fullmatch(txt):
        # Convert to an integer
        return int(txt)
    elif RE_FLOAT.fullmatch(txt):
        # Convert full value to a float
        return float(txt)
    # Split into parts to test for float w/ units
    parts = txt.split()
    # Test for units
    if len(parts) == 2 and RE_FLOAT.fullmatch(parts[0]):
        # Return in MKS units
        return float(parts[0]) * mks(parts[1])
    else:
        # Otherwise interpret as a string (no quotes)
        return txt


# Find a function
def _find_function(
        data: dict,
        name: str,
        comp: Optional[str] = None,
        nmax: Optional[int] = None,
        prefix: Optional[str] = None) -> dict:
    r"""Find a function by name from a :class:`dict`

    :Call:
        >>> funcs = _find_function(data, name, comp=None, nmax=None)
    :Inputs:
        *data*: :class:`dict`
            Any ``.vars`` file data or subsection
        *name*: :class:`str`
            Name of function to search for
        *comp*: {``None``} | :class:`str`
            Optional key name for the function to search for
        *nmax*: {``None``} | :class:`int`
            Optional maximum number of times to find *name* function
        *prefix*: {``None``} | ;class:`str`
            Optional prefix to use for location in output
    :Outputs:
        *funcs*: :class:`dict`\ [:class:`dict`]
            Each instance of *name* function found; key of each item
            says where the function was located w/i *data*
    """
    # Initialize counter and output
    n = 0
    funcs = {}
    # Process prefix
    pre = "" if prefix is None else f"{prefix}."
    # Loop
    for k, v in data.items():
        # Check type
        if not isinstance(v, dict):
            continue
        # Check for function
        if v.get("@function"):
            # Check name
            if v["@function"] != name:
                continue
            # Check component (key)
            if (comp is not None) and (comp != k):
                continue
            # Found one
            funcs[pre + k] = v
            n += 1
        else:
            # Otherwise recurse through subsection
            nmaxj = None if nmax is None else nmax - n
            subfuncs = _find_function(v, name, nmaxj, pre + k)
            # Add any finds
            funcs.update(subfuncs)
            # Update counter
            n += len(subfuncs)
        # Check for exit caused by user limit
        if (nmax is not None) and (n >= nmax):
            break
    # Output
    return funcs


# Read the next "value", which can be nested
def _next_value(fp: IOBase, opt: str):
    r"""Read the next 'value' from a ``.vars`` file; can be recursive

    :Call:
        >>> val = _next_value(fp, opt)
    :Inputs:
        *fp*: :class:`IOBase`
            File open for reading
        *opt*: :class:`str`
            Name of option being read; for error messages
        *role*: :class:`int`
            Role of current value:

            * ``0``: top-level option
            * ``1``: subsection, e.g. <p1=1, p2=air>
            * ``2``: list, e.g. [0, 0, 1.2]
            * ``3``: function, e.g. viscousWall(Twall=300)
    """
    # Read next chunk
    chunk = _next_chunk(fp)
    # Check for empty
    if chunk == '':
        raise ValueError(f"Right-hand side of option '{opt}' is empty")
    # Check for special cases
    if chunk == "<":
        # Go back one char to get beginning of section
        fp.seek(fp.tell() - 1)
        # Read subsection
        return VFileSubsec(fp)
    elif chunk == "[":
        # Go back one char to get beginning of list
        fp.seek(fp.tell() - 1)
        # Read list
        return VFileList(fp)
    # Now check for a "function"
    if RE_WORD.fullmatch(chunk):
        # Read next char
        c = _next_char(fp)
        # Return to original position
        fp.seek(fp.tell() - 1)
        # If it's a '(', we have a function
        if c == "(":
            # Yep it's a function
            return VFileFunction(fp, chunk)
    # Otherwise return raw avlue
    return to_val(chunk)


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
def _next_char(fp: IOBase, newline=True) -> str:
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


# Get magnitude from scalar, list, or polar()
def get_magnitude(v) -> float:
    r"""Get magnitude of a scalar, polar() function, or vector

    :Call:
        >>> a = get_magnitude(v)
    :Inputs:
        *v*: :class:`float` | :class:`list` | :class:`dict`
            Suitable Loci/CHEM "vector"
    :Outputs:
        *a*: :class:`float`
            Magnitude, first arg of polar() function
    :Versions:
        * 2024-02-26 ``@ddalle``: v1.0
    """
    # Check type
    if isinstance(v, dict):
        # Get "polar" function
        if v.get("@function", "").lower() != "polar":
            raise ValueError(
                "Cannot get magnitude of function not named 'polar()'")
        # The magnitude should be the first "arg"
        return v.get("args", [])[0]
    elif isinstance(v, list):
        # Initialize
        a2 = 0.0
        # Loop through entries
        for vj in v:
            a2 += vj*vj
        # Return magnitude
        return np.sqrt(a2)
    else:
        # Assume scalar
        return v


def set_func_arg(
        func: Optional[dict],
        funcname: str,
        j: int,
        v: Any,
        args: Optional[list] = None) -> dict:
    r"""Set an argument value by position

    :Call:
        >>> newfunc = set_func_arg(func, funcname, j, v, args=None)
    :Inputs:
        *func*: ``None`` | :class:`dict`
            Original "function" specification
        *funcname*: :class:`str`
            Name of function
        *j*: :class:`int`
            Position of argument to set
        *v*: :class:`object`
            Value of object to set
        *args*: {``None``} | :class:`list`
            Default list of function args if not present
    :Outputs:
        *newfunc*: :class:`dict`
            Modified *func*, in-place unless *func* is ``None``
    """
    # Check for empty
    func = {} if func is None else func
    # Set parameters
    func["@function"] = funcname
    # Initialize keyword args
    func.setdefault("kwargs", {})
    # Initialize args if necessary
    args = args if args is not None else [0.0] * j
    args = func.setdefault("args", args)
    # Set the value
    args[j] = v
    # Output
    return func


def set_polar_arg(
        func: Optional[dict],
        j: int,
        v: Any,
        args: Optional[list] = None) -> dict:
    r"""Set an argument value by position

    :Call:
        >>> newfunc = set_polar_arg(func, j, v, args=None)
    :Inputs:
        *func*: ``None`` | :class:`dict`
            Original "function" specification
        *j*: :class:`int`
            Position of argument to set
        *v*: :class:`object`
            Value of object to set
        *args*: {``None``} | :class:`list`
            Default list of function args if not present
    :Outputs:
        *newfunc*: :class:`dict`
            Modified *func*, in-place unless *func* is ``None``
    """
    return set_func_arg(func, "polar", j, v, args)
