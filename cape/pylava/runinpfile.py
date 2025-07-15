r"""
:mod:`cape.pylava.runinpfile`: Interface to LAVA-Cartesian input files
=======================================================================

This module provides the class :class:`CartInputFile`, which is used to
parse and modify the C-like input file to LAVA-Cartesian
"""

# Standard library
import os
import re
from io import IOBase, StringIO
from typing import Any, Optional

# Third-party

# Local imports
from ..errors import assert_isinstance


# Other constants
SPECIAL_CHARS = "{}[]:=,;"

# Regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.?[0-9]*([EDed][+-]?[0-9]+)?")
RE_INT = re.compile(r"[+-]?[0-9]+")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_ ]*")
RE_MULTIPLE = re.compile(r"(?P<grp>.+)\s*\^\s*(?P<exp>[0-9]+)")


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
        "_section",
    )

   # --- __dunder__ ---
    def __init__(self, fname: Optional[str] = None):
        self.fdir = None
        self.fname = None
        self.tab = "    "
        # Initialize hidden attributes
        self._section = ""
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
        # Open file
        with open(fname, 'r') as fp:
            # Begin reading sections
            while True:
                # Read section, check status
                q = self._read_next(fp)
                # Check for EOF
                if not q:
                    return

    # Read next thing
    def _read_next(self, fp: IOBase) -> int:
        r"""Read next section of LAVA-cart input file

        :Call:
            >>> opts._read_section(fp)
        """
        # Read next chunk
        chunk = _next_chunk(fp)
        # Check for special cases
        if chunk == "":
            # EOF
            return 0
        # This should be an option name
        opt = chunk
        # Read the xection
        self[opt] = _next_value(fp, opt)
        # Read something if reaching this point
        return 1

   # --- Write ---
    # Write to file
    def write(self, fname: Optional[str] = None):
        r"""Write contents to ``.vars`` file

        :Call:
            >>> opts.write(fname=None)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *fname*: {``None``} | :class:`str`
                File name to write
        :Versions:
            * 2025-07 ``@ddalle``: v1.0
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
        # Loop through main values
        for opt, val in self.items():
            # Write name and value, to_text() may recurse
            fp.write(f"{opt} {to_text(val)}")

   # --- Data ---
    def get_opt(self, sec: str, opt: str, vdef=None) -> Any:
        r"""Get value of an option

        You can use ``.`` in *opt* ot access subsections. For example

        .. code-block:: python

            opts.get_opt("cartesian", "output.surface.frequency")

        :Call:
            >>> v = opts.get_opt(sec, opt, vdef=None)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *sec*: :class:`str`
                Name of section
            *opt*: :class:`str`
                Name of option or subsection(s) and option
            *vdef*: {``None``} | :class:`object`
                Default value if *opt* is not set
        :Outputs:
            *val*: :class:`object`
                Value of *opt*
        :Versions:
            * 2025-05-07 ``@ddalle``: v1.0
        """
        # Get section
        secopts = self.get(sec, {})
        # Check for sub-options
        optparts = opt.split(".")
        # Overall section name
        name = sec
        # Loop through parts
        for subsec in optparts[:-1]:
            # Assert dictionary
            assert_isinstance(secopts, dict, f"{name} section")
            # Recurse
            secopts = secopts.get(subsec, {})
            # Combine names
            name = f"{name}.{subsec}"
        # Check type again
        assert_isinstance(secopts, dict, f"{name} section")
        # Output
        return secopts.get(optparts[-1], vdef)

    def set_opt(self, sec: str, opt: str, val: Any):
        r"""Set value of an option

        You can use ``.`` in *opt* ot access subsections. For example

        .. code-block:: python

            opts.set_opt("cartesian", "output.surface.frequency", 50)

        :Call:
            >>> v = opts.set_opt(sec, opt, val)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *sec*: :class:`str`
                Name of section
            *opt*: :class:`str`
                Name of option or subsection(s) and option
            *val*: :class:`object`
                Value of *opt* to set
        :Versions:
            * 2025-05-07 ``@ddalle``: v1.0
        """
        # Create section if necessary
        if sec not in self:
            # Initialize
            self[sec] = CartFileSection(sec)
        # Get section
        secopts = self[sec]
        # Check for sub-options
        optparts = opt.split(".")
        # Overall section name
        name = sec
        # Loop through parts
        for subsec in optparts[:-1]:
            # Assert dictionary
            assert_isinstance(secopts, dict, f"{name} section")
            # Recurse
            secopts = secopts.setdefault(subsec, {})
            # Combine names
            name = f"{name}.{subsec}"
        # Check type again
        assert_isinstance(secopts, dict, f"{name} section")
        # Get existing section
        secopts[optparts[-1]] = val

   # --- Solver control ---
    def get_nstep(self) -> int:
        r"""Get number of time steps to take

        :Call:
            >>> n = opts.get_nstep()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *n*: :class:`int`
                Number of time steps to take
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_opt("time", "number of steps")

    def set_nstep(self, n: int):
        r"""Set number of time steps to take

        :Call:
            >>> opts.set_nstep(n)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *n*: :class:`int`
                Number of time steps to take
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_opt("time", "number of steps", n)

   # --- Reference conditions ---
    def get_alpha(self):
        r"""Get freestream angle of attack

        :Call:
            >>> a = opts.get_alpha()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *a*: :class:`float`
                Angle of attack [deg]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("alpha")

    def get_beta(self) -> float:
        r"""Set freestream sideslip angle

        :Call:
            >>> b = opts.set_beta()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *b*: :class:`float`
                Sideslip angle [deg]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("beta")

    def get_density(self) -> float:
        r"""Get freestream static density

        :Call:
            >>> rho = opts.get_density()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *rho*: :class:`float`
                Static density [kg/m^3]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("density")

    def get_pressure(self) -> float:
        r"""Get freestream static pressure

        :Call:
            >>> p = opts.get_pressure()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *p*: :class:`float`
                Static pressure [Pa]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("pressure")

    def get_refarea(self) -> float:
        r"""Get solution reference area

        :Call:
            >>> aref = opts.set_reflength()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *aref*: :class:`float`
                Reference area [m^2]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("area")

    def get_refdx(self) -> float:
        r"""Set solution reference grid size

        :Call:
            >>> opts.set_refdx(dx)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *ds*: :class:`float`
                Reference grid scale [m]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("dx")

    def get_reflength(self) -> float:
        r"""Get solution reference length

        :Call:
            >>> lref = opts.set_reflength(lref)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *lref*: :class:`float`
                Reference length [m]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("length")

    def get_temperature(self) -> float:
        r"""Get freestream static temperature

        :Call:
            >>> t = opts.get_temperature()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *t*: :class:`float`
                Static temperature [K]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("temperature")

    def get_uref(self) -> float:
        r"""Set solution reference velocity

        :Call:
            >>> uref = opts.get_uref()
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
        :Outputs:
            *uref*: :class:`float`
                Reference velocity [m/s]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        return self.get_refcond("urefphys")

    def get_refcond(self, opt: str) -> Any:
        r"""Get option from ``reference conditions`` section

        :Call:
            >>> v = opts.get_refcond(opt)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *opt*: :class:`str`
                Name of option within section
        :Outputs:
            *v*: :class:`object`
                Value of the option
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        # Fixed section name
        sec = "reference conditions"
        # Use generic method
        return self.get_opt(sec, opt)

    def set_alpha(self, a: float):
        r"""Set freestream angle of attack

        :Call:
            >>> opts.set_alpha(a)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *a*: :class:`float`
                Angle of attack [deg]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("alpha", a)

    def set_beta(self, b: float):
        r"""Set freestream sideslip angle

        :Call:
            >>> opts.set_beta(b)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *b*: :class:`float`
                Sideslip angle [deg]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("beta", b)

    def set_pressure(self, p: float):
        r"""Set freestream static pressure

        :Call:
            >>> opts.set_pressure(p)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *p*: :class:`float`
                Static pressure [Pa]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("pressure", p)

    def set_density(self, r: float):
        r"""Set freestream static density

        :Call:
            >>> opts.set_density(r)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *r*: :class:`float`
                Static density [kg/m^3]
        :Versions:
            * 2025-07-15 ``@ddalle``: v1.0
        """
        self.set_refcond("density", r)

    def set_mach(self, m: float):
        r"""Set freestream Mach number

        :Call:
            >>> opts.set_mach(m)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *m*: :class:`float`
                Mach number
        :Versions:
            * 2025-07-15 ``@ddalle``: v1.0
        """
        self.set_refcond("mach", m)

    def set_refarea(self, aref: float):
        r"""Set solution reference length

        :Call:
            >>> opts.set_reflength(aref)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *aref*: :class:`float`
                Reference area [m^2]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("area", aref)

    def set_refdx(self, dx: float):
        r"""Set solution reference grid size

        :Call:
            >>> opts.set_refdx(dx)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *ds*: :class:`float`
                Reference grid scale [m]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("dx", dx)

    def set_reflength(self, lref: float):
        r"""Set solution reference length

        :Call:
            >>> opts.set_reflength(lref)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *lref*: :class:`float`
                Reference length [m]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("length", lref)

    def set_temperature(self, t: float):
        r"""Set freestream static temperature

        :Call:
            >>> opts.set_pressure(p)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *t*: :class:`float`
                Static temperature [K]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("temperature", t)

    def set_uref(self, uref: float):
        r"""Set solution reference velocity

        :Call:
            >>> opts.set_uref(uref)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *uref*: :class:`float`
                Reference velocity [m/s]
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        self.set_refcond("urefphys", uref)

    def set_refcond(self, opt: str, v: Any):
        r"""Set option in ``reference conditions`` section

        :Call:
            >>> opts.set_refcond(opt, v)
        :Inputs:
            *opts*: :class:`CartInputFile`
                LAVA-cartesian input file interface
            *opt*: :class:`str`
                Name of option within section
            *v*: :class:`object`
                Value of the option
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        # Section name
        sec = "reference conditions"
        # Generic method
        self.set_opt(sec, opt, v)


class CartFileSection(dict):
    r"""Reader for a "section" of LAVA-Cartesian input files

    This refers to dict-like contents of ``{}`` braces, e.g.

    .. code-block:: none

        {
            mach = 4.5
            alpha = 2.0
        }

    :Call:
        >>> sec = CartFileSection(sec, fp)
    :Inputs:
        *sec*: :class:`str`
            Name of section, used for error messages
        *fp*: :class:`IOBase`
            File open for reading at start of section
    """
    __slots__ = (
        "name",
    )

    def __init__(self, sec: str, fp: Optional[IOBase] = None):
        # Save section name
        self.name = sec
        # Read
        if fp is not None:
            self._read(fp)

    def _read(self, fp: IOBase):
        # Read first chunk
        chunk = _next_chunk(fp)
        # Check it
        assert_nextstr(chunk, "{", f"Start of {self.name} section")
        # Loop until end of section
        while True:
            # Read next option/value pair
            q = self._read_next(fp)
            # Check for read
            if not q:
                return

    def _read_next(self, fp: IOBase) -> bool:
        # Read next chunk
        chunk = _next_chunk(fp)
        # Check for end of section
        if chunk == "":
            # Unexpected EOF ...
            return False
        elif chunk == "}":
            # Proper end of section
            return False
        # Otherwise, option name; must be a word
        assert_regex(chunk, RE_WORD, "left-hand side of LAVA-cartesian option")
        # Save option name
        opt = chunk
        # Read next chunk
        chunk = _next_chunk(fp)
        # Must be either equals sign or new section
        if chunk == "{":
            # Go back on char to pick up start of sec
            fp.seek(fp.tell() - 1)
        else:
            # Must be an equals sign
            assert_nextstr(
                chunk, '=',
                f"text after option '{opt}' in section '{self.name}'")
        # Save value
        self[opt] = _next_value(fp, f"{self.name}.{opt}")
        # Positive result
        return True


class CartFileList(list):
    r"""Reader for a "list" of LAVA-Cartesian input files

    This refers to list-like contents of ``[]`` brackets, e.g.

    .. code-block:: none

        [
            mach,
            cp
        ]

    :Call:
        >>> sec = CartFileList(opt, fp)
    :Inputs:
        *opt*: :class:`str`
            Name of option on left-hand side, used for error messages
        *fp*: :class:`IOBase`
            File open for reading at start of section
    """
    __slots__ = (
        "name",
    )

    def __init__(self, opt: str, fp: Optional[IOBase] = None):
        # Save section name
        self.name = opt
        # Read
        if fp is not None:
            self._read(fp)

    def _read(self, fp: IOBase):
        # Read first chunk
        chunk = _next_chunk(fp)
        # Check it
        assert_nextstr(chunk, "[", f"Start of {self.name} list")
        # Loop until end of section
        while True:
            # Read next option/value pair
            q = self._read_next(fp)
            # Check for read
            if not q:
                return

    def _read_next(self, fp: IOBase) -> bool:
        # Read next chunk
        chunk = _next_chunk(fp)
        # Check for end of section
        if chunk == "":
            # Unexpected EOF ...
            return False
        elif chunk == "]":
            # Proper end of list
            return False
        elif chunk == ",":
            # Continue to next value
            return True
        # Get next value
        v = to_val(chunk)
        # Check for 7^3 notation (returns as tuple)
        if isinstance(v, tuple):
            self.extend(v)
        else:
            # Append value
            self.append(v)
        # Positive result
        return True


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
    breakpoint()
    # Raise an exception
    raise ValueError(msg1 + msg2 + msg3)


# Convert Python value to text
def to_text(val: object, indent: int = 0) -> str:
    r"""Convert appropriate Python value to ``.vars`` file text

    :Call:
        >>> txt = to_text(val, indent=0)
    :Inputs:
        *val*: :class:`object`
            One of several appropriate values for ``.vars`` files
        *indent*: {``0``} | :class:`int`
            Number of levels of indentation
    :Outputs:
        *txt*: :class:`str`
            Converted text
    :Versions:
        * 2025-05-07 ``@ddalle``: v1.0
    """
    # Form tab
    tab = ' ' * indent * 4
    # Check type
    if isinstance(val, list):
        # Convert each element of a list
        txts = [to_text(valj) for valj in val]
        # Join them
        return '[' + ', '.join(txts) + ']'
    elif isinstance(val, dict):
        # Start section
        stream = StringIO()
        stream.write("{\n")
        # Loop through values
        for opt, v in val.items():
            # Print option name
            stream.write(f"{tab}    {opt}")
            # Check value type
            if isinstance(v, dict):
                # Recurse
                stream.write(' ')
                stream.write(to_text(v, indent=indent+1))
            else:
                # Add delimiter
                stream.write(" = ")
                # Recurse
                stream.write(to_text(v, indent=indent))
                stream.write("\n")
        # End section
        stream.write(tab + "}\n\n")
        # Read contents of stream
        stream.seek(0)
        return stream.read()
    elif val is True:
        return "true"
    elif val is False:
        return "false"
    elif val is None:
        return "none"
    # Otherwise convert to string directly (no quotes on strings)
    return f"{val}"


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
        *v*: :class:`object`
            Interpreted value, number, string, boolean, or ``None``
    :Versions:
        * 2025-05-07 ``@ddalle``: v1.0
        * 2025-05-19 ``@ddalle``: v1.1; support ``['a'^3]`` syntax
    """
    # Check if it could be an integer
    if RE_INT.fullmatch(txt):
        # Convert to an integer
        return int(txt)
    elif RE_FLOAT.fullmatch(txt):
        # Convert full value to a float
        return float(txt)
    elif RE_MULTIPLE.fullmatch(txt):
        # Get groups
        mtch = RE_MULTIPLE.fullmatch(txt)
        grp = mtch.group("grp")
        n = mtch.group("exp")
        # Recurse
        v = to_val(grp)
        # Output
        return (v,) * int(n)
    elif txt.lower() == "true":
        # Use literal value
        return True
    elif txt.lower() == "false":
        # Use literal value
        return False
    elif txt.lower() == "none":
        # Use null value
        return None
    # Otherwise use string as-is
    return txt


# Read next value
def _next_value(fp: IOBase, opt: str):
    r"""Read the next 'value' from a LAVA-Cartesian input file; recurse

    :Call:
        >>> val = _next_value(fp, opt)
    :Inputs:
        *fp*: :class:`IOBase`
            File open for reading
        *opt*: :class:`str`
            Name of option being read; for error messages
    :Outputs:
        *val*: :class:`object`, see below
            * :class:`CartFileSection`
            * :class:`CartFileList`
            * :class:`int`
            * :class:`float`
            * :class:`str`
    """
    # Read next chunk
    chunk = _next_chunk(fp)
    # Check for empty
    if chunk == '':
        raise ValueError(f"Right-hand side of option '{opt}' is empty")
    # Store first char
    c = chunk[0]
    # Check for special cases
    if c == "{":
        # Go back one char to get beginning of section
        fp.seek(fp.tell() - 1)
        # Read section
        return CartFileSection(opt, fp)
    elif c == "[":
        # Go back on char to get start of list
        fp.seek(fp.tell() - 1)
        # Read list
        return CartFileList(opt, fp)
    # Convert to numeric type or bool if appropriate
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
        if c in ' \t':
            # Allow white space but normalize
            chunk = f"{chunk.strip()} "
            continue
        elif c in '\r\n':
            # EOL encountered
            return chunk.strip()
        elif c == '/':
            # Read next char to check for comment
            c1 = fp.read(1)
            if c1 == '/':
                # Comment encountered
                fp.readline()
                return chunk.strip()
            elif c1 != '':
                # Single-slash; go back to prev char
                fp.seek(fp.tell() - 1)
        elif c == '#':
            # Comment encountered
            fp.readline()
            return chunk.strip()
        elif chunk.startswith("$") and c in '{}':
            # Allow '${mach}', for example
            pass
        elif c in SPECIAL_CHARS:
            # Revert one character
            fp.seek(fp.tell() - 1)
            # Output
            return chunk.strip()
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
        elif c in ';\n':
            # EOL
            if newline:
                # Return \n as char
                return '\n'
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
