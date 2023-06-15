#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library imports
import re

# Third-party imports
import numpy as np

# Local imports
from .nmlerr import (
    NmlTypeError,
    NmlValueError,
    assert_isinstance,
    assert_nextchar,
    assert_regex)


# Other constants
SPECIAL_CHARS = "&/$=,*(\n"
FORTRAN_TRUE = (".t.", ".true.")
FORTRAN_FALSE = (".f.", ".false.")
SIZE_FROM_RHS = -1

# Types
BOOL_TYPES = (bool, np.bool_)
STR_TYPES = (str, np.str_)
INT_TYPES = (
    int, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64)

# Simple regular expressions
RE_A = re.compile(r'[A-Za-z]')
RE_D = re.compile(r"[0-9DEde.+-]")
RE_N = re.compile(r"[0-9]")
RE_S = re.compile(r'\s')
RE_W = re.compile(r'[A-Za-z0-9_]')
# Characters in an index expression
RE_IND_ALL = re.compile(r'[0-9:, ()]')
# Compound regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.[0-9]*([EDed][+-]?[0-9]+)?")
RE_IND = re.compile(r"\s*(:|[0-9]+\s*:\s*[0-9]+|[0-9]+)\s*([,)])")
RE_INT = re.compile(r"[+-]?[0-9]+")
RE_VEC = re.compile(r"([1-9][0-9]*)\s*\*")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# Class
class NmlFile(dict):
    # Class attributes
    __slots__ = (
        "fdir",
        "fname",
        "section_char",
        "section_order",
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
            raise NmlTypeError(
                "%s() takes 0 to 1 arguments, but %i were given" %
                (type(self).__name__, narg))
        # Check sequential arg type
        if isinstance(a, str):
            # Read namelist
            self.read_nmlfile(a)
        elif isinstance(a, dict):
            # Merge dict
            # self.set_opts(a)
            ...
        elif a is not None:
            # Bad type
            assert_isinstance(
                a, (str, dict),
                "positional input 0 to %s()" % type(self).__name__)
        # Merge in remaining options
        # self.set_opts(kw)

   # --- Read ---
    # Read namelist file
    def read_nmlfile(self, fname: str):
        r"""Read a namelist file

        :Call:
            >>> nml.read_nmlfile(fname)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist index
            *fname*: :class:`str`
                Name of file
        """
        # Open file
        with open(fname, 'r') as fp:
            # Loop until end of file
            while True:
                # Read sections
                n = self._read_nml_section(fp)
                # Exit if no section read
                if n == 0:
                    break

    # Read namelist section
    def _read_nml_section(self, fp) -> int:
        # Read next character
        c = _next_char(fp, newline=False)
        # Check for EOF
        if c == "":
            return 0
        # Check if it's right; if first section, set type for file
        self._check_section_char(c)
        # Initialize section
        data = {}
        # Read name
        sec = _next_chunk(fp)
        # Validate
        assert_regex(sec, RE_WORD)
        # Read next chunk
        chunk = _next_chunk(fp)
        # Must be EOL
        assert_nextchar(chunk, "\n")
        # Save data
        if sec in self:
            # Get current section
            opts = self[sec]
            # Check if it's a list
            if isinstance(opts, list):
                # Append this reading
                opts.append(data)
            else:
                # Convert to list
                self[sec] = [opts, data]
            # Get number of sections
            nsec = len(self[sec])
        else:
            # New section
            self[sec] = data
            # Single section
            nsec = 0
        # Update section ordering
        self.section_order.append((sec, nsec))
        # Loop through data entries
        while True:
            # Read next chunk
            chunk = _next_chunk(fp)
            # Check for end
            if chunk in '$/':
                # Validate end of section
                self._check_section_end(fp, chunk)
                # Go to next section
                break
            # Check following content
            if chunk == '\n':
                # Blank line or line ending w/ comma
                continue
            elif chunk == '':
                # Unterminated section
                msg1 = f" #{nsec + 1}" if nsec else ""
                raise NmlValueError(f"Section {sec}{msg1} is unterminated")
            # Read value name
            name = chunk
            # Validate it
            assert_regex(name, RE_WORD)
            # Read indices if applicable, else '='
            chunk = _next_chunk(fp)
            # Check if it's indices
            if chunk.startswith("("):
                # Parse indices
                nlhs, inds = parse_index_str(chunk)
                # Read next chunk
                chunk = _next_chunk(fp)
            else:
                # Got a scalar
                nlhs = None
            # Validate
            assert_nextchar(chunk, "=")
            # Initialize vector syntax, e.g. the 2 in ``2 * 0.3``
            nrhs = None
            # Initialize final value, may be a list
            vrhs = None
            # Initialize current value
            val = None
            # Flag that value must end next, e.g. can't do ``2*.t. .f.``
            must_end = False
            # Read value(s)
            while True:
                # Read next chunk
                chunk = _next_chunk(fp)
                # Check for end of variable
                if chunk in '\n,' or chunk == '':
                    # End of this value
                    break
                elif must_end:
                    raise NmlValueError(
                        f"Value from '{nrhs} * {val}' " +
                        f"cannot be followed  by '{chunk}'")
                elif chunk == "*":
                    # Got weird F90 vector syntax: 2 * 0.3 -> [0.3, 0.3]
                    assert_isinstance(
                        val, int, "LHS of vector value, 'N * ...'")
                    # Save previous value as *nval*
                    nrhs = val
                    # Read next value
                    chunk = _next_chunk(fp)
                    # Check for hanging vector, like '8 *,'
                    if chunk in ",\n" or chunk == "":
                        raise NmlValueError(
                            f"Unterminated vector '{nrhs} *' for ", +
                            f"'{name} in section '{sec}'")
                    # This value must terminate
                    must_end = True
                # Get value (always returns something)
                val = to_val(chunk)
                # Save vector if appropriate
                if must_end:
                    # Create array
                    vrhs = np.full(nrhs, val)
                    continue
                # Otherwise, got a regular value
                if vrhs is None:
                    # First value (or scalar)
                    vrhs = val
                    nrhs = 1
                else:
                    # Append (works if *vrhs* is scalar or array)
                    vrhs = np.append(vrhs, val)
                    nrhs = vrhs.size
            # Check size if LHS is a `:` slice
            if nlhs == SIZE_FROM_RHS:
                # Copy right-hand side size to left-hand
                nlhs = nrhs
            # Check for scalar left-hand side
            if nlhs is None:
                # Usually a scalar, but allow syntax like...
                #    schedule_cfl = 1.0 10.0
                # which should really be
                #    schedule_cfl(1:2) = 1.0 10.0
                data[name] = vrhs
                continue
            # Ensure *vrhs* is an array for uniform processing
            if not isinstance(vrhs, np.ndarray):
                # Convert to array to get data type if scalar
                vrhs = np.asarray(vrhs)
            # Number of dimensions
            ndim = len(inds)
            # Minimum size required by index for each dimension
            lmin = []
            # Create slices
            slices = []
            # Loop through indices
            for ind in inds:
                # Check which of three index types we got
                if ind is None:
                    # For slices using just ':', size from RHS
                    lmin.append(nrhs)
                    # Convert to '0:5' using size from RHS
                    slices.append(slice(0, nrhs))
                elif isinstance(ind, tuple):
                    # Unpack start, end. like '2:5'
                    ia, ib = ind
                    # Use end for required size
                    lmin.append(ib)
                    # Create slice
                    slices.append(slice(ia - 1, ib))
                else:
                    # Scalar index
                    lmin.append(ind)
                    # Convert to 0-based (Fortran 1-based)
                    slices.append(ind - 1)
            # Get current value (if any)
            v0 = data.get(name)
            # Check for new array (easier case)
            if v0 is None:
                # Allocate with correct data type
                v = np.zeros(tuple(lmin), dtype=vrhs.dtype)
            else:
                # Check existing array has right number of dims
                if v0.ndim != ndim:
                    raise NmlValueError(
                        f"Cannot save {ndim}-D slice to '{name}' in " +
                        f"section '{sec}'; previous value is {v0.ndim}-D")
                # Cumulative max sizes
                lmin1 = tuple(max(l0, l) for l0, l in zip(v0.shape, lmin))
                # Check if this is an expansion
                if v0.shape == lmin1:
                    # Use existing vector
                    v = v0
                else:
                    # Need to append to *v*; find larger dtype
                    dtype = _select_dtype(vrhs, v0)
                    # Initialize new, larger array
                    v = np.zeros(lmin1, dtype=dtype)
                    # Slice to copy *v0*
                    slice0 = tuple(slice(0, lj) for lj in v0.shape)
                    # Save *v0* into *v*
                    v.__setitem__(slice0, v0)
            # Save new data from this line
            v.__setitem__(tuple(slices), vrhs)
            # Save data (unnecessary in some cases)
            data[name] = v
        # Output
        return 1

    # Check section character
    def _check_section_char(self, c: str):
        r"""Check that namelist doesn't switch formats

        For example, you can't have

        .. code-block:: none

            &section1
                myprop = 2,
            /

            $section2
                myname = "joe"
                $end

        For the first section, either character is valid; this function
        will save it in *nml*. If it's the first section and *c* is not
        either ``"&"`` or ``"$"``, an exception will be raised.

        :Call:
            >>> nml._check_section_char(c)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist file interface
            *c*: :class:`str`
                Single character
        :Raises:
            :class:`NmlValueError`
        """
        # First check if we have a section character
        if self.section_char is None:
            # Now it has to be one of two
            if c not in "$&":
                raise NmlValueError(
                    f"Section must start with '$' or '&', not {c}")
            # Save it
            self.section_char = c
        # Otherwise compare it to the standard
        if c != self.section_char:
            raise NmlValueError(
                f"Sections in this file start with '{self.section_char}'" +
                f", not '{c}'")

    # Check end of section
    def _check_section_end(self, fp, c: str):
        r"""Check that namelist section has valid ending

        The *c* should be either ``"/"`` or ``"$"``, depending on the
        namelist format. The following chunk(s) are predetermined based
        on *c*:

        * ``"/"``: ``"\n"``
        * ``"$"``: ``"end"``, ``"\n"``

        :Call:
            >>> nml._check_section_end(c)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist file interface
            *c*: :class:`str`
                Single character
        :Raises:
            :class:`NmlValueError`
        """
        # Filter namelist type
        if self.section_char == '&':
            # Check character
            if c != '/':
                raise NmlValueError(
                    f"Invalid section end, expected '/' but got '{c}'")
        elif self.secion_char == '$':
            # Check character
            if c != '$':
                raise NmlValueError(
                    f"Invalid section end, expected '$end' but got '{c}'")
            # Read next chunk
            chunk = _next_chunk(fp)
            # It should be 'end'
            if chunk.lower() != "end":
                raise NmlValueError(
                    f"Expected '$end' at section end, got '${chunk}'")
        # Read next chunk
        chunk = _next_chunk(fp)
        # Validate
        if not (chunk == '\n' or chunk == ''):
            raise NmlValueError(f"Expected EOL after section, got '{chunk}'")

   # --- Data ---
    # Get one option
    def get_opt(self, sec: str, opt: str, j=None, vdef=None):
        r"""Get value of one variable in one section

        :Call:
            >>> val = nml.set_opt(sec, opt, j=None, vdef=None)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist index
            *sec*: :class:`sec`
                Name of section
            *name*: :class:`str`
                Name of option
            *j*: {``None``} | :class:`int` | :class:`slice` | ``tuple``
                Index or indices of *val* to return ``nml[sec][opt]``
            *vdef*: {``None``} | :class:`object`
                Default value if *opt* not present in ``nml[sec]``
        :Outputs:
            *val*: :class:`object`
                Value to set to ``nml[sec][opt]``
        """
        # Get section
        secnml = self.get(sec, {})
        # Make sure it's a dict
        assert_isinstance(secnml, dict, f"section {sec}")
        # Get value
        v = secnml.get(opt, vdef)
        # Trivial cases
        if v is None or j is None:
            # Not present (v) or return whole value (j is None)
            return
        # If reaching this point, v better be an array
        assert_isinstance(v, np.ndarray, f"section {sec}, option {opt}")
        # Otherwise get indices
        return v.__getitem__(j)

    # Set one option
    def set_opt(self, sec: str, opt: str, val, j=None):
        r"""Set value of one variable in one section

        This can be a partial setting of one entry in a 1-D array.
        Multidimensional arrays cannot be safely set using this
        function.

        :Call:
            >>> nml.set_opt(sec, opt, val, j=None)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist index
            *sec*: :class:`sec`
                Name of section
            *name*: :class:`str`
                Name of option
            *val*: :class:`object`
                Value to set to ``nml[sec][opt]``
            *j*: {``None``} | :class:`int` | :class:`slice`
                Index or indices of where to save *val* in
                ``nml[sec][opt]``
        """
        # Get section
        secnml = self.setdefault(sec, {})
        # Convert list | tuple -> np.ndarray
        if isinstance(val, (list, tuple)):
            val = np.asarray(val)
        # Trivial case where *j* is None
        if j is None:
            secnml[opt] = val
            return
        # If given an index ... get maximum size
        if isinstance(j, INT_TYPES):
            # Given a single index
            jmax = j
        elif isinstance(j, slice):
            # Get upper bound
            jmax = j.stop
            # Don't use syntax like ``opt[:] = val`` or ``opt[3:]``
            if jmax is None:
                raise NmlValueError("Cannot use implicit slice")
        else:
            # Use this function to raise exception
            assert_isinstance(j, (int, slice), "indices *j* to set_opt()")
        # Test if *opt* currently present
        if opt in secnml:
            # Get value
            vcur = secnml[opt]
            # Check if array
            if isinstance(vcur, np.ndarray):
                # Get current size
                ncur = vcur.size
            else:
                # Convert to array of size 1
                vcur = np.array([vcur])
                ncur = 1
            # Current size
            if jmax > ncur:
                # Ensure *val* is an array
                if not isinstance(val, np.ndarray):
                    val = np.asarray(val)
                # Get larger data type
                dtype = _select_dtype(vcur, val)
                # Append
                vnew = np.hstack((vcur, np.zeros(jmax - ncur, dtype=dtype)))
            else:
                # Use current array
                vnew = vcur
            # Save slice
            vnew.__setitem__(j, val)
            # Make sure new slice is saved
            secnml[opt] = vnew

    # Set multiple options
    def apply_dict(self, a: dict):
        r"""Apply multiple settings from a :class:`dict`

        Similar to applying ``nml[sec].update(a[sec])`` for each *sec*
        in *a*.

        :Call:
            >>> nml.apply_dict(a)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist index
            *a*: :class:`dict`\ [:class:`dict`\ [:class:`object`]]
                Two-level dictionary
        """
        # Loop through dictionary
        for k, d in a.items():
            # Get section, or create it
            if k not in self:
                # Append section
                self.section_order.append((k, 0))
            # Get new section
            sec = self.setdefault(k, {})
            # Check for list
            if isinstance(sec, list):
                # For now just get first occurence
                sec = sec[0]
            # Now merge in *d*
            sec.update(d)

   # --- Write ---
    # Write driver
    def write(self, fname: str):
        r"""Write namelist to file

        :Call:
            >>> nml.write(fname)
        :Inputs:
            *nml*: :class:`NmlFile`
                Namelist index
            *fname*: :class:`str`
                Name of file
        """
        # Open file
        with open(fname, 'w') as fp:
            # Loop through sections
            for secname, nsec in self.section_order:
                # Get value
                secval = self[secname]
                # Get subset as approrpriate
                if isinstance(secval, list):
                    # Multiple occurences of *secname*
                    sec = secval[nsec]
                else:
                    # Single occurence
                    sec = secval
                # Single copy of section
                self.write_sec(fp, secname, sec)

    # Write one section
    def write_sec(self, fp, secname: str, sec: dict):
        # Write section header
        fp.write(f" {self.section_char}{secname}\n")
        # Loop through variables
        for name, val in sec.items():
            # Write each value (may take multiple lines)
            self.write_var(fp, name, val)
        # Finish section
        if self.section_char == "&":
            # Section ends with `/`
            fp.write(" /\n\n")
        else:
            # Section ends with `$end` or `$end`
            if secname == secname.lower():
                # Use lower case
                word = "end"
            else:
                # Use upper case
                word = "END"
            # Write ending
            fp.write(f" {self.tab}${word}\n\n")

    # Write a single variable to file
    def write_var(self, fp, name: str, val: object):
        # Convert list|tuple -> array
        if isinstance(val, (list, tuple)):
            val = np.asarray(val)
        # Handle scalars first
        if not isinstance(val, np.ndarray) or val.ndim == 0:
            # Indent first, the wirte name and assignment
            fp.write(f" {self.tab}{name} = ")
            # Convert scalar to value
            txt = to_text(val)
            # Write that
            fp.write(txt)
            fp.write("\n")
            return
        # Get slices
        if val.ndim == 1:
            # Single slice
            slice_inds = np.zeros((0, 1), dtype=np.int64)
        else:
            # Create slices; only using : in first index
            dim_sizes = [np.arange(dim) for dim in val.shape[1:]]
            # Turn the sizes into slices
            dim_inds = np.meshgrid(*dim_sizes)
            # Flatten
            dim_inds = tuple(dim_ind.flatten() for dim_ind in dim_inds)
            # 2D array so that each column is the indices of axes 1-n
            slice_inds = np.stack(dim_inds, axis=0)
        # Loop through slices
        for j in range(slice_inds.shape[1]):
            # Get scalar indices for dimensions 1+
            indsj = slice_inds[:, j]
            # Create slice
            slicej = (slice(None),) + tuple(indsj)
            # Get this slice, same as `val[:, 0, 1]` or `val[:]`
            vj = val.__getitem__(slicej)
            # Write it
            self._write_vec(fp, name, indsj, vj)

    # Write 1D array, with other indices
    def _write_vec(self, fp, name: str, inds, vj):
        # Initialize index
        i = 0
        # Size of array
        n = vj.size
        # Loop until end reached
        while i + 1 <= n:
            # Get next entry
            vi = vj[i]
            # Init ending index for constant-value slice
            ib = i
            # Check if the next entry(s) are the same
            for i1 in range(i + 1, n):
                # Check if equal
                if vj[i1] == vi:
                    # Same value; update slice
                    ib = i1
                else:
                    # Different value; break off search
                    break
            # Write name and indices
            self._write_name_slice(fp, name, i, ib, inds)
            # Get vector size
            ni = ib - i + 1
            # Write multiplier if > 1 (e.g. "8 * .true.")
            if ni > 1:
                fp.write(f"{ni} * ")
            # Write the value
            fp.write(to_text(vi))
            fp.write("\n")
            # Update index
            i = ib + 1

    # Write index specifications
    def _write_name_slice(self, fp, name: str, ia, ib, inds):
        # Write tab, name, and opening paren
        fp.write(f" {self.tab}{name}(")
        # Check if *i* is a range
        if ib is None or ia == ib:
            # Scalar value; just write *ia* but convert -> 1-based
            fp.write('%i' % (ia + 1))
        else:
            # Range, 1-based (0:4 -> 1:4)
            fp.write(f"{ia + 1}:{ib + 1}")
        # Write remaining inds
        for ind in inds:
            fp.write(f",{ind + 1}")
        # Close index parentheses and write equals sign
        fp.write(") = ")


# Read the next chunk
def _next_chunk(fp) -> str:
    r"""Read the next "chunk", word, number, or special character

    Examples of chunks are:
        * Special single characters: ``&$/=,\n``
        * Escaped strings, e.g. ``"my, string"``
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
    c = _next_char(fp)
    # Special situations
    if c in '"\'':
        # Escaped string
        return _read_str(fp, c)
    elif c == "(":
        # Read indices
        return _read_indices(fp)
    # Some single characters are automatically chunks
    if c in SPECIAL_CHARS:
        return c
    # Initialize
    chunk = c
    # Loop until white space encounterd
    while True:
        # Next character
        c = fp.read(1)
        # Check for white space
        if c in ' \r\t':
            # White space encountered
            return chunk
        elif c == '':
            # EOF
            return chunk
        elif c in SPECIAL_CHARS:
            # Revert one character
            fp.seek(fp.tell() - 1)
            # Output
            return chunk
        # Otherwise, append to chunk
        chunk = chunk + c


# Read the next non-space character
def _next_char(fp, newline=True) -> str:
    r"""Read the next character of a namelist file

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
        elif c == '!':
            # Comment, discard line
            fp.readline()
            # Return blank line
            return '\n'
        # Something else... return it
        return c


# Read indices
def _read_indices(fp) -> str:
    r"""Read indices on left-hand side, if any

    :Call:
        >>> chunk = _read_indices(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle open for reading
    :Outputs:
        *chunk*: :class:`str`
            Characters up to closing parenthesis
    :Raises:
        :class:`NmlValueError` if bad character or unterminated index
    :Versions:
        * 2023-06-07 ``@ddalle``: v1.0
    """
    # Initialize output
    c = "("
    chunk = c
    # Loop until parentheses closed
    while True:
        # Read next character
        c = fp.read(1)
        # Check value
        assert_regex(c, RE_IND_ALL, "index or slice")
        # Check for end
        if c == "\n" or c == "":
            # Unterminated
            raise NmlValueError(
                f"Encountered bad index in line ending with '{chunk}'")
        # Append character
        chunk += c
        # Check for end of indices
        if c == ")":
            return chunk


# Read a string
def _read_str(fp, start: str) -> str:
    # Initialize string for output
    txt = ''
    # Initialize escape flag
    escape = False
    # Read until string terminated
    while True:
        # Next character (including white space)
        c = fp.read(1)
        # Check for escapes
        if c == '\\':
            # Save char
            txt += c
            # Set escape flag
            escape = True
        elif escape:
            # Read next character no matter what
            txt += c
            # Unset escape flag
            escape = False
        elif c == start:
            # End of string
            return start + txt + c
        elif c == '\n' or c == '':
            raise NmlValueError(
                f"Unterminated string starting with f{start}{txt}")
        # Normal char
        txt += c


# Get larger data type from two arrays
def _select_dtype(v1: np.ndarray, v2: np.ndarray):
    r"""Select the larger data type from two arrays

    This is mostly useful for string arrays, It will select tye dtype
    with more characters given two arrays.

    :Call:
        >>> dtype = _select_dtype(v1, v2)
    :Inputs:
        *v1*: :class:`np.ndarray`
            An array
        *v2*: :class:`np.ndarray`
            A second array
    """
    # Check which is larger
    if v1.dtype.itemsize >= v2.dtype.itemsize:
        # Use new data type from RHS
        return v1.dtype
    else:
        # Existing array has more bytes
        return v2.dtype


# Read indices
def parse_index_str(txt: str):
    r"""Parse index string

    :Call:
        >>> nval, inds = _read_indices(txt)
    :Inputs:
        *txt*: :class:`str`
            Text of an LHS index, e.g. ``"(:, 2:5, 3)"``
    :Outputs:
        *nval*: :class:`int`
            Size of slice, ``-1`` for slices like ``:``
        *inds*: :class:`list`\ [*ind*]
            List of indices read, ``[]`` if no indices used
        *ind*: ``None`` | :class:`int` | :class:`tuple`
            Individual index, ``:`` -> ``None``; ``3:6`` -> ``(3, 6)``
    :Versions:
        * 2023-06-08 ``@ddalle``: v1.0
    """
    # Get first character
    c = txt[0]
    # Index
    j = 1
    # Make sure it's an opening parenthesis
    assert_nextchar(c, "(")
    # Total Length of string
    n = len(txt)
    # Initialize matches
    inds = []
    # Read remainder of indices
    while j + 1 < n:
        # Match next index
        match = RE_IND.match(txt[j:])
        # Must be a valid match
        if match is None:
            raise NmlValueError(f"Bad index in '{txt}' at column {j + 1}")
        # Unpack two groups
        txtj, delim = match.groups()
        # Update position w/ total length of index
        nj = len(match.group())
        j += nj
        # Parse text of this index
        if txtj == ":":
            # Simple slice, size determined by RHS
            inds.append(None)
        elif RE_INT.fullmatch(txtj):
            # Single index
            inds.append(int(txtj))
        else:
            # Split two sides
            txta, txtb = txtj.split(":", 1)
            # Save integers
            inds.append((int(txta), int(txtb)))
        # Check for end of index
        if delim == ")":
            break
    # Get size of vector from indices
    nval = None
    # Validate each index
    for j, ind in enumerate(inds):
        # Check for single index first
        if isinstance(ind, int):
            # Confirm positive
            if ind <= 0:
                raise NmlValueError(
                    f"Invalid index '{ind}' in position " +
                    f"{j} of {txt}; must be positive")
            # Single index, continue
            continue
        # If not scalar, check for 2D slice
        if nval is not None:
            raise NmlValueError(
                f"In index {txt}; got multiple range dimensions")
        # Check which kind of index we got
        if ind is None:
            # Size determined from rhs
            nval = SIZE_FROM_RHS
        else:
            # Unpack tuple
            ia, ib = ind
            # Confirm positive
            for i in ind:
                if i <= 0:
                    raise NmlValueError(
                        f"Invalid index '{i}' in position " +
                        f"{j} of {txt}; must be positive")
            # Size from slice, e.g. 4:8 -> 5
            nval = ib - ia + 1
    # Convert None -> 1 for scalar assignments
    nval = 1 if nval is None else nval
    # Output
    return nval, inds


# Read value
def to_val(txt: str):
    # Check for a string
    if txt.startswith("'") and txt.endswith("'"):
        # Normal 'string'; remove quotes
        return txt[1:-1]
    elif txt.startswith('"') and txt.endswith('"'):
        # Normal "string"; remove quotes
        return txt[1:-1]
    # Filter lower-case text for simplicity
    txtl = txt.strip().lower()
    # Check for booleans
    if txtl in FORTRAN_TRUE:
        # Boolean True
        return True
    elif txtl in FORTRAN_FALSE:
        # Boolean False
        return False
    elif RE_INT.fullmatch(txt):
        # Valid integer
        return int(txtl)
    elif RE_FLOAT.fullmatch(txt):
        # Valid float, 132.44, +0.33D-10, 1.44e+002, etc.
        return float(txtl.replace("d", "e"))
    else:
        # Invalid entry; return as raw text
        return txt


# Convert value to text
def to_text(val: object):
    # Check for a float
    if isinstance(val, STR_TYPES):
        # Escape the string appropriately
        return repr(val)
    elif isinstance(val, BOOL_TYPES):
        # Write True or False
        if val:
            return ".true."
        else:
            return ".false."
    else:
        # Convert everything else directly
        return str(val)
