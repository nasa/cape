r"""
:mod:`capefile`: Reader for CAPE-specific binary files
=======================================================

"""

# Standard library
import os
import re
from functools import wraps
from io import IOBase
from typing import Optional, Union

# Third-party
import numpy as np
from numpy import ndarray, uint32, uint64


# Common header
HEADER_BYTES = b"#!CAPEDB"

# Bit shifts for record type flags
RT_SHIFT_ELEMTYPE = uint32(0)
RT_SHIFT_ELEMBITS = uint32(8)
RT_SHIFT_ARRAYFLG = uint32(12)
RT_SHIFT_ICOMPLEX = uint32(14)
RT_SHIFT_HASTITLE = uint32(16)
RT_SHIFT_XLONGREC = uint32(17)
RT_SHIFT_BYTEORDR = uint32(18)
RT_SHIFT_RESERVED = uint32(20)
RT_SHIFT_USERMODS = uint32(24)
# Number of bits for record type flags
RT_BITS_ELEMTYPE = uint32(0xff)
RT_BITS_ELEMBITS = uint32(0xf)
RT_BITS_ARRAYFLG = uint32(3)
RT_BITS_ICOMPLEX = uint32(3)
RT_BITS_HASTITLE = uint32(1)
RT_BITS_XLONGREC = uint32(1)
RT_BITS_BYTEORDR = uint32(3)
RT_BITS_RESERVED = uint32(0xf)
RT_BITS_USERMODS = uint32(0xff)
# Record type flags
RT_ELEMTYPE = RT_BITS_ELEMTYPE << RT_SHIFT_ELEMTYPE
RT_ELEMBITS = RT_BITS_ELEMBITS << RT_SHIFT_ELEMBITS
RT_ARRAYFLG = RT_BITS_ARRAYFLG << RT_SHIFT_ARRAYFLG
RT_ICOMPLEX = RT_BITS_ICOMPLEX << RT_SHIFT_ICOMPLEX
RT_HASTITLE = RT_BITS_HASTITLE << RT_SHIFT_HASTITLE
RT_XLONGREC = RT_BITS_XLONGREC << RT_SHIFT_XLONGREC
RT_BYTEORDR = RT_BITS_BYTEORDR << RT_SHIFT_BYTEORDR
RT_RESERVED = RT_BITS_RESERVED << RT_SHIFT_RESERVED
RT_USERMODS = RT_BITS_USERMODS << RT_SHIFT_USERMODS

# Basic element types
ELEMTYPE_NONE = uint32(0)
ELEMTYPE_UINT = uint32(1)
ELEMTYPE_INT = uint32(2)
ELEMTYPE_FLOAT = uint32(3)
ELEMTYPE_COMPLEX = uint32(4)
ELEMTYPE_BOOL = uint32(5)
ELEMTYPE_STR = uint32(8)
ELEMTYPE_BYTES = uint32(9)
ELEMTYPE_LIST = uint32(0x10)
ELEMTYPE_TUPLE = uint32(0x20)
ELEMTYPE_SET = uint32(0x30)
ELEMTYPE_DICT = uint32(0x40)
ELEMTYPE_FLAG_NESTED = uint32(0xf0)
ELEMTYPE_FLAG_STR = uint32(8)
# Commont element size flags
ELEMSIZE_2 = uint32(0) << RT_SHIFT_ELEMBITS
ELEMSIZE_8 = uint32(3) << RT_SHIFT_ELEMBITS
ELEMSIZE_16 = uint32(4) << RT_SHIFT_ELEMBITS
ELEMSIZE_32 = uint32(5) << RT_SHIFT_ELEMBITS
ELEMSIZE_64 = uint32(6) << RT_SHIFT_ELEMBITS
ELEMSIZE_128 = uint32(7) << RT_SHIFT_ELEMBITS
ELEMSIZE_256 = uint32(8) << RT_SHIFT_ELEMBITS
# Data type bits dictionary
DTYPE_DICT = {
    "bool": ELEMTYPE_INT | ELEMSIZE_2,
    "bool_": ELEMTYPE_INT | ELEMSIZE_2,
    "int": ELEMTYPE_INT | ELEMSIZE_64,
    "int8": ELEMTYPE_INT | ELEMSIZE_8,
    "int16": ELEMTYPE_INT | ELEMSIZE_16,
    "int32": ELEMTYPE_INT | ELEMSIZE_32,
    "int64": ELEMTYPE_INT | ELEMSIZE_64,
    "uint8": ELEMTYPE_UINT | ELEMSIZE_8,
    "uint16": ELEMTYPE_UINT | ELEMSIZE_16,
    "uint32": ELEMTYPE_UINT | ELEMSIZE_32,
    "uint64": ELEMTYPE_UINT | ELEMSIZE_64,
    "float": ELEMTYPE_FLOAT | ELEMSIZE_64,
    "float16": ELEMTYPE_FLOAT | ELEMSIZE_16,
    "float32": ELEMTYPE_FLOAT | ELEMSIZE_32,
    "float64": ELEMTYPE_FLOAT | ELEMSIZE_64,
    "complex": ELEMTYPE_COMPLEX | ELEMSIZE_128,
    "complex64": ELEMTYPE_COMPLEX | ELEMSIZE_64,
    "complex128": ELEMTYPE_COMPLEX | ELEMSIZE_128,
    "str": ELEMTYPE_STR,
    "str_": ELEMTYPE_STR,
    "bytes": ELEMTYPE_BYTES,
    "bytes_": ELEMTYPE_BYTES,
}
# Data types not available on all systems
if hasattr(np, "float128"):
    DTYPE_DICT[np.float128] = ELEMTYPE_FLOAT | ELEMSIZE_128
if hasattr(np, "complex256"):
    DTYPE_DICT[np.complex256] = ELEMTYPE_COMPLEX | ELEMSIZE_256

# Numpy character dtype codes
DTYPE_CHARS = {
    ELEMTYPE_UINT: "u",
    ELEMTYPE_INT: "i",
    ELEMTYPE_FLOAT: "f",
    ELEMTYPE_COMPLEX: "c",
    ELEMTYPE_STR: "S",
    ELEMTYPE_BYTES: "S",
}

# Flags for special cases of array dimension
ARRAY_NDIM_DEF = uint32(3) << RT_SHIFT_ARRAYFLG
ARRAY_NDIM_DICT = {
    1: uint32(1) << RT_SHIFT_ARRAYFLG,
    2: uint32(2) << RT_SHIFT_ARRAYFLG,
}

# Max UINT32 value
UINT32MAX = uint64(2**32 - 1)

# Defaults
DEFAULT_ENCODING = "utf-8"

# Patterns
REGEX_NONAME = re.compile("_record[0-9]+")

# STDOUT flags
_MAXLEN_STR = 24


# Error class
class CapeFileError(BaseException):
    r"""Base for all exceptions raised by :mod:`capefile`"""
    pass


# Type error
class CapeFileTypeError(TypeError, CapeFileError):
    r"""Exception class for incorrect type in :mod:`capefile`"""
    pass


# Value error
class CapeFileValueError(ValueError, CapeFileError):
    r"""Exception class for incorrect value in :mod:`capefile`"""
    pass


# Class to contain/read/write data from this file type
class CapeFile(dict):
    r"""Interface to simple binary format specific to CAPE

    :Call:
        >>> cdb = CapeFile(fname=None, meta=False)
        >>> cdb = CapeFile(fp, meta=False)
        >>> cdb = CapeFile(data, meta=False)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *fp*: :class:`io.IOBase`
            File handle open in ``rb`` mode
        *meta*: ``True`` | {``False``}
            Option to only read variable names and sizes
    :Outputs:
        *cdb*: :class:`CapeFile`
            Data from CAPE binary file
        *cdb.filename*: ``None`` | :class:`str`
            Name of file data was read from
        *cdb.filedir*: ``None`` | :class:`str`
            Absolute path to folder with file data was read from
        *cdb.cols*: :class:`list`\ [:class:`str`]
            Ordered list of columns
    """
    __slots__ = (
        "filename",
        "filedir",
        "cols",
        "fp",
        "pos",
    )

    # Initialization
    def __init__(
            self,
            a: Optional[Union[str, dict, IOBase]] = None,
            meta: bool = False):
        # Initialize slots
        self.filename = None
        self.filedir = None
        self.cols = []
        self.pos = {}
        self.fp = None
        # Check for empty
        if a is None:
            # No values to read
            return
        elif isinstance(a, dict):
            # Save quantities from a dict/database
            self.save_dict(a)
            return
        # Read file (or handle)
        self.read(a, meta)

    # Read a whole file
    def read(self, fname: Union[str, IOBase], meta: bool = True):
        r"""Read data from a ``.cdb`` file

        :Call:
            >>> db.read(fname, meta=True)
            >>> db.read(fp, meta=True)
        :Inputs:
            *db*: :class:`CapeFile`
                Instance of ``.cdb`` file interface
            *fname*: :class:`str`
                Name of file to read
            *fp*: :class:`io.IOBase`
                File handle open in ``'rb'`` mode
            *meta*: {``True``} | ``False``
                Option to only read column names and sizes
        """
        # Get file handle
        self.fp = _get_fp(fname, 'rb')
        # Read records from it
        self._read(self.fp, meta)

    # Read file from handle
    def _read(self, fp: IOBase, meta: bool):
        # Save file name
        if hasattr(fp, "name"):
            # Absolute file name
            fabs = os.path.abspath(fp.name)
            # Save parts
            self.filedir, self.filename = os.path.split(fabs)
        # Read and check header
        check_header(fp)
        # Read number of records
        n = read_nrecord(fp)
        # Loop through records
        for j in range(n):
            # Save position
            posj = fp.tell()
            # Read record ane name
            namej, vj = read_record(fp, meta)
            # Expand name if not given one
            colj = self._get_colname(namej, j)
            # Save col
            self.save_col(colj, vj, j)
            # Save positoin
            self.pos[colj] = posj

    # Write entire data set to file
    def write(self, fname: Union[str, IOBase]):
        r"""Write data to a ``.cdb`` file

        :Call:
            >>> db.write(fname)
            >>> db.write(fp)
        :Inputs:
            *db*: :class:`CapeFile`
                Instance of ``.cdb`` file interface
            *fname*: :class:`str`
                Name of file to write
            *fp*: :class:`io.IOBase`
                File handle open in ``'wb'`` mode
        """
        # Get file handle
        with _get_fp(fname, 'wb') as fp:
            # Write to it
            self._write(fp)

    def _write(self, fp: IOBase):
        # Write common header
        fp.write(HEADER_BYTES)
        # Get number of records
        n = len(self.cols)
        # Write that first
        uint64(n).tofile(fp)
        # Loop through records
        for col in self.cols:
            # Assume the name is a title
            name = col
            # Check if this counts as a name
            if REGEX_NONAME.fullmatch(col):
                name = None
            # Write it
            write_record(fp, self[col], name)

    def save_dict(self, a: dict):
        r"""Save all parameters of a :class:`dict`

        :Call:
            >>> db.save_dict(a)
        :Inputs:
            *db*: :class:`CapeFile`
                Instance of ``.cdb`` file interface
            *a*: :class:`dict`
                Any dictionary of serializable keys
        """
        for col, v in a.items():
            self.save_col(col, v)

    def save_col(self, col: Optional[str], v, j=None):
        r"""Save value for one column

        :Call:
            >>> db.save_col(col, v, j=None)
        :Inputs:
            *db*: :class:`CapeFile`
                Instance of ``.cdb`` file interface
            *col*: :class:`str`
                Name of column/field to save
            *v*: :class:`Any`
                Value to save
            *j*: {``None``} | :class:`int`
                Index of which instance of *col* to save
        """
        # Expand name if not given one
        col = self._get_colname(col, j)
        # Check for duplicate
        if (col in self) and (j is not None):
            raise CapeFileValueError(
                f"duplicate record named '{col}' in record {j}")
        # Save the col name
        if col not in self:
            self.cols.append(col)
        # Save the value
        self[col] = v

    def _get_colname(self, col: Optional[str], j: Optional[int]) -> str:
        # Check for nameless record
        if col is None:
            # Create a name
            jname = len(self.cols) if (j is None) else j
            col = f"_record{jname}"
        # Output
        return col


# Class for info on record type
class RecordType(object):
    __slots__ = (
        "rt",
        "element_type",
        "element_bits",
        "record_array",
        "complex_flag",
        "record_title",
        "xlong_record",
        "el_byteorder",
        "unused_flags",
        "usermod_flag",
    )

    def __init__(self, rt: uint32):
        # Save type
        self.rt = rt
        # Read parts
        self.element_type = (rt & RT_ELEMTYPE) >> RT_SHIFT_ELEMTYPE
        self.element_bits = (rt & RT_ELEMBITS) >> RT_SHIFT_ELEMBITS
        self.record_array = (rt & RT_ARRAYFLG) >> RT_SHIFT_ARRAYFLG
        self.complex_flag = (rt & RT_ICOMPLEX) >> RT_SHIFT_ICOMPLEX
        self.record_title = (rt & RT_HASTITLE) >> RT_SHIFT_HASTITLE
        self.xlong_record = (rt & RT_XLONGREC) >> RT_SHIFT_XLONGREC
        self.el_byteorder = (rt & RT_BYTEORDR) >> RT_SHIFT_BYTEORDR
        self.unused_flags = (rt & RT_RESERVED) >> RT_SHIFT_RESERVED
        self.usermod_flag = (rt & RT_USERMODS) >> RT_SHIFT_USERMODS

    @classmethod
    def from_value(cls, v, name: Optional[str] = None):
        # Analyze record type
        rt = genr8_recordtype(v, name)
        # Initialize
        return cls(rt)


# Write a record
def write_record(fp: IOBase, v, name: Optional[str] = None) -> uint64:
    r"""Write a record to file

    :Call:
        >>> size = write_record(fp, v, name=None)
    :Inputs:
        *fp*: :class:`io.IOBase`
            A file open for writing bytes
        *v*: :class:`object`
            Object to be written to file
        *name*: {``None``} | :class:`str`
            Optional title to save *v* as
    :Outputs:
        *size*: :class:`np.uint64`
            Number of bytes written
    """
    # Get record type
    rtyp = RecordType.from_value(v, name)
    # Write the type flag
    rtyp.rt.tofile(fp)
    # Calculate length
    rs = _genr8_recordsize(rtyp, v, name)
    # Write that
    rs.tofile(fp)
    # Initialize with correct type
    if rtyp.xlong_record:
        # Extra long record
        cls = uint64
        size = uint64(12)
    else:
        # "Normal" short record
        cls = uint32
        size = uint64(8)
    # Add in record size
    size += rs
    # Write title if appropriate
    if rtyp.record_title:
        # Convert to bytes
        name_bytes = name.encode(DEFAULT_ENCODING)
        # Write string size
        uint32(len(name_bytes)).tofile(fp)
        # Write the title
        fp.write(name_bytes)
    # Unpack basic element type
    et = rtyp.element_type
    # Check for special types
    if et == ELEMTYPE_NONE:
        return size
    elif et in (ELEMTYPE_LIST, ELEMTYPE_TUPLE, ELEMTYPE_SET):
        # Write length
        uint32(len(v)).tofile(fp)
        # Write each element
        for vj in v:
            write_record(fp, vj)
        # Done
        return size
    elif et == ELEMTYPE_DICT:
        # Write length
        uint32(len(v)).tofile(fp)
        # Write each key
        for k, vk in v.items():
            write_record(fp, vk, k)
        # Done
        return size
    # Ensure array
    if et == ELEMTYPE_STR:
        # Encode
        va = np.char.encode(v, encoding=DEFAULT_ENCODING)
    else:
        # Convert directly to array
        va = np.asarray(v)
    # Write string (element) size if appropriate
    if et in (ELEMTYPE_STR, ELEMTYPE_BYTES):
        # Get item size
        elemsize = uint32(va.dtype.itemsize)
        # Write it
        elemsize.tofile(fp)
    # Unpack array type
    array_flags = rtyp.record_array
    # Write array dimensions if appropriate
    if array_flags == 1:
        # 1D array
        nj = cls(va.size)
        # Write
        nj.tofile(fp)
    elif array_flags == 2:
        # 2D array
        nj = cls(va.shape[0])
        nk = cls(va.shape[1])
        # Write
        nj.tofile(fp)
        nk.tofile(fp)
    elif array_flags == 3:
        # N-D array (includes 0-D)
        nd = uint32(va.ndim)
        na = np.array(va.shape, dtype=cls)
        # Write both
        nd.tofile(fp)
        na.tofile(fp)
    # Write data
    va.tofile(fp)
    # Return the number of bytes written
    return size


# Get record type code from value
def genr8_recordtype(v, name: Optional[str] = None) -> uint32:
    r"""Get capefile record type code based on Python type

    :Call:
        >>> rt = genr8_recordtype(v, name=None)
    :Inputs:
        *v*: :class:`object`
            Any value to analyze type
        *name*: {``None``} | :class:`str`
            Title for *v*, used to determine "title" bit in *rt*
    :Outputs:
        *rt*: :class:`np.uint32`
            Record type code
    """
    # Initialize return type
    rt = uint32(0 if name is None else RT_HASTITLE)
    # Check basic type
    if isinstance(v, list):
        # List (nested, nameless, mutable)
        return rt | ELEMTYPE_LIST | RT_XLONGREC
    elif isinstance(v, tuple):
        # Tuple (nested, nameless, immutable)
        return rt | ELEMTYPE_TUPLE | RT_XLONGREC
    elif isinstance(v, set):
        # Set (nested, unique, nameless, mutable)
        return rt | ELEMTYPE_SET | RT_XLONGREC
    elif isinstance(v, dict):
        # Dict (nested, named, mutable)
        return rt | ELEMTYPE_DICT | RT_XLONGREC
    elif v is None:
        # None
        return rt | ELEMTYPE_NONE
    # Check for array
    if isinstance(v, ndarray):
        # Get flag for number of dimensions *v* has
        nd = ARRAY_NDIM_DICT.get(v.ndim, ARRAY_NDIM_DEF)
        # Check if very long array requires uint64 size
        xl = uint32(v.size*v.dtype.itemsize > UINT32MAX) << RT_SHIFT_XLONGREC
        # Set array bits
        rt |= nd
        # Set extra-long array bit
        rt |= xl
        # Get an element to test data type
        clsname = v.dtype.name
        # Remove length portion for strings
        if clsname.startswith("str"):
            clsname = "str"
        elif clsname.startswith("bytes"):
            clsname = "bytes"
    else:
        # Use scalar to test data type
        clsname = type(v).__name__
    # Check type
    et = DTYPE_DICT.get(clsname, uint32(0))
    # Set first 1.5 bytes based on element type
    rt |= et
    # Output
    return rt


# Calculate record size
def genr8_recordsize(v, name: Optional[str] = None) -> Union[uint32, uint64]:
    # Calculate record type
    rtyp = RecordType.from_value(v, name)
    # Calculate size
    return _genr8_recordsize(rtyp, v, name)


def _genr8_recordsize(rtyp: RecordType, v, name=None) -> Union[uint32, uint64]:
    # Initialize with correct type
    if rtyp.xlong_record:
        # Extra long record
        cls = uint64
        # Use u8 (int64) for array dimensions, too
        len_itemsize = cls(8)
    else:
        # "Normal" short record
        cls = uint32
        # Use u4 (int32) for array dimensions, too
        len_itemsize = cls(4)
    # Initialize size
    rs = cls(0)
    # Check for a name
    if rtyp.record_title:
        # Initialize with length of name
        rs += genr8_rs_str(name)
    # Unpack element basic type
    et = rtyp.element_type
    # Check for special types
    if et == ELEMTYPE_NONE:
        # NoneType; end of record
        return rs
    elif et in (ELEMTYPE_LIST, ELEMTYPE_TUPLE, ELEMTYPE_SET):
        # Add a spot to report the length
        rs += cls(4)
        # Loop through elements
        for vj in v:
            # Calculate type for element
            rtypj = RecordType.from_value(vj)
            # Account for the type and length of record j
            rs += cls(8 + 4*rtypj.xlong_record)
            # Add to total record size
            rs += _genr8_recordsize(rtypj, vj)
        # That's it
        return rs
    elif et == ELEMTYPE_DICT:
        # Add a spot to report the length
        rs += cls(4)
        # Loop through keys
        for k, vk in v.items():
            # Calculate type for item
            rtypk = RecordType.from_value(vk, name=k)
            # Account for the type and length of record k
            rs += cls(8 + 4*rtypk.xlong_record)
            # Add to total record size
            rs += _genr8_recordsize(rtypk, vk, k)
        # That's it
        return rs
    # Get basic size of scalar (ignored if string)
    rs1 = _get_rs_scalar(rtyp)
    # Unpack array type flags
    array_flags = rtyp.record_array
    # Check for string
    if et in (ELEMTYPE_STR, ELEMTYPE_BYTES):
        # Ensure array, with encoding
        if et == ELEMTYPE_STR:
            # Encode strings as UTF-8
            va = np.char.encode(v, encoding=DEFAULT_ENCODING)
        else:
            # Ensure array but don't encode
            va = np.asarray(v)
        # Get size from NumPy data type
        rs1 = uint32(va.dtype.itemsize)
        # Add uint32 for length of string [item]
        rs += uint32(4)
    # Check for array
    if array_flags:
        # Add in number of elements
        if array_flags == 1:
            # Just one size entry
            rs += len_itemsize
        elif array_flags == 2:
            # Two size entries
            rs += cls(len_itemsize * 2)
        elif array_flags == 3:
            # First write dimension, then each item of shape array
            rs += cls(4) + cls(v.ndim * len_itemsize)
        # Add in size for data
        rs += rs1 * cls(v.size)
    else:
        # Just add scalar type
        rs += rs1
    # Output
    return rs


# Calculate record size for a string
def genr8_rs_str(val: str) -> uint32:
    # Calculate length
    return uint32(4 + len(val.encode(DEFAULT_ENCODING)))


# Calculate size of basic scalar
def _get_rs_scalar(rtyp: RecordType) -> uint32:
    # Bit size as exponent of 2, shifted by 3 to convert to bytes
    return uint32(2) ** uint32(max(3, rtyp.element_bits) - 3)


# Add error messages to a read
def check_read(func):
    @wraps(func)
    def wrapper(fp, *a, **kw):
        try:
            return func(fp, *a, **kw)
        except ValueError as err:
            # Generate message
            msg = f"at postion {fp.tell()} of {_genr8_fname(fp)}, "
            raise CapeFileValueError(msg + err.args[0])
    # Return the wrapped function
    return wrapper


# Check header
def check_header(fp: IOBase):
    r"""Check that file has the correct header for file type

    :Call:
        >>> check_header(fp)
    :Inputs:
        *fp*: :class:`io.IOBase`
            File open for reading bytes (``'rb'`` or similar)
    :Raises:
        :class:`CapeFileValueError` if file does not have correct
        header
    """
    # Read first 8 bytes
    flag = fp.read(len(HEADER_BYTES))
    # Check value
    assert_value(flag, HEADER_BYTES, f"header of '{fp.name}'")


# Read a record
@check_read
def read_record(fp: IOBase, meta: bool = False):
    return _read_record(fp, meta)


# Read the number of records
@check_read
def read_nrecord(fp: IOBase) -> uint64:
    r"""Read number of records in file

    :Call:
        >>> nr = read_nrecord(fp)
    :Inputs:
        *fp*: :class:`io.IOBase`
            File handle
    :Outputs:
        *nr*: :class:`np.uint64`
            Number of records
    """
    return _read_nr(fp)


@check_read
def read_recordtype(fp: IOBase) -> uint32:
    return _read_rt(fp)


# Read a record
def _read_record(fp: IOBase, meta: bool = False):
    # Read type code
    rt = _read_rt(fp)
    # Parse type information
    rtyp = RecordType(rt)
    # Check for long record
    xlong = rtyp.xlong_record
    # Unpack basic type
    eltype = rtyp.element_type
    # Length data type
    len_nbyte = 4 + (4*xlong)
    len_dtype = f"<u{len_nbyte}"
    # Read size
    rs, = np.fromfile(fp, len_dtype, count=1)
    # Check for title
    if rtyp.record_title:
        # Read a title
        name, offset = _read_str(fp)
    else:
        # No title
        name = None
        offset = 0
    # Skip
    if meta:
        # Change position
        fp.seek(rs - offset, 1)
        # Return the name w/o value
        return name, None
    # Check for None or nested
    if eltype == ELEMTYPE_NONE:
        # Empty value
        return name, None
    elif eltype & ELEMTYPE_FLAG_NESTED:
        # Nested
        if eltype in (ELEMTYPE_LIST, ELEMTYPE_TUPLE, ELEMTYPE_SET):
            # Initialize list
            v = []
            # Read number of items
            n_item, = np.fromfile(fp, "<u4", count=1)
            # Loop through items
            for j in range(n_item):
                # Read item
                namej, vj = _read_record(fp)
                # Make sure there's no name
                if namej is not None:
                    raise CapeFileValueError(
                        "list-type items cannot have titles" +
                        f"item {j} has name '{namej}'")
                # Save item
                v.append(vj)
            # Convert to appropriate specific type
            if eltype == ELEMTYPE_TUPLE:
                v = tuple(v)
            elif eltype == ELEMTYPE_SET:
                v = set(v)
        elif eltype == ELEMTYPE_DICT:
            # Initialize dict
            v = {}
            # Read number of items
            n_item, = np.fromfile(fp, "<u4", count=1)
            # Loop through items
            for j in range(n_item):
                # Read item
                namej, vj = _read_record(fp)
                # Make sure there's no name
                if namej is None:
                    raise CapeFileValueError(
                        f"dict-type item {j} is missing a name")
                # Save item
                v[namej] = vj
        # Output result of nested read
        return name, v
    # Read string size if necessary
    if eltype in (ELEMTYPE_STR, ELEMTYPE_BYTES):
        # Read length
        n_char, = np.fromfile(fp, "<u4", count=1)
    # Unpack array settings
    array_flags = rtyp.record_array
    # Check if array
    if array_flags == 0:
        # Scalar
        nx = 1
    elif array_flags == 1:
        # Read size
        nx, = np.fromfile(fp, len_dtype, count=1)
    elif array_flags == 2:
        # Read nrow and ncol
        nrow, ncol = np.fromfile(fp, len_dtype, count=2)
        # Total size
        nx = nrow * ncol
    else:
        # N-dimensional array
        nd, = np.fromfile(fp, "<u4", count=1)
        # Read individual dimensions
        dims = np.fromfile(fp, len_dtype, count=nd)
        # Total size
        nx = np.prod(dims)
    # Byte order for NumPy type
    bo = ">" if rtyp.el_byteorder else "<"
    # Get bits
    elem_bits = 2 ** rtyp.element_bits
    # Convert to bytes
    esize = max(1, elem_bits >> 3)
    # Get NumPy type char
    typchar = DTYPE_CHARS.get(eltype)
    # Special cases
    if eltype in (ELEMTYPE_BYTES, ELEMTYPE_STR):
        # Reading individual bytes, so byte order is irrelevant
        bo = "|"
        # Use the specified character length
        esize = n_char
    elif eltype in (ELEMTYPE_UINT, ELEMTYPE_INT):
        if esize == 1:
            typchar = "b"
    # Construct overall NumPy data type
    dtype = f"{bo}{typchar}{esize}"
    # Read
    v = np.fromfile(fp, dtype, nx)
    # Decode string
    if eltype == ELEMTYPE_STR:
        v = np.char.decode(v, encoding=DEFAULT_ENCODING)
    # Unpack if not an array
    if array_flags == 0:
        # Scalar
        v, = v
    elif array_flags == 2:
        # Reshape
        v = v.reshape((nrow, ncol))
    elif array_flags == 3:
        # Reshape
        v = v.reshape(dims)
    # Output
    return name, v


# Read number of records
def _read_nr(fp: IOBase) -> uint64:
    # Read number of records
    nr, = np.fromfile(fp, '<u8', count=1)
    return nr


# Read the record type
def _read_rt(fp: IOBase) -> uint32:
    # Read the type flag; check it, and return
    rt, = np.fromfile(fp, '<u4', count=1)
    return rt


# Read raw bytes
def _read_bytes(fp: IOBase) -> bytes:
    # Read the length
    ns, = np.fromfile(fp, "<u4", count=1)
    # Read the bytes
    return fp.read(ns)


# Read a string
def _read_str(fp: IOBase, encoding: str = DEFAULT_ENCODING):
    # Read the bytes
    raw = _read_bytes(fp)
    # Decode
    return raw.decode(encoding), uint32(len(raw) + 4)


# Generate file handle
def _get_fp(name_or_stream, mode: str = 'rb') -> IOBase:
    r"""Get a file handle with the correct mode

    :Call:
        >>> fp = _get_fp(name, mode="rb")
        >>> fp = _get_fp(fp, mode="rb")
    :Inputs:
        *name*: :class:`str`
            Name of file
        *fp*: :class:`io.IOBase`
            File handle
        *mode*: {``"rb"``} | :class:`str`
            File mode requested
    :Outputs:
        *fp*: :class:`io.IOBase`
            File handle, open in ``'rb'`` mode
    """
    # Check for bad types
    assert_isinstance(name_or_stream, (IOBase, str), "file identifier")
    # Check for file name
    if isinstance(name_or_stream, str):
        return open(name_or_stream, mode)
    # Check mode
    assert_value(name_or_stream.mode, mode, "file mode")
    # Already a handle
    return name_or_stream


# Assert type of a variable
def assert_isinstance(obj, cls_or_tuple, desc=None):
    r"""Conveniently check types

    Applies ``isinstance(obj, cls_or_tuple)`` but also constructs
    a :class:`TypeError` and appropriate message if test fails.

    If *cls* is ``None``, no checks are performed.

    :Call:
        >>> assert_isinstance(obj, cls, desc=None)
        >>> assert_isinstance(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: ``None`` | :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Raises:
        :class:`CapeFileTypeError`
    """
    # Special case for ``None``
    if cls_or_tuple is None:
        return
    # Check for passed test
    if isinstance(obj, cls_or_tuple):
        return
    # Generate type error message
    msg = _genr8_type_error(obj, cls_or_tuple, desc)
    # Raise
    raise CapeFileTypeError(msg)


# Assert value of a variable
def assert_value(obj, val_or_tuple, desc=None):
    r"""Conveniently check values

    If *val* is ``None``, no checks are performed.

    :Call:
        >>> assert_value(obj, val, desc=None)
        >>> assert_value(obj, val_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *val*: ``None`` | :class:`object`
            Single permitted value
        *val_tuple*: :class:`tuple`
            Tuple of allowed values
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Raises:
        :class:`CapeFileValueError`
    """
    # Special case for ``None``
    if val_or_tuple is None:
        return
    # Check if given tuple
    if isinstance(val_or_tuple, tuple):
        # Check for multiple values
        test = obj in val_or_tuple
    else:
        # Test direct value
        test = obj == val_or_tuple
    # Done if test passed
    if test:
        return
    # Generate value error message
    msg = _genr8_value_error(obj, val_or_tuple, desc)
    # Rase
    raise CapeFileValueError(msg)


# Ensure a specific size for an array
def assert_size(v: ndarray, n: int, fp: IOBase, desc=None):
    r"""Check if array has expected size

    :Call:
        >>> assert_size(v, n, fp, desc=None)
    :Inputs:
        *v*: :class:`np.ndarray`
            An array
        *n*: :class:`int`
            Expected size
        *fp*: :class:`io.IOBase`
            File handle to report position
        *desc*: {``None``} | :class:`str`
            Optional text describing *v* for including in error msg
    :Raises:
        :class:`CapeFileValueError`
    """
    # Check for expected size
    if v.size == n:
        return
    # Generate message
    msg1 = f"at position {fp.tell()} of {_genr8_fname(fp)}: "
    msg2 = "" if (not desc) else ("%s: " % desc)
    msg3 = f"got array size={v.size}; expected {n}"
    raise CapeFileValueError(msg1 + msg2 + msg3)


# Create error message for type errors
def _genr8_type_error(obj, cls_or_tuple, desc=None) -> str:
    r"""Create error message for type-check commands

    :Call:
        >>> msg = _genr8_type_error(obj, cls, desc=None)
        >>> msg = _genr8_type_error(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: ``None`` | :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Outputs:
        *msg*: :class:`str`
            Text of an error message explaining available types
    """
    # Check for single type
    if isinstance(cls_or_tuple, tuple):
        # Multiple types
        names = [cls.__name__ for cls in cls_or_tuple]
    else:
        # Single type
        names = [cls_or_tuple.__name__]
    # Create error message
    msg1 = ""
    if desc:
        msg1 = f"{desc}: "
    msg2 = "got type '%s'; " % type(obj).__name__
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3


# Create error message for value errors
def _genr8_value_error(obj, val_or_tuple, desc=None) -> str:
    r"""Create error message for type-check commands

    :Call:
        >>> msg = _genr8_type_error(obj, val, desc=None)
        >>> msg = _genr8_type_error(obj, val_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *val*: :class:`object`
            Single permitted value
        *cls_tuple*: :class:`tuple`
            Tuple of allowed values
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Outputs:
        *msg*: :class:`str`
            Text of an error message explaining available types
    """
    # Check for single type
    if isinstance(val_or_tuple, tuple):
        # Multiple types
        names = [str(val) for val in val_or_tuple]
    else:
        # Single type
        names = [str(val_or_tuple)]
    # Create error message
    msg1 = ""
    if desc:
        msg1 = f"{desc}: "
    msg2 = "got value '%s'; " % obj
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3


# Print file name
def _genr8_fname(fp: IOBase) -> str:
    # Get "name" if any
    name = getattr(fp, "name", None)
    # Check if there was one
    if name is None:
        # This can happen if working with a StringIO or other non-file
        return f"<{fp.__class__.__name__} at 0x{id(fp):x}>"
    else:
        # Use the (base) file name
        return f"<file '{os.path.basename(name)}'>"
