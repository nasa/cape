#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.io`: Binary file input/output tools
==============================================

This is a module to provide fast and convenient utilities for reading
and writing binary data in Cape.  The module relies heavily on the NumPy
functions :func:`fromfile` and :func:`tofile`, but it also performs
checks and conversions for big-ending vs little-endian data in a manner
that can be mostly hidden from the user.  It also handles Fortran
start-of-record and end-of-record markers and performs the available
checks without the need for users to write extra code every time a
Fortran file is accessed.

In most cases classes built upon this library are responsible for
detecting the presence or lack of Fortran record markers and detecting
the endianness of the file automatically.

In addition, users can write to either endianness regardless of the
system byte order.  Although defaulting to the system file format is
recommended for most modern applications, for some software this is not
an available option.

This module frequently utilizes the following naming conventions for
file formats, which are semi-standard but not necessarily universally
recognizable. They take the form of one or two letters for the
endianness of the file and one integer for the number of bytes used to
represent a real-valued :class:`float`.

    ============   ====================================================
    Code           Description
    ============   ====================================================
    ``b4``         Single-precision big-endian C format
    ``b8``         Double-precision big-endian C format
    ``lb4``        Single-precision little-endian C format
    ``lb8``        Double-precision little-endian C format
    ``ne4``        Single-precision native-endian C format
    ``r4``         Single-precision big-endian Fortran format
    ``r8``         Double-precision big-endian Fortran format
    ``lr4``        Single-precision little-endian Fortran format
    ``lr8``        Double-precision little-endian Fortran format
    ============   ====================================================

These codes are used frequently in the names of functions within this
module. In addition, the functions in this module usually contain a
suffix of ``i`` (integer), ``f`` (float), or ``s`` (string).  For
example :func:`read_record_lr4_i` reads a little-endian :class:`int`
record, and :func:`read_record_r8_f` reads a double-precision
:class:`float` record.

By convention, Fortran double-precision files often use single-precision
integers, so functions like :func:`read_record_r8_i` are unlikely to be
utilized.  To add further confusion, Fortran record markers are almost
(?) always 4-byte integers even for double-precision :class:`float`
records. Methods such as :func:`read_record_r8_f2` are provided for the
theoretical case in which the record marker is a :class:`long` (8-byte
integer).  The full table of record-type suffixes for big-endian files
is below. Just prepend the suffix with an ``l`` for the little-endian
versions.

    ========== ================= ======================================
    Suffix     Class             Description
    ========== ================= ======================================
    ``r4_i``   :class:`int32`    Common integer record
    ``r8_i``   :class:`int32`    Long integer record
    ``r8_i2``  :class:`int64`    Long integer with long record markers
    ``r4_f``   :class:`float32`  Common single-precision float
    ``r8_f``   :class:`float32`  Common double-precision float
    ``r8_f2``  :class:`float64`  Double float with long record markers
    ``r4_u``   :class:`uint32`   Common unsigned integer record
    ``r8_u``   :class:`uint32`   Long uint record
    ``r8_u2``  :class:`uint64`   Long uint with long record markers
    ``b4_s``   :class:`str`      String from 4-byte char codes
    ========== ================= ======================================

"""

# Standard library
import os

# Third-party modules
import numpy as np


# Get byte order
LITTLE_ENDIAN = (os.sys.byteorder == 'little')
BIG_ENDIAN = (os.sys.byteorder == 'big')

# *********************************************************************
# ====== environment ==================================================
# Get implied environment byte order
def get_env_byte_order():
    r"""Determine byte order from system and environment variables

    This checks the following environment variables to override the
    system byte order if found. (Listed in order of precedence)

    1. ``F_UFMTENDIAN`` (flag for ``ifort``)
        a. ``"little"`` for little-endian
        b. ``"big"`` for big-endian
    2. ``GFORTRAN_CONVERT_UNIT`` (flag for ``gfortran`` and related)
        a. ``"little_endian"`` for little-endian
        b. ``"big_endian"`` for big-endian

    :Call:
        >>> ebo = get_env_byte_order()
    :Outputs:
        *ebo*: ``"big"`` | ``"little"``
            Implied default byte order
    :Versions:
        * 2021-12-29 ``@ddalle``: Version 1.0
    """
    # System byte order
    ebo = os.sys.byteorder
    # Get relevant environment variables
    ENV_IFORT = os.environ.get('F_UFMTENDIAN')
    ENV_GFORT = os.environ.get('GFORTRAN_CONVERT_UNIT')
    # Check for valid environment variables
    if ENV_IFORT == 'big':
        # IFORT environment variable set to big-endian
        ebo = 'big'
    elif ENV_IFORT == 'little':
        # IFORT environment variable set to little-endian
        ebo = 'little'
    elif ENV_GFORT == 'big_endian':
        # gfortran environment variable set to big-endian
        ebo = 'big'
    elif ENV_GFORT == 'little_endian':
        ebo = "little"
    # Output
    return ebo


# Try to read first record
def get_filetype(fp):
    r"""Get the file type by trying to read first line

    :Call:
        >>> ft = get_filetype(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle open for reading
    :Outputs:
        *ft*: :class:`str`
            File type code:
                * ``""``: empty file
                * ``"|"``: ASCII
                * ``"<4"``: little-endian single-precision (32-bit)
                * ``"<8"``: little-endian double-precision (64-bit)
                * ``">4"``: big-endian single-precision (32-bit)
                * ``">8"``: big-endian double-precision (64-bit)
                * ``"?"``: not ASCII and no Fortran records
    :Versions:
        * 2016-09-04 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 2.0; allow *fp*
    """
    # Get current position
    p = fp.tell()
    # Call main function
    try:
        return _get_filetype(fp)
    finally:
        # Return existing file to original position
        fp.seek(p)


# Try to read first record
def get_filenametype(fname):
    r"""Get the file type by trying to read first line

    :Call:
        >>> ft = get_filenametype(fname)
    :Inputs:
        *fname*: :class:`str`
            File name
    :Outputs:
        *ft*: :class:`str`
            File type code:
                * ``""``: empty file
                * ``"|"``: ASCII
                * ``"<4"``: little-endian single-precision (32-bit)
                * ``"<8"``: little-endian double-precision (64-bit)
                * ``">4"``: big-endian single-precision (32-bit)
                * ``">8"``: big-endian double-precision (64-bit)
                * ``"?"``: not ASCII and no Fortran records
    :Versions:
        * 2016-09-04 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 2.0; use _get_filetype()
    """
    # Call main function
    with open(fname, "rb") as fp:
        return _get_filetype(fp)


def _get_filetype(fp):
    # Check from current position
    p = fp.tell()
    # Get size of file (prevents invalid seeks)
    fp.seek(0, 2)
    p2 = fp.tell()
    # Read first word
    fp.seek(p)
    i4l = np.fromfile(fp, count=1, dtype='<i4')
    # Check for emptiness
    if len(i4l) == 0:
        # Empty file
        return ""
    # Unpack record
    r4l, = i4l
    # Try little-endian single
    # Read end-of-record
    p1 = fp.tell()
    if p1 + r4l <= p2:
        fp.seek(p1 + r4l)
        j2 = np.fromfile(fp, count=1, dtype='<i4')
        # Check consistency
        if len(j2) > 0 and r4l == j2[0]:
            # Consistent markers
            return '<4'
    # Try big-endian single
    fp.seek(p)
    r4b, = np.fromfile(fp, count=1, dtype='>i4')
    # Read end-of-record
    p1 = fp.tell()
    if p1 + r4b <= p2:
        fp.seek(p1 + r4b)
        j2 = np.fromfile(fp, count=1, dtype='>i4')
        # Check consistency
        if len(j2) > 0 and r4b == j2[0]:
            # Consistent markers
            return '>4'
    # Try little-endian double
    fp.seek(p)
    r8l, = np.fromfile(fp, count=1, dtype='<i8')
    # Read end-of-record
    p1 = fp.tell()
    if p1 + r8l <= p2:
        fp.seek(p1 + r8l)
        j2 = np.fromfile(fp, count=1, dtype='<i8')
        # Check consistency
        if len(j2) > 0 and r8l == j2[0]:
            # Consistent markers
            return '<8'
    # Try big-endian double
    fp.seek(p)
    r8b, = np.fromfile(fp, count=1, dtype='>i8')
    # Read end-of-record
    p1 = fp.tell()
    if p1 + r8b <= p2:
        fp.seek(p1 + r8b)
        j2 = np.fromfile(fp, count=1, dtype='>i8')
        # Check consistency
        if len(j2) > 0 and r8b == j2[0]:
            # Consistent markers
            return '>8'
    # Check first 16 bytes
    fp.seek(p)
    x = np.fromfile(fp, count=16, dtype="i1")
    # Check ASCII
    if np.min(x) > 0 and np.max(x) < 128:
        # Apparently ASCII
        return "|"
    else:
        # Failure
        return "?"
# def get_filetype


# *********************************************************************
# ====== string readers ===============================================

# Read string
def read_c_str(fp, encoding="utf-8", nmax=1000):
    r"""Read a C-style string from a binary file

    String is terminated with a null ``\0`` character

    :Call:
        >>> s = read_c_str(fp, encoding="utf-8", nmax=1000)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *encoding*: {``"utf-8"``} | ``"ascii"`` | :class:`str`
            Valid encoding name
        *nmax*: {``1000``} | :class:`int`
            Maximum number of characters, to avoid infinite loops
    :Outputs:
        *s*: :class:`str`
            String read from file until ``\0`` character
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; fix data types
    """
    # Read bytes
    buf = read_c_bytes(fp, nmax=nmax)
    # Decode
    return buf.decode(encoding)


# Read bytes of string
def read_c_bytes(fp, nmax=1000):
    r"""Read bytes of a C-style string from a binary file

    String is terminated with a null ``\0`` character

    :Call:
        >>> s = read_c_str(fp, nmax=1000)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *nmax*: {``1000``} | :class:`int`
            Maximum number of characters, to avoid infinite loops
    :Outputs:
        *s*: :class:`bytes`
            String read from file until ``\0`` character
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; from read_c_str()
    """
    # Initialize array
    buf = bytearray()
    # Loop until termination found
    n = 0
    while n < nmax:
        # Read the next character
        b = fp.read(1)
        n += 1
        # Check for termination (includes EOF)
        if b == b'\0' or b == b'' or b is None:
            # Output
            return bytes(buf)
        else:
            # Append the character to the buffer
            buf.append(ord(b))
    # If this point is reached, we had an overflow
    print("WARNING: More than nmax=%i characters in buffer" % nmax)
    return bytes(buf)


# Read byte string
def read_lb4_s(fp):
    r"""Read C-style string assuming 4 little-endian bytes per char

    :Call:
        >>> s = read_lb4_s(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *s*: :class:`str`
            String read from file
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
    """
    # Initialize array
    buf = bytearray()
    # Loop until termination
    while True:
        # Read the next character
        b = np.fromfile(fp, count=1, dtype="<i4")
        # Check the value
        if (b.size == 0) or (b[0] == 0):
            # End of string
            return buf.decode("utf-8")
        else:
            # Convert to a character and append it
            buf.append(b[0])


# Read byte string
def read_b4_s(fp):
    r"""Read C-style string assuming 4 big-endian bytes per char

    :Call:
        >>> s = read_b4_s(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *s*: :class:`str`
            String read from file
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
    """
    # Initialize array
    buf = bytearray()
    # Loop until termination ofund
    while True:
        # Read the next character
        b = np.fromfile(fp, count=1, dtype=">i4")
        # Check the value
        if (b.size == 0) or (b[0] == 0):
            # End of string
            return buf.decode("utf-8")
        else:
            # Convert to a character and append it
            buf.append(b[0])
# > string read


# ====== string writers ===============================================
# Write byte string
def tofile_lb4_s(fp, s):
    r"""Write C-style string assuming 4 little-endian bytes per char

    :Call:
        >>> tofile_lb4_s(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *s*: :class:`str`
            String to write to binary file
    :Versions:
        * 2017-03-29 ``@ddalle``: Version 1.0
    """
    # Create array
    x = [ord(c) for c in str(s)] + [0]
    # Write it
    tofile_lb4_i(fp, x)


# Write byte string
def tofile_b4_s(fp, s):
    r"""Write C-style string assuming 4 big-endian bytes per char

    :Call:
        >>> tofile_b4_s(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *s*: :class:`str`
            String to write to binary file
    :Versions:
        * 2017-03-29 ``@ddalle``: Version 1.0
    """
    # Create array
    x = [ord(c) for c in str(s)] + [0]
    # Write it
    tofile_b4_i(fp, x)
# > string write


# **********************************************************************
# ====== lb4 write =====================================================
# Write integer as little-endian single-precision
def tofile_lb4_i(fp, x):
    r"""Write an integer or array to single-precision little-endian file

    :Call:
        >>> tofile_lb4_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
    # Write
    X.tofile(fp)


# Write float as little-endian single-precision
def tofile_lb4_f(fp, x):
    r"""Write a float or array to single-precision little-endian file

    :Call:
        >>> tofile_lb4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
    # Write
    X.tofile(fp)
# > lb4 write


# ====== lb8 write =====================================================
# Write integer as little-endian double-precision
def tofile_lb8_i(fp, x):
    r"""Write an integer [array] to double-precision little-endian file

    :Call:
        >>> tofile_lb8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
    # Write
    X.tofile(fp)


# Write float as little-endian double-precision
def tofile_lb8_f(fp, x):
    r"""Write a float [array] to double-precision little-endian file

    :Call:
        >>> tofile_lb4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
    # Write
    X.tofile(fp)
# > lb8 write


# ====== b4 write ======================================================
# Write integer as big-endian single-precision
def tofile_b4_i(fp, x):
    r"""Write an integer or array to single-precision big-endian file

    :Call:
        >>> tofile_b4_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
    # Write
    X.tofile(fp)


# Write float as big-endian single-precision
def tofile_b4_f(fp, x):
    """Write a float or array to single-precision big-endian file

    :Call:
        >>> tofile_b4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
    # Write
    X.tofile(fp)
# > b4 write


# ====== b8 write ======================================================
# Write integer as big-endian double-precision
def tofile_b8_i(fp, x):
    r"""Write an integer or array to double-precision big-endian file

    :Call:
        >>> tofile_b8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
    # Write
    X.tofile(fp)


# Write float as big-endian double-precision
def tofile_b8_f(fp, x):
    r"""Write a float or array to double-precision big-endian file

    :Call:
        >>> tofile_b4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
    # Write
    X.tofile(fp)
# b8 write


# ====== ne4 write =====================================================
# Write integer as native-endian single-precision
def tofile_ne4_i(fp, x):
    r"""Write an integer or array to single-precision native-endian file

    :Call:
        >>> tofile_ne4_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Write
    X.tofile(fp)


# Write float as big-endian single-precision
def tofile_ne4_f(fp, x):
    r"""Write a float or array to single-precision native-endian file

    :Call:
        >>> tofile_ne4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Write
    X.tofile(fp)
# > ne4 write


# ====== ne8 write =====================================================
# Write integer as native-endian double-precision
def tofile_ne8_i(fp, x):
    r"""Write an integer or array to double-precision native-endian file

    :Call:
        >>> tofile_ne8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Write
    X.tofile(fp)


# Write float as native-endian double-precision
def tofile_ne8_f(fp, x):
    r"""Write a float or array to double-precision native-endian file

    :Call:
        >>> tofile_ne8_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Write
    X.tofile(fp)
# > ne8 write


# **********************************************************************
# ====== lr4 record ====================================================
# Write record of single-precision little-endian integers
def write_record_lr4_i(fp, x):
    r"""Write Fortran :class:`int32` record to little-endian file

    The record markers are :class:`int32`.

    :Call:
        >>> write_record_lr4_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Byte counts
    I = np.array(X.size*4, dtype='i4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of single-precision little-endian floats
def write_record_lr4_f(fp, x):
    r"""Write Fortran :class:`float32` record to little-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_lr4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Byte counts
    I = np.array(X.size*4, dtype='i4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)
# > write_record_lr4


# ====== lr8 record ====================================================
# Write record of double-precision little-endian integers
def write_record_lr8_i(fp, x):
    r"""Write Fortran :class:`int64` record to little-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_lr8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Byte counts
    I = np.array(X.size*8, dtype='i4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision little-endian integers
def write_record_lr8_i2(fp, x):
    r"""Write special Fortran :class:`int64` record little-endian file

    The record markers are :class:`int64`.

    :Call:
        >>> write_record_lr8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Byte counts
    I = np.array(X.size*8, dtype='i8')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision little-endian floats
def write_record_lr8_f(fp, x):
    r"""Write Fortran :class:`float64` record to little-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_lr8_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Byte counts
    I = np.array(X.size*8, dtype='i4')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision little-endian floats
def write_record_lr8_f2(fp, x):
    r"""Write special Fortran :class:`float64` record little-endian

    The record markers from this function are :class:`int64`.

    :Call:
        >>> write_record_lr8_f2(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Byte counts
    I = np.array(X.size*8, dtype='i8')
    # Check byte order
    if BIG_ENDIAN: # pragma no cover
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)
# > write_record_lr8


# ====== r4 record ====================================================
# Write record of single-precision big-endian integers
def write_record_r4_i(fp, x):
    r"""Write Fortran :class:`int32` record to big-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_r4_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Byte counts
    I = np.array(X.size*4, dtype='i4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)

# Write record of single-precision big-endian floats
def write_record_r4_f(fp, x):
    """Write Fortran :class:`float32` record to big-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_r4_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Byte counts
    I = np.array(X.size*4, dtype='i4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)
# > write_record_r4


# ====== r8 record ====================================================
# Write record of double-precision big-endian integers
def write_record_r8_i(fp, x):
    r"""Write Fortran :class:`int64` record to big-endian file

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_r8_i(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Byte counts
    I = np.array(X.size*8, dtype='i4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision big-endian integers
def write_record_r8_i2(fp, x):
    r"""Write special Fortran :class:`int64` record big-endian

    The record markers are 8-byte :class:`int64`.

    :Call:
        >>> write_record_r8_i2(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Byte counts
    I = np.array(X.size*8, dtype='i8')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision big-endian floats
def write_record_r8_f(fp, x):
    r"""Write Fortran :class:`float64` record big-endian

    The record markers are 4-byte :class:`int32`.

    :Call:
        >>> write_record_r8_f(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Byte counts
    I = np.array(X.size*8, dtype='i4')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)


# Write record of double-precision big-endian floats
def write_record_r8_f2(fp, x):
    """Write special Fortran :class:`float64` record big-endian

    The record markers are 8-byte :class:`int64`.

    :Call:
        >>> write_record_r8_f2(fp, x)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Byte counts
    I = np.array(X.size*8, dtype='i8')
    # Check byte order
    if LITTLE_ENDIAN:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(fp)
    X.tofile(fp)
    I.tofile(fp)
# > write_record_r8


# ********************************************************************
# ====== lb4 read ====================================================
# Read integer from little-endian single-precision file
def fromfile_lb4_i(fp, n):
    r"""Read *n* 4-byte :class:`int` little-endian

    :Call:
        >>> x = fromfile_lb4_i(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype="<i4")


# Read float from little-endian single-precision file
def fromfile_lb4_f(fp, n):
    r"""Read *n* 4-byte :class:`float` little-endian

    :Call:
        >>> x = fromfile_lb4_f(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype="<f4")
# > lb4 read


# ====== lb8 read ====================================================
# Read integer from little-endian double-precision file
def fromfile_lb8_i(fp, n):
    r"""Read *n* 8-byte :class:`int` little-endian

    :Call:
        >>> x = fromfile_lb8_i(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype="<i8")


# Read float from little-endian double-precision file
def fromfile_lb8_f(fp, n):
    r"""Read *n* 8-byte :class:`float` little-endian

    :Call:
        >>> x = fromfile_lb8_f(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype="<f8")
# > lb4 read


# ====== b4 read ====================================================
# Read integer from big-endian single-precision file
def fromfile_b4_i(fp, n):
    r"""Read *n* 4-byte :class:`int` big-endian

    :Call:
        >>> x = fromfile_b4_i(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype=">i4")


# Read float from big-endian single-precision file
def fromfile_b4_f(fp, n):
    r"""Read *n* 4-byte :class:`float` big-endian

    :Call:
        >>> x = fromfile_b4_f(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype=">f4")
# > b4 read


# ====== b8 read ====================================================
# Read integer from big-endian double-precision file
def fromfile_b8_i(fp, n):
    r"""Read *n* 8-byte :class:`int` big-endian

    :Call:
        >>> x = fromfile_b8_i(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype=">i8")


# Read float from big-endian double-precision file
def fromfile_b8_f(fp, n):
    r"""Read *n* 8-byte :class:`float` big-endian

    :Call:
        >>> x = fromfile_b8_f(fp, n)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read from file
    return np.fromfile(fp, count=n, dtype=">f8")
# > b8 read


# **********************************************************************
# ====== record markers ================================================
# Safely read start-of-record
def read_record_start(fp, dtype):
    r"""Read Fortran-style start-of-record marker

    :Call:
        >>> r1 = read_record_start(fp, dtype)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'rb' or similar
        *dtype*: :class:`str`
            Data type for :func:`np.fromfile`
    :Outputs:
        *r1*: *dtype*
            Start-of-record, usually number of bytes in record
    :Versions:
        * 2021-12-29 ``@ddalle``: Version 1.0
    """
    # Read next int32 or int64
    I1 = np.fromfile(fp, count=1, dtype=dtype)
    # Check for empty file
    if len(I1) == 0:
        return 0
    # Otherwise return record marker
    return I1[0]


# Safely read end-of-record and check
def read_record_end(fp, dtype, r1):
    r"""Read and check Fortran-style end-of-record marker

    :Call:
        >>> r1 = read_record_end(fp, dtype, r1)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'rb' or similar
        *dtype*: :class:`str`
            Data type for :func:`np.fromfile`
        *r1*: :class:`np.int`
            Start-of-record, usually number of bytes in record
    :Outputs:
        *r2*: *r1.__class__*
            End-of-record, matches *r1*
    :Raises:
        *IOError*: if *r1* and *r2* do not match
    :Versions:
        * 2021-12-29 ``@ddalle``: Version 1.0
    """
    # Read next int32 or int64
    I2 = np.fromfile(fp, count=1, dtype=dtype)
    # Check for EOF
    if len(I2) == 0:
        r2 = 0
    else:
        r2 = I2[0]
    # Check
    if r1 != r2:
        raise IOError(
            ("End-of-record-marker '%i' " % r2) +
            ("does not match start '%i'" % r1))
    # Output
    return r2


# Check record marks, lr4
def check_record(fp, dtype):
    r"""Check for consistent record based on record markers

    :Call:
        >>> q = check_record(fp, dtype)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'rb' or similar
        *dtype*: :class:`str`
            Data type for :func:`np.fromfile`
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *fp* has a valid record in the next position
    :Version:
        * 2018-01-11 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 2.0; fork check_record_lr4()
    """
    # Save original position
    p = fp.tell()
    # Read start-of-record marker
    I1 = np.fromfile(fp, count=1, dtype=dtype)
    # Check for end of file
    if I1.size == 0 or I1[0] <= 0:
        fp.seek(p)
        return False
    # Unpack
    r1 = I1[0]
    # Skip to end-of-record
    fp.seek(fp.tell() + r1)
    # Get new position
    p1 = fp.tell()
    # Go to end of file
    fp.seek(0, 2)
    p2 = fp.tell()
    # Check if previous seek advanced beyond end of file
    if p1 > p2:
        fp.seek(p)
        return False
    # Read the end-of-record
    fp.seek(p1)
    I2 = np.fromfile(fp, count=1, dtype=dtype)
    # Return to original position
    fp.seek(p)
    # Check for validity
    return I2.size == 1 and r1 == I2[0]

    
# ====== lr4 record ====================================================
# Read record of single-precision little-endian integers
def read_record_lr4_i(fp):
    r"""Read 4-byte little-endian :class:`int` record

    :Call:
        >>> x = read_record_lr4_i(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i4")
    # Get count
    n = r1 // 4
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<i4")
    # Read the end-of-record
    read_record_end(fp, "<i4", r1)
    # Output
    return x


# Read record of single-precision little-endian integers
def read_record_lr4_f(fp):
    r"""Read 4-byte little-endian :class:`float` record

    :Call:
        >>> x = read_record_lr4_f(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i4")
    # Get count
    n = r1 // 4
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<f4")
    # Read the end-of-record
    read_record_end(fp, "<i4", r1)
    # Output
    return x
# > read_record_lr4


# ====== lr8 record ====================================================
# Read record of double-precision little-endian integers
def read_record_lr8_i(fp):
    r"""Read 8-byte little-endian :class:`int` record

    :Call:
        >>> x = read_record_lr8_i(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i4")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<i8")
    # Read the end-of-record
    read_record_end(fp, "<i4", r1)
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_f(fp):
    r"""Read 8-byte little-endian :class:`float` record

    :Call:
        >>> x = read_record_lr8_f(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i4")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<f8")
    # Read the end-of-record
    read_record_end(fp, "<i4", r1)
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_i2(fp):
    r"""Read 8-byte little-endian :class:`int` record

    with 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_lr8_i2(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i8")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<i8")
    # Read the end-of-record
    read_record_end(fp, "<i8", r1)
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_f2(fp):
    r"""Read 8-byte little-endian :class:`float` record

    with 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_lr8_f2(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, "<i8")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype="<f8")
    # Read the end-of-record
    read_record_end(fp, "<i8", r1)
    # Output
    return x
# > read_record_lr8


# ====== r4 record ====================================================
# Read record of single-precision big-endian integers
def read_record_r4_i(fp):
    r"""Read 4-byte big-endian :class:`int` record

    :Call:
        >>> x = read_record_r4_i(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i4")
    # Get count
    n = r1 // 4
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">i4")
    # Read the end-of-record
    read_record_end(fp, ">i4", r1)
    # Output
    return x


# Read record of single-precision big-endian integers
def read_record_r4_f(fp):
    r"""Read 4-byte big-endian :class:`float` record

    :Call:
        >>> x = read_record_r4_f(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i4")
    # Get count
    n = r1 // 4
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">f4")
    # Read the end-of-record
    read_record_end(fp, ">i4", r1)
    # Output
    return x
# > read_record_r4


# ====== r8 record ====================================================
# Read record of double-precision big-endian integers
def read_record_r8_i(fp):
    r"""Read 8-byte big-endian :class:`int` record

    :Call:
        >>> x = read_record_r8_i(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i4")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">i8")
    # Read the end-of-record
    read_record_end(fp, ">i4", r1)
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_f(fp):
    r"""Read 8-byte big-endian :class:`float` record

    :Call:
        >>> x = read_record_r8_f(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i4")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">f8")
    # Read the end-of-record
    read_record_end(fp, ">i4", r1)
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_i2(fp):
    r"""Read 8-byte big-endian :class:`int` record

    using 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_r8_i2(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i8")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">i8")
    # Read the end-of-record
    read_record_end(fp, ">i8", r1)
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_f2(fp):
    r"""Read 8-byte big-endian :class:`float` record

    using 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_r8_f2(fp)
    :Inputs:
        *fp*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
        * 2021-12-29 ``@ddalle``: Version 1.1; read_record_start()
    """
    # Read start-of-record marker
    r1 = read_record_start(fp, ">i8")
    # Get count
    n = r1 // 8
    # Read that many ints
    x = np.fromfile(fp, count=n, dtype=">f8")
    # Read the end-of-record
    read_record_end(fp, ">i8", r1)
    # Output
    return x
# > read_record_r8
