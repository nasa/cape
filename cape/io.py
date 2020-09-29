#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.io`: Common input/output library
============================================

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
le = (os.sys.byteorder == 'little')
be = (os.sys.byteorder == 'big')

# System byte order
sbo = os.sys.byteorder
# Get relevant environment variables
env_ifort = os.environ.get('F_UFMTENDIAN')
env_gfort = os.environ.get('GFORTRAN_CONVERT_UNIT')
# Check for valid environment variables
if env_ifort == 'big':
    # IFORT environment variable set to big-endian
    sbo = 'big'
elif env_ifort == 'little':
    # IFORT environment variable set to little-endian
    sbo = 'little'
elif env_gfort == 'big_endian':
    # gfortran environment variable set to big-endian
    sbo = 'big'
elif env_gfort == 'little_endian':
    # gfortran environment variable set to little-endian
    sbo = 'little'


# Try to read first record
def get_filetype(fname):
    r"""Get the file type by trying to read first line

    :Call:
        >>> ft = get_filetype(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
    :Outputs:
        *ft*: "|" | "<4" | ">4" | "<8" | ">8"
            File type: ASCII (``"|"``), little-endian single
            (``"<4"``), little-endian double (``"<8"``), big-endian
            single (``">4"``), or big-endian double (``">8"``)
    :Versions:
        * 2016-09-04 ``@ddalle``: Version 1.0
    """
    # Open the file as a binary file (yes, this will work even for ASCII).
    f = open(fname, 'rb')
    # ASCII check: '<i4' and '>i4' are the same!
    i4l = np.fromfile(f, count=1, dtype='<i4'); f.seek(0)
    i4b = np.fromfile(f, count=1, dtype='>i4'); f.seek(0)
    # Check for emptiness
    if len(i4l) == 0:
        # Empty file
        f.close()
        return
    elif i4l[0] == i4b[0] and i4l[0]>0:
        # ASCII
        f.close()
        return '|'
    # Try little-endian single
    f.seek(4+i4l[0])
    # Read end-of-record (maybe) marker
    j4l = np.fromfile(f, count=1, dtype='<i4'); f.seek(0)
    # Check little-endian
    if len(j4l)>0 and j4l[0]==i4l[0]:
        # Consistent markers
        f.close()
        return '<4'
    # Try big-endian single
    f.seek(4+i4b[0])
    # Read end-of-record (maybe) marker
    j4b = np.fromfile(f, count=1, dtype='>i4'); f.seek(0)
    # Check big-endian
    if len(j4b)>0 and j4b[0]==i4b[0]:
        # Consistent markers
        f.close()
        return '>4'
    # Try little-endian double
    i8l = np.fromfile(f, count=1, dtype='<i8')
    # Protect for invalid codes
    try:
        # Read end-of-record
        f.seek(i8l[0], 1)
        j8l = np.fromfile(f, count=1, dtype='<i8')
        # Check consistency
        if len(j8l)>0 and i8l[0]==j8l[0]:
            # Consistent markers
            f.close()
            return '<8'
    except Exception:
        pass
    # Try big-endian double
    f.seek(0)
    i8b = np.fromfile(f, count=1, dtype='>i8')
    # Protect for invalid codes
    try:
        # Read end-of-record
        f.seek(i8b[0], 1)
        j8b = np.fromfile(f, count=1, dtype='>i8')
        # This was the last chance
        f.close()
        # Check consistency
        if len(j8b)>0 and i8b[0]==j8b[0]:
            # Consistent markers
            return '>8'
    except Exception:
        # Failure
        raise ValueError("Could not process file '%s'" % fname)
# def get_filetype


# **********************************************************************
# ====== string readers ===============================================

# Read string
def read_c_str(f, nmax=1000):
    r"""Read a C-style string from a binary file

    String is terminated with a null ``\0`` character

    :Call:
        >>> s = read_c_str(f, nmax=1000)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *nmax*: :class:`int`
            Maximum number of characters, to avoid infinite loops
    :Outputs:
        *s*: :class:`str`
            String read from file until ``\0`` character
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
    """
    # Initialize array
    buf = bytearray()
    # Loop until termination found
    n = 0
    while n < nmax:
        # Read the next character
        b = f.read(1)
        n += 1
        # Check for termination
        if (b == '\0') or (b is None):
            # Output
            return str(buf)
        else:
            # Append the character to the buffer
            buf.append(b)
    # If this point is reached, we had an overflow
    print("WARNING: More than nmax=%i characters in buffer" % nmax)
    return str(buf)


# Read byte string
def read_lb4_s(f):
    r"""Read C-style string assuming 4 little-endian bytes per char

    :Call:
        >>> s = read_lb4_s(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *s*: :class:`str`
            String read from file
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
    """
    # Initialize array
    buf = ''
    # Loop until termination ofund
    while True:
        # Read the next character
        b = np.fromfile(f, count=1, dtype="<i4")
        # Check the value
        if (b.size == 0) or (b[0] == 0):
            # End of string
            return buf
        else:
            # Convert to a character and append it
            buf += chr(b[0])
    # If we reach here, we had overflow
    return buf


# Read byte string
def read_b4_s(f):
    r"""Read C-style string assuming 4 big-endian bytes per char

    :Call:
        >>> s = read_b4_s(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *s*: :class:`str`
            String read from file
    :Versions:
        * 2016-11-14 ``@ddalle``: Version 1.0
    """
    # Initialize array
    buf = ''
    # Loop until termination ofund
    while True:
        # Read the next character
        b = np.fromfile(f, count=1, dtype=">i4")
        # Check the value
        if (b.size == 0) or (b[0] == 0):
            # End of string
            return buf
        else:
            # Convert to a character and append it
            buf += chr(b[0])
    # If we reach here, we had overflow
    return buf
# > string read


# ====== string writers ===============================================

# Write byte string
def tofile_lb4_s(f, s):
    r"""Write C-style string assuming 4 little-endian bytes per char

    :Call:
        >>> tofile_lb4_s(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *s*: :class:`str`
            String to write to binary file
    :Versions:
        * 2017-03-29 ``@ddalle``: Version 1.0
    """
    # Create array
    x = [ord(c) for c in str(s)] + [0]
    # Write it
    tofile_lb4_i(f, x)


# Write byte string
def tofile_b4_s(f, s):
    r"""Write C-style string assuming 4 big-endian bytes per char

    :Call:
        >>> tofile_b4_s(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *s*: :class:`str`
            String to write to binary file
    :Versions:
        * 2017-03-29 ``@ddalle``: Version 1.0
    """
    # Create array
    x = [ord(c) for c in str(s)] + [0]
    # Write it
    tofile_b4_i(f, x)


# Write byte string
def tofile_ne4_s(f, s):
    r"""Write C-style string assuming 4 native-endian bytes per char

    :Call:
        >>> tofile_ne4_s(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *s*: :class:`str`
            String to write to binary file
    :Versions:
        * 2017-03-29 ``@ddalle``: Version 1.0
    """
    # Create array
    x = [ord(c) for c in str(s)] + [0]
    # Write it
    tofile_ne4_i(f, x)
# > string write


# **********************************************************************
# ====== lb4 write =====================================================
# Write integer as little-endian single-precision
def tofile_lb4_i(f, x):
    r"""Write an integer or array to single-precision little-endian file

    :Call:
        >>> tofile_lb4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if be:
        X.byteswap(True)
    # Write
    X.tofile(f)


# Write float as little-endian single-precision
def tofile_lb4_f(f, x):
    r"""Write a float or array to single-precision little-endian file

    :Call:
        >>> tofile_lb4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if be:
        X.byteswap(True)
    # Write
    X.tofile(f)
# > lb4 write


# ====== lb8 write =====================================================
# Write integer as little-endian double-precision
def tofile_lb8_i(f, x):
    r"""Write an integer [array] to double-precision little-endian file

    :Call:
        >>> tofile_lb8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if be:
        X.byteswap(True)
    # Write
    X.tofile(f)


# Write float as little-endian double-precision
def tofile_lb8_f(f, x):
    r"""Write a float [array] to double-precision little-endian file

    :Call:
        >>> tofile_lb4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if be:
        X.byteswap(True)
    # Write
    X.tofile(f)
# > lb8 write


# ====== b4 write ======================================================
# Write integer as big-endian single-precision
def tofile_b4_i(f, x):
    r"""Write an integer or array to single-precision big-endian file

    :Call:
        >>> tofile_b4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if le:
        X.byteswap(True)
    # Write
    X.tofile(f)


# Write float as big-endian single-precision
def tofile_b4_f(f, x):
    """Write a float or array to single-precision big-endian file

    :Call:
        >>> tofile_b4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if le:
        X.byteswap(True)
    # Write
    X.tofile(f)
# > b4 write


# ====== b8 write ======================================================
# Write integer as big-endian double-precision
def tofile_b8_i(f, x):
    r"""Write an integer or array to double-precision big-endian file

    :Call:
        >>> tofile_b8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if le:
        X.byteswap(True)
    # Write
    X.tofile(f)


# Write float as big-endian double-precision
def tofile_b8_f(f, x):
    r"""Write a float or array to double-precision big-endian file

    :Call:
        >>> tofile_b4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if le:
        X.byteswap(True)
    # Write
    X.tofile(f)
# b8 write


# ====== ne4 write =====================================================
# Write integer as native-endian single-precision
def tofile_ne4_i(f, x):
    r"""Write an integer or array to single-precision native-endian file

    :Call:
        >>> tofile_ne4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Write
    X.tofile(f)


# Write float as big-endian single-precision
def tofile_ne4_f(f, x):
    r"""Write a float or array to single-precision native-endian file

    :Call:
        >>> tofile_ne4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Write
    X.tofile(f)
# > ne4 write


# ====== ne8 write =====================================================
# Write integer as native-endian double-precision
def tofile_ne8_i(f, x):
    r"""Write an integer or array to double-precision native-endian file

    :Call:
        >>> tofile_ne8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Write
    X.tofile(f)


# Write float as native-endian double-precision
def tofile_ne8_f(f, x):
    r"""Write a float or array to double-precision native-endian file

    :Call:
        >>> tofile_ne8_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Write
    X.tofile(f)
# > ne8 write


# **********************************************************************
# ====== lr4 record ====================================================
# Write record of single-precision little-endian integers
def write_record_lr4_i(f, x):
    r"""Write Fortran :class:`int` record to little-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_lr4_i(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of single-precision little-endian floats
def write_record_lr4_f(f, x):
    r"""Write Fortran :class:`float` record to little-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_lr4_f(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)
# > write_record_lr4


# ====== lr8 record ====================================================
# Write record of double-precision little-endian integers
def write_record_lr8_i(f, x):
    r"""Write Fortran :class:`long` record to little-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_lr8_i(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision little-endian integers
def write_record_lr8_i2(f, x):
    r"""Write special Fortran :class:`long` record little-endian file

    The record markers are 8 bytes instead of 4.

    :Call:
        >>> write_record_lr8_i(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision little-endian floats
def write_record_lr8_f(f, x):
    r"""Write Fortran :class:`double` record to little-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_lr8_f(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision little-endian floats
def write_record_lr8_f2(f, x):
    r"""Write special Fortran :class:`double` record little-endian

    The record markers from this function are 8 bytes instead of 4.

    :Call:
        >>> write_record_lr8_f2(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if be:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)
# > write_record_lr8


# ====== r4 record ====================================================
# Write record of single-precision big-endian integers
def write_record_r4_i(f, x):
    r"""Write Fortran :class:`int` record to big-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_r4_i(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)

# Write record of single-precision big-endian floats
def write_record_r4_f(f, x):
    """Write Fortran :class:`float` record to big-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_r4_f(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)
# > write_record_r4


# ====== r8 record ====================================================
# Write record of double-precision big-endian integers
def write_record_r8_i(f, x):
    r"""Write Fortran :class:`long` record to big-endian file

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_r8_i(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision big-endian integers
def write_record_r8_i2(f, x):
    r"""Write special Fortran :class:`long` record big-endian

    The record markers are 8-byte :class:`int`.

    :Call:
        >>> write_record_r8_i2(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision big-endian floats
def write_record_r8_f(f, x):
    r"""Write Fortran :class:`double` record big-endian

    The record markers are 4-byte :class:`int`.

    :Call:
        >>> write_record_r8_f(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)


# Write record of double-precision big-endian floats
def write_record_r8_f2(f, x):
    """Write special Fortran :class:`double` record big-endian

    The record markers are 8-byte :class:`int`.

    :Call:
        >>> write_record_r8_f2(f, x)
    :Inputs:
        *f*: :class:`file`
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
    if le:
        X.byteswap(True)
        I.byteswap(True)
    # Write
    I.tofile(f)
    X.tofile(f)
    I.tofile(f)
# > write_record_r8


# ********************************************************************
# ====== lb4 read ====================================================
# Read integer from little-endian single-precision file
def fromfile_lb4_i(f, n):
    r"""Read *n* 4-byte :class:`int` little-endian

    :Call:
        >>> x = fromfile_lb4_i(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype="<i4")


# Read float from little-endian single-precision file
def fromfile_lb4_f(f, n):
    r"""Read *n* 4-byte :class:`float` little-endian

    :Call:
        >>> x = fromfile_lb4_f(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype="<f4")
# > lb4 read


# ====== lb8 read ====================================================
# Read integer from little-endian double-precision file
def fromfile_lb8_i(f, n):
    r"""Read *n* 8-byte :class:`int` little-endian

    :Call:
        >>> x = fromfile_lb8_i(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype="<i8")


# Read float from little-endian double-precision file
def fromfile_lb8_f(f, n):
    r"""Read *n* 8-byte :class:`float` little-endian

    :Call:
        >>> x = fromfile_lb8_f(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype="<f8")
# > lb4 read


# ====== b4 read ====================================================
# Read integer from big-endian single-precision file
def fromfile_b4_i(f, n):
    r"""Read *n* 4-byte :class:`int` big-endian

    :Call:
        >>> x = fromfile_b4_i(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype=">i4")


# Read float from big-endian single-precision file
def fromfile_b4_f(f, n):
    r"""Read *n* 4-byte :class:`float` big-endian

    :Call:
        >>> x = fromfile_b4_f(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype=">f4")
# > b4 read


# ====== b8 read ====================================================
# Read integer from big-endian double-precision file
def fromfile_b8_i(f, n):
    r"""Read *n* 8-byte :class:`int` big-endian

    :Call:
        >>> x = fromfile_b8_i(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype=">i8")


# Read float from big-endian double-precision file
def fromfile_b8_f(f, n):
    r"""Read *n* 8-byte :class:`float` big-endian

    :Call:
        >>> x = fromfile_b8_f(f, n)
    :Inputs:
        *f*: :class:`file`
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
    return np.fromfile(f, count=n, dtype=">f8")
# > b8 read


# **********************************************************************
# ====== lr4 record ====================================================
# Read record of single-precision little-endian integers
def read_record_lr4_i(f):
    r"""Read 4-byte little-endian :class:`int` record

    :Call:
        >>> x = read_record_lr4_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i4')
    # Get count
    n = int(I[0] / 4)
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<i4")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of single-precision little-endian integers
def read_record_lr4_f(f):
    r"""Read 4-byte little-endian :class:`float` record

    :Call:
        >>> x = read_record_lr4_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i4')
    # Get count
    n = I[0] / 4
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<f4")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Check record marks, lr4
def check_record_lr4(f):
    r"""Check for consistent ``lr4`` record based on record markers

    :Call:
        >>> q = check_record_lr4(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *f* has a valid record in the next position
    :Version:
        * 2018-01-11 ``@ddalle``: Version 1.0
    """
    # Save position
    p = f.tell()
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Check for end of file
    if len(I) == 0 or I[0] <= 0:
        f.seek(p)
        return False
    # Skip to end-of-record
    f.seek(I[0], 1)
    # Get new position
    p1 = f.tell()
    # Check for successful seek
    if p1 != p + 4 + I[0]:
        f.seek(p)
        return False
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i4")
    # Return to original position
    f.seek(p)
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # End of record does not match
        return False
    else:
        # Valid record
        return True
# > read_record_lr4


# ====== lr8 record ====================================================
# Read record of double-precision little-endian integers
def read_record_lr8_i(f):
    r"""Read 8-byte little-endian :class:`int` record

    :Call:
        >>> x = read_record_lr8_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<i8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_f(f):
    r"""Read 8-byte little-endian :class:`float` record

    :Call:
        >>> x = read_record_lr8_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='f8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<f8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_i2(f):
    r"""Read 8-byte little-endian :class:`int` record

    with 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_lr8_i2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i8")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<i8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i8")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision little-endian integers
def read_record_lr8_f2(f):
    r"""Read 8-byte little-endian :class:`float` record

    with 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_lr8_f2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i8")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype="<f8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i8")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Check record marks, lr4
def check_record_lr8(f):
    r"""Check for a consistent record by reading record markers only

    :Call:
        >>> q = check_record_lr8(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *f* has a valid record in the next position
    :Version:
        * 2018-01-11 ``@ddalle``: Version 1.0
    """
    # Save position
    p = f.tell()
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i8")
    # Check for end of file
    if len(I) == 0 or I[0] <= 0:
        f.seek(p)
        return False
    # Skip to end-of-record
    f.seek(I[0], 1)
    # Get new position
    p1 = f.tell()
    # Check for successful seek
    if p1 != p + 8 + I[0]:
        f.seek(p)
        return False
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype="<i8")
    # Return to original position
    f.seek(p)
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # End of record does not match
        return False
    else:
        # Valid record
        return True
# > read_record_lr8


# ====== r4 record ====================================================
# Read record of single-precision big-endian integers
def read_record_r4_i(f):
    r"""Read 4-byte big-endian :class:`int` record

    :Call:
        >>> x = read_record_r4_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i4')
    # Get count
    n = I[0] / 4
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">i4")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of single-precision big-endian integers
def read_record_r4_f(f):
    r"""Read 4-byte big-endian :class:`float` record

    :Call:
        >>> x = read_record_r4_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i4')
    # Get count
    n = I[0] / 4
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">f4")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Check record marks, lr4
def check_record_r4(f):
    r"""Check for a consistent record by reading record markers only

    :Call:
        >>> q = check_record_r4(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *f* has a valid record in the next position
    :Version:
        * 2018-01-11 ``@ddalle``: Version 1.0
    """
    # Save position
    p = f.tell()
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i4")
    # Check for end of file
    if len(I) == 0 or I[0] <= 0:
        f.seek(p)
        return False
    # Skip to end-of-record
    f.seek(I[0], 1)
    # Get new position
    p1 = f.tell()
    # Check for successful seek
    if p1 != p + 4 + I[0]:
        f.seek(p)
        return False
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i4")
    # Return to original position
    f.seek(p)
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # End of record does not match
        return False
    else:
        # Valid record
        return True
# > read_record_r4


# ====== r8 record ====================================================
# Read record of double-precision big-endian integers
def read_record_r8_i(f):
    r"""Read 8-byte big-endian :class:`int` record

    :Call:
        >>> x = read_record_r8_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">i8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_f(f):
    r"""Read 8-byte big-endian :class:`float` record

    :Call:
        >>> x = read_record_r8_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='f8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">f8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i4")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_i2(f):
    r"""Read 8-byte big-endian :class:`int` record

    using 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_r8_i2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`int`]
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i8")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">i8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i8")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Read record of double-precision big-endian integers
def read_record_r8_f2(f):
    r"""Read 8-byte big-endian :class:`float` record

    using 8-byte :class:`int` record markers

    :Call:
        >>> x = read_record_r8_f2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray`\ [:class:`float`]
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: Version 1.0
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i8")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i8')
    # Get count
    n = I[0] / 8
    # Read that many ints
    x = np.fromfile(f, count=n, dtype=">f8")
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i8")
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # Consistency
        raise IOError("End-of-record marker does not match start")
    # Output
    return x


# Check record marks, r8
def check_record_r8(f):
    r"""Check for a consistent record by reading record markers only

    :Call:
        >>> q = check_record_lr8(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'rb' or similar
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *f* has a valid record in the next position
    :Version:
        * 2018-01-11 ``@ddalle``: Version 1.0
    """
    # Save position
    p = f.tell()
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype=">i8")
    # Check for end of file
    if len(I) == 0 or I[0] <= 0:
        f.seek(p)
        return False
    # Skip to end-of-record
    f.seek(I[0], 1)
    # Get new position
    p1 = f.tell()
    # Check for successful seek
    if p1 != p + 8 + I[0]:
        f.seek(p)
        return False
    # Read the end-of-record
    J = np.fromfile(f, count=1, dtype=">i8")
    # Return to original position
    f.seek(p)
    # Check for errors
    if len(J)==0 or I[0]!=J[0]:
        # End of record does not match
        return False
    else:
        # Valid record
        return True
# > read_record_r8
