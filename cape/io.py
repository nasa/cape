"""
Common Input/Output Library
===========================

"""

# Numerics
import numpy as np
# Needed for byte order flags
import os

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
    """Get the file type by trying to read first line using various methods
    
    :Call:
        >>> ft = get_filetype(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
    :Outputs:
        *ft*: "|" | "<4" | ">4" | "<8" | ">8"
            File type: ASCII (``"|"``), little-endian single (``"<4"``),
            little-endian double (``"<8"``), big-endian single (``">4"``), or
            big-endian double (``">8"``)
    :Versions:
        * 2016-09-04 ``@ddalle``: First version
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

# Read string
def read_c_str(f, nmax=1000):
    """Read a C-style string from a binary file
    
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
        * 2016-11-14 ``@ddalle``: First version
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
    

# Write integer as little-endian single-precision
def tofile_lb4_i(f, x):
    """Write an integer or array to single-precision little-endian file
    
    :Call:
        >>> tofile_lb4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if be: X.byteswap(True)
    # Write
    X.tofile(f)

# Write float as little-endian single-precision
def tofile_lb4_f(f, x):
    """Write a float or array to single-precision little-endian file
    
    :Call:
        >>> tofile_lb4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if be: X.byteswap(True)
    # Write
    X.tofile(f)

# Write integer as little-endian double-precision
def tofile_lb8_i(f, x):
    """Write an integer or array to double-precision little-endian file
    
    :Call:
        >>> tofile_lb8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if be: X.byteswap(True)
    # Write
    X.tofile(f)

# Write float as little-endian double-precision
def tofile_lb8_f(f, x):
    """Write a float or array to double-precision little-endian file
    
    :Call:
        >>> tofile_lb4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if be: X.byteswap(True)
    # Write
    X.tofile(f)
    
# Write integer as big-endian single-precision
def tofile_b4_i(f, x):
    """Write an integer or array to single-precision big-endian file
    
    :Call:
        >>> tofile_b4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='i4')
    # Check byte order
    if le: X.byteswap(True)
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
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='f4')
    # Check byte order
    if le: X.byteswap(True)
    # Write
    X.tofile(f)

# Write integer as big-endian double-precision
def tofile_b8_i(f, x):
    """Write an integer or array to double-precision big-endian file
    
    :Call:
        >>> tofile_b8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='i8')
    # Check byte order
    if le: X.byteswap(True)
    # Write
    X.tofile(f)

# Write float as big-endian double-precision
def tofile_b8_f(f, x):
    """Write a float or array to double-precision big-endian file
    
    :Call:
        >>> tofile_b4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            Float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Ensure array
    X = np.array(x, dtype='f8')
    # Check byte order
    if le: X.byteswap(True)
    # Write
    X.tofile(f)
    
    
# Write record of single-precision little-endian integers
def write_record_lb4_i(f, x):
    """Write record of integers to Fortran single-precision little-endian file
    
    :Call:
        >>> write_record_lb4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_lb4_f(f, x):
    """Write record of floats to Fortran single-precision little-endian file
    
    :Call:
        >>> write_record_lb4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Write record of double-precision little-endian integers
def write_record_lb8_i(f, x):
    """Write record of integers to Fortran double-precision little-endian file
    
    :Call:
        >>> write_record_lb8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_lb8_f(f, x):
    """Write record of floats to Fortran double-precision little-endian file
    
    Record markers are written as single
    
    :Call:
        >>> write_record_lb8_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_lb8_f2(f, x):
    """Write record of floats to Fortran double-precision little-endian file
    
    :Call:
        >>> write_record_lb8_f2(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Write record of single-precision big-endian integers
def write_record_b4_i(f, x):
    """Write record of integers to Fortran single-precision big-endian file
    
    :Call:
        >>> write_record_b4_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_b4_f(f, x):
    """Write record of floats to Fortran single-precision big-endian file
    
    :Call:
        >>> write_record_b4_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Write record of double-precision big-endian integers
def write_record_b8_i(f, x):
    """Write record of integers to Fortran double-precision big-endian file
    
    :Call:
        >>> write_record_b8_i(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`int` | :class:`np.ndarray`
            Integer or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_b8_f(f, x):
    """Write record of floats to Fortran double-precision big-endian file
    
    Record markers written as single
    
    :Call:
        >>> write_record_b8_f(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
def write_record_b8_f2(f, x):
    """Write record of floats to Fortran double-precision big-endian file
    
    :Call:
        >>> write_record_b8_f2(f, x)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *x*: :class:`float` | :class:`np.ndarray`
            float or array to write to file
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
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
    
    
    
# Read integer from little-endian single-precision file
def fromfile_lb4_i(f, n):
    """Read *n* integers from single-precision little-endian file
    
    :Call:
        >>> x = fromfile_lb4_i(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype="<i4")
    
# Read float from little-endian single-precision file
def fromfile_lb4_f(f, n):
    """Read *n* floats from single-precision little-endian file
    
    :Call:
        >>> x = fromfile_lb4_f(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype="<f4")
    
# Read integer from little-endian double-precision file
def fromfile_lb8_i(f, n):
    """Read *n* integers from double-precision little-endian file
    
    :Call:
        >>> x = fromfile_lb8_i(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype="<i8")
    
# Read float from little-endian double-precision file
def fromfile_lb8_f(f, n):
    """Read *n* floats from double-precision little-endian file
    
    :Call:
        >>> x = fromfile_lb8_f(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype="<f8")
    
# Read integer from big-endian single-precision file
def fromfile_b4_i(f, n):
    """Read *n* integers from single-precision big-endian file
    
    :Call:
        >>> x = fromfile_b4_i(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype=">i4")
    
# Read float from big-endian single-precision file
def fromfile_b4_f(f, n):
    """Read *n* floats from single-precision big-endian file
    
    :Call:
        >>> x = fromfile_b4_f(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype=">f4")
    
# Read integer from big-endian double-precision file
def fromfile_b8_i(f, n):
    """Read *n* integers from double-precision big-endian file
    
    :Call:
        >>> x = fromfile_b8_i(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of *n* integers if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype=">i8")
    
# Read float from big-endian double-precision file
def fromfile_b8_f(f, n):
    """Read *n* floats from double-precision big-endian file
    
    :Call:
        >>> x = fromfile_b8_f(f, n)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
        *n*: :class:`int`
            Number of integers to read
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of *n* floats if possible
    :Versions:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read from file
    return np.fromfile(f, count=n, dtype=">f8")
    
    
# Read record of single-precision little-endian integers
def read_record_lb4_i(f):
    """Read next record from single-precision little-endian file as integers
    
    :Call:
        >>> x = read_record_lb4_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: First version
    """
    # Read start-of-record marker
    I = np.fromfile(f, count=1, dtype="<i4")
    # Process count
    if len(I) == 0 or I[0] == 0:
        return np.array([], dtype='i4')
    # Get count
    n = I[0] / 4
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
def read_record_lb4_f(f):
    """Read next record from single-precision little-endian file as floats
    
    :Call:
        >>> x = read_record_lb4_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Read record of double-precision little-endian integers
def read_record_lb8_i(f):
    """Read next record from double-precision little-endian file as integers
    
    :Call:
        >>> x = read_record_lb8_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
def read_record_lb8_f(f):
    """Read next record from double-precision little-endian file as floats
    
    Record marker read as a single
    
    :Call:
        >>> x = read_record_lb8_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
def read_record_lb8_f2(f):
    """Read next record from double-precision little-endian file as floats
    
    :Call:
        >>> x = read_record_lb8_f2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Read record of single-precision big-endian integers
def read_record_b4_i(f):
    """Read next record from single-precision big-endian file as integers
    
    :Call:
        >>> x = read_record_b4_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
def read_record_b4_f(f):
    """Read next record from single-precision big-endian file as floats
    
    :Call:
        >>> x = read_record_b4_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
    
# Read record of double-precision big-endian integers
def read_record_b8_i(f):
    """Read next record from double-precision big-endian file as integers
    
    :Call:
        >>> x = read_record_b8_i(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`int`)
            Array of integers
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
def read_record_b8_f(f):
    """Read next record from double-precision big-endian file as floats
    
    :Call:
        >>> x = read_record_b8_f(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
def read_record_b8_f2(f):
    """Read next record from double-precision big-endian file as floats
    
    :Call:
        >>> x = read_record_b8_f2(f)
    :Inputs:
        *f*: :class:`file`
            File handle, open 'wb' or similar
    :Outputs:
        *x*: :class:`np.ndarray` (:class:`float`)
            Array of floats
    :Version:
        * 2016-09-05 ``@ddalle``: First version
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
    