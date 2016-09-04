"""
Common Input/Output Library
===========================

"""

# Numerics
import numpy as np
# Needed for byte order flags
import os


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
    
    
    