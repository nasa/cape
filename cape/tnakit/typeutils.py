#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`tnakit.typeutils`: Convenient Type Checking Tools
=========================================================

This module contains convenience methods to check types of objects.   For the
most part, this is a set of predefined calls to :func:`isinstance`, but with
several groups of objects.  Furthermore, it contains several aspects that help
address differences between Python 2 and 3.  For example, it defines the
:class:`unicode` class for Python 3 by simply setting it equal to :class:`str`.

"""

# System modules
import io
import sys

# Import common types for checking
from numpy import ndarray

# Get a variable to hold the "type" of "module"
module = sys.__class__

# Other abbreviations
moduletype = sys.__class__
nonetype = type(None)

# Save major version
try:
    # Get the integer
    PY_MAJOR_VERSION = sys.version_info.major
    PY_MINOR_VERSION = sys.version_info.minor
except Exception:
    # Convert text
    PY_MAJOR_VERSION = int(sys.version[0])
    PY_MINOR_VERSION = int(sys.version.split('.')[1].split(' ')[0])

# Combined version
PY_VERSION = float(PY_MAJOR_VERSION) + 0.1*float(PY_MINOR_VERSION)

# Version checks
if PY_MAJOR_VERSION > 2:
    # Define classes that were deleted in Python 3
    unicode = str
    file    = io.IOBase

# Create tuples of types
if PY_MAJOR_VERSION > 2:
    # Categories of types for Python 3
    strlike = str
    intlike = int
    filelike = io.IOBase
else:
    # Categories of types for Python 2
    strlike = (str, unicode)
    intlike = (int, long)
    filelike = (file, io.IOBase)

# Universal type tuples
arraylike = (list, ndarray)


# Special version of :func:`isinstance` that allows None
def isinstancen(x, types):
    r"""Special version of :func:`isinstance` that allows ``None``

    :Call:
        >>> q = isinstance(x, types)
    :Inputs:
        *x*: :class:`any`
            Any value
        *types*: :class:`type` | :class:`tuple`\ [:class:`type`]
            Type specification as for :func:`isinstance`
    :Outputs:
        *q*: ``True`` | ``False``
            ``True`` if *x* is ``None`` or ``isinstance(x, types)``
    :Versions:
        * 2020-01-13 ``@ddalle``: First version
    """
    # Check for ``None``
    if x is None:
        return True
    # Otherwise call built-in
    return isinstance(x, types)


# Check for a string
def isstr(x):
    r"""Check if a variable is a string (or derivative)
    
    :Call:
        >>> q = isstr(x)
    :Inputs:
        *x*: :class:`any`
            Any variable
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *x* is of type :class:`str` or :class:`unicode` or
            any subclass thereof (and averting disaster caused by lack of
            :class:`unicode` class in Python 3+)
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Check version
    if PY_MAJOR_VERSION > 2:
        # Check just str
        return isinstance(x, str)
    else:
        # Check unicode as well
        return isinstance(x, (str, unicode))


# Check for a "list"
def isarray(x):
    r"""Check if a variable is a list or similar
    
    Accepted types are :class:`list`, :class:`numpy.ndarray`, :class:`tuple`,
    or any subclass thereof.
    
    :Call:
        >>> q = isarray(x)
    :Inputs:
        *x*: :class:`any`
            Any variable
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *x* is of type :class:`list`, :class:`tuple`,
            :class:`np.ndarray`, or any subclass of these three
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Check for traditional list-like things
    if isinstance(x, (list, tuple)):
        # List or tuple; array-like
        return True
    elif not isinstance(x, ndarray):
        # Not a numeric array
        return False
    else:
        # Check for more than 0 dimensions if NumPy array
        return x.ndim > 0


# Test if an object is a file
def isfile(f):
    r"""Check if an object is an instance of a file-like class

    This is complicated by the removal of the :class:`file` type from
    Python 3.  The basic class is :class:`io.IOBase`, but this is not a
    subclass of :class:`file` (or vice versa) in Python 2.

    :Call:
        >>> q = isfile(f)
    :Inputs:
        *f*: :class:`any`
            Any variable
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *f* is an instance of :class:`file`,
            :class:`io.IOBase`, or any subclass
    :Versions:
        * 2019-12-06 ``@ddalle``: First version
    """
    # Check version
    if PY_MAJOR_VERSION > 2:
        # Check just IOBase
        return isinstance(f, io.IOBase)
    else:
        # Check file as well
        return isinstance(f, (io.IOBase, file))
        

