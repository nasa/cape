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

# Version checks
if int(sys.version[0]) > 2:
    # Define classes that were deleted in Python 3
    unicode = str
    file    = io.IOBase
# endif


# Check for a string
def isstr(x):
    """Check if a variable is a string (or derivative)
    
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
    return isinstance(x, (str, unicode))


# Check for a "list"
def isarray(x):
    """Check if a variable is a list or similar
    
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
# def isarray
