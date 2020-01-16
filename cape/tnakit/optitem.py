#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.tnakit.optitem`: Option List Item Access Tools
==============================================================

This module provides several utilities for accessing items from a list
or a so-called "psuedo-list."

For example, if the value of the option is just ``"name"``, this is
considered to be a list with ``"name"`` repeating infinitely many times.
If an option is ``[1, 7]``, this is interpreted as either ``1`` and
``7`` alternating back and forth or ``1`` followed by ``7`` arbitrarily
many times (depending on which access function is used).
"""

# TNA toolkit modules
from . import typeutils


# Utility function to get elements from a list by phase
def getel(x, i=None):
    """Return the *i*th element of an array if possible
    
    :Call:
        >>> xi = getel(x)
        >>> xi = getel(x, i)
    :Inputs:
        *x*: scalar | :class:`list` | :class:`numpy.ndarray`
            A number or list or NumPy vector
        *i*: :class:`int` | ``None``
            Index; if ``None``, entire list is returned
    :Outputs:
        *xi*: scalar | list
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater than the
            length of *x*, ``x`` if *x* is not a :class:`list` or
            :class:`numpy.ndarray` instance, or ``x`` if *i* is ``None``
    :Examples:
        >>> getel('abc', 2)
        'abc'
        >>> getel(1.4, 0)
        1.4
        >>> getel([200, 100, 300], 1)
        100
        >>> getel([200, 100, 300], 15)
        300
        >>> getel([200, 100, 300])
        [200, 100, 300]
    :Versions:
        * 2015-12-08 ``@ddalle``: First version
        * 2019-03-02 ``@ddalle``: Moved to :mod:`tnakit`
    """
    # Check index input
    if i is None:
        return x
    elif not isinstance(i, int):
        raise TypeError("Index input has type '%s', must be int" % type(i))
    # Check for list or similar
    if typeutils.isarray(x):
        # Get length
        L = len(x)
        # Check for empty input.
        if L == 0:
            return None
        # Array-like
        if i:
            # Check the length.
            if i >= L:
                # Take the last element.
                return x[-1]
            else:
                # Take the *i*th element.
                return x[i]
        else:
            # Use the first element.
            return x[0]
    else:
        # Scalar
        return x
# def getel


# Get elements from a list where each item is expected to be a list
def getel_list(x, i=None):
    """Return *i*th element of array, where each element should be an array
    
    :Call:
        >>> xi = getel_list(x)
        >>> xi = getel_list(x, i)
    :Inputs:
        *x*: scalar | :class:`list` | :class:`numpy.ndarray`
            A number or list or NumPy vector
        *i*: :class:`int` | ``None``
            Index; if ``None``, entire list is returned
    :Outputs:
        *xi*: scalar | list
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater than the
            length of *x*, ``x`` if *x[0]* is not a :class:`list` or
            :class:`numpy.ndarray` instance, or ``x`` if *i* is ``None``
    :Examples:
        >>> getel_list('abc', 2)
        'abc'
        >>> getel_list(1.4, 0)
        1.4
        >>> getel_list([1.4], 0)
        [1.4]
        >>> getel_list([[1,4], [2,3], [-1,5]], 1)
        [2, 3]
        >>> getel_list([[1,4], [2,3], [-1,5]], 13)
        [-1, 5]
        >>> getel_list([[1,4], [2,3], [-1,5]], 14)
        [-1, 5]
    :See Also:
        * :func:`getel`
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Check for list or similar
    if typeutils.isarray(x):
        # Get length
        L = len(x)
        # If empty list; return value
        if L == 0:
            return x
        # Check first entry
        if typeutils.isarray(x[0]):
            # List of lists; use regular method
            return getel(x, i)
        else:
            # Just a list; for this function one level of list is a scalar
            return x
    else:
        # Scalar (should this be an error?)
        return x
# def gephasel_list


# Utility function to get elements sanely
def getringel(x, i=None):
    """Return the *i*th element of a "ring", cycling through if appropriate
    
    :Call:
        >>> xi = getringel(x)
        >>> xi = getringel(x, i)
    :Inputs:
        *x*: scalar | :class:`list` | :class:`np.ndarray` | :class:`tuple`
            A scalar or iterable list-like object
        *i*: :class:`int` | ``None``
            Index; if ``None``, entire list is returned
    :Outputs:
        *xi*: scalar | list
            Equal to ``x[mod(i,len(x)]`` unless *i* is ``None``
    :Examples:
        >>> getringel('abc', 2)
        'abc'
        >>> getringel(1.4, 0)
        1.4
        >>> getringel([200, 100, 300], 1)
        100
        >>> getringel([200, 100, 300], 14)
        300
        >>> getringel([200, 100, 300], 15)
        200
        >>> getringel([200, 100, 300])
        [200, 100, 300]
    :Versions:
        * 2019-03-03 ``@ddalle``: First version
    """
    # Check the index
    if i is None:
        return x
    elif not isinstance(i, int):
        raise TypeError("Index input has type '%s', must be int" % type(i))
    # Check for array
    if typeutils.isarray(x):
        # Length
        L = len(x)
        # Check for empty input.
        if L == 0:
            return None
        # Array-like; cycle through indices
        return x[i % L]
    else:
        # Scalar
        return x
# def getringel


# Get elements from a list where each item is expected to be a list
def getringel_list(x, i=None):
    """Return *i*th element of "ring", where each element should be an array
    
    :Call:
        >>> xi = getringel_list(x)
        >>> xi = getringel_list(x, i)
    :Inputs:
        *x*: scalar | :class:`list` | :class:`numpy.ndarray`
            A number or list or NumPy vector
        *i*: :class:`int` | ``None``
            Index; if ``None``, entire list is returned
    :Outputs:
        *xi*: scalar | list
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater than the
            length of *x*, ``x`` if *x[0]* is not a :class:`list` or
            :class:`numpy.ndarray` instance, or ``x`` if *i* is ``None``
    :Examples:
        >>> getringel_list('abc', 2)
        'abc'
        >>> getringel_list(1.4, 0)
        1.4
        >>> getringel_list([1.4], 0)
        [1.4]
        >>> getringel_list([[1,4], [2,3], [-1,5]], 1)
        [2, 3]
        >>> getringel_list([[1,4], [2,3], [-1,5]], 13)
        [2, 3]
        >>> getringel_list([[1,4], [2,3], [-1,5]], 14)
        [-1, 5]
    :See Also:
        * :func:`getel`
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Check for list or similar
    if typeutils.isarray(x):
        # Get length
        L = len(x)
        # If empty list; return value
        if L == 0:
            return x
        # Check first entry
        if typeutils.isarray(x[0]):
            # List of lists; use regular method
            return getringel(x, i)
        else:
            # Just a list; for this function we want the whole list
            return x
    else:
        # Scalar (should this be an error?)
        return x
# def gephasel_list
        

# Utility function to set elements sanely
def setel(x, i, xi):
    """Set the *i*th element of an array if possible
    
    :Call:
        >>> y = setel(x, i, xi)
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: :class:`int` | ``None``
            Index to set.  If *i* is ``None``, the output is reset to *xi*
        *xi*: scalar
            Value to set at scalar
    :Outputs:
        *y*: number-like or list-like
            Input *x* with ``y[i]`` set to ``xi`` unless *i* is ``None``
    :Examples:
        >>> setel(['a', 2, 'c'], 1, 'b')
        ['a', 'b', 'c']
        >>> setel(['a', 'b'], 2, 'c')
        ['a', 'b', 'c']
        >>> setel('a', 2, 'c')
        ['a', None, 'b']
        >>> setel([0, 1], None, 'a')
        'a'
    :Versions:
        * 2015-12-08 ``@ddalle``: First version
    """
    # Check the index input
    if i is None:
        # Raw output
        return xi
    elif not isinstance(i, int):
        raise TypeError("Index input has type '%s', must be int" % type(i))
    # Ensure list
    if typeutils.isarray(x):
        # Copy list or convert
        y = list(x)
    else:
        # Already a list
        y = x
    # Current list length
    L = len(y)
    # Make sure *y* is long enough
    if i > L+1:
        # Append (i-L) copies of the last element
        y += y[-1:]*(i-L-1)
    # Check if we are setting an element or appending it.
    if i > L:
        # Append
        y.append(xi)
    else:
        # Set the value.
        y[i] = xi
    # Output
    return y
# def setel
