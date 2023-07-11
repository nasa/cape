#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`optdict.optitem`: Tools to access items from option lists
==================================================================

This module provides several utilities for accessing items from a list
or a so-called "psuedo-list."

For example, if the value of the option is just ``"name"``, this is
considered to be a list with ``"name"`` repeating infinitely many times.
If an option is ``[1, 7]``, this is interpreted as either ``1`` and
``7`` alternating back and forth or ``1`` followed by ``7`` arbitrarily
many times (depending on which access function is used).
"""

# Standard library
import re

# Third-party
import numpy as np

# Local imports
from .opterror import (
    OptdictExprError,
    OptdictKeyError,
    OptdictTypeError,
    assert_isinstance)


# Defaults
DEFAULT_LISTDEPTH = 0

# Options
SPECIAL_DICT_KEYS = (
    "key",
    "lisdepth",
    "ring",
)
LOOKUP_METHODS = (
    "@expr",
    "@cons",
    "@map",
    "@raw",
)

# Types
ARRAY_TYPES = (
    list,
    tuple,
    np.ndarray)
FLOAT_TYPES = (
    float,
    np.float16,
    np.float32,
    np.float64,
    np.float128)
INT_TYPES = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64)
BOOL_TYPES = (
    bool,
    np.bool_)

# Regular expresstions
REGEX_VAR = re.compile(r"\$([A-Za-z_]+)")


def getel(v, j=None, **kw):
    r"""Return the *j*\ th element of an array if possible

    :Call:
        >>> vj = getel(v, j=None, **kw)
    :Inputs:
        *v*: scalar | :class:`list`
            A scalar or list of items
        *j*: {``None``} | :class:`int`
            Phase index; use ``None`` to just return *v*
        *x*, *values*: {``None``} | :class:`dict`
            Reference conditions to use with ``@expr``, ``@map``, etc.;
            often a run matrix; used in combination with *i*
        *i*, *index*: {``None``} | :class:`int` | :class:`np.ndarray`
            Case index or indices to use with ``@expr``, ``@map``, etc.
        *ring*: ``True`` | {``False``}
            If ``True``, then use ``x[j % len(x)]``; if ``False``,
            use ``x[-1]`` if ``j > len(x)``
        *listdepth*: {``0``} | :class:`int` > 0
            Depth of list to treat as a "scalar;" e.g. if *listdepth*
            is ``1``, then ``[3, 4]`` is a "scalar", but if *listdepth*
            is ``0``, then ``[3, 4]`` is not a scalar
    :Outputs:
        *xj*: :class:`object`
            * ``x`` if *j* is ``None``
            * ``x`` if *x* is scalar (see :func:`check_scalar`)
            * ``None`` if ``x==[]``
            * ``x[j % len(x)]`` if *ring*
            * ``x[j]`` if possible
            * ``x[-1]`` if *j* is greater than ``len(x)-1``
            * ``x[0]`` if *j* is less than ``-len(x)``
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
        >>> getel({"@expr": "$a"}, x={"a": [1, 2]})
        [1, 2]
    :Versions:
        * 2015-12-08 ``@ddalle``: Version 1.0
        * 2019-03-02 ``@ddalle``: Version 1.1; in :mod:`cape.tnakit`
        * 2021-12-15 ``@ddalle``: Version 1.2; in :mod:`optdict`
        * 2021-12-16 ``@ddalle``: Version 2.0; add *ring*, *listdepth*
        * 2022-09-12 ``@ddalle``: Version 3.0; add @map, @expr, @cons
    """
    # Check if *v* is a dict
    if not isinstance(v, dict):
        # Sample phasing
        vj = _getel_phase(v, j=j, **kw)
        # Check again
        if not isinstance(vj, dict):
            return vj
        # Continue using v[j] instead of v
        v = vj
    # Check for special keys
    if "@expr" in v:
        # String expression in Python code
        xtype = "expr"
        vd = v["@expr"]
    elif "@map" in v:
        # Map based on a special key
        xtype = "map"
        vd = v["@map"]
    elif "@cons" in v:
        # Dictionary of constraints
        xtype = "cons"
        vd = v["@cons"]
    elif "@raw" in v:
        # Raw
        return v["@raw"]
    else:
        # General dictionary
        return _getel_phase(v, j=j, **kw)
    # Loop through keys
    for k in v:
        # Check if recognized method
        if k in LOOKUP_METHODS:
            continue
        # Check for unrecognized key
        if k not in SPECIAL_DICT_KEYS:
            raise OptdictKeyError(
                "Unsupported key '%s' for parameter of type '%s'" % (k, xtype))
        # Set value
        kw[k] = v[k]
    # Call appropriate method
    if xtype == "expr":
        vj = _getel_expr(vd, j=j, **kw)
    elif xtype == "map":
        vj = _getel_map(vd, j=j, **kw)
    elif xtype == "cons":
        vj = _getel_constraints(vd, j=j, **kw)
    # Recurse if needed
    if isinstance(vj, dict):
        # Recurse if two special instructions are used
        return getel(vj, j=j, **kw)
    else:
        # Don't recurse lists
        return vj


def _getel_phase(v, j=None, **kw):
    # Check phase input
    j = _check_phase(j)
    # Check null input
    if j is None:
        return v
    # Options
    ring = kw.get("ring", False)
    listdepth = kw.get("listdepth", DEFAULT_LISTDEPTH)
    # Check if *x* is a scalar
    if check_scalar(v, listdepth):
        # Can't use phase index
        return v
    # Get length of list
    L = len(v)
    # Check for empty input
    if L == 0:
        return None
    # Check *ring* (repeat) option
    if ring:
        return v[j % L]
    elif j >= L:
        # Repeat last entry
        return v[-1]
    elif j < -L:
        # Repeat first entry if indexing from back
        return v[0]
    else:
        # Simple case
        return v[j]


def _getel_expr(v, j=None, **kw):
    r"""Evaluate ``@expr`` expressions

    :Call:
        >>> vi = _getel_expr(v, j=None, **kw)
    :Inputs:
        *v*: :class:`str`
            String of Python expression to evaluate
        *values*, *x*: {``{}``} | :class:`dict`
            Dictionary of run matrix or other values to use
        *index*, *i*: :class:`int` | :class:`list` | :class:`np.ndarray`
            Case index if any *x* entries are non-scalar
    :Outputs:
        *vi*: **any**
            Result of evaluating *v* after substitutions
    :Versions:
        * 2022-09-11 ``@ddalle``: Version 1.0
    """
    # Apply phase
    vj = _getel_phase(v, j=j, **kw)
    # Check if *v* is a string
    assert_isinstance(vj, str, '"@expr" value')
    # Get values
    x = _get_kw_x(**kw)
    # Get case index
    i = _get_kw_i(**kw)
    # Find required values
    cols = REGEX_VAR.findall(vj)
    # Initialize relevant values
    __xi = {}
    # Process required columns
    for col in cols:
        # Sample from *x* if possible
        try:
            __xi[col] = _sample_x(x, col, i)
        except OptdictKeyError:
            raise OptdictKeyError(
                ('The "x" keyword for @expr "%s" ' % vj) +
                ('is missing key "%s"' % col))
    # Substitute e.g. "$mach" -> "xi[mach]"
    v2 = REGEX_VAR.sub(r'__xi["\1"]', vj)
    # Evaluate it
    try:
        # Handoff to avoid name conflicts
        vi = eval(v2)
    except Exception:
        raise OptdictExprError('@expr "%s" failed to evaluate' % v)
    # Output
    return vi


def _getel_map(v, j=None, **kw):
    # Get values
    x = _get_kw_x(**kw)
    # Get case index
    i = _get_kw_i(**kw)
    # Get key
    key = kw.get("key")
    # Check if present
    assert_isinstance(key, str, '"key" for a "@map"')
    # Look up value
    try:
        xk = _sample_x(x, key, i)
    except OptdictKeyError:
        raise OptdictKeyError(
            'The "x" keyword for @map is missing map key "%s"' % key)
    # Get the @map for phae *j*
    vj = _getel_phase(v, j=j)
    # Check type
    assert_isinstance(vj, dict, '"@map" value')
    # Check if *xk* is present
    if xk in vj:
        # Directly present
        return vj[xk]
    else:
        # Use default
        return vj.get("_default_")


def _getel_constraints(v, j=None, **kw):
    # Ensure *v* is a dict
    assert_isinstance(v, dict, '"@cons" value')
    # Loop through constraints (assume order works properly)
    for k, vk in v.items():
        # Evaluate the constraint
        qk = _getel_expr(k, **kw)
        # Check if met
        if qk:
            return vk


def _get_kw_x(**kw):
    # Get values
    x = kw.get("values", kw.get("x", {}))
    # Check type
    assert_isinstance(x, dict, 'keyword "x" or "values"')
    # Output
    return x


def _get_kw_i(**kw):
    # Get case index
    i = kw.get("index", kw.get("i"))
    # Check type
    if i is None:
        # Null is ok
        return i
    if isinstance(i, INT_TYPES):
        # Single int
        return i
    if isinstance(i, ARRAY_TYPES):
        # Check entries
        for j, ij in enumerate(i):
            assert_isinstance(ij, INT_TYPES, 'entry %i of "i" option' % j)
        # Output
        return i
    # Bad type
    assert_isinstance(i, (list, int), 'option "i" or "index"')


def _sample_x(x, col, i):
    # Check if present in inputs
    if col in x:
        # Get directly
        vcol = x[col]
    else:
        # Try to "get_values()" it
        try:
            vcol = x.get_values(col)
        except Exception:
            raise OptdictKeyError('The "x" keyword is missing key "%s"' % col)
    # Check if scalar
    if check_scalar(vcol, 0):
        # Already a scalar
        return vcol
    elif i is None:
        # Use all values
        return vcol
    elif check_scalar(i, 0):
        # Single case
        return vcol[i]
    elif isinstance(vcol, np.ndarray):
        # Subset of array
        if isinstance(i, tuple):
            return vcol[list(i)]
        else:
            return vcol[i]
    else:
        # Use multiple cases
        return vcol.__class__(
            [vcol[ii] for ii in i])


def setel(x, xj, j=None, listdepth=DEFAULT_LISTDEPTH):
    r"""Set the *j*\ th element of a list if possible

    :Call:
        >>> y = setel(x, xj, j=None, listdepth=0)
    :Inputs:
        *x*: :class:`list` | :class:`tuple` | :class:`object`
            A list or scalar object
        *j*: {``None``} | :class:`int`
            Phase index; use ``None`` to just return *x*
        *xj*: :class:`object`
            Value to set at ``x[j]`` if possible
        *listdepth*: {``0``} | :class:`int` > 0
            Depth of list to treat as a "scalar;" e.g. if *listdepth*
            is ``1``, then ``[3, 4]`` is a "scalar", but if *listdepth*
            is ``0``, then ``[3, 4]`` is not a scalar
    :Outputs:
        *y*: *xj* | :class:`list`
            Input *x* with ``y[j]`` set to *xj* unless *j* is ``None``
    :Examples:
        >>> setel(['a', 2, 'c'], 'b', 1)
        ['a', 'b', 'c']
        >>> setel(['a', 'b'], 'c', 2)
        ['a', 'b', 'c']
        >>> setel(['a', 'b'], 'c', j=3)
        ['a', 'b', 'b', 'c']
        >>> setel([0, 1], 'a')
        'a'
        >>> setel([0, 1], 'a', -4)
        ['a', 0, 0, 1]
        >>> setel([0, 1], 'a', j=1, listdepth=1)
        [[0, 1], 'a']
        >>> setel([0, 1], 'a', j=1, listdepth=2)
        [[[0, 1]], 'a']
    :Versions:
        * 2015-12-08 ``@ddalle``: Version 1.0; in :mod:`cape.tnakit`
        * 2021-12-17 ``@ddalle``: Version 2.0
    """
    # Check phase input
    j = _check_phase(j)
    # Check the index input
    if j is None:
        # Raw output
        return xj
    # Initialize output by handoff
    y = x
    # Check if *x* is a scalar
    while check_scalar(y, listdepth):
        # Can't use phase index
        y = [y]
    # Convert tuple to list
    if isinstance(x, tuple):
        y = list(x)
    # Current list length
    L = len(y)
    # Check if we are setting an element or appending it.
    if j >= L:
        # Append (j-L) copies of the last element
        if j > L:
            y.extend(y[-1:] * (j-L))
        # Append new *xj*
        y.append(xj)
    elif j < -L:
        # Prepend (L+j-1) copies of first element
        if j - 1 < -L:
            y = (y[:1] * -(L+j+1)) + y
        # Prepend new *xj*
        y.insert(0, xj)
    else:
        # Set the value
        y[j] = xj
    # Output
    return y


def check_array(v, listdepth=DEFAULT_LISTDEPTH):
    r"""Check if *v* is an array type to at least specified depth

    :Call:
        >>> q = check_array(v, listdepth=0)
    :Inputs:
        *v*: :class:`object`
            Any value
        *listdepth*: {``0``} | :class:`int` > 0
            Minimum number of array depth; if ``0``, always returns
            ``True``; if ``2``, then *v* must be a list of lists
    :Outputs:
        *q*: :class:`bool`
            Whether or not *v* is an array to specified depth
    :Versions:
        * 2023-01-25 ``@ddalle``: Version 1.0
    """
    # Hand off for zeroth level
    v0 = v
    # Loop through depth levels
    for i in range(listdepth):
        # Check if current depth is scalar
        if not isinstance(v0, ARRAY_TYPES):
            # Found scalar at depth *j*
            return False
        # Check for empty list
        if len(v0) == 0:
            # Can't check next level; ok if this is requested level
            return i + 1 >= listdepth
        # Otherwise get first element
        v0 = v0[0]
    # If we reached this point, still an array
    return True


def check_scalar(v, listdepth=DEFAULT_LISTDEPTH):
    r"""Check if *v* is a "scalar" to specified depth

    :Call:
        >>> q = check_scalar(v, listdepth=0)
    :Inputs:
        *v*: :class:`object`
            Any value
        *listdepth*: {``0``} | :class:`int` > 0
            Depth of list to treat as a "scalar;" e.g. if *listdepth*
            is ``1``, then ``[3, 4]`` is a "scalar", but if *listdepth*
            is ``0``, then ``[3, 4]`` is not a scalar
    :Outputs:
        *q*: :class:`bool`
            Whether or not *v* is a scalar to specified depth
    :Versions:
        * 2021-12-16 ``@ddalle``: Version 1.0
    """
    # Hand off for zeroth level
    v0 = v
    # Loop through depth levels
    for i in range(listdepth + 1):
        # Check if current depth is scalar
        if not isinstance(v0, ARRAY_TYPES):
            # Found scalar at depth *j*
            return True
        # Check for empty list
        if len(v0) == 0:
            # Can't check first element
            return i < listdepth
        # Otherwise get first element
        v0 = v0[0]
    # If we reached this point, not a scalar
    return False


# Assert array of specified list depth
def assert_array(obj, listdepth: int, desc=None):
    r"""Check that *obj* is an array to depth *listdpeth*

    :Call:
        >>> assert_array(obj, listdepth, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *listdepth*: :class:`int`
            Minimum number of levels of nested arrays required
        *desc*: {``None``} | :class:`str`
            Optional description for *obj* in case of failure
    :Raises:
        :class:`OptdictTypeError`
    :Versions:
        * 2023-01-25 ``@ddalle``: Version 1.0
    """
    # Check depth
    if check_array(obj, listdepth):
        # Success
        return
    # Default description
    if desc is None:
        desc = "Object"
    # Form message
    msg = "%s must be a nested array to depth at least %i" % (desc, listdepth)
    # Raise it
    raise OptdictTypeError(msg)


def _check_phase(j):
    # Check phase input
    if j is None:
        return
    if not isinstance(j, int):
        raise TypeError(
            "Expected 'int' for input 'j'; got '%s'" %
            type(j).__name__)
    return j

