#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`tnakit.kwutils`: Tools for Processing Keyword Arguments
===============================================================

This module contains methods to process keyword argument dictionaries
by checking them against 

    * a list of acceptable names
    * a dictionary of alternate names
    * a dictionary of acceptable types
    * a dictionary of other keys required for any key with dependencies

"""

# Standard library
import difflib
import warnings


# Map keywords
def map_kw(self, kwmap, **kw):
    r"""Map alternate keyword names with no checks

    :Call:
        >>> kwo = map_kw(kwmap, **kw)
    :Inputs:
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Translated keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version
    """
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Save it
        kwo[k] = v
    # Output
    return kwo


# Check valid keyword names, with dependencies
def check_kw(self, kwlist, kwmap, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    :Call:
        >>> kwo = _check_kw(kwlist, kwmap, kwdep, mode, **kw)
    :Inputs:
        *kwlist*: {*db._kw*} | :class:`list`\ [:class:`str`]
            List of acceptable parameters
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kwdep*: {*db._kw_depends*} | :class:`dict`\ [:class:`list`]
            Dictionary of required parameters for some parameters
        *mode*: ``0`` | {``1``} | ``2``
            Flag for quiet (``0``), warn (``1``), or strict (``2``)
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Valid keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version
    """
    # Check mode
    if mode not in [0, 1, 2]:
        raise ValueError("Verbose mode must be 0, 1, or 2")
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Check if present
        if k not in kwlist:
            # Get closet match (n=3 max)
            mtchs = difflib.get_close_matches(k, kwlist)
            # Issue warning
            if len(mtchs) == 0:
                # No suggestions
                msg = "Unrecognized keyword '%s'" % k
            else:
                # Show up to three suggestions
                msg = (
                    ("Unrecognized keyword '%s'" % k) +
                    ("; suggested: %s" % " ".join(mtchs)))
            # Choose warning
            if mode == 2:
                # Exception
                raise KeyError(msg)
            elif mode == 1:
                # Warning
                warnings.warn(msg, UserWarning)
            # Go to next keyword
            continue
        else:
            # Copy to output
            kwo[k] = v
        # Check dependences
        if k in kwdep:
            # Get item
            kdep = kwdep[k]
            # Check if any dependency is present
            if all([ki not in kw for ki in kdep]):
                # Create warning message
                msg = (
                    ("Keyword '%s' depends on one of " % k) +
                    ("the following: %s" % " ".join(kdep)))
                # Choose what to do about it
                if mode == 2:
                    # Exception
                    raise KeyError(msg)
                elif mode == 1:
                    # Warning
                    warnings.warn(msg, UserWarning)
    # Output
    return kwo


# Check valid keyword names, with dependencies
def check_kw_types(kwlist, kwmap, kwtypes, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    :Call:
        >>> kwo = check_kw_types(
            kwlist, kwmap, kwtypes, kwdep, mode, **kw)
    :Inputs:
        *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
            Data file interface
        *kwlist*: {*db._kw*} | :class:`list`\ [:class:`str`]
            List of acceptable parameters
        *kwtypes*: {*db._kw_types*} | :class:`dict`
            Dictionary of :class:`type` or
            :class:`tuple`\ [:class:`type`] for some or all
            keywords, used with :func:`isinstance`
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kwdep*: {*db._kw_depends*} | :class:`dict`\ [:class:`list`]
            Dictionary of required parameters for some parameters
        *mode*: ``0`` | {``1``} | ``2``
            Flag for quiet (``0``), warn (``1``), or strict (``2``)
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Valid keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version
    """
    # Check mode
    if mode not in [0, 1, 2]:
        raise ValueError("Verbose mode must be 0, 1, or 2")
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Check if present
        if k not in kwlist:
            # Get closet match (n=3 max)
            mtchs = difflib.get_close_matches(k, kwlist)
            # Issue warning
            if len(mtchs) == 0:
                # No suggestions
                msg = "Unrecognized keyword '%s'" % k
            else:
                # Show up to three suggestions
                msg = (
                    ("Unrecognized keyword '%s'" % k) +
                    ("; suggested: %s" % " ".join(mtchs)))
            # Choose warning
            if mode == 2:
                # Exception
                raise KeyError(msg)
            elif mode == 1:
                # Warning
                warnings.warn(msg, UserWarning)
            # Go to next keyword
            continue
        # Check for a type
        ktype = kwtypes.get(k, object)
        # Check the type
        if isinstance(v, ktype):
            # Save the value and move on
            kwo[k] = v
        else:
            # Create warning message
            msg = (
                ("Invalid type for keyword '%s'" % k) +
                ("; options are %s" % ktype))
            # Check mode
            if mode == 2:
                # Exception
                raise TypeError(msg)
            elif mode == 1:
                # Warning
                warnings.warn(msg, UserWarning)
        # Check dependences
        if k in kwdep:
            # Get item
            kdep = kwdep[k]
            # Check if any dependency is present
            if all([ki not in kw for ki in kdep]):
                # Create warning message
                msg = (
                    ("Keyword '%s' depends on one of " % k) +
                    ("the following: %s" % " ".join(kdep)))
                # Choose what to do about it
                if mode == 2:
                    # Exception
                    raise KeyError(msg)
                elif mode == 1:
                    # Warning
                    warnings.warn(msg, UserWarning)
    # Output
    return kwo
