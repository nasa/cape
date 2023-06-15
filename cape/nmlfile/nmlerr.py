r"""
Errors for :class:`NmlFile` data

This module provides a collection of error types for reading Fortran
namelist files. It also contains functions that utilize these exception
classes efficiently.
"""

# Standard library
import re


# Basic error family
class NmlError(Exception):
    r"""Parent error class for :mod:`nmlfile`

    Inherits from :class:`Exception`
    """
    pass


class NmlTypeError(TypeError, NmlError):
    r"""Type errors for :class:`NmlFile`

    Inherits from :class:`TypeError`
    """
    pass


class NmlValueError(ValueError, NmlError):
    r"""Value errors for :class:`NmlFile`

    Inherits from :class:`ValueError`
    """
    pass


def assert_isinstance(obj, cls_or_tuple, desc=None):
    r"""Conveniently check types

    Applies ``isinstance(obj, cls_or_tuple)`` but also constructs and
    raises a useful :class:`NmlTypeError` if test failes

    :Call:
        >>> assert_isinstance(obj, cls, desc=None)
        >>> assert_isinstance(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: :class:`type`
            Single allowed class for *obj*
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Typoe of allowed classes
    :Raises:
        :class:`NmlTypeError`
    :Versions:
        * 2023-06-06 ``@ddalle``: v1.0
    """
    # Special case for ``None``
    if cls_or_tuple is None:
        # No checks
        return
    # Normal check
    if isinstance(obj, cls_or_tuple):
        return
    # Generate type error message
    msg = _genr8_type_error(obj, cls_or_tuple, desc)
    # Raise
    raise NmlTypeError(msg)


def assert_nextchar(c: str, chars: str, desc=None):
    # Check if *c* is allowed
    if c in chars:
        return
    # Create text of options
    charlist = (f"'{c}'" for c in chars)
    # Combine
    msg2 = ' or '.join(charlist)
    # Create error message
    if desc is None:
        # Generic message
        msg1 = "Expected next char(s): "
    else:
        # User-provided message
        msg1 = f"After {desc} expected: "
    # Show what we got
    msg3 = f"; got '{c}'"
    # Raise an exception
    raise NmlValueError(msg1 + msg2 + msg3)


def assert_regex(c: str, regex: re.Pattern, desc=None):
    # Check if *c* is allowed
    if regex.fullmatch(c):
        return
    # Combine
    msg2 = f"'{regex.pattern}'"
    # Create error message
    if desc is None:
        # Generic message
        msg1 = "Regex for next char: "
    else:
        # User-provided message
        msg1 = f"After {desc} expected to match: "
    # Show what we got
    msg3 = f"; got '{c}'"
    # Raise an exception
    raise NmlValueError(msg1 + msg2 + msg3)


# Create error message for type errors
def _genr8_type_error(obj, cls_or_tuple, desc=None):
    # Check for single type
    if isinstance(cls_or_tuple, tuple):
        # Multiple types
        names = [cls.__name__ for cls in cls_or_tuple]
    else:
        # Single type
        names = [cls_or_tuple.__name__]
    # Create error message
    if desc is None:
        # No description; *msg2* starts with "Got type"
        msg1 = "G"
    else:
        # Add description
        msg1 = "For %s: g" % desc
    # Rest of message
    msg2 = "ot type '%s'; " % type(obj).__name__
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3

