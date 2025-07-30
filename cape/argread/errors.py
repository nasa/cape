r"""
:mod:`argread.errors`: Error classes for the :mod:`argread` package
====================================================================

This module provides various error classes so that input errors can be
caught instead of raising general exceptions.
"""

# Standard library
from typing import Any, Optional, Union


# Basic error class
class ArgReadError(Exception):
    r"""Parent error class for :mod:`kwparse` errors

    Inherts from :class:`Exception` and enables general catching of all
    errors raised by :mod:`kwparse`
    """
    pass


# Key error
class ArgReadKeyError(KeyError, ArgReadError):
    r"""Errors for missing keys raised by :mod:`kwparse`"""
    pass


# Name error
class ArgReadNameError(NameError, ArgReadError):
    r"""Errors for incorrect names raised by :mod:`kwparse`"""
    pass


# Type error
class ArgReadTypeError(TypeError, ArgReadError):
    r"""Errors for incorrect type raised by :mod:`kwparse`"""
    pass


# Value error
class ArgReadValueError(ValueError, ArgReadError):
    r"""Errors for invalid values raised by :mod:`kwparse`"""
    pass


# Assert type of a variable
def assert_isinstance(
        obj: Any,
        cls_or_tuple: Union[type, tuple],
        desc: Optional[str] = None):
    r"""Conveniently check types

    Applies ``isinstance(obj, cls_or_tuple)`` but also constructs
    a :class:`TypeError` and appropriate message if test fails.

    If *cls* is ``None``, no checks are performed.

    :Call:
        >>> assert_isinstance(obj, cls, desc=None)
        >>> assert_isinstance(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: ``None`` | :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Raises:
        :class:`ArgReadTypeError`
    """
    # Special case for ``None``
    if cls_or_tuple is None or obj is None:
        return
    # Check for passed test
    if isinstance(obj, cls_or_tuple):
        return
    # Generate type error message
    msg = _genr8_type_error(obj, cls_or_tuple, desc)
    # Raise
    raise ArgReadTypeError(msg)


# Create error message for type errors
def _genr8_type_error(
        obj: Any,
        cls_or_tuple, desc: Optional[str] = None):
    r"""Create error message for type-check commands

    :Call:
        >>> msg = _genr8_type_error(obj, cls, desc=None)
        >>> msg = _genr8_type_error(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: ``None`` | :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Optional text describing *obj* for including in error msg
    :Outputs:
        *msg*: :class:`str`
            Text of an error message explaining available types
    """
    # Check for single type
    if isinstance(cls_or_tuple, tuple):
        # Multiple types
        names = [cls.__name__ for cls in cls_or_tuple]
    else:
        # Single type
        names = [cls_or_tuple.__name__]
    # Create error message
    msg1 = ""
    if desc:
        msg1 = f"{desc}: "
    msg2 = "got type '%s'; " % type(obj).__name__
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3

