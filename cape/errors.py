r"""
:mod:`cape.errors`: Customized exceptions for CAPE
=======================================================

This module provides CAPE-specific exceptions and methods to perform
standardized checks throughout :mod:`cape`. One purpose of creating
customized exceptions is to differentiate CAPE bugs and exceptions
raised by CAPE intentionally due to problems with user input or errors
detected in the execution of the integrated CFD solvers.
"""


# Standard libraries
import shutil
from typing import Any, Optional, Union


# Basic error family
class CapeError(Exception):
    r"Base exception class for exceptions intentionally raised by CAPE"
    pass


# File not found error
class CapeFileNotFoundError(FileNotFoundError, CapeError):
    r"""CAPE exception for missing required file"""
    pass


# Runtime error
class CapeRuntimeError(RuntimeError, CapeError):
    r"""CAPE exception that does not fit other categories"""
    pass


# Type error
class CapeTypeError(TypeError, CapeError):
    r"""CAPE exception for objects with the wrong type"""
    pass


# Assert type of a variable
def assert_isinstance(
        obj: Any,
        cls_or_tuple: Union[type, tuple],
        desc: Optional[str] = None):
    r"""Conveniently check types

    Applies ``isinstance(obj, cls_or_tuple)`` but also constructs
    a :class;`TypeError` and appropriate message if test fails

    :Call:
        >>> assert_isinstance(obj, cls, desc=None)
        >>> assert_isinstance(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Brief description of intended use of *obj*
    :Raises:
        * :class:`CapeTypeError` if type of *obj* does not match *cls*
          or *cls_tuple*
    :Version:
        * 2024-06-19 ``@ddalle``: v1.0
    """
    # Special case for ``None``
    if cls_or_tuple is None:
        return
    # Check for passed test
    if isinstance(obj, cls_or_tuple):
        return
    # Generate type error message
    msg = _genr8_type_error(obj, cls_or_tuple, desc)
    # Raise
    raise CapeTypeError(msg)


# Ensure an executable exists
def assert_which(cmdname: str):
    r"""Ensure that a specific executable exists

    :Call:
        >>> assert_which(cmdname)
    :Inputs:
        *cmdname*: :class:`str`
            Name of executable
    :Raises:
        * :class:`CapeFileNotFoundError` if *cmdname* not on PATH
    :Versions:
        * 2025-06-19 ``@ddalle``: v1.0
    """
    # Check if executable was found
    if shutil.which(cmdname):
        return
    # Create error message
    raise CapeFileNotFoundError(f"No '{cmdname}' executable found in path")


# Create error message for type errors
def _genr8_type_error(
        obj: Any,
        cls_or_tuple: Union[type, tuple],
        desc: Optional[str] = None) -> str:
    r"""Create an error message for object not having expected type

    :Call:
        >>> msg = _genr8_type_error(obj, cls, desc=None)
        >>> msg = _genr8_type_error(obj, cls_tuple, desc=None)
    :Inputs:
        *obj*: :class:`object`
            Object whose type is checked
        *cls*: :class:`type`
            Single permitted class
        *cls_tuple*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed classes
        *desc*: {``None``} | :class:`str`
            Brief description of intended use of *obj*
    :Outputs:
        *msg*: :class:`str`
            Error message
    :Version:
        * 2024-06-19 ``@ddalle``: v1.0
    """
    # Check for single type
    if isinstance(cls_or_tuple, tuple):
        # Multiple types
        names = [cls.__name__ for cls in cls_or_tuple]
    else:
        # Single type
        names = [cls_or_tuple.__name__]
    # Create error message
    if desc is None:
        # No description; msg2 is "Got type"
        msg1 = "G"
    else:
        # Add description; msg is "got type"
        msg1 = "For %s: g" % desc
    msg2 = "ot type '%s'; " % type(obj).__name__
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3
