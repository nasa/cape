r"""
:mod:`gruvoc.errors`: Error classes for ``gruvoc`` package
====================================================================

This module provides a collection of error types relevant to the
:mod:`gruvoc` package. They are essentially the same as standard error
types such as :class:`KeyError`, :class:`TypeError`, etc. but with an
extra parent of :class:`GruvocError` to enable catching all errors
specifically raised by this package

"""

# Standard library
import os
import shutil
from typing import Any, Optional

# Third-party
from numpy import ndarray


# Basic error family
class GruvocError(Exception):
    r"""Parent error class for :mod:`gruvoc` errors

    Inherits from :class:`Exception`
    """
    pass


class GruvocAttributeError(AttributeError, GruvocError):
    r"""Error related to accessing attributes

    Inherits from :class:`AttributeError` and :class:`GruvocError`.
    """
    pass


class GruvocFileNotFoundError(FileNotFoundError, GruvocError):
    r"""Exception for missing but required file

    Inherits from :class:`FileNotFoundError` and :class;`GruvocError`
    """
    pass


class GruvocJSONError(ValueError, GruvocError):
    r"""Exception class for errors while parsing JSON files

    Inherits from :class:`ValueError` and :class:`GruvocError`
    """
    pass


class GruvocKeyError(KeyError, GruvocError):
    r"""Exception for missing key in :mod:`gruvoc`

    Inherits from :class:`KeyError` and :class:`GruvocError`
    """
    pass


class GruvocNameError(NameError, GruvocError):
    r"""Error for badly named options in :mod:`gruvoc`

    Inherits from :class:`NameError` and :class:`GruvocError`
    """
    pass


class GruvocNotImplementedError(NotImplementedError, GruvocError):
    r"""Error for features not currently implemented"""
    pass


class GruvocSystemError(SystemError, GruvocError):
    r"""Exception for system errors raised by :mod:`gruvoc`
    """
    pass


class GruvocTypeError(TypeError, GruvocError):
    r"""Exception for unexpected type of parameter in :mod:`gruvoc`
    """
    pass


class GruvocValueError(ValueError, GruvocError):
    r"""Exception for unexpected value of parameter in :mod:`gruvoc`
    """
    pass


# Assert type of a variable
def assert_isinstance(obj, cls_or_tuple, desc=None):
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
    :Raises:
        :class:`GruvocTypeError`
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
    raise GruvocTypeError(msg)


# Assert that file exists
def assert_isfile(fname: str):
    r"""Ensure that a file exists

    :Call:
        >>> assert_isfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of a file
    :Raises:
        :class:`GruvocFileNotFoundError` if *fname* does not exist
    :Versions:
        * 2023-10-25 ``@ddalle``: v1.0
    """
    # Check for file
    if not os.path.isfile(fname):
        # Truncate long file name
        f1 = trunc8_fname(fname, 23)
        # Start message
        msg = f"File '{f1}' does not exist"
        # Check for absolute path
        if not os.path.isabs(fname):
            # Show working directory
            f2 = trunc8_fname(os.getcwd(), 18)
            msg += f"\n  relative to '{f2}'"
        raise GruvocFileNotFoundError(msg)


# Assert than an array has a certain size
def assert_size(v: ndarray, size: int, name: Optional[str] = None):
    # Exit if size is correct
    if v.size == size:
        return
    # Add in name of thing we're checking
    msg = _add_name(f"Expected {size} elements", name)
    # Add in actual size
    msg += f"; got {v.size}"
    # Raise exception
    raise GruvocValueError(msg)


# Assert than an object has a specific value
def assert_value(v: Any, vtarg: Any, name: Optional[str] = None):
    # Check value
    try:
        # Compare values, if possible
        test = (v == vtarg)
    except Exception:
        # Unable to compare *v* and *vtar*, maybe mismatching size
        test = None
        t1 = type(v).__name__
        t2 = type(vtarg).__name__
        raise GruvocTypeError(f"Cannot compare types '{t1}' and '{t2}'")
    # Check test result
    if test:
        return
    else:
        msg = _add_name(f"Expected value {vtarg!r}", name)
        msg += f"; got {v!r}"
    # Raise exception
    raise GruvocValueError(msg)


# Assert that an array has a certain dimension
def assert_nd(v: ndarray, nd: int, name: Optional[str] = None):
    # Check dimension
    if v.ndim == nd:
        return
    # Create error message
    msg = _add_name(f"Expected array dimension of {nd}", name)
    # Add in actual size
    msg += f"; got {v.ndim}"
    # Raise exception
    raise GruvocValueError(msg)


# Assert that an array has a certain dimension
def assert_ndmin(v: ndarray, nd: int, name: Optional[str] = None):
    # Check dimension
    if v.ndim >= nd:
        return
    # Create error message
    msg = _add_name(f"Expected array of dimension {nd} or greater", name)
    # Add in actual size
    msg += f"; got {v.ndim}"
    # Raise exception
    raise GruvocValueError(msg)


# Assert that an array has a certain shape
def assert_shape(
        v: ndarray,
        size: int,
        axis: int = 1,
        nd: Optional[int] = None,
        name: Optional[str] = None):
    # Min dimension
    ndmin = axis if nd is None else nd
    # Ensure min dimension
    assert_ndmin(v, ndmin)
    # Check
    if v.shape[axis] == size:
        return
    # Create error message
    msg = _add_name(f"Expected axis {axis} to have size {size}", name)
    # Show actual
    msg += f"; got {v.shape[axis]}"
    # Raise exception
    raise GruvocValueError(msg)


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
        # No description; msg2 is "Got type"
        msg1 = "G"
    else:
        # Add description; msg is "got type"
        msg1 = "For %s: g" % desc
    msg2 = "ot type '%s'; " % type(obj).__name__
    msg3 = "expected '%s'" % ("' | '".join(names))
    # Output
    return msg1 + msg2 + msg3


def _add_name(msg: str, name: Optional[str] = None) -> str:
    # Add name to string
    if name is None:
        # No name
        return msg
    else:
        # Append name
        return f"{msg} for {name}"


def trunc8_fname(fname: str, n: int) -> str:
    r"""Truncate string so it fits in current terminal with *n* to spare

    :Call:
        >>> fshort = trunc8_fname(fname, n)
    :Inputs:
        *fname*: :class:`str`
            File name or other string to truncate
        *n*: :class:`int`
            Number of chars to reserve for other text
    :Outputs:
        *fshort*: :class:`str`
            *fname* or truncated version if *fname* won't fit in current
            terminal width
    """
    # Length of current name
    l0 = len(fname)
    # Max width allowed (right now)
    maxwidth = shutil.get_terminal_size().columns - n
    # Check if truncation needed
    if l0 < maxwidth:
        # Use existing name
        return fname
    # Try to get leading folder
    if "/" in fname:
        # Get first folder, then everything else
        part1, part2 = fname.split("/", 1)
        # Try to truncate this
        fname = part1 + "/..." + part2[4 + len(part1) - maxwidth:]
    else:
        # Just truncate file name from end
        fname = fname[-maxwidth:]
    # Output
    return fname
