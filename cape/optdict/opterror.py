r"""
:mod:`opterror`: Errors for :class:`OptionsDict` tools
=========================================================

This module provides a collection of error types relevant to the
:mod:`optdict` module. They are essentially the same as standard error
types such as :class:`KeyError` and :class:`TypeError` but with an extra
parent of :class:`OptdictError` to enable catching all errors
specifically raised by this package

"""


# Basic error family
class OptdictError(Exception):
    r"""Parent error class for :mod:`optdict` errors

    Inherits from :class:`Exception`
    """
    pass


class OptdictAttributeError(AttributeError, OptdictError):
    r"""Error related to accessing attributes of :class:`OptionsDict`

    Inherits from :class:`AttributeError` and :class:`OpdictError`.
    """
    pass


class OptdictNameError(NameError, OptdictError):
    r"""Error for badly named options in :class:`OptionsDict`

    Inherits from :class:`NameError` and :class:`OptdictError`
    """
    pass


class OptdictJSONError(ValueError, OptdictError):
    r"""Exception class for errors while parsing JSON files

    Inherits from :class:`ValueError` and :class:`OptdictError`
    """
    pass


class OptdictKeyError(KeyError, OptdictError):
    r"""Exception for missing option in :mod:`optdict.optitem`

    Inherits from :class:`KeyError` and :class:`OptdictError`
    """
    pass


class OptdictTypeError(TypeError, OptdictError):
    r"""Exception for unexpected type of parameter in :mod:`optdict`
    """
    pass


class OptdictValueError(ValueError, OptdictError):
    r"""Exception for unexpected value of parameter in :mod:`optdict`
    """
    pass


class OptdictExprError(ValueError, OptdictError):
    r"""Exception for invalid ``@expr``

    Applies to :func:`optdict.optitem.getel` or :class:`OptionsDict`
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
        :class:`OptdictTypeError`
    :Versions:
        * 2022-09-17 ``@ddalle``: Version 1.0
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
    raise OptdictTypeError(msg)


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
