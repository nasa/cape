r"""
``kwparse``: Tools to map, parse, and validate keyword arguments
=================================================================

This module provides the class :class:`KwargParser` that allows for
convenient and powerful parsing of functions that use the ``**kw``
convention in their signature, such as

.. code-block:: python

    def f(a, b, **kw):
        ...

Users of this module create subclasses of :class:`KwargParser` that
process the expected keyword arguments to :func:`f`. The capabilities
of :class:`KwargParser` include

* only allowing specific keys (:data:`KwargParser._optlist`)
* mapping kwargs to alternate names, e.g. using *v* as a shortcut for
  *verbose* (:data:`KwargParser._optmap`)
* specifying the type(s) allowed for specific options
  (:data:`KwargParser._opttypes`)
* creating aliases for values (:data:`KwargParser._optvalmap`)
* calling converter functions (e.g. ``int()`` to convert a :class:`str`
  to an :class:`int`) (:data:`KwargParser._optconverters`)

Suppose you have a function

.. code-block:: python

    def f(a, b, **kw):
        ...

Where *a* should be a :class:`str`, *b* should be an :class:`int`, and
the only kwargs are *verbose* and *help*, which should both be
:class:`bool`. However, users can use *h* as an alias for *help* and *v*
for *verbose*. Then we could write a subclass of :class:`KwargParser` to
parse and validate args to this function.

.. code-block:: python

    class FKwargs(KwargParser):
        _optlist = ("help", "verbose")
        _optmap = {
            "h": "help",
            "v": "verbose",
        }
        _opttypes = {
            "a": str,
            "b": int,
            "help": bool,
            "verbose": bool,
        }
        _arglist = ("a", "b")
        _nargmin = 2
        _nargmax = 2

Here is how this parser handles an example with expected inputs.

.. code-block:: pycon

    >>> opts = FKwargs("me", 33, v=True)
    >>> print(opts)
    {'verbose': True}
    >>> print(opts.get_args())
    ('me', 33)

In many cases it is preferable to use

* :data:`INT_TYPES` instead of :class:`int`,
* :data:`FLOAT_TYPES` instead of :class:`float`,
* :data:`BOOL_TYPES` instead of :class:`bool`, and
* :data:`STR_TYPES` instead of :class:`str`

within :data:`KwargParser._opttypes`, e.g.

.. code-block:: python

    class FKwargs(KwargParser):
        _optlist = ("help", "verbose")
        _optmap = {
            "h": "help",
            "v": "verbose",
        }
        _opttypes = {
            "a": str,
            "b": INT_TYPES,
            "help": BOOL_TYPES,
            "verbose": BOOL_TYPES,
        }
        _arglist = ("a", "b")
        _nargmin = 2
        _nargmax = 2

so that values taken from :mod:`numpy` arrays are also recognized as
valid "integers," "floats," or "booleans."

Here are some examples of how FKwargs might handle bad inputs.

.. code-block:: pycon

    >>> FKwargs("my", help=True)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.KWTypeError: FKwargs() takes 2 arguments, but 1 were given
    >>> FKwargs(2, 3)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.KWTypeError: FKwargs() arg 0 (name='a'): got type 'int';
    expected 'str'
    >>> FKwargs("my", 10, b=True)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.KWNameError: FKwargs() unknown kwarg 'b'
    >>> FKwargs("my", 10, h=1)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.KWTypeError: FKwargs() kwarg 'help': got type 'int';
    expected 'bool'

In order to use an instance of this :class:`FKwargs` there are several
approaches. The first is to call the parser class directly:

.. code-block:: python

    def f(a, b, **kw):
        opts = FKwargs(a, b, **kw)
        ...

Another method is to use FKwargs as a decorator

.. code-block:: python

    @FKwargs.parse
    def f(a, b, **kw):
        ...

The decorator option ensures that *a*, *b*, and *kw* have all been
validated. Users can then use ``kw.get("help")`` without needing to
check for *h*.
"""

# Standard library
import difflib
import os
from base64 import b32encode
from collections import namedtuple
from functools import wraps

# Third-party
import numpy as np


# Basic error class
class KWParseError(Exception):
    r"""Parent error class for :mod:`kwparse` errors

    Inherts from :class:`Exception` and enables general catching of all
    errors raised by :mod:`kwparse`
    """
    pass


# Key error
class KWKeyError(KeyError, KWParseError):
    r"""Errors for missing keys raised by :mod:`kwparse`"""
    pass


# Name error
class KWNameError(NameError, KWParseError):
    r"""Errors for incorrect names raised by :mod:`kwparse`"""
    pass


# Type error
class KWTypeError(TypeError, KWParseError):
    r"""Errors for incorrect type raised by :mod:`kwparse`"""
    pass


# Value error
class KWValueError(ValueError, KWParseError):
    r"""Errors for invalid values raised by :mod:`kwparse`"""
    pass


# Collections of common types
if hasattr(np, "float128"):
    #: Collection of floating-point types:
    #: :class:`float`
    #: | :class:`numpy.float16`
    #: | :class:`numpy.float32`
    #: | :class:`numpy.float64`
    #: | :class:`numpy.float128`
    FLOAT_TYPES = (
        float,
        np.float16,
        np.float32,
        np.float64,
        np.float128)
else:  # pragma no cover
    FLOAT_TYPES = (
        float,
        np.float16,
        np.float32,
        np.float64)
#: Collection of integer (including unsigned) types:
#: :class:`int`
#: | :class:`numpy.int8`
#: | :class:`numpy.int16`
#: | :class:`numpy.int32`
#: | :class:`numpy.int64`
#: | :class:`numpy.uint8`
#: | :class:`numpy.uint16`
#: | :class:`numpy.uint32`
#: | :class:`numpy.uint64`
INT_TYPES = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64)
#: Collection of boolean-like types:
#: :class:`bool` | :class:`numpy.bool_`
BOOL_TYPES = (
    bool,
    np.bool_)
#: Collection of string-like types:
#: :class:`str` | :class:`numpy.str_`
STR_TYPES = (
    str,
    np.str_)
#: Acceptable types for :data:`KwargParser._optlist`
OPTLIST_TYPES = (
    set,
    tuple,
    frozenset,
    list)


#: Option name/value pair
OptPair = namedtuple("OptPair", ["opt", "val"])


# Decorator to catch KWError
def _wrap_init(func):
    # Define wrapper
    @wraps(func)
    def wrapper(self, *a, **kw):
        # Use a try/catch block
        try:
            # Attempt a normal call
            return func(self, *a, **kw)
        except KWParseError as err:
            # Prepend function name to error message
            msg = f"{type(self).__name__}() {err.args[0]}"
            # Reconstruct error locally to reduce traceback
            raise err.__class__(msg) from None
    # Return the wrapped functions
    return wrapper


# Main class
class KwargParser(dict):
    r"""A class to parse args and keyword args, check types, etc.

    :Call:
        >>> opts = KwargParser(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Arbitrary tuple of positional parameters to a function
        *kw*: :class:`dict`
            Arbitrary dict of keyword arguments to a function
    :Outputs:
        *opts*: :class:`KwargParser`
            Dictionary of validated kwargs and positional parameters
    :Attributes:
        * :attr:`argvals`
    """
  # *** CLASS ATTRIBUTES ***
   # --- General ---
    # Attributes
    __slots__ = (
        "argvals",
    )

    #: Subclass name used in error messages
    #: :class:`str`
    _name = ""

   # --- Options ---
    #: Allowed keyword (option) names:
    #: (:class:`tuple` | :class:`set`)[:class:`str`]
    _optlist = ()

    #: Aliases for kwarg names; key gets replaced with value:
    #: :class:`dict`\ [:class:`str`]
    _optmap = {}

    #: Allowed types for option values, before using converter:
    #: :class:`dict`\ [:class:`type` | :class:`tuple`\ [:class:`type`]]
    _rawopttypes = {}

    #: Aliases for option values:
    #: :class:`dict`\ [:class:`object`]
    _optvalmap = {}

    #: Functions to convert raw value of specified options:
    #: :class:`dict`\ [:class:`callable`]
    _optconverters = {}

    #: Allowed types for option values, after using converter:
    #: :class:`dict`\ [:class:`type` | :class:`tuple`\ [:class:`type`]]
    _opttypes = {}

    #: Specified allowed values for specified options:
    #: :class:`dict`\ [:class:`tuple` | :class:`set`]
    _optvals = {}

    #: Required kwargs:
    #: :class:`tuple`\ [:class:`str`]
    _optlistreq = ()

    #: Names for positional parameters (in order):
    #: :class:`tuple`\ [:class:`set`]
    _arglist = ()

    #: Minimum required number of positional parameters:
    #: :class:`int` >= 0
    _nargmin = 0

    #: Maximum number of positional parameters:
    #: ``None`` | :class:`int` > 0
    _nargmax = None

    #: Default values for specified options:
    #: :class:`dict`\ [:class:`object`]
    _rc = {}

  # *** METHODS ***
   # --- __dunder__ ---
    # Initialization method
    @_wrap_init
    def __init__(self, *args, **kw):
        r"""Initialization method"""
        #: :class:`list` -- List of values of positional parameters
        #: that cannot be aliased to options (not in :data:`_optlist`)
        self.argvals = []
        # Class and name
        cls = self.__class__
        # Allowed args
        nargmin = cls._nargmin
        nargmax = cls._nargmax
        # Process args
        narg = len(args)
        # Format first part of error message for positional params
        if nargmax is None:
            # No upper limit
            msg = f"takes at least {nargmin} arguments,"
        else:
            # Specified upper limit
            ntxt = f"{nargmin} to {nargmax}"
            ntxt = f"{nargmin}" if nargmin == nargmax else ntxt
            msg = f"takes {ntxt} arguments,"
        # Check arg counter
        if narg < nargmin:
            # Not enough args
            raise KWTypeError(f"{msg} but {narg} were given")
        elif (nargmax is not None) and (narg > nargmax):
            # Too many args
            raise KWTypeError(f"{msg} but {narg} were given")
        # Set options from *a* first
        self.set_args(args)
        # Then set options from *kw)
        self.set_opts(kw)
        # Call post-process hook
        self.init_post()

    # Post-initialization hook
    def init_post(self):
        r"""Custom post-initialization hook

        This function is called in the standard :func:`__init__`. The
        default :func:`init_post` does nothing. Users may define custom
        actions in :func:`init_post` in subclasses to make certain
        changes at the end of parsing

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
        """
        pass

  # *** DECORATORS ***
    @classmethod
    def parse(cls, func):
        r"""Decorator for a function to parse and validate its inputs

        :Call:
            >>> wrapper = cls.parse(func)
        :Example:
            .. code-block:: python

                @cls.parse
                def func(*a, **kw):
                    ...

        :Inputs:
            *func*: :class:`callable`
                A function, class, or callable instance
        :Outputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *wrap*: :class:`callable`
                A wrapped version of *func* that parses and validates
                args and kwargs according to *cls* before calling *func*
        """
        # Create wrapper
        @wraps(func)
        def wrapper(*a, **kw):
            # Parse options
            try:
                # Instantiate the requested class
                opts = cls(*a, **kw)
                # Get all options, applying _rc if appropriate
                parsed_kw = opts.get_kwargs()
                # Get positional parameters
                parsed_args = opts.get_args()
            except KWParseError as err:
                # Strip leading *cls.__name__* and use function name
                msg = err.args[0]
                # Check if it starts with class's name
                if msg.startswith(cls.__name__):
                    # Replace with name of function
                    lname = len(cls.__name__)
                    msg = func.__name__ + msg[lname:]
                # Re-raise
                raise err.__class__(msg) from None
            # Call original function with parsed options
            return func(*parsed_args, **parsed_kw)
        # Return wrapper
        return wrapper

  # *** GET/SET ***
   # --- Get ---
    # Get full dictionary
    def get_kwargs(self) -> dict:
        r"""Get dictionary of kwargs, applying defaults

        :Call:
            >>> kwargs = opts.get_kwargs()
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
        :Outputs:
            *kwargs*: :class:`dict`
                Keyword arguments and values currently parsed
        """
        # Get class
        cls = self.__class__
        # Create a copy
        optsdict = dict(self)
        # Get full set of defaults
        rc = self.__class__.getx_cls_dict("_rc")
        # Apply any defaults
        for opt, val in rc.items():
            optsdict.setdefault(opt, val)
        # Get list of required options (don't combine with bases) (?)
        reqopts = cls._optlistreq
        # Loop through the same
        for opt in reqopts:
            # Check if it's present
            if opt not in self:
                raise KWKeyError(
                    f"{cls.__name__}() missing required kwarg '{opt}'")
        # Output
        return optsdict

    # Get list of args, terminating at first None
    def get_args(self) -> tuple:
        r"""Return a copy of the current positional parameter values

        :Call:
            >>> args = opts.get_args()
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
        :Outputs:
            *args*: :class:`tuple`\ [:class:`object`]
                Current values of positional parameters
        """
        # Return current arg values
        return tuple(self.argvals)

    # Get option
    def get_opt(self, opt: str, vdef=None):
        r"""Get value of one option

        :Call:
            >>> val = opts.get_opt(opt, vdef=None)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                Name of option
            *vdef*: {``None``} | :class:`object`
                Default value if *opt* not found in *opts* or
                :data:`_rc`
        """
        # Apply option map
        opt = self.apply_optmap(opt)
        # Check if present
        if opt in self:
            # Get value
            rawval = self[opt]
        elif vdef is not None:
            # Use default value specified by user
            rawval = vdef
        else:
            # Get default
            rawval = self.__class__.getx_cls_key("_rc", opt)
        # Validate and return
        return self.validate_optval(opt, rawval)

   # --- Set ---
    # Set collection of options
    def set_opts(self, a: dict):
        r"""Set a collection of options

        :Call:
            >>> opts.set_opts(a)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *a*: :class:`dict`
                Dictionary of options to update into *opts*
        """
        # Check type
        msg = f"{self.__class__.__name__}.set_opts() arg 1"
        assert_isinstance(a, dict, msg)
        # Loop through option/value paris
        for opt, val in a.items():
            self.set_opt(opt, val)

    # Set single option
    def set_opt(self, rawopt: str, rawval):
        r"""Set the value of a single option

        :Call:
            >>> opts.set_opt(rawopt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *rawopt*: :class:`str`
                Name or alias of option to set
            *rawval*: :class:`object`
                Pre-conversion value of *rawopt*
        """
        # Validate
        opt, val = self.validate_opt(rawopt, rawval)
        # Save value
        self[opt] = val

    # Set list of positional parameter values
    def set_args(self, args):
        r"""Set the values of positional arguments

        :Call:
            >>> opts.set_args()
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *args*: :class:`list` | :class:`tuple`
                Ordered list of positional argument values
        """
        # Loop through args
        for j, rawval in enumerate(args):
            # Save it
            self.set_arg(j, rawval)

    # Set positional parameter value
    def set_arg(self, j: int, rawval):
        r"""Set the value of the *j*-th positional argument

        :Call:
            >>> opts.set_arg(j, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int` >= 0
                Argument index
            *rawval*: :class:`object`
                Value for arg *j*, before :data:`_optconverters`
        """
        # Get class
        cls = self.__class__
        # Get parameter name, if applicable
        argname = cls.get_argname(j)
        # Check for named argument
        if argname is not None:
            # Check if it's a kwarg
            if argname in cls.get_optlist():
                # Save it as kwarg instead of arg
                self.set_opt(argname, rawval)
                return
        # Get number of currently stored args
        nargcur = len(self.argvals)
        # Append ``None`` as needed
        for _ in range(nargcur, j + 1):
            self.argvals.append(None)
        # Validate but save as positional parameter
        val = self.validate_argval(j, argname, rawval)
        # Save that
        self.argvals[j] = val

  # *** VALIDATORS ***
   # --- Combined validator ---
    # Validate an option and raw value
    def validate_opt(self, rawopt: str, rawval) -> OptPair:
        r"""Validate a raw option name and raw value

        Replaces *rawopt* with non-aliased name and applies any
        *optconverter* to *rawval*. Raises an exception if option name,
        type, or value does not match expectations.

        :Call:
            >>> optpair = opts.validate_opt(rawopt, rawval)
            >>> opt, val = opts.validate_opt(rawopt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *rawopt*: :class:`str`
                Name or alias of option, before :data:`_optlist`
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Outputs:
            *optpair*: :class:`OptPair`
                :class:`tuple` of de-aliased option name and converted
                value
            *opt*: :class:`str`
                De-aliased option name (after *optmap* applied)
            *val*: :class:`object`
                Converted value, either *rawval* or
                ``optconverter(rawval)``
        """
        # Apply alias
        opt = self.apply_optmap(rawopt)
        # Check option name
        self.check_optname(opt)
        # Get value
        val = self.validate_optval(opt, rawval)
        # Output
        return OptPair(opt, val)

    # Validate a raw value value
    def validate_optval(self, opt: str, rawval):
        r"""Validate a raw option value

        :Call:
            >>> val = opts.validate_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name (after *optmap* applied)
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Outputs:
            *val*: :class:`object`
                Converted value, either *rawval* or
                ``optconverter(rawval)``
        """
        # Check raw type
        self.check_rawopttype(opt, rawval)
        # Convert value
        aval = self.apply_optconverter(opt, rawval)
        # Apply aliases
        val = self.apply_optvalmap(opt, aval)
        # Check converted type
        self.check_opttype(opt, val)
        # Check value
        self.check_optval(opt, val)
        # Output
        return val

    # Validate a raw argument value
    def validate_argval(self, j: int, argname, rawval):
        r"""Validate a raw positional parameter (arg) value

        :Call:
            >>> val = opts.validate_argval(j, argname, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int`
                Argument index
            *argname*: ``None`` | :class:`str`
                Argument name, if applicable
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Outputs:
            *val*: :class:`object`
                Converted value, either *rawval* or
                ``optconverter(rawval)``
        """
        # Check raw type
        self.check_rawargtype(j, argname, rawval)
        # Convert value
        val = self.apply_argconverter(j, argname, rawval)
        # Check converted type
        self.check_argtype(j, argname, val)
        # Check value
        self.check_argval(j, argname, val)
        # Output
        return val

   # --- Single-property option checkers ---
    # Replace alias if appropriate
    def apply_optmap(self, rawopt: str) -> str:
        r"""Apply alias to raw option name, if applicable

        :Call:
            >>> opt = opts.apply_optmap(rawopt)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *rawopt*: :class:`str`
                Option name or alias, before :data:`_optmap`
        :Outputs:
            *opt*: {*rawopt*} | :class:`str`
                De-aliased option name
        """
        # Get class
        cls = self.__class__
        # Get _optmap key
        return cls.getx_cls_key("_optmap", rawopt, vdef=rawopt)

    # Check _optlist
    def check_optname(self, opt: str):
        r"""Check validity of an option name

        :Call:
            >>> opts.check_optname(opt)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
        :Raises:
            :class:`KWNameError` if *opt* is not recognized
        """
        # Get class
        cls = self.__class__
        # Get list of options allowed
        optlist = cls.get_optlist()
        # Check
        if len(optlist) == 0 or opt in optlist:
            # Valid result
            return
        # Get closest matches
        matches = difflib.get_close_matches(opt, optlist)
        # Common part of warning/error message
        msg = f"unknown kwarg '{opt}'"
        # Add suggestions if able
        if len(matches):
            msg += "; nearest matches: %s" % " ".join(matches[:3])
        # Raise an exception
        raise KWNameError(msg)

    # Check type (before applying converter)
    def check_rawopttype(self, opt: str, rawval):
        r"""Check type of option value prior to conversion function

        :Call:
            >>> opts.check_rawopttype(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *rawval*: :class:`object`
                Raw user value for *opt* before using *optconverter*
        :Raises:
            :class:`KWTypeError` if *rawval* has wrong type
        """
        # Get specified type or tuple of types or None
        cls_or_tuple = self.__class__.get_rawopttype(opt)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Otherwise check types
        assert_isinstance(rawval, cls_or_tuple, f"kwarg '{opt}'")

    # Apply converter, if any
    def apply_optconverter(self, opt: str, rawval):
        r"""Apply option converter function to raw value

        :Call:
            >>> val = opts.apply_optconverter(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *rawval*: :class:`object`
                Raw user value for *opt* before using *optconverter*
        :Outputs:
            *val*: {*rawval*} | :class:`object`
                Result of calling *optconverter* for *opt* on *rawval*
        :Raises:
            :class:`KWTypeError` if *optconverter* for *opt* is not
            callable
        """
        # Get class
        cls = self.__class__
        # Get _optconverter key
        func = cls.get_optconverter(opt)
        # Return original value if not found
        if func is None:
            # No converter
            return rawval
        # Convert
        val = func(rawval)
        # Output
        return val

    # Check type (before applying converter)
    def check_opttype(self, opt: str, val):
        r"""Check type of option value after conversion function

        :Call:
            >>> opts.check_opttype(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *val*: :class:`object`
                Value for *opt*
        :Raises:
            :class:`KWTypeError` if *val* has wrong type
        """
        # Get specified type or tuple of types or None
        cls_or_tuple = self.__class__.get_opttype(opt)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Otherwise check types
        assert_isinstance(val, cls_or_tuple, f"kwarg '{opt}'")

    # Apply option value map, if any
    def apply_optvalmap(self, opt: str, rawval):
        r"""Apply option value map (aliases for value), if any

        :Call:
            >>> val = opts.apply_optconverter(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *rawval*: :class:`object`
                Raw user value for *opt* before using *optconverter*
        :Outputs:
            *val*: {*rawval*} | :class:`object`
                Dealiased (by ``_optvalmap[opt]``) value
        """
        # Get class
        cls = self.__class__
        # Get _optvalmap key
        valmap = cls.get_optvalmap(opt)
        # Return original value if not found
        if valmap is None:
            # No converter
            return rawval
        # Convert (default to original value)
        val = valmap.get(rawval, rawval)
        # Output
        return val

    # Check value
    def check_optval(self, opt: str, val):
        r"""Check option value against list of recognized values

        :Call:
            >>> opts.check_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *val*: :class:`object`
                Value for *opt*
        :Raises:
            :class:`KWValueError` if *opt* has an *optval* setting and
            *val* is not in it
        """
        # Get specified values
        optvals = self.__class__.get_optvals(opt)
        # No checks if *optvals* is not specified
        if optvals is None:
            return
        # Otherwise check value
        if val not in optvals:
            raise KWValueError(f"kwarg '{opt}' invalid value {repr(val)}")

   # --- Single-property arg checkers ---
    # Check type (before applying converter)
    def check_rawargtype(self, j: int, argname, rawval):
        r"""Check type of positional arg prior to conversion function

        :Call:
            >>> opts.check_rawargtype(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Raises:
            :class:`KWTypeError` if *rawval* has wrong type
        """
        # Get class
        cls = self.__class__
        # Get specified type or tuple of types or None
        cls_or_tuple = cls.getx_cls_arg("_rawopttypes", argname)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Form message
        msg = cls._genr8_argmsg(j, argname)
        # Otherwise check types
        assert_isinstance(rawval, cls_or_tuple, msg)

    # Apply converter, if any
    def apply_argconverter(self, j: int, argname, rawval):
        r"""Apply option converter function to raw positional arg value

        :Call:
            >>> val = opts.apply_argconverter(j, argname, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Outputs:
            *val*: {*rawval*} | :class:`object`
                Result of calling *optconverter* for *opt* on *rawval*
        :Raises:
            :class:`KWTypeError` if *optconverter* for *opt* is not
            callable
        """
        # Get class
        cls = self.__class__
        # Get _optconverter key
        func = cls.getx_cls_arg("_optconverters", argname)
        # Return original value if not found
        if func is None:
            # No converter
            return rawval
        # Convert
        val = func(rawval)
        # Output
        return val

    # Check type (after applying converter)
    def check_argtype(self, j: int, argname, val):
        r"""Check type of positional arg after conversion function

        :Call:
            >>> opts.check_argtype(opt, val)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *val*: :class:`object`
                Value for parameter in position *j*, after conversion
        :Raises:
            :class:`KWTypeError` if *val* has wrong type
        """
        # Get class
        cls = self.__class__
        # Get specified type or tuple of types or None
        cls_or_tuple = cls.getx_cls_arg("_opttypes", argname)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Form message
        msg = cls._genr8_argmsg(j, argname)
        # Otherwise check types
        assert_isinstance(val, cls_or_tuple, msg)

    # Check value
    def check_argval(self, j: int, argname, val):
        r"""Check positional arg value against list of recognized values

        :Call:
            >>> opts.check_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`KwargParser`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *val*: :class:`object`
                Value for arg in position *j*
        :Raises:
            :class:`KWValueError` if *argname* has an *optval* setting
            and *val* is not in it
        """
        # Get class
        cls = self.__class__
        # Get specified values
        optvals = cls.getx_cls_arg("_optvals", argname)
        # No checks if *optvals* is not specified
        if optvals is None:
            return
        # Form message
        msg = cls._genr8_argmsg(j, argname)
        # Otherwise check value
        if val not in optvals:
            raise KWValueError(f"{msg} invalid value {repr(val)}")

  # *** CLASS METHODS ***
   # --- ID/naming ---
    # Get class's name
    @classmethod
    def get_cls_name(cls) -> str:
        r"""Get a name to use for a given class

        :Call:
            >>> clsname = cls.get_cls_name()
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
        :Outputs:
            *clsname*: :class:`str`
                *cls._name* if set, else *cls.__name__*
        """
        # Check for name
        if cls._name:
            # Explicitly set
            return cls._name
        # Otherwise just name of class
        return cls.__name__

   # --- Arg (positional) naming ---
    # Get arg mane
    @classmethod
    def get_argname(cls, j: int):
        r"""Get name for an argument by index, if applicable

        :Call:
            >>> argname = cls.get_argname(j)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *j*: :class:`int`
                Positional parameter (arg) index
        :Outputs:
            *argname*: ``None`` | :class:`str`
                Argument option name, if applicable
        """
        # Get argument list (don't combine with subclasses)
        arglist = cls._arglist
        # Check if *j* could be in here
        if j < len(arglist):
            # Get name (can also be ``None``)
            return arglist[j]
        else:
            # No name
            return None

    # Generate error message identifier for arg
    @classmethod
    def _genr8_argmsg(cls, j: int, argname) -> str:
        # Common part (parameter index)
        msg1 = f"arg {j}"
        # Parameter name if appropriate
        msg2 = f" (name='{argname}')" if argname else ""
        # Combine
        return msg1 + msg2

   # --- Option properties ---
    # Get option type(s)
    @classmethod
    def get_rawopttype(cls, opt: str):
        r"""Get the type(s) allowed for the raw value of option *opt*

        If *opttype* is ``None``, no constraints are placed on the raw
        value of *opt*. The "raw value" is the value for *opt* before
        any converters have been applied.

        :Call:
            >>> opttype = cls.get_rawopttype(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *opt*: :class:`str`
                Full (non-aliased) name of option
        :Outputs:
            *opttype*: ``None`` | :class:`type` | :class:`tuple`
                Single type or list of types for raw *opt*
        """
        # Get directly specified type or tuple
        v = cls.getx_cls_key("_rawopttypes", opt)
        # Return if defined
        if v is not None:
            return v
        # Otherwise, check for a _default_
        return cls.getx_cls_key("_rawopttypes", "_default_")

    # Get option type(s)
    @classmethod
    def get_opttype(cls, opt: str):
        r"""Get the type(s) allowed for the value of option *opt*

        If *opttype* is ``None``, no constraints are placed on the
        value of *opt*. The "value" differs from the "raw value" in that
        the "value" is after any converters have been applied.

        :Call:
            >>> opttype = cls.get_opttype(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *opt*: :class:`str`
                Full (non-aliased) name of option
        :Outputs:
            *opttype*: ``None`` | :class:`type` | :class:`tuple`
                Single type or list of types for *opt*
        """
        # Get directly specified type or tuple
        v = cls.getx_cls_key("_opttypes", opt)
        # Return if defined
        if v is not None:
            return v
        # Otherwise, check for a _default_
        return cls.getx_cls_key("_opttypes", "_default_")

    # Get converter
    @classmethod
    def get_optconverter(cls, opt: str):
        r"""Get option value converter, if any, for option *opt*

        Output must be a callable function that takes one argument

        :Call:
            >>> func = cls.get_optconverter(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *opt*: :class:`str`
                Full (non-aliased) name of option
        :Outputs:
            *func*: ``None`` | :class:`function` | **callable**
                Function or other callable object
        """
        # Get converter, if any
        func = cls.getx_cls_key("_optconverters", opt)
        # Done if no converter
        if func is None:
            return
        # Validate
        if not callable(func):
            raise KWTypeError(f"kwarg '{opt}' converter is not callable")
        # Return the function (no way to check if it's unitary?)
        return func

    # Get value map
    @classmethod
    def get_optvalmap(cls, opt: str):
        r"""Get option value aliases, if any, for option *opt*

        Output must be a :class:`dict`

        :Call:
            >>> valmap = cls.get_optvalmap(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *opt*: :class:`str`
                Full (non-aliased) name of option
        :Outputs:
            *valmap*: ``None`` | :class:`dict`
                Map of alias values for *opt*
        """
        # Get converter, if any
        valmap = cls.getx_cls_key("_optvalmap", opt)
        # Done if no map
        if valmap is None:
            return
        # Validate
        assert_isinstance(valmap, dict, f"_optvalmap for '{opt}'")
        # Output
        return valmap

    # Get allowed values
    @classmethod
    def get_optvals(cls, opt: str):
        r"""Get a set/list/tuple of allowed values for option *opt*

        If *optvals* is not ``None``, the full (post-optconverter) value
        will be checked if it is ``in`` *optvals*.

        :Call:
            >>> optvals = cls.get_optvals(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *opt*: :class:`str`
                Full (non-aliased) name of option
        :Outputs:
            *optvals*: ``None`` | :class:`set` | :class:`tuple`
                Tuple, list, set, or frozenset of allowed values
        """
        # Get values, if any (no _default_)
        optvals = cls.getx_cls_key("_optvals", opt)
        # Try for _default_ if applicable
        if optvals is None:
            optvals = cls.getx_cls_key("_optvals", "_default_")
        # Exit if None
        if optvals is None:
            return
        # Checks
        assert_isinstance(optvals, OPTLIST_TYPES, f"kwarg '{opt}' _optvals")
        # Output
        return optvals

   # --- Option specifics ---
    # Get full list of options
    @classmethod
    def get_optlist(cls) -> set:
        r"""Get list of allowed options from *cls* and its bases

        This combines the ``_optlist`` attribute from *cls* and any
        bases it might have that are also subclasses of
        :class:`KwargParser`.

        If *optlist* is an empty set, then no constraints are applied to
        option names.

        :Call:
            >>> optlist = cls.get_opttype(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
        :Outputs:
            *optlist*: :class:`set`\ [:class:`str`]
                Single type or list of types for *opt*
        """
        return cls.getx_cls_set("_optlist")

   # --- Arg lists ---
    # Get value of a class attr dict for an arg
    @classmethod
    def getx_cls_arg(cls, attr: str, argname, vdef=None):
        r"""Get :class:`dict` class attribute for positional parameter

        If *argname* is ``None``, the parameter (arg) has no name, and
        only ``"_arg_default_"`` and ``"_default_"`` can be used from
        ``getattr(cls, attr)``.

        Otherwise, this will look in the bases of *cls* if
        ``getattr(cls, attr)`` does not have *argname*. If *cls* is a
        subclass of another :class:`KwargParser` class, it will search
        through the bases of *cls* until the first time it finds a class
        attribute *attr* that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.getx_cls_key(attr, key, vdef=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *attr*: :class:`str`
                Name of class attribute to search
            *key*: :class:`str`
                Key name in *cls.__dict__[attr]*
            *vdef*: {``None``} | :class:`object`
                Default value to use if not found in class attributes
        :Outputs:
            *v*: ``None`` | :class:`ojbect`
                Any value, ``None`` if not found
        """
        # Get default value
        v0 = cls.getx_cls_key(attr, "_default_", vdef=vdef)
        # Get default specific to parameters
        v1 = cls.getx_cls_key(attr, "_arg_default_", vdef=v0)
        # Try to get option-specific value if applicable
        if argname:
            # Use *argname* to search for values
            return cls.getx_cls_key(attr, argname, vdef=v1)
        else:
            # Without a parameter name, can only use defaults
            return v1

   # --- Option lists ---
    # Get value of a class attr dict
    @classmethod
    def getx_cls_key(cls, attr: str, key: str, vdef=None):
        r"""Access *key* from a :class:`dict` class attribute

        This will look in the bases of *cls* if ``getattr(cls, attr)``
        does not have *key*. If *cls* is a subclass of another
        :class:`KwargParser` class, it will search through the bases of
        *cls* until the first time it finds a class attribute *attr*
        that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.getx_cls_key(attr, key, vdef=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *attr*: :class:`str`
                Name of class attribute to search
            *key*: :class:`str`
                Key name in *cls.__dict__[attr]*
            *vdef*: {``None``} | :class:`object`
                Default value to use if not found in class attributes
        :Outputs:
            *v*: ``None`` | :class:`ojbect`
                Any value, ``None`` if not found
        """
        # Get cls's attribute if possible
        clsdict = cls.__dict__.get(attr)
        # Check if found
        if isinstance(clsdict, dict) and key in clsdict:
            return clsdict[key]
        # Otherwise loop through bases until found
        for clsj in cls.__bases__:
            # Only process subclass
            if not issubclass(clsj, KwargParser):
                continue
            # Generate random string
            vdefj = randomstr()
            # Recurse
            vj = clsj.getx_cls_key(attr, key, vdef=vdefj)
            # Test if something was found (else we'll get the rand str)
            if vj is not vdefj:
                return vj
        # Not found
        return vdef

    # Get full list of options
    @classmethod
    def getx_cls_set(cls, attr: str) -> set:
        r"""Get combined :class:`set` for *cls* and its bases

        This allows a subclass of :class:`KwargParser` to only add to
        the ``_optlist`` attribute rather than manually include the
        ``_optlist`` of all the bases.

        :Call:
            >>> v = cls.getx_cls_set(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *attr*: :class:`str`
                Name of class attribute to search
        :Outputs:
            *v*: :class:`set`
                Combination of ``getattr(cls, attr)`` and
                ``getattr(base, attr)`` for each ``base`` in
                ``cls.__bases__``, etc.
        """
        # Initialize output
        clsset = set()
        # Get attribute
        v = cls.__dict__.get(attr)
        # Update set
        if v:
            clsset.update(v)
        # Loop through bases
        for clsj in cls.__bases__:
            # Only recurse if KwargParser
            if issubclass(clsj, KwargParser):
                # Recurse
                clssetj = clsj.getx_cls_set(attr)
                # Combine
                clsset.update(clssetj)
        # Output
        return clsset

    # Get full dictionary of options
    @classmethod
    def getx_cls_dict(cls, attr: str) -> dict:
        r"""Get combined :class:`dict` for *cls* and its bases

        This allows a subclass of :class:`KwargParser` to only add to
        the ``_opttypes`` or ``_optmap`` attribute rather than manually
        include contents of all the bases.

        :Call:
            >>> clsdict = cls.getx_cls_dict(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`KwargParser`
            *attr*: :class:`str`
                Name of class attribute to search
        :Outputs:
            *clsdict*: :class:`dict`
                Combination of ``getattr(cls, attr)`` and
                ``getattr(base, attr)`` for each ``base`` in
                ``cls.__bases__``, etc.
        """
        # Get attribute
        clsdict = cls.__dict__.get(attr, {})
        # Loop through bases
        for clsj in cls.__bases__:
            # Only recurse if KwargParser
            if issubclass(clsj, KwargParser):
                # Recurse
                clsdictj = clsj.getx_cls_dict(attr)
                # Combine, but don't overwrite *clsdict*
                for kj, vj in clsdictj.items():
                    clsdict.setdefault(kj, vj)
        # Output
        return clsdict


# Assert type of a variable
def assert_isinstance(obj, cls_or_tuple, desc=None):
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
        :class:`KWTypeError`
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
    raise KWTypeError(msg)


# Create a random string
def randomstr(n=15) -> str:
    r"""Create a random string encded as a hexadecimal integer

    :Call:
        >>> txt = randomstr(n=15)
    :Inputs:
        *n*: {``15``} | :class:`int`
            Number of random bytes to encode
    """
    return b32encode(os.urandom(n)).decode().lower()


# Create error message for type errors
def _genr8_type_error(obj, cls_or_tuple, desc=None):
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
