#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
``argread``: Parse command-line arguments and options
==========================================================

This class provides the :class:`ArgReader` class, instances of
which can be used to customize available command-line options for a
given interface.

The :class:`ArgReader` class has the function :func:`ArgReader.parse`,
which interprets ``sys.argv`` and splits it into a list of args and
kwargs. The base :class:`ArgReader` class will parse command-line
arguments in useful ways, but one of the main intents is to subclass
:class:`ArgReader` and customize how various CLI arguments are parsed.

Here are some examples of the base interpreter, which can also be used
with the convenience function :func:`readkeys`.

.. code-block:: pycon

    >>> readkeys(["prog", "arg1", "arg2", "-v", "--name", "argread"])
    ("arg1", "arg2"), {"v": True, "name": "argread"}
    >>> readkeys(["prog", "-v", "1", "2", "-name", "argread", "d=3"])
    ("2",), {"v": "1", "name": "argread", "d": "3"}

Notice that the ``1`` goes as the value of the option ``v``. The ``2``
is not preceded by a :class:`str` starting with a dash, so it gets
interpreted as an arg.

There is an alternate function :func:`readflags` that interprets
multi-letter args starting with a single dash differently:

.. code-block:: pycon

    >>> readflags(["prog", "-lh", "fname"])
    ("fname",), {"l": True, "h": True}

There is a third function :func:`readflagstar` that interprets arguments
like ``tar``

.. code-block:: pycon

    >>> readflagstar(["prog", "-cvf", "fname"])
    (), {"c": True, "v": True, "f": "fname"}

These convenience functions make it easy to parse many command-line
inputs with minimal effort, but it is also possible to declare richer
and more specific interfaces by subclassing :class:`ArgReader`.

For example

.. code-block:: python

    class MyParser(ArgReader):
        _optlist_noval = (
            "default",
            "verbose",
        )
        _optmap = {
            "d": "default",
            "v": verbose",
        }

has two properties. First ``-d`` and ``-v`` are abbreviations for
``--default`` and ``--verbose``, respectively, because of the entries in
:data:`MyParser._optmap`. Second, neither option can take a "value", so

.. code-block:: console

    $ prog -d hub url

would be parsed as

.. code-block::

    ("hub", "url"), {"default": True}

In other words, the word ``hub`` didn't get interpreted as the value for
the option ``d`` as it would with the default :class:`ArgReader`.

Another common use case is to convert CLI strings to another type. For
example suppose you have an option ``i`` that takes in the index of
something you want to work with, like

.. code-block:: console

    $ prog -i 3

The default :func:`readkeys` in this case would return

.. code-block:: python

    (), {"i": "3"}

you can convert this :class:`str` ``"3"`` to an :class:`int` with the
following subclass

.. code-block::

    class MyConverter(ArgReader):
        _optconverters = {
            "i": int,
        }

This :class:`ArgReader` also allows for convenient and powerful parsing
of functions arguments. See for example

.. code-block:: python

    def f(a, b, **kw):
        ...

Users of this module create subclasses of :class:`ArgReader` that
process the expected keyword arguments to :func:`f`. The
argument-parsing capabilities of :class:`ArgReader` include

* only allowing specific keys (:data:`ArgReader._optlist`)
* mapping kwargs to alternate names, e.g. using *v* as a shortcut for
  *verbose* (:data:`ArgReader._optmap`)
* specifying the type(s) allowed for specific options
  (:data:`ArgReader._opttypes`)
* creating aliases for values (:data:`ArgReader._optvalmap`)
* calling converter functions (e.g. ``int()`` to convert a :class:`str`
  to an :class:`int`) (:data:`ArgReader._optconverters`)

Suppose you have a function

.. code-block:: python

    def f(a, b, **kw):
        ...

Where *a* should be a :class:`str`, *b* should be an :class:`int`, and
the only kwargs are *verbose* and *help*, which should both be
:class:`bool`. However, users can use *h* as an alias for *help* and *v*
for *verbose*. Then we could write a subclass of :class:`ArgReader` to
parse and validate args to this function.

.. code-block:: python

    class FKwargs(ArgReader):
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
    >>> print(opts.get_argvals())
    ('me', 33)

In many cases it is preferable to use

* :data:`INT_TYPES` instead of :class:`int`,
* :data:`FLOAT_TYPES` instead of :class:`float`,
* :data:`BOOL_TYPES` instead of :class:`bool`, and
* :data:`STR_TYPES` instead of :class:`str`

within :data:`ArgReader._opttypes`, e.g.

.. code-block:: python

    class FKwargs(ArgReader):
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
    kwparse.ArgReadTypeError: FKwargs() takes 2 arguments, but 1 were given
    >>> FKwargs(2, 3)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.ArgReadTypeError: FKwargs() arg 0 (name='a'): got type 'int';
    expected 'str'
    >>> FKwargs("my", 10, b=True)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.ArgReadNameError: FKwargs() unknown kwarg 'b'
    >>> FKwargs("my", 10, h=1)
    File "kwparse.py", line 172, in wrapper
        raise err.__class__(msg) from None
    kwparse.ArgReadTypeError: FKwargs() kwarg 'help': got type 'int';
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
import re
import sys
from base64 import b32encode
from collections import namedtuple
from functools import wraps
from typing import Any, Callable, Optional

# Third-party imports
import numpy as np

# Local imports
from .clitext import compile_rst
from .errors import (
    ArgReadError,
    ArgReadKeyError,
    ArgReadNameError,
    ArgReadTypeError,
    ArgReadValueError,
    assert_isinstance
)


__version__ = "1.3.1"

# Constants
TAB = '    '
IERR_OK = 0
IERR_CMD = 16
IERR_OPT = 32


# Regular expression for options like "cdfr=1.3"
REGEX_EQUALKEY = re.compile(r"(\w+)=([^=].*)")

# A name for the ``a, kw`` tuple
ArgTuple = namedtuple("ArgTuple", ("a", "kw"))
SubCmdTuple = namedtuple("SubCmdTuple", ("cmdname", "argv"))
SubParserTuple = namedtuple("SubParserTuple", ("cmdname", "subparser"))
SubParserCheck = namedtuple("SubParserCheck", ("cmdname", "subparser", "ierr"))


#: Collection of floating-point types:
#: :class:`float`
#: | :class:`numpy.float16`
#: | :class:`numpy.float32`
#: | :class:`numpy.float64`
#: | :class:`numpy.float128`
FLOAT_TYPES = (
    float,
    np.floating)

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
    np.integer)
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
#: Acceptable types for :data:`ArgReader._optlist`
OPTLIST_TYPES = (
    set,
    tuple,
    frozenset,
    list)


#: Option name/value pair
OptPair = namedtuple("OptPair", ["opt", "val"])


# Decorator to catch ArgReadError
def _wrap_init(func):
    # Define wrapper
    @wraps(func)
    def wrapper(self, *a, **kw):
        # Use a try/catch block
        try:
            # Attempt a normal call
            return func(self, *a, **kw)
        except ArgReadError as err:
            # Prepend function name to error message
            msg = f"{type(self).__name__}() {err.args[0]}"
            # Reconstruct error locally to reduce traceback
            raise err.__class__(msg) from None
    # Return the wrapped functions
    return wrapper


# Metaclass to combine _optlist and other class attributes
class MetaArgReader(type):
    r"""Metaclass for :class:`ArgReader`

    This metaclass combines attributes w/ bases. For example if creating
    a new class :class:`Class2` that inherits from :class:`Class1`, this
    will automatically combine ``Class1._optlist` and
    ``Class2._optlist`` and save the result as ``Class2._optlist``. This
    happens behind the scenes so that users do not need to worry about
    repeating ``_optlist`` entries.
    """

    #: List of tuple-like class attributes
    _tuple_attrs = (
        "_optlist",
        "_optlist_noval",
    )

    #: List of dict-like class attributes
    _dict_attrs = (
        "_help_opt",
        "_help_optarg",
        "_optconverters",
        "_optmap",
        "_opttypes",
        "_optvalmap",
        "_optvals",
        "_rawopttypes",
        "_rc",
    )

    def __new__(metacls, name: str, bases: tuple, namespace: dict):
        r"""Initialize a new subclass, but combine ``_optlist`` attr

        :Call:
            >>> cls = metacls.__new__(name, bases, namespace)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *name*: :class:`str`
                Name of new class being created
            *bases*: :class:`tuple`\ [:class:`type`]
                Bases for new class
            *namespace*: :class:`dict`
                Attributes, methods, etc. for new class
        :Outputs:
            *cls*: :class:`type`
                New class using *metacls* instead of :class:`type`
        """
        # Initialize the new class
        cls = type.__new__(metacls, name, bases, namespace)
        # Check for attribute entries to inherit from bases
        for clsj in bases:
            cls.combine_attrs(clsj, cls)
        # Return the new class
        return cls

    @classmethod
    def combine_attrs(metacls, clsj: type, cls: type):
        r"""Combine attributes of *clsj* and *cls*

        :Call:
            >>> metacls.combine_attrs(clsj, cls)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *clsj*: :class:`type`
                Parent class (basis) to combine into *cls*
            *cls*: :class:`type`
                New class in which to save combined attributes
        """
        # Combine tuples
        for attr in metacls._tuple_attrs:
            metacls.combine_tuple(clsj, cls, attr)
        # Combine dict/map
        for attr in metacls._dict_attrs:
            metacls.combine_dict(clsj, cls, attr)

    @classmethod
    def combine_tuple(metacls, clsj: type, cls: type, attr: str):
        r"""Combine one tuple-like class attribute of *clsj* and *cls*

        :Call:
            >>> metacls.combine_tuple(clsj, cls, attr)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *clsj*: :class:`type`
                Parent class (basis) to combine into *cls*
            *cls*: :class:`type`
                New class in which to save combined attributes
            *attr*: :class:`str`
                Name of attribute to combine
        """
        # Get initial properties
        vj = getattr(clsj, attr, None)
        vx = cls.__dict__.get(attr)
        # Check for both
        qj = isinstance(vj, OPTLIST_TYPES)
        qx = isinstance(vx, OPTLIST_TYPES)
        if not (qj and qx):
            return
        # Initialize with (copy of) the parent
        combined_list = list(vj)
        # Loop through child
        for v in vx:
            if v not in combined_list:
                combined_list.append(v)
        # Save combined list
        setattr(cls, attr, tuple(combined_list))

    @classmethod
    def combine_dict(metacls, clsj: type, cls: type, attr: str):
        r"""Combine one dict-like class attribute of *clsj* and *cls*

        :Call:
            >>> metacls.combine_dict(clsj, cls, attr)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *clsj*: :class:`type`
                Parent class (basis) to combine into *cls*
            *cls*: :class:`type`
                New class in which to save combined attributes
            *attr*: :class:`str`
                Name of attribute to combine
        """
        # Get initial properties
        vj = getattr(clsj, attr, None)
        vx = cls.__dict__.get(attr)
        # Check for both
        qj = isinstance(vj, dict)
        qx = isinstance(vx, dict)
        if not (qj and qx):
            return
        # Copy dict from basis
        combined_dict = dict(vj)
        # Combine results
        combined_dict.update(vx)
        # Save combined list
        setattr(cls, attr, combined_dict)


# Argument read class
class ArgReader(dict, metaclass=MetaArgReader):
    r"""Class to parse command-line interface arguments

    :Call:
        >>> parser = ArgReader(**kw)
    :Outputs:
        *parser*: :class:`ArgReader`
            Instance of command-line argument parser
    :Attributes:
        * :attr:`argv`
        * :attr:`argvals`
        * :attr:`cmdname`
        * :attr:`prog`
        * :attr:`kwargs_sequence`
        * :attr:`kwargs_replaced`
        * :attr:`kwargs_single_dash`
        * :attr:`kwargs_double_dash`
        * :attr:`kwargs_equal_sign`
        * :attr:`param_sequence`
    """
  # *** CLASS ATTRIBUTES ***
   # --- General ---
    # List of instance attributes
    __slots__ = (
        "argv",
        "argvals",
        "cmdname",
        "prog",
        "kwargs_sequence",
        "kwargs_replaced",
        "kwargs_single_dash",
        "kwargs_double_dash",
        "kwargs_equal_sign",
        "param_sequence",
    )

    #: Name of program for which arguments are being parsed
    _name = "argread"

   # --- Options ---
    #: Allowed keyword (option) names:
    #: (:class:`tuple` | :class:`set`)[:class:`str`]
    _optlist = ()

    #: List of options that cannot take a value:
    #: (:class:`tuple` | :class:`set`)\ [:class:`str`]
    _optlist_noval = ()

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

    #: Option to enforce ``_optlist``
    _restrict = False

    #: Default values for specified options:
    #: :class:`dict`\ [:class:`object`]
    _rc = {}

   # --- Positional Arguments ---
    #: Names for positional parameters (in order):
    #: :class:`tuple`\ [:class:`set`]
    _arglist = ()

    #: Minimum required number of positional parameters:
    #: :class:`int` >= 0
    _nargmin = 0

    #: Maximum number of positional parameters:
    #: ``None`` | :class:`int` > 0
    _nargmax = None

   # --- CLI: front desk ---
    #: List of available commands
    _cmdlist = None

    #: Aliases for command names
    _cmdmap = {}

    #: Parser classes for sub-commands
    _cmdparsers = {}

   # --- CLI: parsing ---
    #: Option to interpret multi-char words with a single dash
    #: as single-letter boolean options, e.g.
    #: ``-lh`` becomes ``{"l": True, "h": True}``
    single_dash_split = False

    #: Option to interpret multi-char words with a single dash
    #: the way POSIX command ``tar`` does, e.g.
    #: ``-cvf val`` becomes ``{"c": True, "v": True, "f": "val"}``
    single_dash_lastkey = False

    #: Option to allow equal-sign options, e.g.
    #: ``key=val`` becomes ``{"key": "val"}``
    equal_sign_key = True

   # --- Errors ---
    #: Base exception class: :class:`Exception`
    exc_cls = ArgReadError

   # --- CLI: help messages ---
    #: Optional list and sequence of options to show in ``-h`` output
    #: (default is to use ``_optlist``)
    _help_optlist = None

    #: Optional sequence of arguments
    _help_arglist = None

    #: Description of each option, for creation of automatic "-h" output
    _help_opt = {}

    #: List of options to display as negative
    _help_opt_negative = ()

    #: Names for arguments of options that take arguments, to be used in
    #: automatically generated help messages
    _help_optarg = {}

    #: Descriptions for sub-command names
    _help_cmd = {}

    #: Optional list and sequence of sub-commands to show in ``-h``
    _help_cmdlist = None

    #: Short description of program for title line
    _help_title = ""

    #: Optional longer description of program to add to ``-h`` output
    _help_description = ""

    #: Prompt character to use in usage line of help message
    _help_prompt = '$'

    #: Even more help information to write after the list of options
    _help_extra = ""

  # *** METHODS ***
   # --- __dunder__ ---
    @_wrap_init
    def __init__(self, *args, **kw):
        r"""Initialization method"""
        #: :class:`list`\ [:class:`str`] --
        #: List of raw CLI commands parsed
        self.argv = []
        #: :class:`str` -- Name of program read from ``argv[0]``
        self.prog = None
        #: :class:`str` -- Name of subcommand to use, if any
        self.cmdname = None
        #: :class:`list` -- Current values of non-keyword arguments
        self.argvals = []
        #: :class:`list`\ [:class:`str`] --
        #: List of options in original order
        self.kwargs_sequence = []
        #: :class:`list`\ [:class:`str`] --
        #: List of options that have duplicates
        self.kwargs_replaced = []
        #: :class:`list`\ [:class:`str`] --
        #: List of options that were entered with a single dash
        self.kwargs_single_dash = {}
        #: :class:`list`\ [:class:`str`] --
        #: List of options that were entered with a double dash
        self.kwargs_double_dash = {}
        #: :class:`list`\ [:class:`str`] --
        #: List of options that were entered using ``key=val`` syntax
        self.kwargs_equal_sign = {}
        #: :class:`list`\ [:class:`str`, :class:`object`] --
        #: List of option name and value as parsed (includes duplicates
        #: in their original order)
        self.param_sequence = []
        # Parse positional parameters
        if len(args):
            self.parse_args(args)
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
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
        """
        pass

  # *** DECORATORS ***
    @classmethod
    def check(cls: type, func: Callable):
        r"""Decorator for a function to parse and validate its inputs

        :Call:
            >>> wrapper = cls.parse(func)
        :Example:
            .. code-block:: python

                @cls.check
                def func(*a, **kw):
                    ...

        :Inputs:
            *func*: :class:`callable`
                A function, class, or callable instance
        :Outputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
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
                parsed_args, parsed_kw = opts.get_a_kw()
            except ArgReadError as err:
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
    def get_args(self) -> ArgTuple:
        r"""Get full list of args and options from parsed inputs

        :Call:
            >>> args, kwargs = parser.get_args()
        :Outputs:
            *args*: :class:`list`\ [:class:`str`]
                List of positional parameter argument values
            *kwargs*: :class:`dict`
                Dictionary of named options and their values
        """
        # Get list of arguments
        args = list(self.argvals)
        # Get full dictionary of outputs, applying defaults
        kwargs = self.get_kwargs()
        # Output
        return ArgTuple(args, kwargs)

    # Get list of args, terminating at first None
    def get_argvals(self) -> tuple:
        r"""Return a copy of the current positional parameter values

        :Call:
            >>> args = opts.get_argvals()
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
        :Outputs:
            *args*: :class:`tuple`\ [:class:`object`]
                Current values of positional parameters
        """
        # Return current arg values
        return tuple(self.argvals)

    # Get full dictionary
    def get_kwargs(self) -> dict:
        r"""Get dictionary of kwargs, applying defaults

        :Call:
            >>> kwargs = opts.get_kwargs()
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
        :Outputs:
            *kwargs*: :class:`dict`
                Keyword arguments and values currently parsed
        """
        # Get class
        cls = self.__class__
        # List of options
        optlist = cls._optlist
        # Create a copy
        optsdict = dict(self)
        # Get full set of defaults
        rc = cls._rc
        rc = {} if rc is None else rc
        # Apply any defaults
        for opt, val in rc.items():
            if (optlist is None) or opt in optlist:
                optsdict.setdefault(opt, val)
        # Get list of required options (don't combine with bases) (?)
        reqopts = cls._optlistreq
        # Loop through the same
        for opt in reqopts:
            # Check if it's present
            if opt not in self:
                raise ArgReadKeyError(
                    f"{cls.__name__}() missing required kwarg '{opt}'")
        # Output
        return optsdict

    # Get full dictionary
    def get_a_kw(self) -> ArgTuple:
        r"""Get dictionary of kwargs, applying defaults

        :Call:
            >>> a, kw = opts.get_a_kw()
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
        :Outputs:
            *kwargs*: :class:`dict`
                Keyword arguments and values currently parsed
        """
        # Get class
        cls = self.__class__
        # List of options
        optlist = cls._optlist
        # Get args
        a = self.get_argvals()
        # Get list of arguments defined positionally
        argnames = list(self._arglist)
        argnames = argnames[:len(a)]
        # Create a copy
        kw = {}
        # Loop through current kwargs
        for opt, val in self.items():
            if opt not in argnames:
                kw[opt] = val
        # Get full set of defaults
        rc = cls._rc
        rc = {} if rc is None else rc
        # Apply any defaults
        for opt, val in rc.items():
            if (optlist is None) or (opt in optlist):
                if opt not in argnames:
                    kw.setdefault(opt, val)
        # Get list of required options (don't combine with bases) (?)
        reqopts = cls._optlistreq
        # Loop through the same
        for opt in reqopts:
            # Check if it's present
            if opt not in self:
                raise ArgReadKeyError(
                    f"{cls.__name__}() missing required kwarg '{opt}'")
        # Output
        return ArgTuple(a, kw)

    # Get option
    def get_opt(self, opt: str, vdef: Optional[Any] = None):
        r"""Get value of one option

        :Call:
            >>> val = opts.get_opt(opt, vdef=None)
        :Inputs:
            *opts*: :class:`ArgReader`
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

   # --- Parse --
    def parse_args(self, args: tuple):
        r"""Parse positional parameters

        :Call:
            >>> opts.parse_args(args)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *args*: :class:`list` | :class:`tuple`
                Ordered list of positional argument values
        """
        # Re-initialize argument list
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
            raise ArgReadTypeError(f"{msg} but {narg} were given")
        elif (nargmax is not None) and (narg > nargmax):
            # Too many args
            raise ArgReadTypeError(f"{msg} but {narg} were given")
        # Set options from *a* first
        self.set_args(args)

   # --- Set ---
    # Set collection of options
    def set_opts(self, a: dict):
        r"""Set a collection of options

        :Call:
            >>> opts.set_opts(a)
        :Inputs:
            *opts*: :class:`ArgReader`
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
    def set_opt(self, rawopt: str, rawval: Any):
        r"""Set the value of a single option

        :Call:
            >>> opts.set_opt(rawopt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
    def set_args(self, args: tuple):
        r"""Set the values of positional arguments

        :Call:
            >>> opts.set_args(args)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *args*: :class:`list` | :class:`tuple`
                Ordered list of positional argument values
        """
        # Loop through args
        for j, rawval in enumerate(args):
            # Save it
            self.set_arg(j, rawval)

    # Set positional parameter value
    def set_arg(self, j: int, rawval: Any):
        r"""Set the value of the *j*-th positional argument

        :Call:
            >>> opts.set_arg(j, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
                # Get validated value
                rawval = self[argname]
        # Get number of currently stored args
        nargcur = len(self.argvals)
        # Append ``None`` as needed
        for _ in range(nargcur, j + 1):
            self.argvals.append(None)
        # Validate but save as positional parameter
        val = self.validate_argval(j, argname, rawval)
        # Save that
        self.argvals[j] = val

  # *** CLI ***
   # --- CLI front desk parsing ---
    def fullparse(self, argv: Optional[list] = None) -> SubParserTuple:
        r"""Identify sub-command and use appropriate parser

        :Call:
            >>> cmdname, subparser = parser.fullparse(argv=None)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *argv*: {``None``} | :class:`list`\ [:class:`str`]
                Optional arguments to parse, else ``sys.argv``
        :Outputs:
            *cmdname*: ``None`` | :class:`str`
                Name of command, if identified or inferred
            *subparser*: :class:`ArgReadder`
                Parser for *cmdname* applied to remaining CLI args
        """
        # Decide command name
        cmdname, argvcmd = self.decide_cmdname(argv)
        # Check for a subcommand
        if cmdname is None:
            return cmdname, self
        # Get default class
        clsdef = self._cmdparsers.get("_default_", self.__class__)
        # Otherwise get sub-parser class
        cls = self._cmdparsers.get(cmdname, clsdef)
        # Create new instance
        subparser = cls()
        # Parse reduced set of commands
        subparser.parse(argvcmd)
        # Output
        return SubParserTuple(cmdname, subparser)

    def fullparse_check(self, argv: Optional[list] = None) -> SubParserCheck:
        r"""Identify sub-command and use appropriate parser

        :Call:
            >>> cmdname, subparser, ierr = parser.fullparse_check(argv)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *argv*: {``None``} | :class:`list`\ [:class:`str`]
                Optional arguments to parse, else ``sys.argv``
        :Outputs:
            *cmdname*: ``None`` | :class:`str`
                Name of command, if identified or inferred
            *subparser*: :class:`ArgReadder`
                Parser for *cmdname* applied to remaining CLI args
            *ierr*: :class:`int`
                Return code
        """
        # Default args
        argv = _get_argv(argv)
        # Run parser with error handling
        try:
            # Attempt to parse
            cmdname, subparser = self.fullparse(argv)
            # Standard output
            return SubParserCheck(cmdname, subparser, IERR_OK)
        except (NameError, ValueError, TypeError) as e:
            # Parse function name
            if os.path.isabs(argv[0]):
                argv[0] = os.path.basename(argv[0])
            # Error message
            print("In command:\n")
            print("  " + " ".join(argv) + "\n")
            print(e.args[0])
            # Output
            return SubParserCheck(0, 0, IERR_OPT)

   # --- CLI parsing ---
    def parse(self, argv: Optional[list] = None) -> ArgTuple:
        r"""Parse CLI args

        :Call:
            >>> a, kw = parser.parse(argv=None)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *argv*: {``None``} | :class:`list`\ [:class:`str`]
                Optional arguments to parse, else ``sys.argv``
        :Outputs:
            *a*: :class:`list`
                List of positional arguments
            *kw*: :class:`dict`
                Dictionary of options and their values
        """
        # Parse CLI args
        self._parse(argv)
        # Output current values
        return self.get_args()

    def parse_cli_full(self, argv: Optional[list] = None) -> ArgTuple:
        r"""Parse CLI args

        :Call:
            >>> a, kw = parser.parse_cli_full(argv=None)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *argv*: {``None``} | :class:`list`\ [:class:`str`]
                Optional arguments to parse, else ``sys.argv``
        :Outputs:
            *a*: :class:`list`
                List of positional arguments
            *kw*: :class:`dict`
                Dictionary of options and their values
            *kw["__replaced__"]*: :class:`list`\ [(:class:`str`, *any*)]
                List of any options replaced by later values
        """
        # Parse CLI args
        self._parse(argv)
        # Output current values
        return self.get_cli_args()

    def _parse(self, argv: Optional[list] = None):
        # Process optional args
        if argv is None:
            # Copy *sys.argv*
            argv = list(sys.argv)
        else:
            # Check type of *argv*
            assert_isinstance(argv, list, "'argv'")
            # Check each arg is a string
            for j, arg in enumerate(argv):
                # Check type
                assert_isinstance(arg, str, f"argument {j}")
            # Copy args
            argv = list(argv)
        # Save copy of args to *self*
        self.argv = list(argv)
        # (Re)initialize attributes storing parsed arguments
        self.argvals = []
        self.kwargs_sequence = []
        self.kwargs_replaced = []
        self.kwargs_single_dash = {}
        self.kwargs_double_dash = {}
        self.kwargs_equal_sign = {}
        self.param_sequence = []
        # Check for command name
        if len(argv) == 0:
            raise ArgReadTypeError(
                "Expected at least one argv entry (program name)")
        # Save command name
        self.prog = argv.pop(0)
        # Loop until args are gone
        while argv:
            # Extract first argument
            arg = argv.pop(0)
            # Parse argument
            prefix, key, val, flags = self._parse_arg(arg)
            # Check for flags
            if flags:
                # Set all to ``True``
                for flag in flags:
                    self.save_single_dash(flag, True)
            # Check if arg
            if prefix == "":
                # Positional parameter
                self.save_arg(val)
                continue
            # Check option type: "-opt", "--opt", "opt=val"
            if prefix == "=":
                # Equal-sign option
                self.save_equal_key(key, val)
                continue
            elif key is None:
                # This can happen when only flags, like ``"-lh"``
                continue
            # Determine save function based on prefix
            if prefix == "-":
                save = self.save_single_dash
            else:
                save = self.save_double_dash
            # Check for "--no-mykey"
            if key.startswith("no-"):
                # This is interpreted "mykey=False"
                save(key[3:], False)
                continue
            # Apply _optmap (aliases)
            key = self.apply_optmap(key)
            # Check for "noval" options, or if next arg is available
            if key in self._optlist_noval or (len(argv) == 0):
                # No following arg to check
                save(key, True)
                continue
            # Check next arg
            prefix1, _, val1, _ = self._parse_arg(argv[0])
            # If it is not a key, save the value
            if prefix1 == "":
                # Save value like ``--extend 2``
                save(key, val1)
                # Pop argument
                argv.pop(0)
            else:
                # Save ``True`` for ``--qsub``
                save(key, True)

    def _parse_arg(self, arg: str):
        r"""Parse type for a single CLI arg

        :Call:
            >>> prefix, key, val, flags = parser._parse_arg(arg)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *arg*: :class:`str`
                Single arg to parse, usually from ``sys.argv``
        :Outputs:
            *prefix*: ``""`` | ``"-'`` | ``"--"`` | ``"="``
                Argument prefix
            *key*: ``None`` | :class:`str`
                Option name if *arg* is ``--key`` or ``key=val``
            *val*: ``None`` | :class:`str`
                Option value or positional parameter value
            *flags* ``None`` | :class:`str`
                List of single-character flags, e.g. for ``-lh``
        """
        # Global settings
        splitflags = self.single_dash_split
        lastflag = self.single_dash_lastkey
        equalkey = self.equal_sign_key
        # Check for options like "cdfr=1.2"
        if equalkey:
            # Test the arg
            match = REGEX_EQUALKEY.match(arg)
        else:
            # Do not test
            match = None
        # Check if it starts with a hyphen
        if match:
            # Already processed key and value
            key, val = match.groups()
            flags = None
            prefix = "="
        elif not arg.startswith("-"):
            # Positional parameter
            key = None
            val = arg
            flags = None
            prefix = ""
        elif arg.startswith("--"):
            # A normal, long-form key
            key = arg.lstrip("-")
            val = None
            flags = None
            prefix = "--"
        elif splitflags:
            # Single-dash option, like '-d' or '-cvf'
            prefix = "-"
            val = None
            # Check for special handling of last flag
            if len(arg) == 1:
                # No flags, no key
                flags = ""
                key = ""
            elif len(arg) == 2:
                # No flags, one key
                flags = ""
                key = arg[-1]
            elif lastflag:
                # Last "flag" is special
                flags = arg[1:-1]
                key = arg[-1]
            else:
                # Just list of flags
                flags = arg[1:]
                key = None
        else:
            # Single-dash opton
            prefix = "-"
            key = arg[1:]
            val = None
            flags = None
        # Output
        return prefix, key, val, flags

   # --- Python arg retrieval ---

   # --- CLI arg retrieval
    def get_cli_args(self) -> ArgTuple:
        r"""Get full list of args and options from parsed inputs

        :Call:
            >>> args, kwargs = parser.get_cli_args()
        :Outputs:
            *args*: :class:`list`\ [:class:`str`]
                List of positional parameter argument values
            *kwargs*: :class:`dict`
                Dictionary of named options and their values
            `kwargs["__replaced__"]`: :class:`tuple`
                Overwritten repeat CLI options
        """
        # Get list of arguments
        args = list(self.argvals)
        # Get full dictionary of outputs, applying defaults
        kwargs = self.get_cli_kwargs()
        # Output
        return ArgTuple(args, kwargs)

    def get_cli_kwargs(self) -> dict:
        r"""Get full list of kwargs, including repeated values

        :Call:
            >>> args, kwargs = parser.get_args()
        :Outputs:
            *kwargs*: :class:`dict`
                Dictionary of named options and their values
        """
        # Get full dictionary of outputs, applying defaults
        kwargs = self.get_kwargs()
        # Set __replaced__
        kwargs["__replaced__"] = [
            tuple(opt) for opt in self.kwargs_replaced]
        # Output
        return kwargs

    def get_argtuple(self) -> tuple:
        r"""Get list of all args and kwargs by name

        Unnamed positional arguments will have names like ``arg5``.

        :Call:
            >>> argtuple = parser.get_argtuple()
        :Outputs:
            *argtuple*: :class:`tuple`\ [:class:`str`, :class:`object`]
                Tuple of name/value pairs, including overwritten kwargs
        """
        # Initialize output
        arglist = []
        # Get args and kwargs
        args = self.get_argvals()
        kwargs = self.get_kwargs()
        # Argument names
        argnames = self._arglist
        # Loop through args
        for j, arg in enumerate(args):
            # Get name
            name = f"arg{j+1}" if j >= len(argnames) else argnames[j]
            # Check if present
            if name not in kwargs:
                arglist.append((name, arg))
        # Loop through replaced kwargs
        for opt in self.kwargs_replaced:
            arglist.append(tuple(opt))
        # Append actual kwargs
        for name, arg in kwargs.items():
            arglist.append((name, arg))
        # Output
        return tuple(arglist)

    def get_argdict(self) -> dict:
        r"""Get dictionary of all args and kwargs by name

        Unnamed positional arguments will have names like ``arg5``.

        :Call:
            >>> kw = parser.get_argdict()
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of positional and keyword names and values
        """
        # Initialize output
        argdict = {}
        # Get args and kwargs
        args = self.get_argvals()
        kwargs = self.get_kwargs()
        # Argument names
        argnames = self._arglist
        # Loop through args
        for j, arg in enumerate(args):
            # Get name
            name = f"arg{j+1}" if j >= len(argnames) else argnames[j]
            # Just save it
            argdict[name] = arg
        # Apply non-overwritten kwargs
        argdict.update(kwargs)
        # Output
        return argdict

   # --- Subcommand interface ---
    def decide_cmdname(self, argv: Optional[list] = None) -> SubCmdTuple:
        r"""Identify sub-command if appropriate

        :Call:
            >>> cmdname, subargv = parser.decide_cmdname(argv=None)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *argv*: {``None``} | :class:`list`\ [:class:`str`]
                Raw command-line args to parse {``sys.argv``}
        :Outputs:
            *cmdname*: :class:`str`
                (Standardized) name of sub-command as supplied by user
            *subargv*: :class:`list`\ [:class:`str`]
                Command-line arguments for sub-command to parse
        """
        # Expand CLI list if necessary
        argv = self.argv if argv is None else argv
        argv = sys.argv if argv is None else argv
        # Parse commands as given
        self.parse(argv)
        # Check for a command
        if len(self.argvals):
            # Use first argument as default method
            cmdname = self.argvals[0]
        else:
            # Attempt to infer
            cmdname = self.infer_cmdname()
        # Check for a command
        if cmdname is None:
            # No subcommand
            return SubCmdTuple(None, argv)
        # Identify command
        prog = self.prog
        # Copy parameter sequence
        params = list(self.param_sequence)
        # The parameter that defines this arg
        target_param = (None, cmdname)
        # Loop through parameters to pop-out the first arg
        for j, param in enumerate(params):
            # Check
            if param == target_param:
                params.pop(j)
                break
        # Reconstruct
        argv = self.reconstruct(params)
        # Get full command name (apply aliases)
        fullcmdname = self.apply_cmdmap(cmdname)
        # Replace program name
        argv[0] = f"{prog}>{fullcmdname}"
        # Output
        return SubCmdTuple(fullcmdname, argv)

    def infer_cmdname(self) -> Optional[str]:
        r"""Infer sub-command if not determined by first argument

        This command is usually overwritten in subclasses for special
        command-line interfaces where the primary task is inferred from
        options. This version always returns ``None``

        :Call:
            >>> cmdname = parser.infer_cmdname()
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
        :Outputs:
            *cmdname*: :class:`str`
                Name of sub-command
        """
        return None

    def apply_cmdmap(self, cmdname: str) -> str:
        r"""Apply aliases for sub-command name

        :Call:
            >>> fullcmdname = parser.apply_cmdmap(cmdname)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *cmdname*: :class:`str`
                Name of sub-command as supplied by user
        :Outputs:
            *fullcmdname*: :class:`str`
                Standardized name of *cmdname*, usually just *cmdname*
        """
        # Check for alternates
        return self._cmdmap.get(cmdname, cmdname)

   # --- CLI reconstruction ---
    def reconstruct(self, params: Optional[list] = None) -> list:
        r"""Recreate a command from parsed information

        :Call:
            >>> cmdlist = parser.reconstruct()
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *params*: {``None``} | :class:`list`\ [:class:`tuple`]
                Optional list of parameters (default from *parser*)
        :Outputs:
            *cmdlist*: :class:`list`\ [:class:`str`]
                Reconstruction of originally parsed command
        """
        # Start with programname
        cmdlist = [self.prog.replace('>', '-')]
        # Use default parameter sequence
        param_sequence = self.param_sequence if params is None else params
        # Loop through parameters in order they were read
        for k, v in param_sequence:
            # Check for an arg (kwarg is None)
            if k is None:
                cmdlist.append(v)
                continue
            # Check which kind of kwarg it is
            if k in self.kwargs_equal_sign:
                cmdlist.append(f"{k}={v}")
                continue
            # Check which prefix to use
            prefix = '--' if k in self.kwargs_double_dash else '-'
            # Check for value
            if (v is None) or (v is True):
                # No value
                cmdlist.append(f"{prefix}{k}")
            elif v is False:
                # Negated
                cmdlist.append(f"{prefix}no-{k}")
            else:
                # With value
                cmdlist.append(f"{prefix}{k}")
                cmdlist.append(str(v))
        # Output
        return cmdlist

   # --- Arg/Option interface ---
    def save_arg(self, arg):
        r"""Save a positional argument

        :Call:
            >>> parser.save_arg(arg, narg=None)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *arg*: :class:`str`
                Name/value of next parameter
        """
        self._save(None, arg)

    def save_double_dash(self, k, v=True):
        r"""Save a double-dash keyword and value

        :Call:
            >>> parser.save_double_dash(k, v=True)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: {``True``} | ``False`` | :class:`str`
                Value to save
        """
        self._save(k, v)
        self.kwargs_double_dash[k] = v

    def save_equal_key(self, k, v):
        r"""Save an equal-sign key/value pair, like ``"mach=0.9"``

        :Call:
            >>> parser.save_equal_key(k, v)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: :class:`str`
                Value to save
        """
        self._save(k, v)
        self.kwargs_equal_sign[k] = v

    def save_single_dash(self, k, v=True):
        r"""Save a single-dash keyword and value

        :Call:
            >>> parser.save_single_dash(k, v=True)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: {``True``} | ``False`` | :class:`str`
                Value to save
        """
        self._save(k, v)
        self.kwargs_single_dash[k] = v

    def _save(self, rawopt: str, rawval):
        # Append to universal list of args
        self.param_sequence.append((rawopt, rawval))
        # Check option vs arg
        if rawopt is None:
            # Get index
            j = len(self.argvals)
            # Save arg
            self.set_arg(j, rawval)
        else:
            # Validate value
            opt, val = self.validate_opt(rawopt, rawval)
            # Universal keyword arg sequence
            self.kwargs_sequence.append((opt, val))
            # Check if a previous key
            if opt in self:
                # Save to "replaced" options
                self.kwargs_replaced.append((opt, self[opt]))
            # Save to current kwargs
            self[opt] = val

    def get_aliases(self, opt: str) -> list:
        r"""Get list of aliases for a particular option

        :Call:
            >>> names = parser.get_aliases(opt)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *names*: :class:`list`\ [:class:`str`]
                List of aliases, including *opt*; primary name first
        """
        # Get primary name
        mainopt = self._optmap.get(opt, opt)
        # Initialize list
        names = [mainopt]
        # Loop through aliases
        for alias, fullopt in self._optmap.items():
            # Check for match
            if fullopt == mainopt:
                names.append(alias)
        # Output
        return names

   # --- CLI help ---
    def show_help(self, opt: str = "help") -> bool:
        r"""Display help message for non-front-desk parser if requested

        :Call:
            >>> q = parser.show_help(opt="help")
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *opt*: {``"help"``} | :class:`str`
                Name of option to trigger help message
        :Outputs:
            *q*: :class:`str`
                Whether front-desk help was triggered
        """
        # Check for help option
        if self.get(opt, False) and self._cmdlist is None:
            # Print help message
            print(compile_rst(self.genr8_help()))
            return True
        else:
            # No "help" requested
            return False

    def help_frontdesk(
            self,
            cmdname: Optional[str],
            opt: str = "help") -> bool:
        r"""Display help message for front-desk parser, if appropriate

        :Call:
            >>> q = parser.help_frontdesk(cmdname)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *cmdname*: ``None`` | :class:`str`
                Name of sub-command, if specified
            *opt*: {``"help"``} | :class:`str`
                Name of option to trigger help message
        :Outputs:
            *q*: :class:`str`
                Whether front-desk help was triggered
        """
        # Get class
        cls = self.__class__
        # Check if this is a front desk
        if cls._cmdlist is None:
            return False
        # Check for null commands
        if cmdname is None or cmdname == opt:
            print(compile_rst(self.genr8_help()))
            return True
        # Check if command was recognized
        if cmdname not in cls._cmdlist:
            # Get closest matches
            close = difflib.get_close_matches(
                cmdname, cls._cmdlist, n=4, cutoff=0.3)
            # Use all if no matches
            close = close if close else cls._cmdlist
            # Generate list as text
            matches = " | ".join(close)
            # Display them
            print(f"Unexpected '{cls._name}' command '{cmdname}'")
            print(f"Closest matches: {matches}")
            return True
        # No problems
        return False

    def genr8_help(self) -> str:
        r"""Generate automatic help message to use w/ ``-h``

        :Call:
            >>> msg = parser.genr8_optshelp()
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
        :Outputs:
            *msg*: :class:`str`
                Help message
        """
        # Generate parts
        title = self._genr8_help_title()
        descr = self._genr8_help_description()
        usage = self._genr8_help_usage()
        parms = self._genr8_help_args()
        subcs = self._genr8_help_cmdlist()
        optns = self._genr8_help_options()
        extra = self._genr8_help_coda()
        # Combine results
        return title + descr + usage + parms + subcs + optns + extra

    def genr8_optshelp(self) -> str:
        r"""Generate help message for all the options in _optlist

        :Call:
            >>> msg = parser.genr8_optshelp()
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
        :Outputs:
            *msg*: :class:`str`
                Help message for all options
        """
        # Get option list
        optlist = self._help_optlist
        # Default to _optlist if not defined
        optlist = optlist if optlist is not None else self._optlist
        # Generate text for each option
        msgs = [self.genr8_opthelp(opt) for opt in optlist]
        # Add header and join mesages
        return "\n\n".join(msgs)

    def genr8_opthelp(self, opt: str) -> str:
        r"""Generate a help message for a particular option

        :Call:
            >>> msg = parser.genr8_opthelp(opt)
        :Inputs:
            *parser*: :class:`ArgReader`
                Command-line argument parser
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *msg*: :class:`str`
                Help message for option *opt*
        """
        # Get all option names, with single- or double-dashes
        optname = self._genr8_help_optnames(opt)
        # Get main option name
        mainopt = self._optmap.get(opt, opt)
        # Initialize
        msg = TAB + optname
        # Check for option name
        argname = self._help_optarg.get(mainopt)
        # Append if necessary
        if argname:
            msg = f"{msg} {argname}"
        # Get description
        optdescr = self._help_opt.get(mainopt)
        # Append if necessary
        if optdescr:
            msg += f"\n{TAB}{TAB}{optdescr}"
        # Get default value
        vdef = self._rc.get(mainopt)
        # Check for default value
        if vdef is not None:
            # Use old-fashioned string format b/c using braces in str
            msg += " {%s}" % vdef
        # Output
        return msg

    def _genr8_help_title(self) -> str:
        r"""Generate header portion of ``-h`` output"""
        # Initialize with name of program
        title = f"``{self._name}``"
        # Get short description/title
        short_descr = self._help_title
        # Append if appropriate
        title += '' if not short_descr else f": {short_descr}"
        # Add divider to mark as title
        hline = '=' * len(title)
        # Return with a
        return f"{title}\n{hline}"

    def _genr8_help_description(self) -> str:
        r"""Generate longer description if necessary"""
        # Get description
        descr = self._help_description
        # Replace None -> ''
        descr = '' if descr is None else descr
        # Strip newline chars
        descr = descr.strip('\n')
        # Prepend two newline chars if content
        prefix = '\n\n' if descr else ''
        return prefix + descr

    def _genr8_help_usage(self) -> str:
        r"""Create the ``Usage`` portion of help message"""
        # Initialize message
        msg = f"\n\n:Usage:\n{TAB}.. code-block:: console\n\n"
        # Get character for prompt
        c = self._help_prompt
        # Generate prompt char(s) and space, if necessary
        strt = '' if not c else f"{c} "
        # Add prompt and program name
        msg += f"{TAB*2}{strt}{self._name}"
        # Get lists of args and options
        args = self._arglist
        opts = self._optlist
        # Check if we're doing a sub-command
        if self._cmdlist is not None:
            # Add message name
            return msg + " CMD [ARGS] [OPTIONS]"
        # Loop through required args
        for j in range(self._nargmin):
            # Add argument name
            msg += f" {args[j].upper()}"
        # Cover optional arguments
        if len(args) > self._nargmin:
            # Loop through optional args
            for j in range(self._nargmin, len(args)):
                msg += f" [{args[j].upper()}"
            # Close all the optional args
            msg += ']'*(len(args) - self._nargmin)
        # Append [OPTIONS] if necessary
        msg += " [OPTIONS]" if opts else ""
        # Output
        return msg

    def _genr8_help_cmdlist(self) -> str:
        r"""Create the list of available commands in *Inputs* section"""
        # Exit if none
        if self._cmdlist is None:
            return ""
        # Initialize list
        msg = "\n\n:Sub-commands:"
        # Get option list
        cmdlist = self._help_cmdlist
        # Default to _optlist if not defined
        cmdlist = cmdlist if cmdlist is not None else self._cmdlist
        # Loop through commands
        for cmdname in cmdlist:
            # Base default description
            cmdhelp0 = f"Run ``{cmdname}`` command"
            # Get default parser
            clsdef = self._cmdparsers.get("_default_", self.__class__)
            # Otherwise get sub-parser class
            cls = self._cmdparsers.get(cmdname, clsdef)
            # Get description from that parser
            cmdhelp1 = getattr(cls, "_help_title", cmdhelp0)
            # Get descriptionv
            cmdhelp = self._help_cmd.get(cmdname, cmdhelp1)
            # Display
            msg += f"\n{TAB}``{cmdname}``\n"
            msg += f"{TAB*2}{cmdhelp}\n"
        # Output
        return msg.rstrip('\n')

    def _genr8_help_args(self) -> str:
        # Exit if doing sub-commands
        if self._cmdlist is not None:
            return ''
        # Initialize empty message
        msg = ''
        # Add argument descriptions
        for j, arg in enumerate(self._arglist):
            # Add section header
            if j == 0:
                msg = "\n\n:Arguments:"
            # Get description
            descr = self._help_opt.get(arg, '')
            # Get default value
            vdef = self._rc.get(arg)
            # Initialize message
            msgj = f"\n{TAB}**{arg.upper()}**: {descr}"
            # Add default value
            msgvdef = '' if vdef is None else f" {vdef}"
            # Append
            msg += msgj + msgvdef
        # Output
        return msg

    def _genr8_help_options(self) -> str:
        # Get options formatting
        optmsg = self.genr8_optshelp()
        # Add section title
        msg = f"\n\n:Options:\n{optmsg}" if optmsg else ""
        # Output
        return msg

    def _genr8_help_optnames(self, opt: str) -> str:
        r"""Create option names for all aliases, ``-h, --help``"""
        # Get list of aliases
        names = self.get_aliases(opt)
        # Create message for each
        helpnames = [self._genr8_help_optname(name) for name in names]
        # Join them
        return ', '.join(helpnames)

    def _genr8_help_optname(self, opt: str) -> str:
        # Add --no- if *opt* is default-true (e.g. show ``--no-start``)
        optname = f"no-{opt}" if opt in self._help_opt_negative else opt
        prefix = '--' if len(optname) > 1 else '-'
        return prefix + optname

    def _genr8_help_coda(self) -> str:
        r"""Generate additional help at the end"""
        # Get description
        descr = self._help_extra
        # Replace None -> ''
        descr = '' if descr is None else descr
        # Strip newline chars
        descr = descr.strip('\n')
        # Prepend two newline chars if content
        prefix = '\n\n' if descr else ''
        return prefix + descr

  # *** VALIDATORS ***
   # --- Combined validator ---
    # Validate an option and raw value
    def validate_opt(self, rawopt: str, rawval: Any) -> OptPair:
        r"""Validate a raw option name and raw value

        Replaces *rawopt* with non-aliased name and applies any
        *optconverter* to *rawval*. Raises an exception if option name,
        type, or value does not match expectations.

        :Call:
            >>> optpair = opts.validate_opt(rawopt, rawval)
            >>> opt, val = opts.validate_opt(rawopt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
    def validate_optval(self, opt: str, rawval: Any):
        r"""Validate a raw option value

        :Call:
            >>> val = opts.validate_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
    def validate_argval(self, j: int, argname: str, rawval: Any):
        r"""Validate a raw positional parameter (arg) value

        :Call:
            >>> val = opts.validate_argval(j, argname, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
            *opts*: :class:`ArgReader`
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
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
        :Raises:
            :class:`ArgReadNameError` if *opt* is not recognized
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
        msg = f"unknown kwarg '{opt}' for parser '{self.__class__.__name__}'"
        # Add suggestions if able
        if len(matches):
            msg += "; nearest matches: %s" % " ".join(matches[:3])
        # Raise an exception
        raise ArgReadNameError(msg)

    # Check type (before applying converter)
    def check_rawopttype(self, opt: str, rawval: Any):
        r"""Check type of option value prior to conversion function

        :Call:
            >>> opts.check_rawopttype(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *rawval*: :class:`object`
                Raw user value for *opt* before using *optconverter*
        :Raises:
            :class:`ArgReadTypeError` if *rawval* has wrong type
        """
        # Get specified type or tuple of types or None
        cls_or_tuple = self.__class__.get_rawopttype(opt)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Otherwise check types
        assert_isinstance(rawval, cls_or_tuple, f"kwarg '{opt}'")

    # Apply converter, if any
    def apply_optconverter(self, opt: str, rawval: Any):
        r"""Apply option converter function to raw value

        :Call:
            >>> val = opts.apply_optconverter(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *rawval*: :class:`object`
                Raw user value for *opt* before using *optconverter*
        :Outputs:
            *val*: {*rawval*} | :class:`object`
                Result of calling *optconverter* for *opt* on *rawval*
        :Raises:
            :class:`ArgReadTypeError` if *optconverter* for *opt* is not
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
    def check_opttype(self, opt: str, val: Any):
        r"""Check type of option value after conversion function

        :Call:
            >>> opts.check_opttype(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *val*: :class:`object`
                Value for *opt*
        :Raises:
            :class:`ArgReadTypeError` if *val* has wrong type
        """
        # Get specified type or tuple of types or None
        cls_or_tuple = self.__class__.get_opttype(opt)
        # Check if there's a constraint
        if cls_or_tuple is None:
            return
        # Otherwise check types
        assert_isinstance(val, cls_or_tuple, f"kwarg '{opt}'")

    # Apply option value map, if any
    def apply_optvalmap(self, opt: str, rawval: Any):
        r"""Apply option value map (aliases for value), if any

        :Call:
            >>> val = opts.apply_optconverter(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
    def check_optval(self, opt: str, val: Any):
        r"""Check option value against list of recognized values

        :Call:
            >>> opts.check_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *opt*: :class:`str`
                De-aliased option name
            *val*: :class:`object`
                Value for *opt*
        :Raises:
            :class:`ArgReadValueError` if *opt* has an *optval* setting and
            *val* is not in it
        """
        # Get specified values
        optvals = self.__class__.get_optvals(opt)
        # No checks if *optvals* is not specified
        if optvals is None:
            return
        # Otherwise check value
        if val not in optvals:
            raise ArgReadValueError(f"kwarg '{opt}' invalid value {repr(val)}")

   # --- Single-property arg checkers ---
    # Check type (before applying converter)
    def check_rawargtype(self, j: int, argname: str, rawval: Any):
        r"""Check type of positional arg prior to conversion function

        :Call:
            >>> opts.check_rawargtype(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *rawval*: :class:`object`
                Value of option, before :data:`_optconverters`
        :Raises:
            :class:`ArgReadTypeError` if *rawval* has wrong type
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
    def apply_argconverter(self, j: int, argname: str, rawval: Any):
        r"""Apply option converter function to raw positional arg value

        :Call:
            >>> val = opts.apply_argconverter(j, argname, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
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
            :class:`ArgReadTypeError` if *optconverter* for *opt* is not
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
    def check_argtype(self, j: int, argname: str, val: Any):
        r"""Check type of positional arg after conversion function

        :Call:
            >>> opts.check_argtype(opt, val)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *val*: :class:`object`
                Value for parameter in position *j*, after conversion
        :Raises:
            :class:`ArgReadTypeError` if *val* has wrong type
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
    def check_argval(self, j: int, argname: str, val: Any):
        r"""Check positional arg value against list of recognized values

        :Call:
            >>> opts.check_optval(opt, rawval)
        :Inputs:
            *opts*: :class:`ArgReader`
                Keyword argument parser instance
            *j*: :class:`int`
                Positional parameter (arg) index
            *argname*: ``None`` | :class:`str`
                Positional parameter (arg) name, if appropriate
            *val*: :class:`object`
                Value for arg in position *j*
        :Raises:
            :class:`ArgReadValueError` if *argname* has an *optval*
            setting and *val* is not in it
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
            raise ArgReadValueError(f"{msg} invalid value {repr(val)}")

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
                A subclass of :class:`ArgReader`
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
                A subclass of :class:`ArgReader`
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
                A subclass of :class:`ArgReader`
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
                A subclass of :class:`ArgReader`
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
                A subclass of :class:`ArgReader`
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
            raise ArgReadTypeError(f"kwarg '{opt}' converter is not callable")
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
                A subclass of :class:`ArgReader`
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
                A subclass of :class:`ArgReader`
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
        :class:`ArgReader`.

        If *optlist* is an empty set, then no constraints are applied to
        option names.

        :Call:
            >>> optlist = cls.get_opttype(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
        :Outputs:
            *optlist*: :class:`set`\ [:class:`str`]
                Single type or list of types for *opt*
        """
        return cls.getx_cls_set("_optlist")

   # --- Arg lists ---
    # Get value of a class attr dict for an arg
    @classmethod
    def getx_cls_arg(
            cls: type,
            attr: str,
            argname: str,
            vdef: Optional[Any] = None) -> Any:
        r"""Get :class:`dict` class attribute for positional parameter

        If *argname* is ``None``, the parameter (arg) has no name, and
        only ``"_arg_default_"`` and ``"_default_"`` can be used from
        ``getattr(cls, attr)``.

        Otherwise, this will look in the bases of *cls* if
        ``getattr(cls, attr)`` does not have *argname*. If *cls* is a
        subclass of another :class:`ArgReader` class, it will search
        through the bases of *cls* until the first time it finds a class
        attribute *attr* that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.getx_cls_key(attr, key, vdef=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
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
    def getx_cls_key(
            cls: type,
            attr: str,
            key: str,
            vdef: Optional[Any] = None) -> Any:
        r"""Access *key* from a :class:`dict` class attribute

        This will look in the bases of *cls* if ``getattr(cls, attr)``
        does not have *key*. If *cls* is a subclass of another
        :class:`ArgReader` class, it will search through the bases of
        *cls* until the first time it finds a class attribute *attr*
        that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.getx_cls_key(attr, key, vdef=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
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
            if not issubclass(clsj, ArgReader):
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

        This allows a subclass of :class:`ArgReader` to only add to
        the ``_optlist`` attribute rather than manually include the
        ``_optlist`` of all the bases.

        :Call:
            >>> v = cls.getx_cls_set(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
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
            # Only recurse if ArgReader
            if issubclass(clsj, ArgReader):
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

        This allows a subclass of :class:`ArgReader` to only add to
        the ``_opttypes`` or ``_optmap`` attribute rather than manually
        include contents of all the bases.

        :Call:
            >>> clsdict = cls.getx_cls_dict(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`ArgReader`
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
            # Only recurse if ArgReader
            if issubclass(clsj, ArgReader):
                # Recurse
                clsdictj = clsj.getx_cls_dict(attr)
                # Combine, but don't overwrite *clsdict*
                for kj, vj in clsdictj.items():
                    clsdict.setdefault(kj, vj)
        # Output
        return clsdict


# Class with single_dash_split=False (default)
class KeysArgReader(ArgReader):
    r"""Subclass of :class:`ArgReader` for :func:`readkeys`

    The class attribute ``KeysArgRead.single_dash_split`` is set to
    ``False`` so that ``-opt val`` is interpreted as

    .. code-block:: python

        {"opt": "val"}
    """
    __slots__ = ()
    single_dash_split = False


# Class with single_dash_split=True (flags)
class FlagsArgReader(ArgReader):
    r"""Subclass of :class:`ArgReader` for :func:`readflags`

    The class attribute ``FlagsArgRead.single_dash_split`` is set to
    ``True`` so that ``-opt val`` is interpreted as

    .. code-block:: python

        {"o": True, "p": True, "t": True}

    and ``"val"`` becomes an argument.
    """
    __slots__ = ()
    single_dash_split = True


# Class with args like ``tar -cvf this.tar``
class TarFlagsArgReader(ArgReader):
    r"""Subclass of :class:`ArgReader` for :func:`readflags`

    The class attributes are

    .. code-block:: python

        TarArgRead.single_dash_split = True
        TarArgRead.single_dash_lastkey = True

    so that ``-opt val`` is interpreted as

    .. code-block:: python

        {"o": True, "p": True, "t": "val"}
    """
    __slots__ = ()
    single_dash_split = True
    single_dash_lastkey = True


# Read keys
def readkeys(argv=None):
    r"""Parse args where ``-cj`` becomes ``cj=True``

    :Call:
        >>> a, kw = readkeys(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of args other than ``sys.argv``
    :Outputs:
        *a*: :class:`list`\ [:class:`str`]
            List of positional args
        *kw*: :class:`dict`\ [:class:`str` | :class:`bool`]
            Keyword arguments
    """
    # Create parser
    parser = KeysArgReader()
    # Parse args
    return parser.parse(argv)


# Read keys
def readkeys_full(argv=None):
    r"""Parse args where ``-cj`` becomes ``cj=True``

    :Call:
        >>> a, kw = readkeys_full(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of args other than ``sys.argv``
    :Outputs:
        *a*: :class:`list`\ [:class:`str`]
            List of positional args
        *kw*: :class:`dict`\ [:class:`str` | :class:`bool`]
            Keyword arguments
    """
    # Create parser
    parser = KeysArgReader()
    # Parse args
    return parser.parse_cli_full(argv)


# Read flags
def readflags(argv=None):
    r"""Parse args where ``-cj`` becomes ``c=True, j=True``

    :Call:
        >>> a, kw = readflags(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of args other than ``sys.argv``
    :Outputs:
        *a*: :class:`list`\ [:class:`str`]
            List of positional args
        *kw*: :class:`dict`\ [:class:`str` | :class:`bool`]
            Keyword arguments
    """
    # Create parser
    parser = FlagsArgReader()
    # Parse args
    return parser.parse(argv)


# Read flags like ``tar`
def readflagstar(argv=None):
    r"""Parse args where ``-cf a`` becomes ``c=True, f="a"``

    :Call:
        >>> a, kw = readflags(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of args other than ``sys.argv``
    :Outputs:
        *a*: :class:`list`\ [:class:`str`]
            List of positional args
        *kw*: :class:`dict`\ [:class:`str` | :class:`bool`]
            Keyword arguments
    """
    # Create parser
    parser = TarFlagsArgReader()
    # Parse args
    return parser.parse(argv)


# Create a random string
def randomstr(n: int = 15) -> str:
    r"""Create a random string encded as a hexadecimal integer

    :Call:
        >>> txt = randomstr(n=15)
    :Inputs:
        *n*: {``15``} | :class:`int`
            Number of random bytes to encode
    """
    return b32encode(os.urandom(n)).decode().lower()


# Get default CLI args
def _get_argv(argv: Optional[list]) -> list:
    # Get sys.argv if needed
    argv = list(sys.argv) if argv is None else argv
    # Check for name of executable
    cmdname = argv[0]
    if cmdname.endswith("__main__.py"):
        # Get module name
        argv[0] = os.path.basename(os.path.dirname(cmdname))
    # Output
    return argv
