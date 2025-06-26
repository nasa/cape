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
"""

# Standard library
import difflib
import os
import re
import sys
from collections import namedtuple
from typing import Optional

# Local imports
from .clitext import compile_rst
from ._vendor.kwparse import (
    MetaKwargParser,
    KWTypeError,
    KwargParser,
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


# Custom error class
class ArgReadError(Exception):
    r"""Base error class for this package
    """
    pass


class ArgReadValueError(ValueError, Exception):
    pass


# Metaclass to combine _optlist and other class attributes
class MetaArgReader(MetaKwargParser):
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
    )


# Argument read class
class ArgReader(KwargParser, metaclass=MetaArgReader):
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
    :See also:
        * :class:`_vendor.kwparse.KwargParser`
    """
   # --- Class attributes ---
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

    #: List of options that cannot take a value:
    #: (:class:`tuple` | :class:`set`)\ [:class:`str`]
    _optlist_noval = ()

    #: Option to enforce ``_optlist``
    _restrict = False

    #: List of available commands
    _cmdlist = None

    #: Aliases for command names
    _cmdmap = {}

    #: Parser classes for sub-commands
    _cmdparsers = {}

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

    #: Base exception class: :class:`Exception`
    exc_cls = ArgReadError

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

   # --- __dunder__ ---
    def __init__(self):
        r"""Initialization method

        :Versions:
            * 2021-11-21 ``@ddalle``: v1.0
        """
        # Initialize attributes
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

   # --- Parsers ---
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
        :Versions:
            * 2024-11-11 ``@ddalle``: v1.0
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
        :Versions:
            * 2025-06-25 ``@ddalle``: v1.0
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
            *kw["__replaced__"]*: :class:`list`\ [(:class:`str`, *any*)]
                List of any options replaced by later values
        :Versions:
            * 2021-11-21 ``@ddalle``: v1.0
        """
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
            raise KWTypeError(
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
        # Output current values
        return self.get_args()

    def get_args(self) -> ArgTuple:
        r"""Get full list of args and options from parsed inputs

        :Call:
            >>> args, kwargs = parser.get_args()
        :Outputs:
            *args*: :class:`list`\ [:class:`str`]
                List of positional parameter argument values
            *kwargs*: :class:`dict`
                Dictionary of named options and their values
        :Versions:
            * 2023-11-08 ``@ddalle``: v1.0
        """
        # Get list of arguments
        args = list(self.argvals)
        # Get full dictionary of outputs, applying defaults
        kwargs = self.get_kwargs()
        # Output
        return ArgTuple(args, kwargs)

    def get_kwargs(self) -> dict:
        r"""Get full list of kwargs, including repeated values

        :Call:
            >>> args, kwargs = parser.get_args()
        :Outputs:
            *kwargs*: :class:`dict`
                Dictionary of named options and their values
        :Versions:
            * 2024-12-19 ``@ddalle``: v1.0
        """
        # Get full dictionary of outputs, applying defaults
        kwargs = KwargParser.get_kwargs(self)
        # Set __replaced__
        kwargs["__replaced__"] = [
            tuple(opt) for opt in self.kwargs_replaced]
        # Output
        return kwargs

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
        :Versions:
            * 2021-11-23 ``@ddalle``: v1.0
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
        :Versions:
            * 2024-11-11 ``@ddalle``: v1.0
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
        :Versions:
            * 2024-11-11 ``@ddalle``: v1.0
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
        :Versions:
            * 2024-11-11 ``@ddalle``: v1.0
        """
        # Check for alternates
        return self._cmdmap.get(cmdname, cmdname)

   # --- Reconstruction ---
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
        :Versions:
            * 2024-11-11 ``@ddalle``: v1.0
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
        :Versions:
            * 2021-11-23 ``@ddalle``: v1.0
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
        :Versions:
            * 2021-11-23 ``@ddalle``: v1.0
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
        :Versions:
            * 2021-11-23 ``@ddalle``: v1.0
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
        :Versions:
            * 2021-11-23 ``@ddalle``: v1.0
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

   # --- Help ---
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
    :Versions:
        * 2021-12-01 ``@ddalle``: v1.0
    """
    # Create parser
    parser = KeysArgReader()
    # Parse args
    return parser.parse(argv)


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
    :Versions:
        * 2021-12-01 ``@ddalle``: v1.0
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
    :Versions:
        * 2021-12-01 ``@ddalle``: v1.0
    """
    # Create parser
    parser = TarFlagsArgReader()
    # Parse args
    return parser.parse(argv)


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
