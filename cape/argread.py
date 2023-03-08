r"""
Parse command-line inputs based on one of two methods.  The first method
counts both "-" and "--" as prefixes for keyword names; this is common
among many advanced programs.  For example, the two following examples
would be treated as equivalent (assuming it is called by some script
```myScript.py``.

    .. code-block:: console

        $ myScript.py --v --i test.txt
        $ myScript.py -v -i test.txt

The second method assumes single-hyphen options are single-character
flags that can be combined.  This is common in many built-in Unix/Linux
utilities. Consider how ``ls -lh`` is interpreted.  The following two
examples would be interpreted equivalently.

    .. code-block:: console

        $ myScript.py -v -i
        $ myScript.py -vi

A third method is provided to have similar behavior to the Unix ``tar``
command. In this case, the following two commands will be different.

    .. code-block:: console

        $ myScript.py -cf mytar.tar
        $ myScript.py --cf mytar.tar

The first example sets *c* to ``True`` and *f* to ``"mytar.tar"``; the
second command sets *cf* to ``"mytar.tar"``.
"""

# Standard library
import re
import sys


# Create fake "unicode" type for checking strings in all versions
if sys.version_info.major > 2:
    unicode = str

# Create a "string" type
strlike = (str, unicode)


# Regular expression for options like "cdfr=1.3"
REGEX_EQUALKEY = re.compile(r"(\w+)=([^=].*)")


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
        * 2021-12-01 ``@ddalle``: Version 1.0
    """
    # Create parser
    parser = ArgumentReader(single_dash_split=False)
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
        * 2021-12-01 ``@ddalle``: Version 1.0
    """
    # Create parser
    parser = ArgumentReader(
        single_dash_split=True,
        single_dash_lastkey=False)
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
        * 2021-12-01 ``@ddalle``: Version 1.0
    """
    # Create parser
    parser = ArgumentReader(
        single_dash_split=True,
        single_dash_lastkey=True)
    # Parse args
    return parser.parse(argv)


# Argument read class
class ArgumentReader(object):
    __slots__ = (
        "argv",
        "args",
        "kwargs",
        "prog",
        "kwargs_sequence",
        "kwargs_replaced",
        "kwargs_single_dash",
        "kwargs_double_dash",
        "kwargs_equal_sign",
        "param_sequence",
        "single_dash_split",
        "single_dash_lastkey",
        "equal_sign_key")

    r"""Class to parse arguments

    :Call:
        >>> parser = ArgumentReader(**kw)
    :Inputs:
        *single_dash_split*: ``True`` | {``False``}
            Option to split ``-cj`` into ``c=True, j=True``
        *single_dash_lastkey*: ``True`` | {``False``}
            Option to interpret ``-cf 1`` as ``c=True, f="1"``
        *equal_sign_key*: {``True``} | ``False``
            Option to interpret ``a=1`` as ``a="1"`` (keyword)
    :Outputs:
        *parser*: :class:`ArgumentReader`
            Instance of command-line argument parser
    :Versions:
        * 2021-12-03 ``@ddalle``: Version 1.0
    """
    def __init__(self, **kw):
        r"""Initialization method

        :Versions:
            * 2021-11-21 ``@ddalle``: Version 1.0
        """
        # Initialize attributes
        self.argv = []
        self.prog = None
        self.args = []
        self.kwargs = {}
        self.kwargs_sequence = []
        self.kwargs_replaced = []
        self.kwargs_single_dash = {}
        self.kwargs_double_dash = {}
        self.kwargs_equal_sign = {}
        self.param_sequence = []
        # Parse modes
        self.single_dash_split = kw.pop("single_dash_split", False)
        self.single_dash_lastkey = kw.pop("single_dash_lastkey", False)
        self.equal_sign_key = kw.pop("equal_sign_key", True)

    def parse(self, argv=None, **kw):
        r"""Parse args

        :Call:
            >>> a, kw = parser.parse(argv=None)
        :Inputs:
            *parser*: :class:`ArgumentReader`
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
            * 2021-11-21 ``@ddalle``: Version 0.1; started
        """
        # Process optional args
        if argv is None:
            # Copy *sys.argv*
            argv = list(sys.argv)
        elif not isinstance(argv, list):
            # Wrong type
            raise TypeError(
                "Expected arg 'argv' to be type 'list'; " +
                "got '%s'" % type(argv).__name__)
        else:
            # Check each arg is a string
            for j, arg in enumerate(argv):
                # Check type
                if isinstance(arg, strlike):
                    continue
                # Bad type
                raise TypeError(
                    ("Argument %i: expected type 'list' " % j) +
                    ("but got '%s'" % type(arg).__name__))
            # Copy args
            argv = list(argv)
        # Save copy of args to *self*
        self.argv = list(argv)
        # (Re)initialize attributes storing parsed arguments
        self.args = []
        self.kwargs = {}
        self.kwargs_sequence = []
        self.kwargs_replaced = []
        self.kwargs_single_dash = {}
        self.kwargs_double_dash = {}
        self.kwargs_equal_sign = {}
        self.param_sequence = []
        # Check for command name
        if len(argv) == 0:
            raise IndexError("Expected at least one argv entry (program name)")
        # Save command name
        self.prog = argv.pop(0)
        # Global parse modes
        splitflags = kw.get("single_dash_split")
        lastflag = kw.get("single_dash_lastkey")
        equalkey = kw.get("equal_sign_key")
        # Save if needed
        if splitflags is not None:
            self.single_dash_split = not not splitflags
        if lastflag is not None:
            self.single_dash_lastkey = not not lastflag
        if equalkey is not None:
            self.equal_sign_key = not not equalkey
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
            # Check option/arg type
            if prefix == "":
                # Positional parameter
                self.save_arg(val)
                continue
            elif prefix == "=":
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
            # Check if next arg is available
            if len(argv) == 0:
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
        # Form output
        a = list(self.args)
        kw = dict(self.kwargs)
        kw["__replaced__"] = [tuple(opt) for opt in self.kwargs_replaced]
        # Output
        return a, kw

    # Parse a single arg
    def _parse_arg(self, arg):
        r"""Parse type for a single CLI arg

        :Call:
            >>> prefix, key, val, flags = parser._parse_arg(arg)
        :Inputs:
            *parser*: :class:`ArgumentReader`
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
            * 2021-11-23 ``@ddalle``: Version 1.0
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

    def save_arg(self, arg):
        r"""Save a positional argument

        :Call:
            >>> parser.save_arg(arg, narg=None)
        :Inputs:
            *parser*: :class:`ArgumentReader`
                Command-line argument parser
            *arg*: :class:`str`
                Name/value of next parameter
        :Versions:
            * 2021-11-23 ``@ddalle``: Version 1.0
        """
        self._save(None, arg)

    def save_double_dash(self, k, v=True):
        r"""Save a double-dash keyword and value

        :Call:
            >>> parser.save_double_dash(k, v=True)
        :Inputs:
            *parser*: :class:`ArgumentReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: {``True``} | ``False`` | :class:`str`
                Value to save
        :Versions:
            * 2021-11-23 ``@ddalle``: Version 1.0
        """
        self._save(k, v)
        self.kwargs_double_dash[k] = v

    def save_equal_key(self, k, v):
        r"""Save an equal-sign key/value pair, like ``"mach=0.9"``

        :Call:
            >>> parser.save_equal_key(k, v)
        :Inputs:
            *parser*: :class:`ArgumentReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: :class:`str`
                Value to save
        :Versions:
            * 2021-11-23 ``@ddalle``: Version 1.0
        """
        self._save(k, v)
        self.kwargs_equal_sign[k] = v

    def save_single_dash(self, k, v=True):
        r"""Save a single-dash keyword and value

        :Call:
            >>> parser.save_single_dash(k, v=True)
        :Inputs:
            *parser*: :class:`ArgumentReader`
                Command-line argument parser
            *k*: :class:`str`
                Name of key to save
            *v*: {``True``} | ``False`` | :class:`str`
                Value to save
        :Versions:
            * 2021-11-23 ``@ddalle``: Version 1.0
        """
        self._save(k, v)
        self.kwargs_single_dash[k] = v

    def _save(self, k, v):
        # Append to universal list of args
        self.param_sequence.append((k, v))
        # Check option vs arg
        if k is None:
            # Save arg
            self.args.append(v)
        else:
            # Universal keyword arg sequence
            self.kwargs_sequence.append((k, v))
            # Check if a previous key
            if k in self.kwargs:
                # Save to "replaced" options
                self.kwargs_replaced.append((k, self.kwargs[k]))
            # Save to current kwargs
            self.kwargs[k] = v

