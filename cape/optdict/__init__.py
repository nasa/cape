#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`optdict`: Advanced :class:`dict`-based options interface
===============================================================

This class provides the module :class:`OptionsDict`, which is a Python
class based on :class:`dict` with a variety of advanced features and an
interface to and from JSON files.

The :class:`OptionsDict` can be used in isolation as an alternative to
:class:`dict` but that can be initiated from a JSON file name. For
example if the file ``settings.json`` has the contents

.. code-block:: javascript

    {
        "a": 3,
        "b": true,
        "c": ["fun", "bad"],
        "d": JSONFile("d.json")
    }

and ``d.json`` contains

.. code-block:: javascript

    {
        "north": 0.0,
        "south": 180.0 // heading angle
    }

Then an :class:`OptionsDict` can be instantiated using

.. code-block:: python

    opts = OptionsDict("settings.json")

Note two features of this example:

    * including other JSON files using the :func:`JSONFile` directive
    * comments using the ``//`` syntax

The other way to use :class:`OptionsDict` is to create subclasses, and
the reason to do this is to only allow specified option names (keys) or
to only allow certain types and/or values for some keys.

.. code-block:: python

    class MyOpts(OptionsDict):
        _optlist = {
            "a",
            "b",
            "c"
        }
        _opttypes = {
            "a": int,
            "b": str,
            "c": (dict, list)
        }
        _optvals = {
            "a": {0, 1, 2}
        }
        _rc = {
            "b": "fun"
        }

    opts = MyOpts(c={"on": True}, a=2)

Now if we try to set a parameter that's not in ``_optlist``, we'll
either get ignored, warned, or raise an exception, depending on the
"warning mode":

.. code-block:: pycon

    >>> opts.set_opt("d", 1)
    Unknown 'MyOpts' option 'd'

We can also access the default value for *b* if it has not been set:

.. code-block:: pycon

    >>> opts.get_opt("b")
    "fun"

Types and values can be constrained and checked seperately on assignment
and retrieval.

.. code-block:: pycon

    >>> opts.set_opt("a", 1.0)
    For 'MyOpts' option 'a': got type 'float'; expected 'int'
    >>> opts.set_opt("b", 1.0, mode=WARNMODE_NONE)
    >>> opts["b"]
    1.0
    >>> opts.get_opt("b")
    For 'MyOpts' option 'b': got type 'float'; expected 'str'
    >>> opts.set_opt("a", 3)
    Unknown value 3 for MyOpts option 'a'; available options: 0 1 2

The type checkers can have somewhat surprising behavior if trying to
use a :class:`list`.

.. code-block:: pycon

    >>> opt.set_opt("c", [1, 2], mode=WARNMODE_WARN)
    ... option 'c' phase 0: got type 'int'; expected 'dict' | 'list'

This seems confusing because ``[1, 2]`` is a :class:`list`, which seems
to be allowed. However, :mod:`optdict` (by default, anyway) interprets
``[1, 2]`` as ``1`` for phase 0 and ``2`` for phases 1 and beyond:

.. code-block:: pycon

    >>> opts.set_opt("c", [1, 2], mode=WARNMODE_NONE)
    >>> opts.get_opt('c", j=0, mode=WARNMODE_NONE)
    1

There are two solutions to these cases where a :class:`list` might be
intended as an atomic input value (not a sequence of phase-specific
values):

.. code-block:: pycon

    >>> opts.set_opt("c", [[1, 2]]
    >>> opts.get_opt("c", j=0)
    [1, 2]
    >>> opts.get_opt("c")
    [[1, 2]]
    >>> MyOpts._optlistdepth["c"] = 1
    >>> opts.set_opt("c", [1, 2])
    >>> opts.get_opt("c", j=0)
    [1, 2]
    >>> opts.get_opt("c")
    [1, 2]

.. note::

    Type checking for both :class:`int` and :class:`float` can be tricky
    when dealing with numeric data, which may originate from
    :class:`numpy` types that look like integers but fail the
    :func:`isinstance` test. For example

    .. code-block:: pycon

        >>> isinstance(np.arange(4)[0], int)
        False

    To help in this situation, :mod:`optdict` provides some extra
    :class:`tuple`\ s of :class:`type`\ s that include most
    :class:`numpy` types such as :class:`int32`, :class:`int64`,
    :class:`float32`, etc.

    .. code-block:: pycon

        >>> isinstance(np.arange(4)[0], INT_TYPES)
        True

    The special type lists provided are

    +-----------------+----------------------------------------------+
    | ``ARRAY_TYPES`` | :class:`list`, :class:`tuple`,               |
    |                 | :class:`np.ndarray`                          |
    +-----------------+----------------------------------------------+
    | ``BOOL_TYPES``  | :class:`bool`, :class:`bool_`                |
    +-----------------+----------------------------------------------+
    | ``FLOAT_TYPES`` | :class:`float`, :class:`float16`,            |
    |                 | :class:`float32`, :class:`float64`,          |
    |                 | :class:`float128`                            |
    +-----------------+----------------------------------------------+
    | ``INT_TYPES``   | :class:`int`, :class:`int8`, :class:`int16`, |
    |                 | :class:`int32`, :class:`int64`,              |
    |                 | :class:`uint8`, :class:`uint16`,             |
    |                 | :class:`uint32`, :class:`uint64`             |
    +-----------------+----------------------------------------------+


.. _optdict-phases:

Phases
--------
Each entry in an instance of :class:`OptionsDict` is interpreted not as
a static value but a sequence of inputs for different "phases" of some
operation such as running a Computational Fluid Dynamics (CFD) solver.
For example, (by default) a :class:`list` entry such as

.. code-block:: python

    {"PhaseIters": [100, 200]}

means that the setting *PhaseIters* will take the value ``100`` for
phase 0 (using Python's 0-based indexing) and ``200`` for phase 1. In
addition, this ``200`` setting is automatically extended to phases 2, 3,
etc. to allow for more consolidated input. If the options instance
instead has

.. code-block:: python

    {"PhaseIters": 100}

then the value of *PhaseIters* is ``100`` for all possible phases.

Note that the duality of this example means that :mod:`optdict` must
distinguish between array-like and non-array-like entries, which in
some cases can introduce extra considerations. For example, consider a
setting called *Schedule* that for every phase is a list of two
:class:`int`\ s. Setting a value of ``[50, 100]`` will not produce the
desired results because it will be interpreted as ``50`` for phase 0
rather than the full :class:`list` for all phases. The simplest solution
is to use

.. code-block:: python

    {"Schedule": [[50, 100]]}

It is also possible to use the *listdepth* keyword argument to
:func:`OptionsDict.get_opt` or define *Schedule* to be special by
setting the ``_listdepth`` :ref:`class attribute <optdict-cls-attr>`
in a subclass.

.. code-block:: python

    class MyOpts(OptionsDict):
        _listdepth = {
            "Schedule": 1
        }

    opts = MyOpts(Schedule=[50, 100])
    assert opts.get_opt("Schedule", j=0) == [50, 100]

This element-access approach is implemented by the function
:func:`optitem.getel`.


.. _optdict-special-dict:

Special Dictionaries
-------------------------
Options in an instance of :class:`OptionsDict` can also be defined as
"special :class:`dict`\ s," of which there are are four types. These
work in combination with a run matrix that's stored in the *x* attribute
of the options instance.

.. code-block:: python

    # Define run matrix
    x = {
        "mach": [0.5, 0.75, 1.0, 1.25, 1.5],
        "alpha": 0.0,
        "beta": 0.0,
        "arch": ["cas", "rom", "rom", "bro", "sky"],
    }
    # Define options
    opts = OptionsDict({
        "my_expression": {"@expr": "$mach * $mach - 1"},
        "my_constrained_expr": {
            "@cons": {
                "$mach <= 1": 0.0,
                "True": {"@expr": "$mach * $mach - 1"}
            }
        },
        "my_map": {
            "@map": {"cas": 10, "rom": 5, "sky": 20, "_default_": 40},
            "key": "arch"
        },
        "my_raw_value": {
            "@raw": {"@expr": "$mach * $mach - 1"}
        },
    })
    # Save trajectory
    opts.set_x(x)

The ``@expr`` tag allows users to define an arbitrary expression with
run matrix keys from *opts.x* denoted using a ``$`` sigil. Users may
also assume that :mod:`numpy` has been imported as ``np``. The following
example evaluates :math:`M^2-1` for case ``i=3``, where ``mach=1.25``.

.. code-block:: pycon

    >>> opts.get_opt("my_expression", i=3)
    0.5625

The next type of special dict is the ``@cons`` tag, which allows for
similar expressions on the left-hand side. The definition for
*my_constrained_expr* above is equivalent to the regular code

.. code-block:: python

    if mach <= 1:
        return 0.0
    else:
        return mach * mach - 1

Here the expression ``True`` serves as an ``else`` clause because it
is evaluated as ``elif True``.

The third type of special dict is ``@map``, which allows a user to
define specific values (or another special dict) based on the value of
some other key in *opts.x*, and the value of that key must be a
:class:`str`.

.. code-block:: pycon

    >>> opts.x["arch"][2]
    'rom'
    >>> opts.get_opt("my_map", i=2)
    5
    >>> opts.x["arch"][3]
    'bro'
    >>> opts.get_opt("my_map", i=3)
    40
    >>> opts.x["arch"][4]
    'sky'
    >>> opts.get_opt("my_map", i=4)
    20

Finally, a ``@raw`` value allows users to specify a raw value that might
otherwise get expanded.

.. code-block:: pycon

    >>> opts.get_opt("my_raw_value", i=1)
    {"@expr": "$mach * $mach - 1"}

These special dicts may also recurse.


.. _optdict-cls-attr:

Class Attributes
-------------------
Below is a discussion of the data attributes of the :class:`OptionsDict`
class and their purposes. The class attributes, despite having names
starting with ``_``, are intended for use when developers or users
create subclasses of :class:`OptionsDict`.

The class attributes naturally combine, so that second- or higher-order
subclasses need not repeat attributes like ``_optlist``. So in the
example

.. code-block:: python

    class MyOpts1(OptionsDict):
        _optlist = {"a"}
        _opttypes = {"a": str}

    class MyOpts2(MyOpts1):
        _optlist = {"b"}
        _opttypes = {"b": (int, float)}

instances of :class:`MyOpts2` will allow options named either *a* or *b*
and check the types of both, but options with any other name will not be
allowed in such :class:`MyOpts2` instances.

``OptionsDict._optlist``: :class:`set`
    If nonempty, a :class:`set` of allowed option names.

    See also:
        * :func:`OptionsDict.get_optlist`
        * :func:`OptionsDict.check_optname`

``OptionsDict._opttypes``: :class:`dict`
    Specified :class:`type` or :class:`tuple`\ [:class:`type`] that
    restricts the allowed type for each specified option.  For example

    .. code-block:: python

        _opttypes = {
            "a": str,
            "b": (int, float)
        }

    will require the two tests

    .. code-block:: python

        isinstance(opts["a"], str)
        isinstance(opts["b"], (int, float))

    to both return ``True``.

    See also:
        * :func:`OptionsDict.get_opttype`
        * :func:`OptionsDict.check_opttype`

``OptionsDict._optmap``: :class:`dict`\ [:class:`str`]
    Dictionary of aliases where the key is the alias and the value is
    the full name of an option. In other words, any attempt to get or
    set an option ``key`` will really affect ``_optmap.get(key, key)``.

    .. code-block:: python

        class MyOpts(OptionsDict):
            _optlist = {"Alpha", "Beta"}
            _optmap = {
                "a": "Alpha",
                "A": "Alpha",
                "b": "Beta",
                "B": "Beta"
            }

    This class has two options, ``"Alpha"`` and ``"Beta"``, but six ways
    to set them.

    .. code-block:: pycon

        >>> opts = MyOpts(a=3)
        >>> opts
        {'Alpha': 3}

    See also:
        * :func:`OptionsDict.apply_optmap`

``OptionsDict._optvals``: :class:`dict`\ [:class:`set`]
    Dictionary of specific values allowed for certain options. This
    attribute is mostly suited for string options, for which a limited
    set of possible values is more likely.

    When a user tries to set a value that's not in the possibilities
    specified in ``_optvals``, they will also see a helpful warning of
    the closest allowed option(s).

    The canonical type for the values in ``_optvals`` is :class:`set`,
    but a :class:`tuple` may be better suited when only a few values
    are permitted.

    See also:
        * :func:`OptionsDict.check_optval`

``OptionsDict._optlistdepth``: :class:`dict`\ [:class:`int`]
    List depth for named options. If ``_optlistdepth[opt]`` is ``1``,
    then a :class:`list` for *opt* is interpreted as a scalar.

    See also:
        * :func:`OptionsDict.get_opt`
        * :func:`OptionsDict.get_listdepth`

``OptionsDict._optlist_ring``: :class:`set`
    A :class:`set` of options for which a phase list repeats from the
    beginning. Normally the last entry would just repeat indefinitely

    See also:
        * :func:`OptionsDict.get_opt`

``OptionsDict._rc``: :class:`dict`
    Default value for any option

    See also:
        * :func:`OptionsDict.get_opt`
        * :func:`OptionsDict.get_cls_key`

``OptionsDict.__slots__``: :class:`tuple`\ [:class:`str`]
    Tuple of attributes allowed in instances of :class:`OptionsDict`.
    This prevents users from setting arbitrary attributes of
    :class:`OptionsDict` instances.

    When subclassing :class:`OptionsDict`, this behavior will go away
    unless the following syntax is used:

    .. code-block:: python

        class MyOpts(OptionsDict):
            __slots__ = tuple()

    Alternatively, developers may allow specified **additional** slots
    by listing those in a tuple as a class attribute of the subclass.

``OptionsDict._rst_types``: :class:`dict`\ [:class:`str`]
    Prespecified string to use as the type and/or default value for
    automatic property get/set functions. A default can be constructed
    based on ``OptionsDict._opttypes``.

    See also:
        * :func:`OptionsDict.add_property`
        * :func:`OptionsDict.add_getter`
        * :func:`OptionsDict.add_setter`

``OptionsDict._rst_descriptions``: :class:`dict`\ [:class:`str`]
    Documentation string used to describe one or more options, to be
    used in automatic property get/set functions.

    See also:
        * :func:`OptionsDict.add_property`
        * :func:`OptionsDict.add_getter`
        * :func:`OptionsDict.add_setter`


Classes and Methods
----------------------

"""

# Standard library
import copy
import difflib
import functools
import io
import json
import math
import os
import re
import sys

# Third-party
import numpy as np

# Local imports
from . import opterror
from . import optitem
from .opterror import (
    OptdictAttributeError,
    OptdictKeyError,
    OptdictJSONError,
    OptdictNameError,
    OptdictTypeError,
    OptdictValueError,
    assert_isinstance)
from .optitem import check_scalar


# Regular expression for JSON file inclusion
REGEX_JSON_INCLUDE = re.compile(
    r'(?P<cmd>JSONFile\("(?P<json>[-\w.+= /\\]+)"\))')
# Regular expression to recognize a quote
REGEX_QUOTE = re.compile('"(.*?)"')
# Regular expression to strip comment to end of line
REGEX_COMMENT = re.compile("//.*$")
# Simple finder of all non-word chars
_REGEX_W = re.compile(r"\W")
_REGEX_D = re.compile(r"^\d+")

# Warning mode flags
WARNMODE_NONE = 0
WARNMODE_QUIET = 1
WARNMODE_WARN = 2
WARNMODE_ERROR = 3

# Defaults
DEFAULT_WARNMODE = WARNMODE_WARN
DEFAULT_WARNMODE_INAME = WARNMODE_WARN
DEFAULT_WARNMODE_ITYPE = WARNMODE_NONE
DEFAULT_WARNMODE_ONAME = WARNMODE_ERROR
DEFAULT_WARNMODE_OTYPE = WARNMODE_ERROR

# Index for overall warning mode tuple
INDEX_INAME = 0
INDEX_ITYPE = 1
INDEX_ONAME = 2
INDEX_OTYPE = 3

# Types
ARRAY_TYPES = (
    list,
    tuple,
    np.ndarray)
FLOAT_TYPES = (
    float,
    np.float16,
    np.float32,
    np.float64,
    np.float128)
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
BOOL_TYPES = (
    bool,
    np.bool_)

# Other settings (internal)
_MAX_OPTVALS_DISPLAY = 4

# Text descriptions
_RST_WARNMODE_NONE = f":``{WARNMODE_NONE}``: no checks"
_RST_WARNMODE_QUIET = f":``{WARNMODE_QUIET}``: validate silently"
_RST_WARNMODE_WARN = f":``{WARNMODE_WARN}``: validate and show warnings"
_RST_WARNMODE_ERROR = f":``{WARNMODE_ERROR}``: raise an exception if invalid"
_RST_WARNMODE_LIST = "``0`` | ``1`` | ``2`` | ``3``"
_RST_WARNMODE1 = f"""Warning mode code

            {_RST_WARNMODE_NONE}
            {_RST_WARNMODE_QUIET}
            {_RST_WARNMODE_WARN}
            {_RST_WARNMODE_ERROR}"""
_RST_WARNMODE2 = f"""Warning mode code

                {_RST_WARNMODE_NONE}
                {_RST_WARNMODE_QUIET}
                {_RST_WARNMODE_WARN}
                {_RST_WARNMODE_ERROR}"""
_RST_FLOAT_TYPES = ":class:`float` | :class:`float32`"
_RST_INT_TYPES = ":class:`int` | :class:`int32` | :class:`int64`"
# Dictionary of various text to expand
_RST = {
    "_RST_WARNMODE_LIST": _RST_WARNMODE_LIST,
    "_RST_WARNMODE1": _RST_WARNMODE1,
    "_RST_WARNMODE2": _RST_WARNMODE2,
}
_RST_GETOPT = r"""*j*: {``None``} | :class:`int`
                Phase index; use ``None`` to just return *v*
            *i*: {``None``} | :class:`int` | :class:`np.ndarray`
                *opts.x* index(es) to use with ``@expr``, ``@map``, etc.
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
            *ring*: {*opts._optring[key]*} | ``True`` | ``False``
                Override option to loop through phase inputs
            *listdepth*: {``0``} | :class:`int` > 0
                Depth of list to treat as a scalar
            *x*: {``None``} | :class:`dict`
                Reference conditions to use with ``@expr``, ``@map``, etc.;
                often a run matrix; used in combination with *i*""" % _RST
_RST_SETOPT = r"""*j*: {``None``} | :class:`int`
                Phase index; use ``None`` to just return *v*
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
            *listdepth*: {``0``} | :class:`int` > 0
                Depth of list to treat as a scalar""" % _RST
_RST["_RST_GETOPT"] = _RST_GETOPT
_RST["_RST_SETOPT"] = _RST_SETOPT


# Expand docstring
def expand_doc(func_or_cls):
    # Expand the docstring
    func_or_cls.__doc__ = func_or_cls.__doc__ % _RST
    # Return existing object
    return func_or_cls


# Main class
@expand_doc
class OptionsDict(dict):
    r"""Advanced :class:`dict`-based options interface

    :Call:
        >>> opts = OptionsDict(fjson, **kw)
        >>> opts = OptionsDict(mydict, **kw)
    :Inputs:
        *fjson*: :class:`str`
            Name of JSON file to read
        *mydict*: :class:`dict` | :class:`OptionsDict`
            Another :class:`dict` instance
        *_name*: {``None``} | :class:`str`
            Name to use in error messages and warnings
        *_x*: {``None``} | :class:`dict`
            Supporting values such as run matrix
        *_warnmode*: {``None``} | %(_RST_WARNMODE_LIST)s
            %(_RST_WARNMODE1)s
        *_warnmode_iname*: {*_warnmode*} | %(_RST_WARNMODE_LIST)s
            Warning mode for checking input option names
        *_warnmode_itype*: {*_warnmode*} | %(_RST_WARNMODE_LIST)s
            Warning mode for checking input value and type
        *_warnmode_oname*: {*_warnmode*} | %(_RST_WARNMODE_LIST)s
            Warning mode for checking output option names
        *_warnmode_otype*: {*_warnmode*} | %(_RST_WARNMODE_LIST)s
            Warning mode for checking output value and type
        *kw*: :class:`dict`
            Keyword arguments interpreted as option/value pairs
    :Outputs:
        *opts*: :class:`OptionsDict`
            Options interface
    :Slots:
        *opts.x*: ``None`` | :class:`dict`
            Supporting values such as run matrix
        *opts._xoptlist*: ``None`` | :class:`set`
            Option names specifically allowed for this instance
        *opts._xopttypes*: :class:`dict`
            Type or tuple of types allowed for some or all options
        *opts._xoptvals*: :class:`dict`
            Instance-specific values allowed for some or all options
    :Versions:
        * 2021-12-05 ``@ddalle``: Version 0.1; started
        * 2021-12-06 ``@ddalle``: Version 1.0
        * 2022-09-20 ``@ddalle``: Version 1.1; get_opt() w/ x
        * 2022-09-24 ``@ddalle``: Version 1.2: *_warnmode*
        * 2022-09-30 ``@ddalle``: Version 1.3: *_rc* and four warnmodes
    """
  # *** CLASS ATTRIBUTES ***
   # --- Slots ---
    # Allowed attributes
    __slots__ = (
        "name",
        "x",
        "_xoptlist",
        "_xopttypes",
        "_xoptvals",
        "_xrc",
        "_xwarnmode",
        "_lastwarnmsg",
        "_lastwarnmode",
        "_lastwarncls",
        "_jsondir",
        "_filenames",
        "_lines",
        "_code",
        "_filenos",
        "_linenos",
        "_sourcelinenos",
    )

   # --- Option lists ---
    # All accepted options
    _optlist = set()

    # No non-default list depth declarations
    _optlistdepth = {}

    # No keys sampled as ring by default
    _optlist_ring = set()

    # Alternate names
    _optmap = {}

    # Type(s) for each option
    _opttypes = {}

    # Specifically allowed values
    _optvals = {}

    # Transformations/aliases for option values
    _optvalmap = {}

    # Converters (before value checking)
    _optval_converters = {}

    # Defaults
    _rc = {}

   # --- Labeling ---
    _name = ""

   # --- Documentation ---
    # Documentation type strings
    _rst_types = {}

    # Option descriptions
    _rst_descriptions = {}

   # --- Sections ---
    # Section class instantiators
    _sec_cls = {}

    # Prefix for each section
    _sec_prefix = {}

    # Parent section spec
    _sec_parent = {}

   # --- Settings ---
    _warnmode = DEFAULT_WARNMODE

   # --- Auto-properties and documentation ---
    # Pre-written strings for available types/values (can be automated)
    _rst_types = {}

    # Verbal descriptions of options
    _rst_descriptions = {}

  # *** CONFIG ***
   # --- __dunder__ ---
    def __init__(self, *args, **kw):
        r"""Initialization method

        :Versions:
            * 2021-12-06 ``@ddalle``: Version 1.0
            * 2022-09-19 ``@ddalle``: Version 1.1; fix *a* name conflict
            * 2022-09-20 ``@ddalle``: Version 1.2; allow *x* as args[1]
            * 2022-09-30 ``@ddalle``: Version 1.3; eliminate args[1]
            * 2022-10-04 ``@ddalle``: Version 1.4; int_post hook
            * 2022-10-10 ``@ddalle``: Version 1.5; init sections, name
        """
        # Initialize attributes
        self.name = ""
        self._init_json_attributes()
        self._init_optlist_attributes()
        self._init_lastwarn()
        # Process args
        narg = len(args)
        # Process up to one arg
        if narg == 0:
            # No args
            a = None
        elif narg == 1:
            # One arg
            a = args[0]
        else:
            # Too many args
            raise OptdictTypeError(
                "%s() takes 0 to 1 arguments, but %i were given" %
                (type(self).__name__, narg))
        # Save run matrix
        self.x = kw.pop("_x", None)
        self.name = kw.pop("_name", None)
        # get warning mode options
        xwarnmode = kw.pop("_warnmode", None)
        xmode_iname = kw.pop("_warnmode_iname", xwarnmode)
        xmode_itype = kw.pop("_warnmode_itype", xwarnmode)
        xmode_oname = kw.pop("_warnmode_oname", xwarnmode)
        xmode_otype = kw.pop("_warnmode_otype", xwarnmode)
        # Save warning mode
        self._xwarnmode = (xmode_iname, xmode_itype, xmode_oname, xmode_otype)
        # Check sequential input type
        if isinstance(a, str):
            # Read JSON file
            self.read_jsonfile(a)
        elif isinstance(a, dict):
            # Merge into self
            self.set_opts(a)
        elif a is not None:
            # Bad type
            raise OptdictTypeError(
                "Expected file name or dict, got '%s'" % type(a).__name__)
        # Merge in *kw*
        self.set_opts(kw)
        # Process sections
        self.init_sections()
        # Run final hook for custom actions
        self.init_post()

   # --- Init assistants ---
    def _init_optlist_attributes(self):
        self._xoptlist = None
        self._xopttypes = None
        self._xoptvals = None
        self._xrc = None
        self._xwarnmode = None

    def _init_lastwarn(self):
        self._lastwarnmsg = None
        self._lastwarnmode = None
        self._lastwarncls = None

    def _init_json_attributes(self):
        # Delete/reset attributes
        self._jsondir = None
        self._filenames = None
        self._lines = None
        self._code = None
        self._filenos = None
        self._linenos = None
        self._sourcelinenos = None

   # --- Hooks ---
    def init_post(self):
        r"""Custom function to run at end of :func:`__init__`

        This function can be used on subclasses of :class:`OptionsDict`
        to do extra special initiation without needing to rewrite the
        instantiation function.

        The :func:`OptionsDict.init_post` function performs no actions,
        but a subclass can redefine this function to do custom tasks for
        every new instance of that class.

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Versions:
            * 2022-10-04 ``@ddalle``: Version 1.0
        """
        pass

   # --- Sections ---
    def init_sections(self):
        r"""Initialize standard sections for a class

        :Call:
            >>> opts.init_sections()
        :Inputs:
            *opts*: :class:`odict`
                Options interface
        :Versions:
            * 2022-10-10 ``@ddalle``: Version 1.0
        """
        # Class handle
        cls = self.__class__
        # Loop through sections
        for sec, seccls in cls._sec_cls.items():
            # Get prefix and parent
            prefix = cls._sec_prefix.get(sec)
            parent = cls._sec_parent.get(sec)
            # Initialize the section
            self.init_section(seccls, sec, parent=parent, prefix=prefix)

    def init_section(self, cls, sec=None, parent=None, prefix=None):
        r"""Initialize a generic section

        :Call:
            >>> opts.init_section(cls, sec=None, **kw)
        :Inputs:
            *opts*: :class:`odict`
                Options interface
            *cls*: :class:`type`
                Class to use for *opts[sec]*
            *sec*: {*cls.__name__*} | :class:`str`
                Specific key name to use for subsection
            *parent*: {``None``} | :class:`str`
                Other subsection from which to inherit defaults
            *prefix*: {``None``} | :class:`str`
                Prefix to add at beginning of each key
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0 (:class:`odict`)
            * 2022-10-10 ``@ddalle``: Version 1.0; append name
        """
        # Process warning mode
        mode = self._get_warnmode(None, INDEX_ITYPE)
        # Default name
        if sec is None:
            # Use the name of the class
            sec = cls.__name__
        # Create name for section
        if self.name:
            # Prepend parent name to section name
            secname = self.name + " > " + sec
        else:
            # Section name is just name of the section
            secname = sec
        # Get value of self[sec] if possible
        v = self.get(sec)
        # Check if present
        if sec not in self:
            # Create empty instance
            self[sec] = cls(_name=secname)
        elif isinstance(v, cls):
            # Already initialized
            return
        elif isinstance(v, dict):
            # Convert :class:`dict` to special class
            if prefix is None:
                # Transfer keys into new class
                self[sec] = cls(**v)
            else:
                # Create dict with prefixed key names
                tmp = {
                    prefix + k: vk
                    for k, vk in v.items()
                }
                # Convert *tmp* instead of *v*
                self[sec] = cls(tmp, _name=secname)
        else:
            # Got something other than a mapping
            msg = opterror._genr8_type_error(
                v, (dict, cls), "section '%s'" % sec)
            # Save warning
            self._save_lastwarn(msg, mode, OptdictTypeError)
            # Process it if mode indicates such
            self._process_lastwarn()
            return
        # Check for *parent* to define default settings
        if parent:
            # Get the settings of parent
            vp = self.get(parent)
            # Ensure it's a dict
            if not isinstance(vp, dict):
                return
            # Loop through *vp*, but don't overwrite
            for k, vpk in vp.items():
                self[sec].setdefault(k, vpk)

   # --- Copy ---
    # Copy
    def copy(self):
        r"""Create a copy of an options interface

        :Call:
            >>> opts1 = opts.copy()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options instance
        :Outputs:
            *opts1*: :class:`OptionsDict`
                Deep copy of options instance
        :Versions:
            * 2019-05-10 ``@ddalle``: Version 1.0 (:class:`odict`)
            * 2022-10-03 ``@ddalle``: Version 1.0
        """
        # Initialize copy
        opts = self.__class__()
        # Loop through keys
        for k, v in self.items():
            # Check the type
            if not isinstance(v, dict):
                # Save a copy of the key
                opts[k] = copy.copy(v)
            else:
                # Recurse
                opts[k] = v.copy()
        # Copy all slots
        for attr in self.__slots__:
            setattr(opts, attr, copy.copy(getattr(self, attr)))
        # Output
        return opts

  # *** FILE I/O ***
   # --- JSON ---
    def read_jsonfile(self, fname):
        r"""Read a JSON file (w/ comments and includes)

        :Call:
            >>> opts.read_jsonfile(fname)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *fname*: :class:`str`
                Name of JSON file to read
        :Versions:
            * 2021-12-06 ``@ddalle``: Version 1.0
            * 2021-12-14 ``@ddalle``: Version 2.0; helpful JSON errors
        """
        # Strip comments and expand JSONFile() includes
        self.expand_jsonfile(fname)
        # Process code
        try:
            d = json.loads("".join(self._code))
        except Exception as e:
            # Get error text
            if len(e.args) == 0:  # pragma no cover
                raise
            etxt = e.args[0]
            # Check for line number
            match = re.search("line ([0-9]+)", etxt)
            # If no line number mentioned, raise original error
            if match is None:  # pragma no cover
                raise
            # None parsed
            d = None
            # Get line number
            n = int(match.group(1)) - 1
            # Start and end line number for details
            n0 = max(n-2, 0)
            n1 = min(n+3, len(self._lines))
            # Get file name
            fileno = self._filenos[n]
            filenos = self._filenos[n0:n1]
            filename = self._filenames[fileno]
            # Line number w/i original file
            m = self._sourcelinenos[n] + 1
            # Get max line number
            maxlineno = max(self._sourcelinenos[n0:n1])
            logmaxl = math.ceil(math.log10(maxlineno + 2))
            # Create print formats with line numbers
            if any([fn != fileno for fn in filenos]):
                # At least two files shown in surrounding text
                qfileno = True
                # Max file number
                maxfileno = max(filenos)
                logmaxf = math.ceil(math.log10(maxfileno + 2))
                # Use at least four chars
                logmaxf = max(4, logmaxf)
                logmaxl = max(4, logmaxl)
                # Include file index
                msgprefix = "\n    %%-%is %%-%is  Code" % (logmaxf, logmaxl)
                msgprefix = msgprefix % ("File", "Line")
                # Include file number for each line
                fmt1 = "\n--> %%%ii %%%ii  %%s" % (logmaxf, logmaxl)
                fmt2 = "\n    %%%ii %%%ii  %%s" % (logmaxf, logmaxl)
            else:
                # No file number
                qfileno = False
                fmt1 = "\n--> %%%ii %%s" % logmaxl
                fmt2 = "\n    %%%ii %%s" % logmaxl
            # Start message
            msg = "Error reading JSON file '%s'\n" % self._filenames[0]
            msg += "Error occurred on line %i" % m
            # File name
            if filename != self._filenames[0]:
                msg += (" of '%s'" % filename)
            # Try to get meaningful part of original message
            match = re.match("(.+[a-z]):", e.args[0])
            # Add original report if readable
            if match:
                msg += "\n  %s\n" % match.group(1)
            # Subheader
            msg += "\nLines surrounding problem area:"
            # Column headers if needed
            if qfileno:
                msg += msgprefix
            # Loop through lines near problem
            for i in range(n0, n1):
                # Line number w/i original file
                mi = self._sourcelinenos[i] + 1
                # Include file number?
                if qfileno:
                    # Get file number
                    fi = self._filenos[i]
                    # Include it in options
                    args = (fi, mi, self._lines[i].rstrip())
                else:
                    # No file number in args
                    args = (mi, self._lines[i].rstrip())
                # Check if it's the line identified
                if i == n:
                    # Highlight this line
                    msg += fmt1 % args
                else:
                    # Surrounding line
                    msg += fmt2 % args
            # Display file names
            if qfileno:
                # Header
                msg += "\n\nFile name key:"
                # Format for each line
                fmt3 = "\n  %%%ii: %%s" % math.ceil(math.log10(maxfileno + 2))
                for fi in set(filenos):
                    msg += fmt3 % (fi, self._filenames[fi])
        finally:
            # Clear out data attributes
            self._init_json_attributes()
        # Reraise
        if d is None:
            raise OptdictJSONError(msg)
        # Merge
        self.set_opts(d)

    def expand_jsonfile(self, fname, **kw):
        r"""Recursively read a JSON file

        :Call:
            >>> opts.expand_jsonfile(fname)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *fname*: :class:`str`
                Name of JSON file to read
        :Attributes:
            *opts.lines*: :class:`list`\ [:class:`str`]
                Original code with comments
            *opts.code*: :class:`list`\ [:class:`str`]
                Original code with comments stripped
            *opts.filenames*: :class:`list`\ [:class:`str`]
                List of one or more JSON file names read
            *opts.linenos*: :class:`list`\ [:class:`int`]
                Line number of each line within original file
            *opts.filenos*: :class:`list`\ [:class:`int`]
                Index of file from which each line came
        :Versions:
            * 2021-12-06 ``@ddalle``: Version 1.0
        """
        # Index for this file
        fileno = kw.get("fileno", 0)
        # Global line number for beginning of file
        lineno = kw.get("lineno", 0)
        # Process prefix
        prefix = kw.get("prefix", "")
        # Initialize if necessary
        if fileno == 0:
            # Reset line-tracking attributes
            self._filenames = [fname]
            self._lines = []
            self._code = []
            self._filenos = []
            self._linenos = []
            self._sourcelinenos = []
            # Save folder containing *fname*
            self._jsondir = os.path.dirname(os.path.abspath(fname))
        # Initialize line number w/in ifle
        locallineno = 0
        # Open input file
        with io.open(fname, mode="r", encoding="utf-8") as fp:
            # Loop through file
            while True:
                # Read line
                line = fp.readline()
                # Check for EOF
                if line == "":
                    break
                # Save index information for this line
                self._filenos.append(fileno)
                self._linenos.append(lineno + locallineno)
                self._sourcelinenos.append(locallineno)
                # Strip comment
                code = strip_comment(line)
                # Check for JSONFile()
                match = REGEX_JSON_INCLUDE.search(code)
                # Expand include statements
                ninclude = 0
                while match:
                    # Counter for included files
                    ninclude += 1
                    # Get file number
                    j = len(self._filenames)
                    # Get name of file
                    fj = match.group("json")
                    # Save file name
                    self._filenames.append(fj)
                    # Use "/" on all systems
                    fj = fj.replace("/", os.sep)
                    # Get absolute path
                    if not os.path.isabs(fj):
                        fj = os.path.join(self._jsondir, fj)
                    # Indices of group
                    ia = match.start()
                    ib = match.end()
                    # Save prefix as line
                    prefix = code[:ia] + "\n"
                    self._lines.append(prefix)
                    self._code.append(prefix)
                    # Recurse
                    self.expand_jsonfile(
                        fj, fileno=j,
                        lineno=lineno+locallineno)
                    # Check for additional groups
                    match = REGEX_JSON_INCLUDE.search(code[ib:])
                # Save the line if no inclusions
                if ninclude:
                    # Lines w/ >0 JSONFile() commands, append suffix
                    self._lines[-1] += code[ib:]
                    self._code[-1] += code[ib:]
                elif locallineno == 0:
                    # Save raw text and comment-stripped version
                    self._lines.append(prefix + line)
                    self._code.append(prefix + code)
                else:
                    # Save raw text and comment-stripped version
                    self._lines.append(line)
                    self._code.append(code)
                # Increment line counter
                locallineno += 1

  # *** CONDITIONS INTERFACE ***
   # --- Set ---
    def set_x(self, x: dict):
        r"""Set full conditions dict

        :Call:
            >>> opts.set_x(x)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *x*: :class:`dict`
                Supporting conditions or run matrix
        :Versions:
            * 2022-09-20 ``@ddalle``: Version 1.0
        """
        # Check input type
        assert_isinstance(x, dict, desc="supporting values, *x*")
        # Save
        self.x = x

   # --- Get ---
    def get_xvals(self, col: str, i=None):
        r"""Get values for one run matrix key

        :Call:
            >>> v = opts.get_xvals(col, i=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *col*: :class:`str`
                Name of *opts.x* key to sample
            *i*: {``None``} | :class:`int` | :class:`np.ndarray`
                Optional mask of cases to sample
        :Outputs:
            *v*: **any**
                Values of *opts.x* or *opts.x[i]*, usually an array
        :Versions:
            * 2022-09-20 ``@ddalle``: Version 1.0
        """
        # Test for empty conditions
        if (self.x is None) or col not in self.x:
            return
        # Use rules from optitem
        return optitem._sample_x(self.x, col, i)

  # *** OPTION INTERFACE ***
   # --- Get option ---
    @expand_doc
    def get_opt(self, opt: str, j=None, i=None, **kw):
        r"""Get named property, by phase, case index, etc.

        :Call:
            >>> val = opts.get_opt(opt, j=None, i=None, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to access
            %(_RST_GETOPT)s
        :Outputs:
            *val*: :class:`object`
                Value of *opt* for given conditions, in the simplest
                case simply ``opts[opt]``
        :Versions:
            * 2022-09-20 ``@ddalle``: Version 1.0
        """
        # Get notional value
        if opt in self:
            # Get directly-specified option, even if ``None``
            v = self[opt]
        elif isinstance(self._xrc, dict) and opt in self._xrc:
            # Get default from this instance
            v = self._xrc[opt]
        else:
            # Attempt to get from default, search bases if necessary
            v = self.__class__.get_cls_key("_rc", opt)
        # Set values
        kw.setdefault("x", self.x)
        # Check option
        mode = kw.pop("mode", None)
        # Apply getel() for details
        val = optitem.getel(v, j=j, i=i, **kw)
        # Check *val*
        valid = self.check_opt(opt, val, mode, out=True)
        # Test validity of *val*
        if not valid:
            # Process error/warning/nothing
            self._process_lastwarn()
            # Don't return incorrect result
            return
        # Output
        return val

    @expand_doc
    def get_subopt(self, sec: str, opt: str, key="Type", **kw):
        r"""Get subsection option, applying cascading definitions

        This function allows for cascading section definitions so that
        common settings can be defined in a parent section only once.

        :Examples:
            >>> opts = OptionsDict(
                A={"a": 1, "b": 2},
                B={"c": 3, "Type": "A"},
                C={"a": 17, "Type": "B"})
            >>> opts.get_subopt("C", "a")
            17
            >>> opts.get_subopt("C", "b")
            2
            >>> opts.get_subopt("C", "c")
            3
        :Call:
            >>> v = opts.get_subopt(sec, opt, key="Type", **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *sec*: :class:`str`
                Name of subsection to access
            *opt*: :class:`str`
                Name of option to access
            *key*: {``"Type"``} | :class:`str`
                Key in ``opts[sec]`` that defines parent of *sec*
            %(_RST_GETOPT)s
        :Outputs:
            *val*: :class:`object`
                Value of *opt* for given conditions, in the simplest
                case simply ``opts[sec][opt]``
        :Versions:
            * 2022-10-05 ``@ddalle``: Version 1.0
        """
        # Warning mode
        mode = self._get_warnmode(kw.get("mode"), INDEX_OTYPE)
        # Get section
        subopts = self.get(sec, {})
        # Check type
        if not isinstance(subopts, dict):
            # Create error message
            msg = opterror._genr8_type_error(
                subopts, dict, "section '%s'" % sec)
            # Save warning
            self._save_lastwarn(msg, mode, OptdictTypeError)
            # Exit
            self._process_lastwarn()
            return
        # Check for value
        if opt in subopts:
            # Check for built-in checks
            if isinstance(subopts, OptionsDict):
                # Use get_opt from section
                return subopts.get_opt(opt, **kw)
            else:
                # Include *x* to getel() commands if needed
                kw.setdefault("x", self.x)
                # Use phasing and special dict tool for direct value
                return optitem.getel(subopts[opt], **kw)
        # Get name of parent, if possible
        parent = subopts.get(key)
        # Check if that section is also present
        if parent in self:
            # Recurse (cascading definitions)
            return self.get_subopt(parent, opt, key, **kw)
        # If *parent* not found, **then** fall back to self._rc
        # Otherwise return ``None``
        if isinstance(subopts, OptionsDict):
            return subopts.get_opt(opt, **kw)

   # --- Set option(s) ---
    @expand_doc
    def set_opts(self, opts: dict, mode=None):
        r"""Set values of several options

        This command is similar to ``opts.update(a)``, but with checks
        performed prior to assignment.

        :Call:
            >>> opts.set_opts(a, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *a*: :class:`dict`
                Dictionary of options to merge into *opts*
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
        """
        # Check types
        assert_isinstance(opts, dict)
        # Loop through option/value pairs
        for opt, val in opts.items():
            self.set_opt(opt, val, mode=mode)

    @expand_doc
    def set_opt(self, opt: str, val, j=None, mode=None):
        r"""Set value of one option

        This command is similar to ``opts[opt] = val``, but with checks
        performed prior to assignment.

        :Call:
            >>> opts.set_opt(val, j=None, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to set
            *val*: **any**
                Value to save in *opts[opt]*
            %(_RST_SETOPT)s
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 1.1: _process_lastwarn()
        """
        # Apply alias to name
        opt = self.apply_optmap(opt)
        # Check value
        if not self.check_opt(opt, val, mode, out=False):
            # Process error/warning
            self._process_lastwarn()
            return
        # If *opt* already set, use setel
        if opt in self and j is not None:
            # Get list depth
            listdepth = self.get_listdepth(opt)
            # Apply setel() to set phase *j*
            val = optitem.setel(self[opt], val, j=j, listdepth=listdepth)
        # If all tests passed, set the value
        self[opt] = val

  # *** OPTION PROPERTIES ***
   # --- Option checkers ---
    @expand_doc
    def check_opt(self, opt: str, val, mode=None, out=False) -> bool:
        r"""Check if *val* is consistent constraints for option *opt*

        :Call:
            >>> valid = opts.check_opt(opt, val, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to test
            *val*: **any**
                Value for *opt* to test
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
            *out*: ``True`` | {``False``}
                Option to use output/input warning modes
        :Outputs:
            *valid*: ``True`` | ``False``
                Whether or not value *val* is allowed for option *opt*
        :Versions:
            * 2022-09-25 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 2.0; expanded warnmodes
        """
        # Apply alias to name
        opt = self.apply_optmap(opt)
        # Expand warning modes for input/output
        if out:
            mode_name = self._get_warnmode(mode, INDEX_ONAME)
            mode_type = self._get_warnmode(mode, INDEX_OTYPE)
        else:
            mode_name = self._get_warnmode(mode, INDEX_INAME)
            mode_type = self._get_warnmode(mode, INDEX_ITYPE)
        # Check option name
        if not self.check_optname(opt, mode_name):
            return False
        # Check option types
        if not self.check_opttype(opt, val, mode_type):
            return False
        # Check option values
        if not self.check_optval(opt, val, mode_type):
            return False
        # Passed all tests
        return True

   # --- Part checkers ---
    def apply_optmap(self, opt: str):
        r"""Replace *opt* with *opt1* if suggested using the *_optmap*

        :Call:
            >>> opt1 = opts.apply_optmap(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to test
        :Outputs:
            *opt1*: :class:`str`
                Final name of option, *opt* or a standardized alias
        :Versions:
            * 2022-09-18 ``@ddalle``: Version 1.0
        """
        # Get dict of aliases
        optmap = self.__class__._optmap
        # Apply it
        return optmap.get(opt, opt)

    @expand_doc
    def check_optname(self, opt: str, mode=None) -> bool:
        r"""Check validity of option named *opt*

        :Call:
            >>> valid = opts.check_optname(opt, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to test
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Outputs:
            *valid*: ``True`` | ``False``
                Whether or not option name *opt* is allowed
        :Versions:
            * 2022-09-18 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 2.0: _save_lastwarn()
        """
        # Process warning mode
        mode = self._get_warnmode(mode, INDEX_INAME)
        # Check mode
        if mode == WARNMODE_NONE:
            # No checks!
            return True
        # Get set of accepted option names
        optlist = self.get_optlist()
        # Basic checks
        if len(optlist) == 0:
            # No list of allowed options
            return True
        elif opt in optlist:
            # Accepted optio nname
            return True
        # Get closest matches
        matches = difflib.get_close_matches(opt, optlist)
        # Common part of warning/error message
        msg = "Unknown %s" % self._genr8_opt_msg(opt)
        # Add suggestions if able
        if len(matches):
            msg += "; nearest matches: %s" % " ".join(matches[:3])
        # Save error
        self._save_lastwarn(msg, mode, OptdictNameError)
        # For mode=1, negative result
        return False

    @expand_doc
    def check_opttype(self, opt: str, val, mode=None) -> bool:
        r"""Check type of a named option's value

        This uses the options type dictionary in *type(opts)._opttypes*
        and/or *opts._xopttypes*. If *opt* is present in neither of
        these dicts, no checks are performed.

        Special dicts, e.g

        .. code-block:: python

            {
                "@map": {
                    "a": 2,
                    "b": 3,
                },
                "key": "arch"
            }

        are checked against their own special rules, for example

        * ``"@expr"`` must be a :class:`str`
        * ``"@map"`` must be a :class:`dict`
        * ``"@cons"`` must be a :class:`dict`
        * ``"@map"`` must be accompanied by ``"map"``, a :class:`str`

        :Call:
            >>> valid = opts.opts.check_opttype(opt, val, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to test
            *val*: **any**
                Value for *opt* to test
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Outputs:
            *valid*: ``True`` | ``False``
                Whether or not option name *opt* is allowed
        :Versions:
            * 2022-09-18 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 1.1; use ``_lastwarn``
        """
        # Fill out mode
        mode = self._get_warnmode(mode, INDEX_ITYPE)
        # Apply main checker, which is recurvsive
        return self._check_opttype(opt, val, mode)

    def _check_opttype(self, opt, val, mode, j=None) -> bool:
        # Don't check types on mode 0
        if mode == WARNMODE_NONE:
            return True
        # Get allowed type(s)
        opttype = self.get_opttype(opt)
        # Get list depth
        listdepth = self.get_listdepth(opt)
        # Burrow
        if check_scalar(val, listdepth):
            # Check the type of a scalar
            if isinstance(val, dict):
                # Check for raw
                if "@raw" in val:
                    # Validate
                    valid = self._validate_raw(opt, val, mode, j)
                    if not valid:
                        return False
                    # Get @raw value
                    val = val["@raw"]
                elif "@expr" in val:
                    # Validate
                    valid = self._validate_expr(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid
                elif "@cons" in val:
                    # Validate
                    valid = self._validate_cons(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid
                elif "@map" in val:
                    # Validate
                    valid = self._validate_map(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid
            # Check if all types are allowed
            if opttype is None:
                return True
            # Check scalar value
            if isinstance(val, opttype):
                # Accepted type
                return True
            # Form and save error message
            self._save_opttype_error(opt, val, opttype, mode, j)
            # Test failed
            return False
        else:
            # Recurse
            for j, vj in enumerate(val):
                # Test *j*th entry
                qj = self._check_opttype(opt, vj, mode, j=j)
                # Check for failure
                if not qj:
                    return False
            # Each entry passed
            return True

    @expand_doc
    def check_optval(self, opt: str, val, mode=None) -> bool:
        r"""Check if *val* is an acceptable value for option *opt*

        No checks are performed *val* if it is a special dict that
        cannot be evaluated to a single value.

        :Call:
            >>> valid = opts.check_optval(opt, val, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to test
            *val*: **any**
                Value for *opt* to test
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Outputs:
            *valid*: ``True`` | ``False``
                Whether or not value *val* is allowed for option *opt*
        :Versions:
            * 2022-09-25 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 1.1; use ``_lastwarn``
        """
        # Fill out mode
        mode = self._get_warnmode(mode, INDEX_ITYPE)
        # Don't check for mode 0
        if mode == WARNMODE_NONE:
            return True
        # Get acceptable values
        optvals = self.get_optvals(opt)
        # Check for no constraints (all values accepted)
        if not optvals:
            # No constraints
            return True
        # Check for special dicts
        if isinstance(val, dict):
            # Check for special keys
            if "@raw" in val:
                # Check the raw value only
                val = val["@raw"]
                # Recurse if necessary
                if isinstance(val, dict):
                    return self.check_optval(opt, val, mode)
            elif ("@expr" in val) or ("@cons" in val) or ("@map" in val):
                # Cannot check
                return True
        # Get list depth
        listdepth = self.get_listdepth(opt)
        # Burrow dwon if given an array
        if not check_scalar(val, listdepth):
            # Loop through entries
            for vj in val:
                # Validate
                valid = self.check_optval(opt, vj, mode)
                if not valid:
                    return False
            # Otherwise valid
            return True
        # Check *val* against *vals*
        if type(val).__hash__ is None:
            # Expecting set of allowed values, but unhashable *val*
            msg0 = (
                ("Value for %s " % self._genr8_opt_msg(opt)) +
                ("has unhashable type '%s'; " % type(val).__name__))
            msg1 = self._show_options(val, optvals)
            # Handle error/warning
            self._save_lastwarn(msg0 + msg1, mode, OptdictTypeError)
            # Negative result
            return False
        elif val in optvals:
            # Valid
            return True
        # Otherwise form an error message
        msg1 = self._show_options(val, optvals)
        # Complete error message
        msg0 = "Unknown value %s for %s ; " % (
            repr(val), self._genr8_opt_msg(opt))
        # Handle error/warning
        self._save_lastwarn(msg0 + msg1, mode, OptdictValueError)
        # Negative result
        return False

   # --- Special dict validators ---
    def _validate_raw(self, opt, val, mode, j=None) -> bool:
        # Key name
        key = "@raw"
        # Allowed extra keys
        okeys = []
        # Check for additional keys
        valid_k = self._validate_sdict_okeys(opt, val, key, okeys, mode, j)
        # Combine tests
        return valid_k

    def _validate_expr(self, opt, val, mode, j=None) -> bool:
        # Key name
        key = "@expr"
        # Allowed extra keys
        okeys = []
        # Check type
        valid_t = self._validate_sdict_type(opt, val, key, str, mode, j)
        # Check for additional keys
        valid_k = self._validate_sdict_okeys(opt, val, key, okeys, mode, j)
        # Combine tests
        return valid_t and valid_k

    def _validate_cons(self, opt, val, mode, j=None) -> bool:
        # Key name
        key = "@cons"
        # Allowed extra keys
        okeys = []
        # Check type
        valid_t = self._validate_sdict_type(opt, val, key, dict, mode, j)
        # Check for additional keys
        valid_k = self._validate_sdict_okeys(opt, val, key, okeys, mode, j)
        # Combine tests
        return valid_t and valid_k

    def _validate_map(self, opt, val, mode, j=None) -> bool:
        # Key name
        key = "@map"
        # Required keys
        rkeys = {
            "key": str
        }
        # Allowed extra keys
        okeys = ["key"]
        # Check type
        valid_t = self._validate_sdict_type(opt, val, key, dict, mode, j)
        # Check for additional keys
        valid_r = self._validate_sdict_rkeys(opt, val, key, rkeys, mode, j)
        # Check for additional keys
        valid_k = self._validate_sdict_okeys(opt, val, key, okeys, mode, j)
        # Combine tests
        return valid_t and valid_r and valid_k

    def _validate_sdict_type(self, opt, val, key, cls, mode, j):
        r"""Validate that special dict @ value has correct type

        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 1.1; no raises
        """
        # Get @raw, @expr, etc.
        v = val[key]
        # Check type
        if isinstance(v, cls):
            # Correct type
            return True
        else:
            # Wrong type
            # Generate message for option name
            msg1 = self._genr8_opt_msg(opt, j) + " " + key
            # Complete the TypeError message
            msg = opterror._genr8_type_error(val, cls, msg1)
            # Save warning/exception
            self._save_lastwarn(msg, mode, OptdictTypeError)
            return False

    def _validate_sdict_rkeys(self, opt, val, key, rkeytypes, mode, j):
        r"""Validate special dict has required keys

        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
            * 2022-09-30 ``@ddalle``: Version 1.1; no raises
        """
        # Loop through required keys
        for k, clsk in rkeytypes.items():
            # Test of required key is present
            if k not in val:
                # Generate message for option name
                msg1 = self._genr8_opt_msg(opt, j) + " " + key
                # Complete missing key message
                msg = msg1 + (" is missing required key '%s'" % k)
                # Handle warning/exception
                self._save_lastwarn(msg, mode, OptdictKeyError)
                return False
            # Get value of required key
            vk = val[k]
            # Test type
            if (clsk is not None) and (not isinstance(vk, clsk)):
                # Generate message for option name
                msg1 = self._genr8_opt_msg(opt, j)
                # Generate message for *k*
                msg2 = ' %s key "%s"' % (key, k)
                # Complete the TypeError message
                msg = opterror._genr8_type_error(val, clsk, msg1 + msg2)
                # Handle error
                self._save_lastwarn(msg, mode, OptdictTypeError)
                return False
        # Otherwise good result
        return True

    def _validate_sdict_okeys(self, opt, val, key, okeys, mode, j):
        r"""Validate special dict has no unrecognized keys

        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
        """
        # Loop through required keys
        for k in val:
            # Check if allowed
            if k == key:
                # Name of special key; valid
                continue
            elif k not in okeys:
                # Generate message for option name
                msg1 = self._genr8_opt_msg(opt, j)
                # Generate message for *k*
                msg2 = ' %s has unrecognized extra key "%s"' % (key, k)
                # Handle error
                self._save_lastwarn(msg1 + msg2, mode, OptdictKeyError)
                return False
        # Otherwise good result
        return True

   # --- Option properties ---
    def get_optlist(self) -> set:
        r"""Get list of explicitly named options

        :Call:
            >>> optlist = opts.get_optlist()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Outputs:
            *optlist*: :class:`set`
                Allowed option names; if empty, all options allowed
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
            * 2022-10-04 ``@ddalle``: Version 2.0: recurse throubh bases
        """
        # Get list of options from class
        optlist = self.__class__.get_cls_set("_optlist")
        # Get instance-specific list
        xoptlist = self._xoptlist
        # Add them
        if isinstance(xoptlist, set):
            # Combine
            return optlist | xoptlist
        else:
            # Just the class's
            return optlist

    def get_opttype(self, opt: str):
        r"""Get allowed type(s) for *opt*

        :Call:
            >>> opttype = opts.get_opttype(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *opttype*: ``None`` | :class:`type` | :class:`tuple`
                Type or tuple thereof of allowed type(s) for *opt*
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
        """
        # Get instance-specific
        opttypes = self._xopttypes
        # Check if present
        if (opttypes is not None) and (opt in opttypes):
            # Instance-specific overrides
            return opttypes[opt]
        # Use class's map
        opttypes = self.__class__.get_cls_key("_opttypes", opt)
        # Check for a "_default_" if missing
        if opttypes is None:
            opttypes = self.__class__.get_cls_key("_opttypes", "_default_")
        # Output
        return opttypes

    def get_listdepth(self, opt: str) -> int:
        r"""Get list depth for a specified key

        :Call:
            >>> depth = opts.get_listdepth(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to query
        :Outputs:
            *depth*: {``0``} | :class:`int` >= 0
                Depth of expected list-type values, for example if ``1``
                a :class:`list` is expected for this option
        :Versions:
            * 2022-09-09 ``@ddalle``: Version 1.0
            * 2022-09-18 ``@ddalle``: Version 1.1; simple dict
        """
        # Check input type
        assert_isinstance(opt, str)
        # Get option from attribute
        optlistdepth = self._optlistdepth
        # Check if directly present
        if opt in optlistdepth:
            return optlistdepth[opt]
        # Check for default
        if "_default_" in optlistdepth:
            return optlistdepth["_default_"]
        # Default
        return optitem.DEFAULT_LISTDEPTH

    def get_optvals(self, opt: str):
        r"""Get set of acceptable values for option *opt*

        :Call:
            >>> vals = opts.get_optvals(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *vals*: ``None`` | :class:`set`
                Allowed values for option *opt*
        :Versions:
            * 2022-09-24 ``@ddalle``: Version 1.0
        """
        # Get set of options from class
        optvals = self.__class__._optvals
        # Get instance-specific set
        xoptvals = self._xoptvals
        # Check if instance-specific values are available
        if xoptvals is None:
            # No instance-specific defintions
            return optvals.get(opt)
        else:
            # Get instance-specific options
            return xoptvals.get(opt, optvals.get(opt))

   # --- Option list modifiers ---
    def add_xopt(self, opt: str):
        r"""Add an additional instance-specific allowed option name

        :Call:
            >>> opts.add_xopt(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
        """
        # Check input
        assert_isinstance(opt, str, "option name, *opt*,")
        # Initialize if needed
        if self._xoptlist is None:
            # Initialize with single option
            self._xoptlist = {opt}
        else:
            # Add *opt* to list
            self._xoptlist.add(opt)

    def add_xopttype(self, opt: str, opttype):
        r"""Add instance-specific allowed type(s) for *opt*

        :Call:
            >>> opts.add_xopttype(opt, opttype)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
            *opttype*: :class:`type` | :class:`tuple`
                Type or tuple thereof of allowed type(s) for *opt*
        :Versions:
            * 2022-09-19 ``@ddalle``: Version 1.0
        """
        # Check input(s)
        assert_isinstance(opt, str, "option name, *opt*,")
        assert_isinstance(opttype, (type, tuple), "option type, *opttype*,")
        # Initialize if needed
        if self._xopttypes is None:
            # Initialize with singleton dict
            self._xopttypes = {opt: opttype}
        else:
            # Add to dict
            self._xopttypes[opt] = opttype

   # --- Low-level ---
    def _get_warnmode(self, mode, imode):
        # Check for explicit mode
        mode = _access_warnmode(mode, imode)
        if mode is not None:
            return mode
        # Check for instance-specific mode
        mode = _access_warnmode(self._xwarnmode, imode)
        if mode is not None:
            return mode
        # Get class-specific mode
        mode = _access_warnmode(self._warnmode, imode)
        if mode is None:
            # Global default
            return DEFAULT_WARNMODE
        else:
            # Class-defined
            return mode

    def _save_lastwarn(self, msg, mode, cls):
        self._lastwarnmsg = msg
        self._lastwarnmode = mode
        self._lastwarncls = cls

    def _process_lastwarn(self):
        # Get mode from last warning
        mode = self._lastwarnmode
        # Exit if no warning
        if mode is None:
            # No failure
            return
        elif mode in (WARNMODE_QUIET, WARNMODE_NONE):
            # No warnings for possible failure
            return
        elif mode == WARNMODE_WARN:
            # Show error
            sys.stderr.write(self._lastwarnmsg)
            sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            # Exception
            sys.tracebacklimit = 1
            raise self._lastwarncls(self._lastwarnmsg)

    def _genr8_opt_msg(self, opt, j=None):
        # Get name to use
        if self.name:
            name = self.name
        else:
            name = type(self).__name__
        # Start error message
        msg1 = "'%s' option '%s'" % (name, opt)
        # Test for a phase
        if j is None:
            # No list index
            return msg1
        else:
            return msg1 + (" phase %i" % j)

    def _save_opttype_error(self, opt, val, opttype, mode, j):
        # Generate message for option name
        msg1 = self._genr8_opt_msg(opt, j)
        # Otherwise, format error message
        msg = opterror._genr8_type_error(val, opttype, msg1)
        # Error/warning
        self._save_lastwarn(msg, mode, OptdictTypeError)

    def _show_options(self, val, optvals):
        # Otherwise form an error message
        if isinstance(val, str) and len(optvals) > _MAX_OPTVALS_DISPLAY:
            # Get close matches only
            matches = difflib.get_close_matches(val, optvals)
            # Show only close matches
            msg = "close available options: '%s'" % "' '".join(matches)
        else:
            # Convert all options to string representation
            matches = [repr(v) for v in optvals]
            # Show all options
            msg = "available options: %s" % " ".join(matches)
        # Output
        return msg

  # *** CLASS METHODS ***
   # --- Class attribute access --
    @classmethod
    def get_cls_key(cls, attr: str, key: str, vdef=None):
        r"""Access *key* from a :class:`dict` class attribute

        This will look in the bases of *cls* if ``getattr(cls, attr)``
        does not have *key*. If *cls* is a subclass of another
        :class:`OptionsDict` class, it will search trhough the bases of
        *cls* until the first time it finds a class attribute *attr*
        that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.get_cls_key(attr, key, vdef=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *attr*: :class:`str`
                Name of class attribute to search
            *key*: :class:`str`
                Key name in *cls.__dict__[attr]*
            *vdef*: {``None``} | :class:`object`
                Default value to use if not found in class attributes
        :Outputs:
            *v*: ``None`` | :class:`ojbect`
                Any value, ``None`` if not found
        :Versions:
            * 2022-10-01 ``@ddalle``: Version 1.0
        """
        # Get property specifically from *cls* (if any)
        clsvals = cls.__dict__.get(attr)
        # Check if found
        if isinstance(clsvals, dict) and key in clsvals:
            return clsvals[key]
        # Otherwise loop through bases in attempt to find it
        for clsj in cls.__bases__:
            # Only process if OptionsDict
            if issubclass(clsj, OptionsDict):
                # Recurse
                return clsj.get_cls_key(attr, key, vdef=vdef)
        # Not found
        return vdef

    @classmethod
    def get_cls_set(cls, attr: str):
        r"""Get combined :class:`set` for *cls* and its bases

        This allows a subclass of :class:`OptionsDict` to only add to
        the ``_optlist`` attribute rather than manually include the
        ``_optlist`` of all the bases.

        :Call:
            >>> v = cls.get_cls_set(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *attr*: :class:`str`
                Name of class attribute to search
        :Outputs:
            *v*: :class:`set`
                Combination of ``getattr(cls, attr)`` and
                ``getattr(cls.__bases__[0], attr)``, etc.
        :Versions:
            * 2022-10-04 ``@ddalle``: Version 1.0
        """
        # Initialize values
        clsset = set()
        # Get attribute
        v = cls.__dict__.get(attr)
        # Update
        if v:
            clsset.update(v)
        # Loop through bases
        for clsj in cls.__bases__:
            # Only process if OptionsDict
            if issubclass(clsj, OptionsDict):
                # Recurse
                clsset.update(clsj.get_cls_set(attr))
        # Output
        return clsset

   # --- Get/Set properties ---
    @classmethod
    def add_properties(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of getters and setters with common settings

        :Call:
            >>> cls.add_properties(optlist, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *optlist*: :class:`list`\ [:class:`str`]
                Name of options to process
            *prefix*: {``None``} | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to functions
        :Versions:
            * 2022-10-14 ``@ddalle``: Version 1.0
        """
        for opt in optlist:
            cls.add_property(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_setters(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of property setters with common settings

        :Call:
            >>> cls.add_setters(optlist, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *optlist*: :class:`list`\ [:class:`str`]
                Name of options to process
            *prefix*: {``None``} | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to functions
        :Versions:
            * 2022-10-14 ``@ddalle``: Version 1.0
        """
        for opt in optlist:
            cls.add_setter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_getters(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of property getters with common settings

        :Call:
            >>> cls.add_getters(optlist, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *optlist*: :class:`list`\ [:class:`str`]
                Name of options to process
            *prefix*: {``None``} | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to functions
        :Versions:
            * 2022-10-14 ``@ddalle``: Version 1.0
        """
        for opt in optlist:
            cls.add_getter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_property(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add getter and setter methods for option *opt*

        For example ``cls.add_property("a")`` will add functions
        :func:`get_a` and :func:`set_a`, which have signatures like
        :func:`OptionsDict.get_opt` and :func:`OptionsDict.set_opt`
        except that they don't have the *opt* input.

        :Call:
            >>> cls.add_property(opt, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to functions
        :Versions:
            * 2022-09-30 ``@ddalle``: Version 1.0
            * 2022-10-03 ``@ddalle``: Version 1.1; docstrings
            * 2022-10-10 ``@ddalle``: Version 1.2; metadata, try/catch
        """
        cls.add_getter(opt, prefix=prefix, name=name, doc=doc)
        cls.add_setter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_getter(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add getter method for option *opt*

        For example ``cls.add_property("a")`` will add a function
        :func:`get_a`, which has a signatures like
        :func:`OptionsDict.get_opt` except that it doesn't have the
        *opt* input.

        :Call:
            >>> cls.add_property(opt, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix in method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to getter function
        :Versions:
            * 2022-09-30 ``@ddalle``: Version 1.0
            * 2022-10-03 ``@ddalle``: Version 1.1; docstrings
            * 2022-10-10 ``@ddalle``: Version 1.2; metadata, try/catch
        """
        # Check if acting on original OptionsDict
        cls._assert_subclass()
        # Default name
        name, fullname = cls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname
        # Check if already present
        if funcname in cls.__dict__:
            raise OptdictAttributeError(
                "Method '%s' for class '%s' already exists"
                % (funcname, cls.__name__))

        # Define function
        def func(self, j=None, i=None, **kw):
            try:
                return self.get_opt(opt, j=j, i=i, **kw)
            except Exception:
                raise

        # Generate docstring if requrested
        if doc:
            func.__doc__ = cls.genr8_getter_docstring(opt, name, prefix)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)

    @classmethod
    def add_setter(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add getter and setter methods for option *opt*

        For example ``cls.add_property("a")`` will add a function
        :func:`set_a`, which has a signature like
        :func:`OptionsDict.set_opt` except that it doesn't have the
        *opt* input.

        :Call:
            >>> cls.add_property(opt, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix for method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to setter function
        :Versions:
            * 2022-09-30 ``@ddalle``: Version 1.0
            * 2022-10-03 ``@ddalle``: Version 1.1; docstrings
            * 2022-10-10 ``@ddalle``: Version 1.2; metadata, try/catch
        """
        # Check if acting on original OptionsDict
        cls._assert_subclass()
        # Default name
        name, fullname = cls._get_funcname(opt, name, prefix)
        funcname = "set_" + fullname
        # Check if already present
        if funcname in cls.__dict__:
            raise OptdictAttributeError(
                "Method '%s' for class '%s' already exists"
                % (funcname, cls.__name__))

        # Define function
        def func(self, v, j=None, mode=None):
            try:
                return self.set_opt(opt, v, j=j, mode=mode)
            except Exception:
                raise

        # Generate docstring if requrested
        if doc:
            func.__doc__ = cls.genr8_setter_docstring(opt, name, prefix)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)

    @classmethod
    def _get_funcname(cls, opt: str, name=None, prefix=None):
        # Default name
        if name is None:
            name = normalize_optname(opt)
        # Name of function
        if prefix:
            funcname = prefix + name
        else:
            funcname = name
        # Output
        return name, funcname

    @classmethod
    def _assert_subclass(cls):
        # Check if acting on original OptionsDict
        if cls is OptionsDict:
            raise OptdictTypeError(
                "Can't set attributes directly for OptionsDict class")

   # --- Low-level: docstring ---
    @classmethod
    def genr8_getter_docstring(cls, opt: str, name, prefix, indent=8, tab=4):
        r"""Create automatic docstring for getter function

        :Call:
            >>> txt = cls.genr8_getter_docstring(opt, name, prefx, **kw)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of functions
            *prefx*: ``None`` | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
        :Outputs:
            *txt*: :class:`str`
                Contents for ``get_{opt}`` function docstring
        :Versions:
            * 2022-10-03 ``@ddalle``: Version 1.0
        """
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.get_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get value of option "%s"\n\n' % fullopt
        # Generate signature
        signature = (
            "%s>>> %s = opts.get_%s(j=None, i=None, **kw)\n"
            % (tab2, name, funcname))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent, tab=tab)
        # Form full docstring
        return (
            title +
            tab1 + ":Call:\n" +
            signature +
            tab1 + ":Inputs:\n" +
            rst_cls +
            tab2 + _RST_GETOPT + "\n" +
            tab1 + ":Outputs:\n" +
            rst_opt
        )

    @classmethod
    def genr8_setter_docstring(cls, opt: str, name, prefix, indent=8, tab=4):
        r"""Create automatic docstring for setter function

        :Call:
            >>> txt = cls.genr8_setter_docstring(opt, name, prefx, **kw)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of functions
            *prefx*: ``None`` | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
        :Outputs:
            *txt*: :class:`str`
                Contents for ``set_{opt}`` function docstring
        :Versions:
            * 2022-10-03 ``@ddalle``: Version 1.0
        """
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.get_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get value of option "%s"\n\n' % fullopt
        # Generate signature
        signature = (
            "%s>>> opts.set_%s(%s, j=None, i=None, **kw)\n"
            % (tab2, funcname, name))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent, tab=tab)
        # Form full docstring
        return (
            title +
            tab1 + ":Call:\n" +
            signature +
            tab1 + ":Inputs:\n" +
            rst_cls +
            rst_opt +
            tab2 + _RST_SETOPT
        )

    @classmethod
    def _genr8_rst_cls(cls, name="opts", indent=8, tab=4):
        # Two-line description
        return (
            ("%*s*%s*: %s\n" % (indent + tab, "", name, cls.__name__)) +
            ("%*soptions interface\n" % (indent + 2*tab, "")))

    @classmethod
    def _genr8_rst_opt(cls, opt: str, indent=8, tab=4) -> str:
        # Generate the name
        name = normalize_optname(opt)
        # Get description of values/types
        opttypes = cls._genr8_rst_opttypes(opt)
        # Get decription of parameter
        optdesc = cls._genr8_rst_desc(opt)
        # Two-line description
        return (
            ("%*s*%s*: %s\n" % (indent + tab, "", name, opttypes)) +
            ("%*s%s\n" % (indent + 2*tab, "", optdesc)))

    @classmethod
    def _genr8_rst_desc(cls, opt: str) -> str:
        # Check for user-defined descripiton
        txt = cls.get_cls_key("_rst_descriptions", opt)
        if txt:
            return txt
        # Otherwise produce a generic description
        return 'value of option "%s"' % opt

    @classmethod
    def _genr8_rst_opttypes(cls, opt: str) -> str:
        # Check for user-defined string
        txt = cls.get_cls_key("_rst_types", opt)
        if txt:
            return txt
        # Get a default value
        vdef = cls.get_cls_key("_rc", opt)
        # Check for values
        optvals = cls.get_cls_key("_optvals", opt)
        # Check for values
        if optvals:
            return genr8_rst_value_list(optvals, vdef)
        # Check for types
        opttypes = cls.get_cls_key("_opttypes", opt)
        # Convert opttypes to text
        return genr8_rst_type_list(opttypes, vdef)


# Apply all methods of one subsection class to parent
def promote_subsec(cls1, cls2, sec=None, skip=[], **kw):
    r"""Promote all methods of a subsection class to parent class

    Methods of parent class will not be overwritten

    :Call:
        >>> promote_subsec(cls1, cls2, sec=None, skip=[], **kw)
    :Inputs:
        *cls1*: :class:`type`
            Parent class
        *cls2*: :class:`type`
            Subsection class
        *sec*: {``None``} | :class:`str`
            Name of subsection, defaults to *cls2.__name__*
        *skip*: {``[]``} | :class:`list`
            List of methods from *cls2* not to add to *cls1*
        *init*: {``True``} | ``False``
            If ``True``, initialize subsection when *cls1* methods used
        *parent*: {``None``} | :class:`str`
            Name of section from which to get default settings
    :Versions:
        * 2019-01-10 ``@ddalle``: Version 1.0
    """
    # Get property dictionaries
    dict1 = cls1.__dict__
    dict2 = cls2.__dict__
    # Create the decorator to promote each method (function)
    f_deco = subsec_func(cls2, sec, **kw)
    # Loop through methods of *cls2*
    for fn in dict2:
        # Manual skipping
        if fn in skip:
            continue
        # Get value of *cls2* attribute
        func = dict2[fn]
        # Skip if not a function
        if not callable(func):
            continue
        # Check if already present
        if fn in dict1:
            continue
        # Set attribute to decorated function
        setattr(cls1, fn, f_deco(func))


# Strip a comment
def strip_comment(line):
    r"""Strip a comment from a line

    :Call:
        >>> code = strip_comment(line)
    :Examples:
        >>> strip_comment('  "a": 2,\n')
        '  "a": 2,\n'
        >>> strip_comment('  // "a": 2,\n')
        '\n'
        >>> strip_comment('  "a": ["a//b", // comment\n')
        '  "a": ["a//b", \n'
    :Inputs:
        *line*: :class:`str`
            A line of text, possibly including a // comment
    :Outputs:
        *code*: :class:`str`
            Input *line* with comments removed
    :Versions:
        * 2021-12-06 ``@ddalle``: Version 1.0
    """
    # Check for simple case
    if "//" not in line:
        # No comment
        return line
    # Check for whole-line comments
    if line.strip().startswith("//"):
        return "\n"
    # Check for quotes, which might confuse things
    match_list = REGEX_QUOTE.findall(line)
    # Check if any quotes were found
    if match_list:
        # Substitute quotes with reinsertion string instructions
        line1 = REGEX_QUOTE.sub('"{}"', line)
        # Strip comments
        line1 = REGEX_COMMENT.sub("", line1)
        # Reinsert quotes
        return line1.format(*match_list)
    else:
        # Just strip comments (no quotes to worry about)
        return REGEX_COMMENT.sub("", line)


# Decorator to get function from subclass
def subsec_func(cls, sec=None, parent=None, init=True):
    r"""Decorator (w/ args) to apply a function from a subsection class

    :Call:
        >>> f = subsec_func(cls, sec=None, parent=None, init=True)
    :Inputs:
        *cls*::class:`type`
            Class to apply to subsection
        *sec*: {*cls.__name*} | :class:`str`
            Name of subsection
        *init*: {``True``} | ``False``
            If ``True`` and nontrivial *cls*, initialize subsection
        *parent*: {``None``} | :class:`str`
            Name of section from which to get default settings
    :Outputs:
        *f*: :class:`function`
            Decorator with arguments expanded
    :Examples:
        .. code-block:: python

            @subsec_func("RunControl", RunControl)
            def get_PhaseSequence(self, *a, **kw):
                pass

    :Versions:
        * 2019-01-10 ``@ddalle``: Version 1.0
        * 2021-10-18 ``@ddalle``: Version 1.1; default *sec*
    """
    # Default *sec*
    if sec is None:
        sec = cls.__name__

    # Decorator for the function
    def decorator_subsec(func):
        # Inherit metadata from func
        @functools.wraps(func)
        # The before and after function
        def wrapper(self, *a, **kw):
            # Initialize the section
            if init and (cls is not None):
                self.init_section(cls, sec, parent=parent)
            # Get the function from the subsection
            f = getattr(self[sec], func.__name__)
            # Call the function from the subsection
            v = f(*a, **kw)
            # Return value
            return v
        # Copy the docstring
        if cls is not None:
            wrapper.__doc__ = getattr(cls, func.__name__).__doc__
        # Output
        return wrapper
    # Return decorator
    return decorator_subsec


# Get warning mode for certain role
def _access_warnmode(mode, index: int) -> int:
    # Check type
    if isinstance(mode, tuple):
        return mode[index]
    else:
        return mode


# Normalize an option name
def normalize_optname(opt: str) -> str:
    r"""Normalize option name to a valid Python variable name

    For example, you can have an option name ``"@1d+2"`` but not a
    method called ``get_@1d+2()``. This function will normalize the
    above to ``d2`` by eliminating non-word characters and stripping
    leading digits. This is so that :func:`OptionsDict.add_property`
    will suggest a valid function name :func:`get_d2`.

    :Call:
        >>> name = normalize_optname(opt)
    :Inputs:
        *opt*: :class:`str`
            Option name
    :Outputs:
        *name*: :class:`str`
            Normalized version, valid Python expression
    :Versions:
        * 2022-10-02 ``@ddalle``: Version 1.0
    """
    # Eliminate all non-word characters
    name = _REGEX_W.sub("", opt)
    # Strip leading digits
    name = _REGEX_D.sub("", name)
    # Output
    return name


# Generate string for list of values if applicable
def genr8_rst_value_list(optvals, vdef=None):
    r"""Format a string to represent several possible values in reST

    :Call:
        >>> txt = genr8_rst_value_list(optvals, vdef=None)
    :Inputs:
        *optvals*: :class:`set` | **iterable**
            Possible values for some parameter to take
        *vdef*: {``None``} | :class:`object`
            A default value to highlight with {} characters
    :Outputs:
        *txt*: :class:`str`
            reStructuredText representation of *optvals*
    :Versions:
        * 2022-10-02 ``@ddalle``: Version 1.0
    """
    # Replace each value with representation
    strvals = [
        ("{``%r``}" if v == vdef else "``%r``") % v
        for v in optvals
    ]
    # Assemble string
    return " | ".join(strvals)


# Generate string for list of types
def genr8_rst_type_list(opttypes, vdef=None):
    r"""Format a string to represent one or more types in reST

    Examples of potential output:

    * {``None``} | :class:`object`
    * {``"on"``} | :class:`str`
    * {``None``} | :class:`int` | :class:`int64`

    :Call:
        >>> txt = genr8_rst_type_list(opttype, vdef=None)
        >>> txt = genr8_rst_type_list(opttypes, vdef=None)
    :Inputs:
        *opttype*: :class:`type`
            Single allowed type
        *opttypes*: :class:`tuple`\ [:class:`type`]
            Tuple of allowed types
        *vdef*: {``None``} | :class:`object`
            Default value
    :Outputs:
        *txt*: :class:`str`
            reStructuredText representation of *opttypes*
    :Versions:
        * 2022-10-03 ``@ddalle``: Version 1.0
    """
    # Always show default value
    txt = "{``%r``} | " % vdef
    # Convert types to string
    if isinstance(opttypes, type):
        # Single type
        return txt + ":class:`%s`" % opttypes.__name__
    elif opttypes == INT_TYPES:
        # Special case for many int types
        return txt + _RST_INT_TYPES
    elif opttypes == FLOAT_TYPES:
        # Special case for many float types
        return txt + _RST_FLOAT_TYPES
    elif opttypes:
        # Convert each type to a string
        strtypes = [":class:`%s`" % clsj.__name__ for clsj in opttypes]
        # Add types to string
        return txt + " | ".join(strtypes)
    else:
        # Assume all types are allowed
        return txt + ":class:`object`"