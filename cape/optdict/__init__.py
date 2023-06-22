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
    opts.save_x(x)

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
        * :func:`OptionsDict.getx_optlist`
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
        * :func:`OptionsDict.getx_opttype`
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
        * :func:`OptionsDict.getx_listdepth`

``OptionsDict._optring``: :class:`dict`\ [:class:`bool`]
    A :class:`dict` of whether each option should be treated as a
    "ring", meaning a :class:`list` is repeated in full. If
    ``_optring[opt]`` is true, then *opt* is treated as a ring.
    Otherwise the last entry will be repeated. The default is ``False``,
    but this can be overridden using ``_optlist["_default_"] = True``.

    See also:
        * :func:`OptionsDict.get_opt`
        * :func:`OptionsDict.getx_optring`

``OptionsDict._rc``: :class:`dict`
    Default value for any option

    See also:
        * :func:`OptionsDict.get_opt`
        * :func:`OptionsDict.getx_cls_key`

``OptionsDict._xoptkey``: :class:`str`
    Name of option (including its aliases) for which the value (which
    must be a :class:`list`, :class:`set`, or :class:`tuple`) is
    appended to an instance's ``_xoptlist``.

    In other words, if ``_xoptkey`` is ``"Components"``, and
    ``opts["Components"]`` is ``["a", "b", "c"]``, then *a*, *b*, and
    *c* will be allowed as option names for *opts*.

``OptionsDict._sec_cls``: :class:`dict`\ [:class:`type`]
    Dictionary of classes to use as type for named subsections

    See also:
        * :func:`OptionsDict.init_sections`

``OptionsDict._sec_cls_opt``: :class:`str`
    Name of option to use as determining which class to use from
    ``_sec_cls_optmap`` to initialize each subsection.

    A typical value for this option is ``"Type"``. If this attribute is
    set to ``None``, then no value-dependent subsection class
    conversions are attempted.

``OptionsDict._sec_cls_optmap``: :class:`dict`\ [:class:`type`]
    Dictionary of classes to use to convert each subsection according
    to its base value of of ``_sec_cls_opt``.

    This attribute is used if some sections should be converted to
    different types (for example force & moment options vs line load
    options) depending on their contents.

    See also:
        * :func:`OptionsDict.get_subkey_base`

``OptionsDict._sec_prefix``: :class:`dict`\ [:class:`str`]
    Optional prefixes to use for each option in a subsection controlled
    by ``_sec_cls``.

    See also:
        * :func:`OptionsDict.init_sections`

``OptionsDict._sec_initfrom``: :class:`dict`\ [:class:`str`]
    If used with ``_sec_cls``, sections will be initialized with values
    from a parent section named in this attribute.

``OptionsDict._sec_parent``: :class:`dict`
    Defines some subsections to fall back to either their parent section
    or another subsection for default values, before trying locally
    defined defaults.

    Can be either ``-1`` (for parent section) or :class:`str`

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
from .optitem import check_array, check_scalar


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

# Constant to use parent section as _xparent for subseciton
USE_PARENT = -1

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
            *vdef*: {``None``} | :class:`object`
                Manual default
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
            *ring*: {*opts._optring[key]*} | ``True`` | ``False``
                Override option to loop through phase inputs
            *listdepth*: {``0``} | :class:`int` > 0
                Depth of list to treat as a scalar
            *x*: {``None``} | :class:`dict`
                Ref conditions to use with ``@expr``, ``@map``, etc.;
                often a run matrix; used in combination with *i*
            *sample*: {``True``} | ``False``
                Apply *j*, *i*, and other settings recursively if output
                is a :class:`list` or :class:`dict`""" % _RST
_RST_SETOPT = r"""*j*: {``None``} | :class:`int`
                Phase index; use ``None`` to just return *v*
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
            *listdepth*: {``0``} | :class:`int` > 0
                Depth of list to treat as a scalar""" % _RST
_RST_ADDOPT = r"""*mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s""" % _RST
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
        * 2021-12-05 ``@ddalle``: v0.1; started
        * 2021-12-06 ``@ddalle``: v1.0
        * 2022-09-20 ``@ddalle``: v1.1; get_opt() w/ x
        * 2022-09-24 ``@ddalle``: v1.2: *_warnmode*
        * 2022-09-30 ``@ddalle``: v1.3: *_rc* and four warnmodes
    """
  # *** CLASS ATTRIBUTES ***
   # --- Slots ---
    # Allowed attributes
    __slots__ = (
        "i",
        "name",
        "x",
        "_xoptlist",
        "_xopttypes",
        "_xoptvals",
        "_xparent",
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
    _optring = {}

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

    # Key used to add to instance's _xoptlist
    _xoptkey = None

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
    # Section class instantiators by name
    _sec_cls = {}

    # Prefix for each section
    _sec_prefix = {}

    # Section to inherit defaults from
    _sec_initfrom = {}

    # Parent to define fall-back settings before *rc*
    _sec_parent = {}

    # Sections w/ value-dependent classes
    _sec_cls_opt = None
    _sec_cls_optmap = {}

   # --- Settings ---
    _warnmode = DEFAULT_WARNMODE

  # *** CONFIG ***
   # --- __dunder__ ---
    def __init__(self, *args, **kw):
        r"""Initialization method

        :Versions:
            * 2021-12-06 ``@ddalle``: v1.0
            * 2022-09-19 ``@ddalle``: v1.1; fix *a* name conflict
            * 2022-09-20 ``@ddalle``: v1.2; allow *x* as args[1]
            * 2022-09-30 ``@ddalle``: v1.3; eliminate args[1]
            * 2022-10-04 ``@ddalle``: v1.4; int_post hook
            * 2022-10-10 ``@ddalle``: v1.5; init sections, name
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
        self.setx_i(kw.pop("_i", None))
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
            assert_isinstance(
                a, (str, dict), "positional input to %s" % type(self).__name__)
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
        self._xparent = None
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
            * 2022-10-04 ``@ddalle``: v1.0
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
            * 2022-10-10 ``@ddalle``: v1.0
            * 2022-12-03 ``@ddalle``: v1.1; parent -> initfrom
        """
        # Class handle
        cls = self.__class__
        # Get compound dictionary of section names and classes
        sec_cls = cls.getx_cls_dict("_sec_cls")
        # Loop through sections
        for sec, seccls in sec_cls.items():
            # Get prefix and parent
            prefix = cls._sec_prefix.get(sec)
            initfrom = cls._sec_initfrom.get(sec)
            # Initialize the section
            self.init_section(seccls, sec, initfrom=initfrom, prefix=prefix)
        # Do the same for value-dependent class subsections
        self._init_secmap()
        # Set up fall-back values by defining "parent" for some sections
        self._init_sec_parents()

    def init_section(self, cls, sec=None, **kw):
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
            *initfrom*: {``None``} | :class:`str`
                Other subsection from which to inherit defaults
            *prefix*: {``None``} | :class:`str`
                Prefix to add at beginning of each key
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0 (:class:`odict`)
            * 2022-10-10 ``@ddalle``: v1.0; append name
            * 2022-12-03 ``@ddalle``: v1.1; parent -> initfrom
        """
        # Process warning mode
        mode = self._get_warnmode(None, INDEX_ITYPE)
        # Use same warning modes
        w_iname, w_itype, w_oname, w_otype = self._xwarnmode
        kwcls = {
            "_warnmode_iname": w_iname,
            "_warnmode_itype": w_itype,
            "_warnmode_oname": w_oname,
            "_warnmode_otype": w_otype,
        }
        # Options
        initfrom = kw.get("initfrom")
        prefix = kw.get("prefix")
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
            self[sec] = cls(_name=secname, **kwcls)
        elif isinstance(v, cls):
            # Already initialized
            return
        else:
            # Convert *v* to special class
            if isinstance(v, dict) and (prefix is not None):
                # Create dict with prefixed key names
                tmp = {
                    prefix + k: vk
                    for k, vk in v.items()
                }
            else:
                # In all other cases, use *v* as-is
                tmp = v
            # Create class, i.e. perform conversion
            try:
                # Depending on *cls*, *tmp* MAY not have to be a dict
                self[sec] = cls(tmp, _name=secname, **kwcls)
            except OptdictTypeError:
                # Got something other than a mapping
                msg = opterror._genr8_type_error(
                    v, (dict, cls), "section '%s'" % sec)
                # Save warning
                self._save_lastwarn(msg, mode, OptdictTypeError)
                # Process it if mode indicates such
                self._process_lastwarn()
                return
        # Check for *initfrom* to define default settings
        if initfrom:
            # Get the settings of parent
            vp = self.get(initfrom)
            # Ensure it's a dict
            if not isinstance(vp, dict):
                return
            # Loop through *vp*, but don't overwrite
            for k, vpk in vp.items():
                self[sec].setdefault(k, vpk)

    def _init_secmap(self):
        r"""Initialize value-dependent subsection classes

        :Call:
            >>> opts._init_secmap()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Versions:
            * 2022-11-06 ``@ddalle``: v1.0
        """
        # Class handle
        cls = self.__class__
        # Get key for section class map
        opt = cls._sec_cls_opt
        # Exit if appropriate
        if opt is None:
            return
        # Get compound dictionary of map between section types and class
        secmap = cls.getx_cls_dict("_sec_cls_optmap")
        # Get regular (not value-dependent) subsecs to avoid conflict
        subsecs = cls.getx_cls_dict("_sec_cls")
        # Current list of declared options
        optlist = cls.getx_cls_set("_optlist")
        # Default class
        clsdef = secmap.get("_default_")
        # Use same warning modes
        w_iname, w_itype, w_oname, w_otype = self._xwarnmode
        kwcls = {
            "_warnmode_iname": w_iname,
            "_warnmode_itype": w_itype,
            "_warnmode_oname": w_oname,
            "_warnmode_otype": w_otype,
        }
        # Loop through sections
        for sec in self:
            # Check if already initiated
            if sec in subsecs:
                continue
            # Check for pre-declared option (not a section)
            if sec in optlist:
                continue
            # Get the value of *opt*, cascading if necessary
            secopt = self.get_subkey_base(sec, opt)
            # Check for class from the map
            seccls = secmap.get(secopt, clsdef)
            # Check for miss
            if seccls is None:
                # Handle this elsewhere
                continue
            # Otherwise initiate
            self[sec] = seccls(self[sec], _name=sec, **kwcls)

    def _init_sec_parents(self):
        # Class handle
        cls = self.__class__
        # Get map for parents of each section
        sec_parents = cls._sec_parent
        # Exit if appropriate
        if not sec_parents:
            return
        # Ensure type
        assert_isinstance(sec_parents, dict, "parents for named sections")
        # Get default
        default_parent = sec_parents.get("_default_")
        # Loop through sections
        for sec in self:
            # Get section
            secopts = self[sec]
            # Only process if type is correct
            if not isinstance(secopts, OptionsDict):
                continue
            # Get value
            parent = sec_parents.get(sec, default_parent)
            # Check parent type and value
            if parent is None:
                # No parent
                continue
            elif isinstance(parent, str):
                # Other named section
                secopts.setx_parent(self[parent])
            elif parent == USE_PARENT:
                # Default defined in parent
                secopts.setx_parent(self)
            else:
                raise OptdictValueError(
                    "Unrecognized '%s' parent for section '%s'"
                    % (type(parent).__name__, sec))

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
            * 2019-05-10 ``@ddalle``: v1.0 (:class:`odict`)
            * 2022-10-03 ``@ddalle``: v1.0
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
            * 2021-12-06 ``@ddalle``: v1.0
            * 2021-12-14 ``@ddalle``: v2.0; helpful JSON errors
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
            * 2021-12-06 ``@ddalle``: v1.0
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
            # Save current folder
            self._jsondir = os.getcwd()
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
    def save_x(self, x: dict, recursive=True):
        r"""Set full conditions dict

        :Call:
            >>> opts.save_x(x, recursive=True)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *x*: :class:`dict`
                Supporting conditions or run matrix
            *recursive*: {``True``} | ``False``
                Option ro apply *x* to subsections of *opts*
        :Versions:
            * 2022-09-20 ``@ddalle``: v1.0
            * 2022-10-28 ``@ddalle``: v1.1; *recursive*
        """
        # Check input type
        assert_isinstance(x, dict, desc="supporting values, *x*")
        # Save
        self.x = x
        # Recurse if appropriate
        if recursive:
            # Loop through items
            for sec, v in self.items():
                # Check for subsection
                if isinstance(v, OptionsDict):
                    # Apply
                    v.save_x(x, True)

    def setx_i(self, i=None, recursive=True):
        r"""Set current default run matrix index

        :Call:
            >>> opts.setx_i(i=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *i*: {``None``} | :class:`int`
                Run matrix index
            *recursive*: {``True``} | ``False``
                Option ro apply *x* to subsections of *opts*
        :Versions:
            * 2023-05-02 ``@ddalle``: v1.0
            * 2023-06-02 ``@ddalle``: v1.1; recursive
        """
        # Set index
        self.i = i
        # Recurse if appropriate
        if recursive:
            # Loop through items
            for sec, v in self.items():
                # Check for subsection
                if isinstance(v, OptionsDict):
                    # Apply
                    v.setx_i(i, True)

   # --- Get ---
    def getx_xvals(self, col: str, i=None):
        r"""Get values for one run matrix key

        :Call:
            >>> v = opts.getx_xvals(col, i=None)
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
            * 2022-09-20 ``@ddalle``: v1.0
        """
        # Test for empty conditions
        if (self.x is None) or col not in self.x:
            return
        # Get current index
        i = self.getx_i(i)
        # Use rules from optitem
        return optitem._sample_x(self.x, col, i)

    def getx_i(self, i=None):
        r"""Get run matrix index to use for option expansion

        This function returns ``None`` only if *i* is ``None`` and
        ``opts.i`` is also ``None``.

        :Call:
            >>> i1 = opts.getx_i(i=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *i*: {``None``} | :class:`int` | :class:`np.ndarray`
                Optional mask of cases to sample
        :Outputs:
            *i1*: *opts.i* | *i*
                *i* if other than ``None``; else **opts.i**
        :Versions:
            * 2023-05-02 ``@ddalle``: v1.0
        """
        # Check
        if i is None:
            # Use current default value
            return self.i
        else:
            # Return user value
            return i

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
                case ``opts[opt]``
        :Versions:
            * 2022-09-20 ``@ddalle``: v1.0
            * 2023.06-13 ``@ddalle``: v1.1; implement *ring* opt
            * 2023-06-14 ``@ddalle``: v1.2; enforce *listdepth* fully
        """
        # Get notional value
        if opt in self:
            # Get directly-specified option, even if ``None``
            v = self[opt]
        elif self._check_parent(opt):
            # Use fall-backs from some other interface
            v = self.getx_opt_parent(opt)
        elif "vdef" in kw:
            # Use manual default
            v = kw["vdef"]
        else:
            # Get default, search _xrc, then _rc, then _rc of __bases__
            v = self.get_opt_default(opt)
        # Set values
        kw.setdefault("x", self.x)
        # Expand index
        i = self.getx_i(i)
        # Set list depth option
        listdepth = kw.setdefault("listdepth", self.getx_listdepth(opt))
        # Set ring (vs repeat-last) option
        kw.setdefault("ring", self.getx_optring(opt))
        # Check option
        mode = kw.pop("mode", None)
        # Apply getel() for details
        val = self._sample_val(v, j, i, **kw)
        # Don't check if ``None``
        if val is None:
            return
        # Make a list if we got a scalar compared to listdepth
        if listdepth > 0:
            # Loop until sufficient list depth achieved
            while not check_array(val, listdepth):
                # Add another layer of list depth
                val = [val]
        # Check *val*, potentially converting dict->OptionsDict
        valid, val = self.check_opt(opt, val, mode, out=True)
        # Test validity of *val*
        if not valid:
            # Process error/warning/nothing
            self._process_lastwarn()
            # Don't return invalid result
            return
        # If *val* is a dict, sample it
        if isinstance(val, dict) and kw.get("sample", True):
            # Remove kwargs specific to *opt*
            kw.pop("ring")
            kw.pop("listdepth")
            # Apply *j* (phase), *i* (case index), etc. to contents
            val = self.sample_dict(val, i=i, j=j, **kw)
        # Return value
        return val

    @expand_doc
    def sample_dict(self, v: dict, j=None, i=None, _depth=0, **kw):
        r"""Expand a value, selectin phase, applying conditions, etc.

        :Call:
            >>> val = opts.expand_val(v, j=None, i=None, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *v*: :class:`dict` | :class:`list`\ [:class:`dict`]
                Initial raw option value
            *f*: ``True`` | {``False``}
                Force *j* and *i* to be integers
            %(_RST_GETOPT)s
        :Outputs:
            *val*: :class:`dict`
                Value of *opt* for given conditions, in the simplest
                case ``v``, perhaps ``v[j]``
        :Versions:
            * 2023-05-15 ``@ddalle``: v1.0
            * 2023-06-14 ``@ddalle``: v1.1; apply ``_rc`` if possible
        """
        # Set values
        kw.setdefault("x", self.x)
        # Expand index
        i = self.getx_i(i)
        # Check types
        if kw.get("f", False):
            # Change default, None -> 0 for *i* if no run matrix
            if i is None and kw["x"] is None:
                i = 0
            # Check types
            assert_isinstance(j, INT_TYPES, "phase index")
            assert_isinstance(i, INT_TYPES, "case index")
        # Sample list -> scalar, etc.
        vj = optitem.getel(v, j=j, i=i, **kw)
        # Check types again for parent dictionary
        if _depth == 0:
            # For initial pass, must be a dict
            assert_isinstance(vj, dict, "sampled dictionary")
        elif not isinstance(vj, dict):
            # Otherwise, if not a dict, done
            return vj
        # Initialize output
        val = v.__class__()
        # Loop through entries
        for k, vk in vj.items():
            # Apply extra options if *vk* is OptionsDict
            if isinstance(vj, OptionsDict):
                # Set listdepth and ring options
                kw["ring"] = vj.getx_optring(k)
                kw["listdepth"] = vj.getx_listdepth(k)
            # Recurse
            valk = self.sample_dict(vk, j, i, _depth + 1, **kw)
            # Save
            val[k] = valk
        # Apply defaults if appropriate
        if isinstance(val, OptionsDict):
            # Get full _rc
            rc = val.getx_cls_dict("_rc")
            # Loop throug defaults
            for k, vk in rc.items():
                # Apply but don't overwrite
                val.setdefault(k, vk)
        # Output
        return val

    def _sample_val(self, v, j, i, **kw):
        return optitem.getel(v, j=j, i=i, **kw)

    def get_opt_default(self, opt: str):
        r"""Get default value for named option

        :Call:
            >>> vdef = opts.get_opt_default(self, opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to access
        :Outputs:
            *vdef*: :class:`object`
                Default value if any, else ``None``
        :Versions:
            * 2022-10-30 ``@ddalle``: v1.0
        """
        if isinstance(self._xrc, dict) and opt in self._xrc:
            # Get default from this instance
            v = copy.deepcopy(self._xrc[opt])
        else:
            # Attempt to get from default, search bases if necessary
            v = copy.deepcopy(self.__class__.getx_cls_key("_rc", opt))
        # Output
        return v

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
            * 2022-10-05 ``@ddalle``: v1.0
            * 2022-11-06 ``@ddalle``: v1.1; special opt==key case
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
        if opt == key:
            # Special case; for opt=key, get deepest, not shallowest
            pass
        elif opt in subopts:
            # Check for built-in checks
            if isinstance(subopts, OptionsDict):
                # Use get_opt from section
                return subopts.get_opt(opt, **kw)
            else:
                # Include *x* to getel() commands if needed
                kw.setdefault("x", self.x)
                # Use phasing and special dict tool for direct value
                val = optitem.getel(subopts[opt], **kw)
                # Sample if appropriate
                if isinstance(val, dict) and kw.get("sample", True):
                    # Apply *j* (phase), *i*, etc. to contents of dict
                    val = self.sample_dict(val, **kw)
                # Output
                return val
        # Get name of parent, if possible
        parent = subopts.get(key)
        # Check if that section is also present
        if parent in self:
            # Recurse (cascading definitions)
            return self.get_subopt(parent, opt, key, **kw)
        elif opt == key:
            # End of recursion for special case of opt==key
            if parent is None:
                # No parent; name of this section is end of recursion
                return sec
            else:
                # Deepest section has *key*
                return parent
        # If *parent* not found, **then** fall back to self._rc
        # Otherwise return ``None``
        if isinstance(subopts, OptionsDict):
            return subopts.get_opt(opt, **kw)

    @expand_doc
    def get_subkey_base(self, sec: str, key: str, **kw) -> str:
        r"""Get root value of cascading subsection key

        For example, with the options

        .. code-block:: pycon

            >>> opts = OptionsDict({
                "A": {"Type": "BaseType"},
                "B": {"Type": "A"},
                "C": {"Type": "C"}
            })
            >>> opts.get_subkey_base("C", "Type")
            'BaseType'

        The function will find the deepest-level parent of section
        ``"C"`` using ``"Type"`` as the name of the parent section. When
        no parents remain, it will return either the final ``"Type"``
        value or the name of the last parent. The other possibility is

        .. code-block:: pycon

            >>> opts = OptionsDict({
                "A": {},
                "B": {"Type": "A"},
                "C": {"Type": "C"}
            })
            >>> opts.get_subkey_base("C", "Type")
            'A'

        :Call:
            >>> base = opts.get_subkey_base(sec, key, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *sec*: :class:`str`
                Name of subsection to access
            *key*: {``"Type"``} | :class:`str`
                Key in ``opts[sec]`` that defines parent of *sec*
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Outputs:
            *base*: :class:`str`
                Value of *key* from deepest parent of *sec* found
        :Versions:
            * 2022-11-06 ``@ddalle``: v1.0
        """
        return self.get_subopt(sec, key, key, **kw)

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
            * 2022-09-19 ``@ddalle``: v1.0
        """
        # Check types
        assert_isinstance(opts, dict)
        # Add to _xoptlist if appropriate
        self._process_xoptkey(opts)
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
            * 2022-09-19 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v1.1: _process_lastwarn()
        """
        # Apply alias to name
        opt = self.apply_optmap(opt)
        # Check validity of attempted setting
        valid, val = self.check_opt(opt, val, mode, out=False)
        # Check value
        if not valid:
            # Process error/warning
            self._process_lastwarn()
            return
        # If *opt* already set, use setel
        if opt in self and j is not None:
            # Get list depth
            listdepth = self.getx_listdepth(opt)
            # Apply setel() to set phase *j*
            val = optitem.setel(self[opt], val, j=j, listdepth=listdepth)
        # If all tests passed, set the value
        self[opt] = val

   # --- Extend option ---
    # Add to general key
    @expand_doc
    def extend_opt(self, opt: str, val, mode=None):
        r"""Extend an array-like or dict-like option

        This function allows for targeted additions to :class:`list` and
        :class:`dict` values for *opt* without redefining the entire
        entry.

        :Call:
            >>> opts.extend_opt(opt, val, mode=None)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to extend
            *val*: **any**
                Value to append to *opts[opt]*. If *opts[opt]* is a
                :class:`dict`, *val* must also be a :class:`dict`
            *mode*: {``None``} | %(_RST_WARNMODE_LIST)s
                %(_RST_WARNMODE2)s
        :Versions:
            * 2022-10-14 ``@ddalle``: v1.0
        """
        # Check for null extension
        if val is None:
            return
        # Get Current list
        vcur = self.get_opt(opt, sample=False)
        # Check key type
        if isinstance(vcur, dict):
            # Check input type
            if not isinstance(val, dict):
                raise OptdictTypeError(
                    ("Cannot append type '%s' " % type(val).__name__) +
                    ("to %s, " % self._genr8_opt_msg(opt)) +
                    ("with type '%s'\n" % type(vcur).__name__))
            # Append dictionary
            for kp, vp in val.items():
                vcur.setdefault(kp, vp)
        elif opt not in self:
            # Set value from scratch
            vcur = val
        elif isinstance(vcur, (tuple, set)):
            # Unextendable type
            raise OptdictTypeError(
                ("Cannot extend %s " % self._genr8_opt_msg(opt)) +
                ("with type '%s'" % type(vcur).__name__))
        else:
            # Check if already a list
            if not isinstance(vcur, list):
                # Make a list
                vcur = [vcur]
                # Resave the listified value
                self.set_opt(opt, vcur, mode=mode)
            # Check for scalar
            if isinstance(val, ARRAY_TYPES):
                for vj in val:
                    # Check if the file is already there
                    vcur.append(vj)
            else:
                # Append single value
                vcur.append(val)
        # Get full name
        fullopt = self.apply_optmap(opt)
        # Save value if new
        if fullopt not in self:
            self.set_opt(opt, vcur, mode=mode)

  # *** OPTION PROPERTIES ***
   # --- Option checkers ---
    @expand_doc
    def check_opt(self, opt: str, val, mode=None, out=False) -> bool:
        r"""Check if *val* is consistent constraints for option *opt*

        :Call:
            >>> valid, v2 = opts.check_opt(opt, val, mode=None)
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
            *v2*: **any**
                Either *val* or converted to new class
        :Versions:
            * 2022-09-25 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v2.0; expanded warnmodes
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
        valid = self.check_optname(opt, mode_name)
        if not valid:
            return False, val
        # Check option types
        valid, val = self.check_opttype(opt, val, mode_type)
        if not valid:
            return False, val
        # Check option values
        valid = self.check_optval(opt, val, mode_type)
        # Return result of last test
        return valid, val

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
            * 2022-09-18 ``@ddalle``: v1.0
        """
        # Get dict of aliases
        optmap = self.getx_cls_dict("_optmap")
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
            * 2022-09-18 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v2.0: _save_lastwarn()
        """
        # Process warning mode
        mode = self._get_warnmode(mode, INDEX_INAME)
        # Check mode
        if mode == WARNMODE_NONE:
            # No checks!
            return True
        # Get set of accepted option names
        optlist = self.getx_optlist()
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
            >>> valid, v2 = opts.opts.check_opttype(opt, val, mode=None)
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
            *v2*: **any**
                Either *val* or converted to new class
        :Versions:
            * 2022-09-18 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v1.1; use ``_lastwarn``
        """
        # Fill out mode
        mode = self._get_warnmode(mode, INDEX_ITYPE)
        # Apply main checker, which is recurvsive
        return self._check_opttype(opt, val, mode)

    def _check_opttype(self, opt, val, mode, j=None) -> bool:
        # Don't check types on mode 0 or if *val* is ``None``
        if mode == WARNMODE_NONE:
            return True, val
        # Get allowed type(s)
        opttype = self.getx_opttype(opt)
        # Don't check ``None``
        if val is None:
            return True, val
        # Check list depth if *j* is None
        if j is None:
            # Get list depth
            listdepth = self.getx_listdepth(opt)
            # Check it
            if not check_array(val, listdepth):
                self._save_listdepth_error(opt, val, listdepth, mode)
        # Burrow
        if check_scalar(val, 0):
            # Pass to another name to let @raw test work
            v1 = val
            # Check the type of a scalar
            if isinstance(val, dict):
                # Check for raw
                if "@raw" in val:
                    # Validate
                    valid = self._validate_raw(opt, val, mode, j)
                    if not valid:
                        return False, val
                    # Get @raw value
                    v1 = val["@raw"]
                elif "@expr" in val:
                    # Validate
                    valid = self._validate_expr(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid, val
                elif "@cons" in val:
                    # Validate
                    valid = self._validate_cons(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid, val
                elif "@map" in val:
                    # Validate
                    valid = self._validate_map(opt, val, mode, j)
                    # Unless we have access to *self.x* and *i*, done
                    return valid, val
            # Check if all types are allowed
            if opttype is None:
                return True, val
            # Check scalar value
            if isinstance(v1, opttype):
                # Accepted type
                return True, val
            # Special allowances for converting dict -> OptionsDict
            if isinstance(val, dict) and v1 is val:
                # Ensure opttype is a tuple
                if not isinstance(opttype, tuple):
                    opttype = opttype,
                # Loop through types
                for opttypej in opttype:
                    # Check for allowed OptionsDict subtype
                    if issubclass(opttypej, OptionsDict):
                        # Found one! Convert the dict -> *opttypej*
                        return True, opttypej(val)
            # Form and save error message
            self._save_opttype_error(opt, val, opttype, mode, j)
            # Test failed
            return False, val
        else:
            # Recurse
            for j, vj in enumerate(val):
                # Test *j*th entry
                qj, v2j = self._check_opttype(opt, vj, mode, j=j)
                # Save new value if appropriate
                if vj is not v2j:
                    # Using 'is' test limits fail for *val* tuple
                    val[j] = v2j
                # Check for failure
                if not qj:
                    return False, val
            # Each entry passed
            return True, val

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
            * 2022-09-25 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v1.1; use ``_lastwarn``
        """
        # Fill out mode
        mode = self._get_warnmode(mode, INDEX_ITYPE)
        # Don't check for mode 0
        if mode == WARNMODE_NONE or val is None:
            return True
        # Get acceptable values
        optvals = self.getx_optvals(opt)
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
        listdepth = self.getx_listdepth(opt)
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
            * 2022-09-19 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v1.1; no raises
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
            * 2022-09-19 ``@ddalle``: v1.0
            * 2022-09-30 ``@ddalle``: v1.1; no raises
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
            * 2022-09-19 ``@ddalle``: v1.0
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
    def getx_optlist(self) -> set:
        r"""Get list of explicitly named options

        :Call:
            >>> optlist = opts.getx_optlist()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Outputs:
            *optlist*: :class:`set`
                Allowed option names; if empty, all options allowed
        :Versions:
            * 2022-09-19 ``@ddalle``: v1.0
            * 2022-10-04 ``@ddalle``: v2.0: recurse throubh bases
        """
        # Get list of options from class
        optlist = self.__class__.getx_cls_set("_optlist")
        # Get instance-specific list
        xoptlist = self._xoptlist
        # Add them
        if isinstance(xoptlist, set):
            # Combine
            return optlist | xoptlist
        else:
            # Just the class's
            return optlist

    def getx_aliases(self, opt: str) -> set:
        r"""Get list of aliases for an option

        :Call:
            >>> aliases = opts.getx_aliases(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                (Final) option name
        :Outputs:
            *aliases*: :class:`set`\ [:class:`str`]
                Set of unique aliases for *opt*, including *opt*
        :Versions:
            * 2022-11-04 ``@ddalle``: v1.0
        """
        # Class
        cls = self.__class__
        # Get full optmap
        optmap = cls.getx_cls_dict("_optmap")
        # Initialize alias list
        aliases = {opt}
        # Loop through optmap
        for alias, optj in optmap.items():
            # Check for match
            if opt == optj:
                aliases.add(alias)
        # Output
        return aliases

    def getx_opttype(self, opt: str):
        r"""Get allowed type(s) for *opt*

        :Call:
            >>> opttype = opts.getx_opttype(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *opttype*: ``None`` | :class:`type` | :class:`tuple`
                Type or tuple thereof of allowed type(s) for *opt*
        :Versions:
            * 2022-09-19 ``@ddalle``: v1.0
        """
        # Get instance-specific
        opttypes = self._xopttypes
        # Check if present
        if (opttypes is not None) and (opt in opttypes):
            # Instance-specific overrides
            return opttypes[opt]
        # Use class's map
        opttypes = self.__class__.getx_cls_key("_opttypes", opt)
        # Check for a "_default_" if missing
        if opttypes is None:
            opttypes = self.__class__.getx_cls_key("_opttypes", "_default_")
        # Output
        return opttypes

    def getx_listdepth(self, opt: str) -> int:
        r"""Get list depth for a specified key

        :Call:
            >>> depth = opts.getx_listdepth(opt)
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
            * 2022-09-09 ``@ddalle``: v1.0
            * 2022-09-18 ``@ddalle``: v1.1; simple dict
            * 2023-06-13 ``@ddalle``: v1.2; use ``getx_cls_dict()``
        """
        # Check input type
        assert_isinstance(opt, str)
        # Get option from attribute
        optlistdepth = self.getx_cls_dict("_optlistdepth")
        # Check if directly present
        if opt in optlistdepth:
            return optlistdepth[opt]
        # Check for default
        if "_default_" in optlistdepth:
            return optlistdepth["_default_"]
        # Default
        return optitem.DEFAULT_LISTDEPTH

    def getx_optring(self, opt: str) -> bool:
        r"""Check if *opt* should be looped

        If ``True``, cycle through entire list. If ``False``, just
        repeat the final entry.

        :Call:
            >>> ring = opts.getx_optring(opt)
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
            * 2023-06-13 ``@ddalle``: v1.0
        """
        # Check input type
        assert_isinstance(opt, str)
        # Apply optmap
        opt = self.apply_optmap(opt)
        # Get _optring dict
        optring = self.getx_cls_dict("_optring")
        # Get default
        ringdef = optring.get("_default_", False)
        # Process *opt* specifically
        ring = optring.get(opt, ringdef)
        # Output
        return ring

    def getx_optvals(self, opt: str):
        r"""Get set of acceptable values for option *opt*

        :Call:
            >>> vals = opts.getx_optvals(opt)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *vals*: ``None`` | :class:`set`
                Allowed values for option *opt*
        :Versions:
            * 2022-09-24 ``@ddalle``: v1.0
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
    def _process_xoptkey(self, a):
        r"""Add to allowed options by processing ``_xoptkey``

        :Call:
            >>> opts._process_xoptkey(a)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *a*: :class:`dict`
                Raw (unvalidated) options to process
        :Versions:
            * 2022-11-05 ``@ddalle``: v1.0
        """
        # Get key
        key = self._xoptkey
        # Check if missing
        if key is None:
            return
        # Get aliases
        aliases = self.getx_aliases(key)
        # Loop through aliases to see if any are present
        for opt in aliases:
            # Check if present
            if opt not in a:
                continue
            # Add to _xoptlist all the values from *opt*
            self.add_xopts(a[opt])

    def add_xopts(self, optlist):
        r"""Add several instance-specific allowed option names

        :Call:
            >>> opts.add_xopts(optlist)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *optlist*: :class:`list` | :class:`set` | :class:`tuple`
                List of options to combine
        :Versions:
            * 2022-09-19 ``@ddalle``: v1.0
        """
        # Check input
        assert_isinstance(
            optlist, (set, list, tuple), "option name list, *opts*")
        # Check each entry (str)
        for j, opt in enumerate(optlist):
            assert_isinstance(opt, str, "option list entry %i" % j)
        # Initialize if needed
        if self._xoptlist is None:
            self._xoptlist = set(optlist)
        else:
            # Append
            self._xoptlist.update(optlist)

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
            * 2022-09-19 ``@ddalle``: v1.0
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
            * 2022-09-19 ``@ddalle``: v1.0
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

   # --- Parent ---
    def setx_parent(self, parent: dict):
        r"""Set an object to define fall-back values

        This takes precedence over _rc

        :Call:
            >>> opts.setx_parent(parent)
        :Inputs:
            *opts* :class:`OptionsDict`
                Options interface
            *parent*: :class:`dict` | :class:`OptionsDict`
                Fall-back options :class:`dict`
        :Versions:
            * 2022-12-02 ``@ddalle``: v1.0
        """
        # Check type
        assert_isinstance(parent, dict, "parent for fall-back values")
        # Save it
        self._xparent = parent

    def _check_parent(self, opt: str):
        r"""Check if *opts* has an option in its fall-back

        :Call:
            >>> q = opts._check_parent(opt)
        :Inputs:
            *opts* :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Option name
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *opt* is in *opts._xparent*
        :Versions:
            * 2022-12-02 ``@ddalle``: v1.0
        """
        # Check type
        assert_isinstance(opt, str, "option name")
        # Get parent
        parent = self._xparent
        # Check type
        if isinstance(parent, dict):
            return opt in parent
        # No parent
        return False

    def getx_opt_parent(self, opt: str):
        r"""Get value from fall-back *opts._xparent*

        :Call:
            >>> v = opts.getx_opt_parent(opt, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Name of option to access
        :Outputs:
            *val*: :class:`object`
                Value of *opt* for given conditions, in the simplest
                case simply ``opts[opt]``
        :Versions:
            * 2022-12-02 ``@ddalle``: v1.0
        """
        # Check type
        assert_isinstance(opt, str, "option name")
        # Get parent
        parent = self._xparent
        # Check type
        if isinstance(parent, dict):
            return parent.get(opt)

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

    def _save_listdepth_error(self, opt, val, listdepth, mode):
        # Generate message
        msg = f"Option '{opt}' must have array depth >= {listdepth}"
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
    def getx_cls_key(cls, attr: str, key: str, vdef=None):
        r"""Access *key* from a :class:`dict` class attribute

        This will look in the bases of *cls* if ``getattr(cls, attr)``
        does not have *key*. If *cls* is a subclass of another
        :class:`OptionsDict` class, it will search trhough the bases of
        *cls* until the first time it finds a class attribute *attr*
        that is a :class:`dict` containing *key*.

        :Call:
            >>> v = cls.getx_cls_key(attr, key, vdef=None)
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
            * 2022-10-01 ``@ddalle``: v1.0
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
                return clsj.getx_cls_key(attr, key, vdef=vdef)
        # Not found
        return vdef

    @classmethod
    def getx_cls_set(cls, attr: str):
        r"""Get combined :class:`set` for *cls* and its bases

        This allows a subclass of :class:`OptionsDict` to only add to
        the ``_optlist`` attribute rather than manually include the
        ``_optlist`` of all the bases.

        :Call:
            >>> v = cls.getx_cls_set(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *attr*: :class:`str`
                Name of class attribute to search
        :Outputs:
            *v*: :class:`set`
                Combination of ``getattr(cls, attr)`` and
                ``getattr(base, attr)`` for each ``base`` in
                ``cls.__bases__``, etc.
        :Versions:
            * 2022-10-04 ``@ddalle``: v1.0
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
                clsset.update(clsj.getx_cls_set(attr))
        # Output
        return clsset

    @classmethod
    def getx_cls_dict(cls, attr: str):
        r"""Get combined :class:`dict` for *cls* and its bases

        This allows a subclass of :class:`OptionsDict` to only add to
        the ``_opttypes`` or ``_sec_cls`` attribute rather than manually
        include contents of all the bases.

        :Call:
            >>> clsdict = cls.getx_cls_dict(attr)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *attr*: :class:`str`
                Name of class attribute to search
        :Outputs:
            *clsdict*: :class:`dict`
                Combination of ``getattr(cls, attr)`` and
                ``getattr(base, attr)`` for each ``base`` in
                ``cls.__bases__``, etc.
        :Versions:
            * 2022-10-24 ``@ddalle``: v1.0
        """
        # Get attribute
        clsdict = cls.__dict__.get(attr)
        # Initialize if necessary
        if not isinstance(clsdict, dict):
            clsdict = {}
        # Loop through bases
        for clsj in cls.__bases__:
            # Only process if OptionsDict
            if issubclass(clsj, OptionsDict):
                # Recurse
                clsdictj = clsj.getx_cls_dict(attr)
                # Update, but don't overwrite
                for kj, vj in clsdictj.items():
                    clsdict.setdefault(kj, vj)
        # Output
        return clsdict

   # --- Subsections ---
    @classmethod
    def promote_sections(cls, skip=[]):
        r"""Promote all sections based on class attribute *_sec_cls*

        :Call:
            >>> cls.promote_sections(skip=[])
        :Inputs:
            *cls*: :class:`type`
                Parent class
            *skip*: {``[]``} | :class:`list`
                List of sections to skip
        :Versions:
            * 2022-10-23 ``@ddalle``: v1.0
        """
        # Loop through sections
        for secj, clsj in cls._sec_cls.items():
            # Check if skipped
            if secj in skip:
                continue
            # Promote
            cls.promote_subsec(clsj, secj)

    @classmethod
    def promote_subsec(cls, cls2, sec=None, skip=[], **kw):
        r"""Promote all methods of a subsection class to parent class

        Methods of parent class will not be overwritten

        :Call:
            >>> promote_subsec(cls1, cls2, sec=None, skip=[], **kw)
        :Inputs:
            *cls*: :class:`type`
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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        # Prevent operations on OptionsDict directly
        cls._assert_subclass()
        # Call module function
        promote_subsec(cls, cls2, sec=sec, skip=skip, **kw)

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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        for opt in optlist:
            cls.add_property(opt, prefix=prefix, name=name, doc=doc)

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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        for opt in optlist:
            cls.add_getter(opt, prefix=prefix, name=name, doc=doc)

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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        for opt in optlist:
            cls.add_setter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_extenders(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of property extenders with common settings

        :Call:
            >>> cls.add_extenders(optlist, prefix=None, name=None)
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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        for opt in optlist:
            cls.add_extender(opt, prefix=prefix, name=name, doc=doc)

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
            * 2022-09-30 ``@ddalle``: v1.0
            * 2022-10-03 ``@ddalle``: v1.1; docstrings
            * 2022-10-10 ``@ddalle``: v1.2; metadata, try/catch
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
            >>> cls.add_getter(opt, prefix=None, name=None)
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
            * 2022-09-30 ``@ddalle``: v1.0
            * 2022-10-03 ``@ddalle``: v1.1; docstrings
            * 2022-10-10 ``@ddalle``: v1.2; metadata, try/catch
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
            >>> cls.add_setter(opt, prefix=None, name=None)
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
            * 2022-09-30 ``@ddalle``: v1.0
            * 2022-10-03 ``@ddalle``: v1.1; docstrings
            * 2022-10-10 ``@ddalle``: v1.2; metadata, try/catch
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
    def add_extender(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add extender method for option *opt*

        For example ``cls.add_extender("a")`` will add a function
        :func:`add_a`, which has a signatures like
        :func:`OptionsDict.extend_opt` except that it doesn't have the
        *opt* input.

        :Call:
            >>> cls.add_extender(opt, prefix=None, name=None)
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
            * 2022-10-14 ``@ddalle``: v1.0
        """
        # Check if acting on original OptionsDict
        cls._assert_subclass()
        # Default name
        name, fullname = cls._get_funcname(opt, name, prefix)
        funcname = "add_" + fullname
        # Check if already present
        if funcname in cls.__dict__:
            raise OptdictAttributeError(
                "Method '%s' for class '%s' already exists"
                % (funcname, cls.__name__))

        # Define function
        def func(self, val, **kw):
            try:
                return self.extend_opt(opt, val, **kw)
            except Exception:
                raise

        # Generate docstring if requrested
        if doc:
            func.__doc__ = cls.genr8_extender_docstring(opt, name, prefix)
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

   # --- Help ---
    def help_opt(self, opt: str, **kw):
        r"""Open interactive help for option *opt*

        :Call:
            >>> opts.help_opt(opt, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Option name
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
            *v*: {``True``} | ``False``
                Verbose flag, returns ``show_option`` output
        :Versions:
            * 2023-06-22 ``@ddalle``: v1.0
        """
        # Create a temporary class
        class tmpcls(object):
            __slots__ = ()
        # For this message, set default to verbose
        kw.setdefault("v", True)
        # Generate message
        txt = self.getx_optinfo(opt, **kw)
        # Prepend title if not verbose
        if not kw.get("verbose", kw.get("v", False)):
            txt = "Description:\n\n" + txt
        # Assign the docstring
        tmpcls.__doc__ = txt
        # Reset the name
        tmpcls.__name__ = f"Option{opt}"
        # Create interactive hlep
        help(tmpcls)

    def getx_optinfo(self, opt: str, **kw) -> str:
        r"""Create output description for option

        :Call:
            >>> txt = opts.getx_optinfo(opt, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Option name
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
            *v*: ``True`` | {``False``}
                Verbose flag, returns :func:`show_option` output
        :Outputs:
            *txt*: :class:`str`
                Text for ``opt`` description
        :Versions:
            * 2023-06-14 ``@aburkhea``: v1.0
            * 2023-06-22 ``@ddalle``: v1.1
                * default indent 8->0
                * name ``help_opt()`` -> ``getx_optinfo()``
        """
        # Get verbose option
        v = kw.get("verbose", kw.get("v", False))
        # Use different function if verbose
        if v:
            return self.show_opt(opt, **kw)
        # Other options
        indent = kw.get("indent", 0)
        tab = kw.get("tab", 4)
        # Get class
        cls = self.__class__
        # Apply aliases if any
        fullopt = cls.getx_cls_key("_optmap", opt, vdef=opt)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(fullopt, indent=indent, tab=tab)
        # Output
        return rst_opt

    def show_opt(self, opt: str, **kw):
        r"""Display verbose help information for *opt*

        :Call:
            >>> txt = opts.show_opt(opt, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *opt*: :class:`str`
                Option name
            *indent*: {``0``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
            *c*: {``-``} | ``"="`` | ``"*``" | ``"^"``
                Character to use for reST title lines
            *overline*: ``True`` | {``False``}
                Option to have overline and underline for title
            *underline*: {``True``} | ``False``
                Option to print *opt* as a reST section title
        :Outputs:
            *txt*: :class:`str`
                Text of *opt* help message
        :Versions:
            * 2023-06-14 ``@aburkhea``: v1.0
            * 2023-06-22 ``@ddalle``: v2.0; only show nontrivial items
        """
        # Convert a type to a strint
        def typnam(cls: type) -> str:
            # Get module and name
            modname = cls.__module__
            clsname = cls.__name__
            # Check for built-ins
            if modname == "builtins":
                # Just use the name, e.g. "int"
                return clsname
            else:
                # Include module, e.g "numpy.int64"
                return f"{modname}.{clsname}"
        # Other options
        indent = kw.get("indent", 0)
        tab = kw.get("tab", 4)
        c = kw.get("c", '-')
        overline = kw.get("overline", False)
        underline = kw.get("underline", True)
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Apply aliases if any
        fullopt = self.apply_optmap(opt)
        # Form a line with sufficient length
        hline = c * len(fullopt)
        # Initialize title if not a section
        title = f"{fullopt}\n\n"
        # Make name string
        if overline:
            # --------
            # {fullopt}
            # ---------
            title = f"{hline}\n{fullopt}\n{hline}\n\n"
        elif underline:
            # {fullopt}
            # ---------
            title = f"{fullopt}\n{hline}\n\n"
        # Get description
        desc = self._genr8_rst_desc(fullopt)
        # Form message for description
        descmsg = f"{tab1}:Description:\n{tab2}{desc}\n"
        # Get alias list
        aliases = self.getx_aliases(fullopt)
        # Don't count *fullopt* as an aliase
        aliases.remove(fullopt)
        # Initialize empty alias text
        aliasmsg = ""
        # Form message about aliases
        if len(aliases):
            # Determine whether or not to include "es"
            plural = "es" if len(aliases) > 1 else ""
            # Form message
            aliasmsg = (
                f":Alias{plural}:\n" +
                "".join(f"{tab2}* {alias}\n" for alias in aliases))
        # Get types
        opttype = self.getx_cls_key("_opttypes", fullopt)
        # Initialize default type text: "object"
        typtxt = f"{tab2}* object\n"
        # Assume only one type
        plural = ""
        # If type is tuple
        if isinstance(opttype, tuple):
            # Multiple types
            typtxt = "".join(f"{tab2}* {typnam(cls)}\n" for cls in opttype)
            # Check for multiple types
            if len(opttype) > 1:
                plural = "s"
        elif isinstance(opttype, type):
            # Make type string
            typtxt = f"{tab2}* {typnam(opttype)}\n"
        # Add header to _opttype section
        typmsg = tab1 + f":Allowed Type{plural}:\n" + typtxt
        # Get defaults
        optrc = self.getx_cls_key("_rc", fullopt)
        # Initialize empty message for defaults
        rcmsg = ""
        # Format default value if any
        if optrc is not None:
            # Make default string
            rcs = tab2 + repr(optrc) + "\n"
            # Add default
            rcmsg = tab1 + ":Default Value:\n" + rcs
        # Get permitted values
        optvals = self.getx_cls_key("_optvals", fullopt)
        # Initialize empty message for _optvals
        valmsg = ""
        # Check for specified _optvals
        if optvals is not None:
            # Form list
            valtxt = "".join(f"{tab2}* {repr(val)}\n" for val in optvals)
            # Add header
            valmsg = tab1 + ":Permitted Values:\n" + valtxt
        # Get list depth
        listdepth = self.getx_listdepth(fullopt)
        # Initialize empty messag
        ldmsg = ""
        # Check for positive list depth
        if listdepth:
            # Add text
            ldmsg = f"{tab1}:List Depth:\n{tab2}{listdepth}\n"
        # Assemble help message
        helpmsg = (
            title + descmsg + aliasmsg + typmsg + rcmsg + valmsg + ldmsg)
        # Form full help message
        return helpmsg

   # --- Low-level: docstring ---
    @classmethod
    def genr8_getter_docstring(cls, opt: str, name, prefix, **kw):
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
            *prefix*: ``None`` | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
            *extra_args*: {``None``} | :class:`dict`\ [:class:`tuple`]
                Dictionary of args and their types and descriptions
        :Outputs:
            *txt*: :class:`str`
                Contents for ``get_{opt}`` function docstring
        :Versions:
            * 2022-10-03 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v1.1; extra inputs
        """
        # Other options
        indent = kw.get("indent", 8)
        tab = kw.get("tab", 4)
        extra_args = kw.get("extra_args", {})
        # Expand tabs
        tab0 = " " * tab
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.getx_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get %s\n\n' % cls._genr8_rst_desc(fullopt)
        # Form signature for extra args
        narg = len(extra_args)
        args_signature = ", ".join(extra_args) + (", " * min(1, narg))
        # Generate signature
        signature = (
            "%s>>> %s = opts.get_%s(%sj=None, i=None, **kw)\n"
            % (tab2, name, funcname, args_signature))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent+tab, tab=tab)
        # Initialize extra args args
        rst_args_list = []
        # Loop through args
        for arg in extra_args:
            # Check type
            if isinstance(extra_args, dict):
                # Get values from inputs
                arg_raw = extra_args[arg]
            else:
                # No description; declare a default
                arg_raw = (":class:`object`", arg)
            # Check if given a string
            if isinstance(arg_raw, str) or len(arg_raw) != 2:
                # Map that single string to a description
                arg_raw = (":class:`object`", arg_raw)
            # Unpack type and description
            rst_type, rst_descr = arg_raw
            # Append to args
            rst_args_list.append(tab2 + ("*%s*: %s" % (arg, rst_type)))
            rst_args_list.append(tab2 + tab0 + rst_descr)
        # Create inputs for extra args
        rst_extra_args = "\n".join(rst_args_list) + ("\n" * min(1, narg))
        # Form full docstring
        return (
            title +
            tab1 + ":Call:\n" +
            signature +
            tab1 + ":Inputs:\n" +
            rst_cls +
            rst_extra_args +
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
            * 2022-10-03 ``@ddalle``: v1.0
        """
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.getx_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get %s\n\n' % cls._genr8_rst_desc(fullopt)
        # Generate signature
        signature = (
            "%s>>> opts.set_%s(%s, j=None, i=None, **kw)\n"
            % (tab2, funcname, name))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent+tab, tab=tab)
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
    def genr8_extender_docstring(cls, opt: str, name, prefix, indent=8, tab=4):
        r"""Create automatic docstring for extender function

        :Call:
            >>> txt = cls.genr8_extender_docstring(opt, name, pre, **kw)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of functions
            *pre*: ``None`` | :class:`str`
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
            * 2022-10-03 ``@ddalle``: v1.0
        """
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.getx_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get %s\n\n' % cls._genr8_rst_desc(fullopt)
        # Generate signature
        signature = (
            "%s>>> opts.add_%s(%s, **kw)\n"
            % (tab2, funcname, name))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent+tab, tab=tab)
        # Form full docstring
        return (
            title +
            tab1 + ":Call:\n" +
            signature +
            tab1 + ":Inputs:\n" +
            rst_cls +
            rst_opt +
            tab2 + _RST_ADDOPT
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
        # Tabs
        # Two-line description
        return (
            ("%*s*%s*: %s\n" % (indent, "", name, opttypes)) +
            ("%*s%s\n" % (indent + tab, "", optdesc)))

    @classmethod
    def _genr8_rst_desc(cls, opt: str) -> str:
        # Check for user-defined descripiton
        txt = cls.getx_cls_key("_rst_descriptions", opt)
        if txt:
            return txt
        # Otherwise produce a generic description
        return 'value of option "%s"' % opt

    @classmethod
    def _genr8_rst_opttypes(cls, opt: str) -> str:
        # Check for user-defined string
        txt = cls.getx_cls_key("_rst_types", opt)
        if txt:
            return txt
        # Get a default value
        vdef = cls.getx_cls_key("_rc", opt)
        # Check for values
        optvals = cls.getx_cls_key("_optvals", opt)
        # Check for values
        if optvals is not None:
            return genr8_rst_value_list(optvals, vdef)
        # Check for types
        opttypes = cls.getx_cls_key("_opttypes", opt)
        # Get list depth
        listdepth = cls.getx_cls_key(
            "_optlistdepth", opt, vdef=optitem.DEFAULT_LISTDEPTH)
        # Convert opttypes to text
        return genr8_rst_type_list(opttypes, vdef, listdepth)


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
        * 2019-01-10 ``@ddalle``: v1.0
    """
    # Get property dictionaries
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
        if hasattr(cls1, fn):
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
        * 2021-12-06 ``@ddalle``: v1.0
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
        * 2019-01-10 ``@ddalle``: v1.0
        * 2021-10-18 ``@ddalle``: v1.1; default *sec*
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
        * 2022-10-02 ``@ddalle``: v1.0
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
        * 2022-10-02 ``@ddalle``: v1.0
    """
    # Replace each value with representation
    strvals = [
        ("{``%r``}" if v == vdef else "``%r``") % v
        for v in optvals
    ]
    # Assemble string
    return " | ".join(strvals)


# Generate string for list of types
def genr8_rst_type_list(opttypes, vdef=None, listdepth=0):
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
        *listdepth*: {``0``} | ``1``
            List depth; serves as flag that function returns list
    :Outputs:
        *txt*: :class:`str`
            reStructuredText representation of *opttypes*
    :Versions:
        * 2022-10-03 ``@ddalle``: v1.0
        * 2023-04-20 ``@ddalle``: v1.1; add *listdepth*
    """
    # Always show default value
    vdef_txt = "{``%r``} | " % vdef
    # Convert types to string
    if isinstance(opttypes, type):
        # Single type
        type_txt = ":class:`%s`" % opttypes.__name__
    elif opttypes == INT_TYPES:
        # Special case for many int types
        type_txt = _RST_INT_TYPES
    elif opttypes == FLOAT_TYPES:
        # Special case for many float types
        type_txt = _RST_FLOAT_TYPES
    elif opttypes:
        # Convert each type to a string
        strtypes = [":class:`%s`" % clsj.__name__ for clsj in opttypes]
        # Add types to string
        type_txt = " | ".join(strtypes)
    else:
        # Assume all types are allowed
        type_txt = ":class:`object`"
    # Check for listdepth
    listflag = int(listdepth > 0)
    types_txt = (r":class:`list`\ ["*listflag) + type_txt + ("]"*listflag)
    # Output
    return vdef_txt + types_txt

