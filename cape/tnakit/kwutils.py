#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods to process keyword argument dictionaries
by checking them against

    * a list (ideally a set) of acceptable names
    * a dictionary of alternate names
    * a dictionary of acceptable types
    * a dictionary of other keys required for any key with dependencies

"""

# Standard library
import sys
import copy
import difflib

# Local modules
from . import optitem
from . import rstutils
from . import typeutils


# Map keywords
def map_kw(kwmap, **kw):
    r"""Map alternate keyword names with no checks

    :Call:
        >>> kwo = map_kw(kwmap, **kw)
    :Inputs:
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Translated keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version
    """
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Save it
        kwo[k] = v
    # Output
    return kwo


# Check valid keyword names, with dependencies
def check_kw(kwlist, kwmap, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    :Call:
        >>> kwo = check_kw(kwlist, kwmap, kwdep, mode, **kw)
    :Inputs:
        *kwlist*: {*db._kw*} | :class:`list`\ [:class:`str`]
            List of acceptable parameters
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kwdep*: {*db._kw_depends*} | :class:`dict`\ [:class:`list`]
            Dictionary of required parameters for some parameters
        *mode*: ``0`` | {``1``} | ``2``
            Flag for quiet (``0``), warn (``1``), or strict (``2``)
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Valid keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version
    """
    # Check mode
    if mode not in [0, 1, 2]:
        raise ValueError("Verbose mode must be 0, 1, or 2")
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Check if present
        if k not in kwlist:
            # Get closet match (n=3 max)
            mtchs = difflib.get_close_matches(k, kwlist)
            # Issue warning
            if len(mtchs) == 0:
                # No suggestions
                msg = "Unrecognized keyword '%s'" % k
            else:
                # Show up to three suggestions
                msg = (
                    ("Unrecognized keyword '%s'" % k) +
                    ("; suggested: %s" % " ".join(mtchs)))
            # Choose warning
            if mode == 2:
                # Exception
                raise KeyError(msg)
            elif mode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            # Go to next keyword
            continue
        else:
            # Copy to output
            kwo[k] = v
        # Check dependences
        if k in kwdep:
            # Get item
            kdep = kwdep[k]
            # Check if any dependency is present
            if all([ki not in kw for ki in kdep]):
                # Create warning message
                msg = (
                    ("Keyword '%s' depends on one of " % k) +
                    ("the following: %s" % " ".join(kdep)))
                # Choose what to do about it
                if mode == 2:
                    # Exception
                    raise KeyError(msg)
                elif mode == 1:
                    # Warning
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
    # Output
    return kwo


# Check valid keyword names, with dependencies
def check_kw_types(kwlist, kwmap, kwtypes, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    :Call:
        >>> kwo = check_kw_types(|args1|, **kw)
    :Inputs:
        *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
            Data file interface
        *kwlist*: {*db._kw*} | :class:`list`\ [:class:`str`]
            List of acceptable parameters
        *kwtypes*: {*db._kw_types*} | :class:`dict`
            Dictionary of :class:`type` or
            :class:`tuple`\ [:class:`type`] for some or all
            keywords, used with :func:`isinstance`
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kwdep*: {*db._kw_depends*} | :class:`dict`\ [:class:`list`]
            Dictionary of required parameters for some parameters
        *mode*: ``0`` | {``1``} | ``2``
            Flag for quiet (``0``), warn (``1``), or strict (``2``)
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Valid keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version

    .. |args1| replace:: kwlist, kwmap, kwtypes, kwdep, mode
    """
    # Check mode
    if mode not in [0, 1, 2]:
        raise ValueError("Verbose mode must be 0, 1, or 2")
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Check if present
        if k not in kwlist:
            # Get closet match (n=3 max)
            mtchs = difflib.get_close_matches(k, kwlist)
            # Issue warning
            if len(mtchs) == 0:
                # No suggestions
                msg = "Unrecognized keyword '%s'" % k
            else:
                # Show up to three suggestions
                msg = (
                    ("Unrecognized keyword '%s'" % k) +
                    ("; suggested: %s" % " ".join(mtchs)))
            # Choose warning
            if mode == 2:
                # Exception
                raise KeyError(msg)
            elif mode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            # Go to next keyword
            continue
        # Check for a type
        ktype = kwtypes.get(k, object)
        # Check the type
        if isinstance(v, ktype):
            # Save the value and move on
            kwo[k] = v
        else:
            # Create warning message
            msg = (
                ("Invalid type for keyword '%s'" % k) +
                ("; options are %s" % ktype))
            # Check mode
            if mode == 2:
                # Exception
                raise TypeError(msg)
            elif mode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
        # Check dependences
        if k in kwdep:
            # Get item
            kdep = kwdep[k]
            # Check if any dependency is present
            if all([ki not in kw for ki in kdep]):
                # Create warning message
                msg = (
                    ("Keyword '%s' depends on one of " % k) +
                    ("the following: %s" % " ".join(kdep)))
                # Choose what to do about it
                if mode == 2:
                    # Exception
                    raise KeyError(msg)
                elif mode == 1:
                    # Warning
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
    # Output
    return kwo


# Check valid keyword names, with dependencies
def check_kw_eltypes(kwlist, kwmap, kwtypes, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    Each keyword is permitted to be a :class:`list` of the required
    type in addition to just being the correct type.

    :Call:
        >>> kwo = check_kw_eltypes(|args2|, **kw)
    :Inputs:
        *kwlist*: :class:`set`\ [:class:`str`]
            List (set) of acceptable parameters
        *kwtypes*: {*db._kw_types*} | :class:`dict`
            Dictionary of :class:`type` or
            :class:`tuple`\ [:class:`type`] for some or all
            keywords, used with :func:`isinstance`
        *kwmap*: {*db._kw_map*}: :class:`dict`\ [:class:`str`]
            Map of *alternate*: *primary* abbreviations
        *kwdep*: {*db._kw_depends*} | :class:`dict`\ [:class:`list`]
            Dictionary of required parameters for some parameters
        *mode*: ``0`` | {``1``} | ``2``
            Flag for quiet (``0``), warn (``1``), or strict (``2``)
        *kw*: :class:`dict`
            Any keyword arguments
    :Outputs:
        *kwo*: :class:`dict`
            Valid keywords and their values from *kw*
    :Versions:
        * 2019-12-13 ``@ddalle``: First version

    .. |args2| replace:: kwlist, kwmap, kwtypes, kwdep, mode
    """
    # Check mode
    if mode not in [0, 1, 2]:
        raise ValueError("Verbose mode must be 0, 1, or 2")
    # Initialize output
    kwo = {}
    # Loop through keys
    for (k0, v) in kw.items():
        # Map names if appropriate
        k = kwmap.get(k0, k0)
        # Check if present
        if k not in kwlist:
            # Get closet match (n=3 max)
            mtchs = difflib.get_close_matches(k, kwlist)
            # Issue warning
            if len(mtchs) == 0:
                # No suggestions
                msg = "Unrecognized keyword '%s'" % k
            else:
                # Show up to three suggestions
                msg = (
                    ("Unrecognized keyword '%s'" % k) +
                    ("; suggested: %s" % " ".join(mtchs)))
            # Choose warning
            if mode == 2:
                # Exception
                raise KeyError(msg)
            elif mode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            # Go to next keyword
            continue
        # Check for a type
        ktype = kwtypes.get(k, object)
        # Check the type
        if isinstance(v, ktype):
            # Save the value and move on (single item/scalar)
            kwo[k] = v
        elif isinstance(v, list) and all([isinstance(vi, ktype) for vi in v]):
            # Save the list of values
            kwo[k] = v
        else:
            # Type string
            if isinstance(ktype, tuple):
                # Join names
                typestr = " ".join([cls.__name__ for cls in ktype])
            else:
                # Single name
                typestr = ktype.__name__
            # Create warning message
            msg = (
                ("Invalid type for keyword '%s'" % k) +
                ("; options are: %s" % typestr))
            # Check mode
            if mode == 2:
                # Exception
                raise TypeError(msg)
            elif mode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
        # Check dependences
        if k in kwdep:
            # Get item
            kdep = kwdep[k]
            # Check if any dependency is present
            if all([ki not in kw for ki in kdep]):
                # Create warning message
                msg = (
                    ("Keyword '%s' depends on one of " % k) +
                    ("the following: %s" % " ".join(kdep)))
                # Choose what to do about it
                if mode == 2:
                    # Exception
                    raise KeyError(msg)
                elif mode == 1:
                    # Warning
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
    # Output
    return kwo


# Class to contain processed keywords
class KwargHandler(dict):
    r"""Class to process kwargs against preset key lists and types

    :Call:
        >>> opts = KwargHandler(_optsdict=None, _warnmode=1, **kw)
    :Inputs:
        *kw*: :class:`dict`
            Keyword arguments turned into options
        *_optsdict*: {``None``} | :class:`dict`
            Dictionary of previous options (overwritten by *kw*)
        *_warnmode*: ``0`` | {``1``} | ``2``
            Warning mode from :mod:`kwutils`
        *_section*: {``None``} | :class:`str`
            Name of options section to restrict to
        *_optlist*: {``None``} | :class:`set`\ [:class:`str`]
            Specified list of allowed options
    :Outputs:
        *opts*: :class:`MPLOpts`
            Options interface from *kw* with checks and applied defaults
    :Versions:
        * 2020-01-16 ``@ddalle``: Generalized from :mod:`plot_mpl`
    """
  # =================
  # Class Attributes
  # =================
  # <
   # --- Global Keywords ---
    # All options
    _optlist = set()

    # Options for which ``None`` is allowed
    _optlist_none = set()

    # Options for which a singleton should be a list
    _optlist_list = set()

    # Options for which lists are rings
    _optlist_ring = set()
    
    # Options for which lists are hold-last
    _optlist_holdlast = set()

    # Default list type
    _optlist_type = 0
    
    # Alternate names
    _optmap = {}

    # Types
    _opttypes = {}

    # Dependencies
    _optdependencies = {}

    # Sets of allowed values
    _optvals = {}

    # Transformations for option values
    _optvalmap = {}

    # Converters (before value checking)
    _optval_converters = {}

   # --- Option Sublists ---
    # Dictionary of options for certain functions (ordered)
    _optlists = {}

   # --- Cascading Options ---
    # Global options mapped to subcategory options
    _kw_submap = {}

   # --- Conflicting Options ---
    # Aliases to merge for subcategory options
    _kw_subalias = {}

   # --- Documentation Data ---
    # Type strings
    _rst_types = {}

    # Option descriptions
    _rst_descriptions = {}

   # --- Default Control ---
    # Global defaults
    _rc = {}

    # Subdefaults
    _rc_sections = {}
  # >
  
  # ============
  # Config
  # ============
  # <
    # Initialization method
    def __init__(self, _optsdict=None, _warnmode=1, **kw):
        r"""Initialization method

        :Versions:
            * 2019-12-19 ``@ddalle``: First version (plot_mpl.MPLOpts)
        """
        # Get class
        cls = self.__class__
        # Initialize an unfiltered dict
        if isinstance(_optsdict, dict):
            # Initialize with dictionary
            optsdict = dict(_optsdict, **kw)
        else:
            # Initialize from just keywords
            optsdict = kw
        # Remove anything that's ``None``
        opts = cls.denone(optsdict)

        # Check for a section
        sec = kw.pop("_section", None)
        secs = kw.pop("_sections", None)
        # Check for a specified list
        optlist = kw.pop("_optlist", set())
        
        # Ensure set
        if not isinstance(optlist, set):
            optlist = set(optlist)

        # Process section list
        if secs is not None:
            # Loop through sections
            for sec in secs:
                # Get options for that section
                secopts = cls._optlists.get(sec)
                # Union the options
                if secopts:
                    optlist |= set(secopts)
        elif sec is not None:
            # Get options for one section
            secopts = cls._optlists.get(sec)
            # Union the options
            if secopts:
                optlist |= set(secopts)

        # Default option list
        if len(optlist) == 0:
            # All options
            optlist = cls._optlist

        # Save settings
        self._optlist_check = optlist
        self._warnmode = _warnmode

        # Check keywords
        opts = check_kw_eltypes(
            optlist,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies, _warnmode, **opts)

        # Copy entries
        for (k, v) in opts.items():
            # Check value before saving
            self._set_option(k, v)
  # >

  # ==================
  # Change Settings
  # ==================
  # <
   # --- Set ---
    # Set an option, with checks
    def set_option(self, opt, val):
        r"""Set an option, with checks

        The option is first looked up in *_optmap* and saved with the
        full name found there, if any.  After mapping, *opt* is checked
        against the *_optlist* and *_opttypes* of the class.

        :Call:
            >>> opts.set_option(opt, val)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *opt*: :class:`str`
                Name of option
            *val*: :class:`any`
                Specified value
        :Versions:
            * 2020-01-26 ``@ddalle``: First version
        """
        # Check for trivial input
        if val is None:
            return
        # Class
        cls = self.__class__
        # Create simple keyword dict
        kw = {opt: val}
        # Check validity, apply maps
        opts = check_kw_eltypes(
            self._optlist_check,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies,
            self._warnmode, **kw)
        # Expand abbreviation or alternate name
        opt = cls._optmap.get(opt, opt)
        # If that survived, save the value
        if opts:
            self._set_option(opt, val)

    # Set an option while checking the *value*
    def _set_option(self, opt, val):
        r"""Set an option, with checks on *val*

        The option is looked up in *_optvalmap* followed by
        *_optval_converters* and altered if indicated by those class
        attributes.  The final value is then checked against *_optvals*
        before finally saving as a key if that test is passed.

        :Call:
            >>> opts.set_option(opt, val)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *opt*: :class:`str`
                Name of option
            *val*: :class:`any`
                Specified value
        :Versions:
            * 2020-01-31 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Warning mode
        _warnmode = getattr(cls, "_warnmode", 1)
        # Value-check dicts
        optvmap = cls._optvalmap
        optvals = cls._optvals
        optconv = cls._optval_converters
        # Get map, converter, and set of allowed values
        M = optvmap.get(opt)
        f = optconv.get(opt)
        V = optvals.get(opt) 
        # Apply map
        if (M is not None) and typeutils.isstr(val):
            # Only works for strings
            val = M.get(val, val)
        # Perform conversion
        if f is not None:
            val = f(val)
        # Value check
        if (V is not None) and (val not in V):
            # Get closet match (n=3 max)
            if typeutils.isstr(val):
                # Get closest values
                try:
                    # Assume *V* has correct types
                    mtchs = difflib.get_close_matches(val, V)
                except TypeError:
                    # Only works if all of *V* is strings
                    mtches = []
                # Choose best warning
                if len(mtchs) == 0:
                    # No suggestions
                    msg = (
                        ("Unrecognized value '%s' " % val) +
                        ("for keyword '%s'" % opt))
                else:
                    # Show up to three suggestions
                    msg = (
                        ("Unrecognized value '%s' " % val) +
                        ("for keyword '%s'" % opt) +
                        ("; suggested: %s" % " ".join(mtchs)))
            else:
                # Close matches only for strings
                msg = "Keyword '%s' has invalid value" % opt
            # Choose what to do about it
            if _warnmode == 2:
                # Exception
                raise ValueError(msg)
            elif _warnmode == 1:
                # Warning
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            # Don't save it
            return
        # Save value
        self[opt] = val

    # Set an option, with checks
    def setdefault_option(self, opt, val):
        r"""Set an option, with checks, but without overwriting

        The option is first looked up in *_optmap* and saved with the
        full name found there, if any.  After mapping, *opt* is checked
        against the *_optlist* and *_opttypes* of the class.

        :Call:
            >>> v = opts.setdefault_option(opt, val)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *opt*: :class:`str`
                Name of option
            *val*: :class:`any`
                Specified value
        :Outputs:
            *v*: *opts[opt]* | *val*
                Same as output from :func:`setdefault`
        :Versions:
            * 2020-01-26 ``@ddalle``: First version
        """
        # Class
        cls = self.__class__
        # Expand abbreviation or alternate name
        opt = cls._optmap.get(opt, opt)
        # Check for trivial input
        if val is None:
            return self.get(opt)
        # Create simple keyword dict
        kw = {opt: val}
        # Check validity, apply maps
        opts = check_kw_eltypes(
            self._optlist_check,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies,
            self._warnmode, **kw)
        # If that survived, save the value
        if opts:
            return self.setdefault(opt, val)

   # --- Update Many ---
    # Apply several settings
    def update(self, **kw):
        r"""Apply several settings, with checks, from kwargs

        This works like the standard :func:`dict.apply` with two
        differences:

            * The same keyword checks and name changes that take place
              during :func:`__init__` are also done on *kw*

            * If both *opts* and *kw* have a value for key *k*, with
              *v0* for *opts[k]* and *v* for *kw[k]*, then the two
              values are blended using ``v0.update(v)``

        :Call:
            >>> opts.update(**kw)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *kw*: :class:`dict`
                Additional options added to *opts*, with checks
        :Versions:
            * 2020-01-24 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Warning mode
        warnmode = kw.pop("_warnmode", self._warnmode)
        # Remove anything that's ``None``
        opts = cls.denone(kw)
        # Check validity, apply maps
        opts = check_kw_eltypes(
            self._optlist_check,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies,
            warnmode, **opts)
        # Update settings
        for k, v in opts.items():
            # Get current setting
            v0 = self.get(k)
            # Check if present
            if isinstance(v, dict) and isinstance(v0, dict):
                # Meld settings
                v0.update(**v)
            else:
                # New key or overwritten key
                self._set_option(k, v)
  # >

  # ============
  # Utilities
  # ============
  # <
   # --- Filtering ---
    # Remove ``None`` keys
    @classmethod
    def denone(cls, opts):
        r"""Remove any keys whose value is ``None``
    
        :Call:
            >>> opts = denone(opts)
        :Inputs:
            *opts*: :class:`dict`
                Any dictionary
        :Outputs:
            *opts*: :class:`dict`
                Input with any keys whose value is ``None`` removed;
                returned for convenience but input is also affected
        :Versions:
            * 2019-03-01 ``@ddalle``: First version
            * 2019-12-19 ``@ddalle``: From :mod:`mplopts`
            * 2020-01-16 ``@ddalle``: From :mod:`plot_mpl`
        """
        # Loop through keys
        for (k, v) in dict(opts).items():
            # Check if ``None``
            if v is None:
                opts.pop(k)
        # Output
        return opts

   # --- Phase ---
    # Select options for phase *i*
    @classmethod
    def select_phase(cls, kw, i=0):
        r"""Select option *i* for each option in *kw*
    
        This cycles through lists of options for named options such as
        *color* and repeats if *i* is longer than the list of options.
        Special options like *dashes* that accept a list as a value are
        handled automatically.  If the value of the option is a single
        value, it is returned regardless of the value of *i*.
    
        :Call:
            >>> kw_p = cls.select_plotphase(kw, i=0)
        :Inputs:
            *kw*: :class:`dict`
                Dictionary of options for one or more graphics features
            *i*: {``0``} | :class:`int`
                Index
        :Outputs:
            *kw_p*: :class:`dict`
                Dictionary of options with lists replaced by scalar values
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2020-01-16 ``@ddalle``: From :mod:`plot_mpl`
        """
        # Initialize plot options
        kw_p = {}
        # Loop through options
        for (k, V) in kw.items():
            # Get ring-vs-holdlast type
            listtype = cls._get_listtype(k)
            # Check if this is a "list" option
            if k in cls._optlist_list:
                # Get value as a list
                if listtype == 0:
                    # Repeat entire list
                    v = optitem.getringel_list(V, i)
                else:
                    # Repeat last value
                    v = optitem.getel_list(v, i)
            else:
                # Get value as a scalar
                if listtype == 0:
                    # Repeat entire list
                    v = optitem.getringel(V, i)
                else:
                    # Repeat last value
                    v = optitem.getel(v, i)
            # Set option
            kw_p[k] = v
        # Output
        return kw_p

    # Get rule for phase beyond end of list
    @classmethod
    def _get_listtype(cls, k):
        r"""Get "ring" or "holdlast"  option for named key

        :Call:
            >>> ringcode = cls._get_listtype(k)
        :Inputs:
            *cls*: :class:`type`
                Options class
            *k*: :class:`str`
                Name of key
        :Outputs:
            *ringcode*: ``0`` | ``1``
                ``0`` for a ring, which repeats, ``1`` for a list
                where calling for indices beyond length of list
                returns last value
        :Versions:
            * 2020-01-16 ``@ddalle``: First version
        """
        # Check explicit options
        if k in cls._optlist_ring:
            # Explicitly a ring
            return 0
        elif k in cls._optlist_holdlast:
            # Explicitly a hold-last
            return 1
        else:
            # Use the default
            return cls._optlist_type
  # >

  # ===============
  # Item Retrieval
  # ===============
  # <
   # --- Individual Option ---
    # Get individual option
    def get_option(self, opt, vdef=None, parents=None):
        r"""Get value of a specific option, ignoring section

        :Call:
            >>> optval = opts.get_option(opt, vdef=None, parents=None)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *opt*: :class:`str`
                Name of option
            *vdef*: {``None``} | :class:`any`
                Default value, ignored if *opt* in *_rc*
            *parents*: ``None`` | :class:`set`\ [:class:`str`]
                List of parents, used to detect recursion
        :Outputs:
            *optval*: :class:`any`
                Processed value, from *opts* or *opts._rc*, with
                additional processing from *opts._kw_submap*, and others
        :Versions:
            * 2020-01-17 ``@ddalle``: First version
        """
        # Class
        cls = self.__class__
        # Default value
        optdef = cls._rc.get(opt, vdef)
        # Get value
        optval = self.get(opt)
        # Apply default
        if optval is None:
            # Copy the default
            optval = copy.copy(optdef)
        # Check for dict
        if optval is None:
            # Exit on trivial option
            return optval
        elif not isinstance(optval, dict):
            # Return the nontrivial, nondict value
            return optval
        # Default parent set
        if parents is None:
            # This option becomes the parent for later cals
            parents = set()
        # Get dictionary of options to inherit from
        kw_submap = cls._kw_submap.get(opt, {})
        # Loop through primary submap options
        for (fromopt, subopt) in kw_submap.items():
            # Check for "."
            ndot = fromopt.count(".")
            # Filter "."
            if ndot == 0:
                # Direct access simple option
                subval = self.get_option(fromopt, parents=parents)
            elif ndot == 1:
                # Process later (to set priority)
                continue
            else:
                # Not recursing on dicts of dicts or something
                raise ValueError("Cannot process option '%s'" % fromopt)
            # One more check for ``None``
            if subval is not None:
                # Don't override
                optval[subopt] = subval
        # Store cascading option parents to avoid multiple access
        kw_cascade = {}
        # Loop through cascade options
        for (fromopt, subopt) in kw_submap.items():
            # Check for "."
            ndot = fromopt.count(".")
            # Filter "."
            if ndot == 0:
                # Previously accessed
                continue
            if ndot == 1:
                # Split
                k0, k1 = fromopt.split(".")
                # Check for recursion
                if k0 in parents:
                    raise ValueError(
                        ("Detected recursion while accessing '%s' " % opt) +
                        ("with parents %s" % parents))
                # Get dictionary of options
                if k0 in kw_cascade:
                    # Previously calculated
                    subdict = kw_cascade[k0]
                else:
                    # Get parent options for first time
                    subdict = self.get_option(k0, parents=parents | {opt})
                    # Save them
                    kw_cascade[k0] = subdict
                # Check for ``None``
                if subdict is None:
                    continue
                # Check for a dictionary
                if not isinstance(subdict, dict):
                    raise TypeError(
                        ("Cannot access '%s' because " % fromopt) +
                        ("'%s' option is not a dict" % k0))
                # Get the value
                subval = subdict.get(k1)
            # One more check for ``None``
            if subval is not None:
                # Don't override
                optval.setdefault(subopt, subval)
        # Get aliases
        kw_alias =  cls._kw_subalias.get(opt)
        # Check for aliases
        if isinstance(kw_alias, dict):
            # Apply aliases
            optval = {
                kw_alias.get(k, k): v for (k, v) in optval.items()
            }
        # Check for valid default
        if isinstance(optdef, dict):
            # Apply defaults, but override with explicit values
            optval = dict(optdef, **optval)
        # Output
        return cls.denone(optval)
        
   # --- Predeclared Sections ---
    # Generic function
    def section_options(self, sec, mainopt=None):
        r"""Process options for a particular section

        :Call:
            >>> kw = opts.section_options(sec, mainopt=None)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *sec*: :class:`str`
                Name of options section
            *mainopt*: {``None``} | :class:`str`
                If specified, return value of this option instead of
                :class:`dict` of all options from section; this is most
                often used when *mainopt* itself has a :class:`dict`
                type where the other *sec* options get mapped into it
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options in *opts* relevant to *sec*
        :Versions:
            * 2020-01-16 ``@ddalle``: First version
        """
       # --- Prep ---
        # Class
        cls = self.__class__
        # Get list of options for section
        optlist = cls._optlists.get(sec, [])
        # Get defaults
        rc = cls._rc
       # --- Select Options ---
        # Create dictionary of all hits for this ection
        kw_sec = {}
        # Loop through option list
        for opt in optlist:
            # Get value for this option
            kw_sec[opt] = self.get_option(opt)
       # --- Defaults ---
        # Get defaults
        rc_sec = cls._rc_sections.get(sec, {})
        # Apply defaults
        for (k, v) in rc_sec.items():
            # Set default
            if kw_sec.get(k) is None:
                # use a shallow copy to avoid changing defaults
                kw_sec[k] = copy.copy(v)
       # --- Output ---
        # Remove ``None``
        kw = cls.denone(kw_sec)
        # Check for *mainopt*
        if mainopt is None:
            # No selection
            return kw
        else:
            # Return primary key only (default to ``{}``)
            return kw.get(mainopt, {})

   # --- Option Lists by Option ---
    # Get list of option that affect a section
    @classmethod
    def _get_optlist_section(cls, sec):
        r"""Get list of options that affect one section

        :Call:
            >>> optlist = cls._get_optlist_section(sec)
        :Inputs:
            *cls*: :class:`type`
                Options subclass of :class:`KwargHandler`
            *sec*: :class:`str`
                Name of option section
        :Outputs:
            *optlist*: :class:`set`\ [:class:`str`]
                List of options affection at least one option in *sec*
        :Versions:
            * 2020-01-23 ``@ddalle``: First version
        """
        # Get list
        opts = cls._optlists.get(sec, [])
        # Process list
        return cls._get_optlist_list(opts)
        
    # Get list of options that a list of options
    @classmethod
    def _get_optlist_list(cls, opts):
        r"""Get list of options that affect a list of options

        :Call:
            >>> optlist = cls._get_optlist_list(opts)
        :Inputs:
            *cls*: :class:`type`
                Options subclass of :class:`KwargHandler`
            *opts*: :class:`iterable`\ [:class:`str`]
                Name of option
            *parents*: ``None`` | :class:`set`\ [:class:`str`]
                List of parents, used to detect recursion
        :Outputs:
            *optlist*: :class:`set`\ [:class:`str`]
                List of options that can impact *opt*
        :Versions:
            * 2020-01-23 ``@ddalle``: First version
        """
        # Initialize list
        optlist = set()
        # Loop through options
        for opt in opts:
            # Combine affecting options
            optlist |= cls._get_optlist_option(opt)
        # Output
        return optlist
        
    # Get list of options that affect one option
    @classmethod
    def _get_optlist_option(cls, opt, parents=None):
        r"""Get list of options that affect one option

        :Call:
            >>> optlist = cls._get_optlist_option(opt, parents=None)
        :Inputs:
            *cls*: :class:`type`
                Options subclass of :class:`KwargHandler`
            *opt*: :class:`str`
                Name of option
            *parents*: ``None`` | :class:`set`\ [:class:`str`]
                List of parents, used to detect recursion
        :Outputs:
            *optlist*: :class:`set`\ [:class:`str`]
                List of options that can impact *opt*
        :Versions:
            * 2020-01-23 ``@ddalle``: First version
        """
        # Initialize list
        optlist = {opt}
        # Get dictionary of options to inherit from
        kw_submap = cls._kw_submap.get(opt)
        # If there's no map, this is a trivial operation
        return optlist
        # Default parent set
        if parents is None:
            # This option becomes the parent for later calls
            parents = set()
        # Loop through primary submap options
        for (fromopt, subopt) in kw_submap.items():
            # Check for "."
            ndot = fromopt.count(".")
            # Filter "."
            if ndot > 0:
                # Just get name of main option
                fromopt, _ = fromopt.split(".", 1)
            # Check for recursion
            if fromopt in parents:
                raise ValueError(
                    ("Detected recursion in submap of '%s' " % opt) +
                    ("with inheritance from '%s'" % fromopt))
            # Direct access simple option
            suboptlist = cls._get_optlist_option(fromopt, parents | {opt})
            # Combine lists
            optlist |= suboptlist
        # Output
        return optlist
  # >

  # ====================
  # Class Modification
  # ====================
  # <
   # --- Combine with Parent ---
    # Combine all primary attributes
    @classmethod
    def combine_optdefs(cls, parentcls=None, f=False):
        r"""Combine *_optlist* and all other :class:`KwargHandler` attrs

        :Call:
            >>> cls.combine_opts(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        # Call specific function for each attribute
        cls.combine_optlist(parentcls, f)
        cls.combine_optmap(parentcls, f)
        cls.combine_opttypes(parentcls, f)
        cls.combine_optdependencies(parentcls, f)
        cls.combine_optlists(parentcls, f)
        cls.combine_optvals(parentcls, f)
        cls.combine_tagmap(parentcls, f)
        cls.combine_kw_submap(parentcls, f)
        cls.combine_kw_subalias(parentcls, f)
        cls.combine_rst_types(parentcls, f)
        cls.combine_rst_descriptions(parentcls, f)
        cls.combine_rc(parentcls, f)
        cls.combine_rc_sections(parentcls, f)

    # Combine options list
    @classmethod
    def combine_optlist(cls, parentcls=None, f=False):
        r"""Combine *_optlist* from two classes

        :Call:
            >>> cls.combine_optlist(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optset("_optlist", parentcls, f)
        cls.combine_optset("_optlist_holdlast", parentcls, f)
        cls.combine_optset("_optlist_list", parentcls, f)
        cls.combine_optset("_optlist_none", parentcls, f)
        cls.combine_optset("_optlist_ring", parentcls, f)

    # Combine options alternate names
    @classmethod
    def combine_optmap(cls, parentcls=None, f=False):
        r"""Combine *_optmap* from two classes

        :Call:
            >>> cls.combine_optmap(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optdict("_optmap", parentcls, f)

    # Combine options allowed types
    @classmethod
    def combine_opttypes(cls, parentcls=None, f=False):
        r"""Combine *_opttypes* from two classes

        :Call:
            >>> cls.combine_opttypes(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optdict("_opttypes", parentcls, f)

    # Combine options required dependencies
    @classmethod
    def combine_optdependencies(cls, parentcls=None, f=False):
        r"""Combine *_optdependencies* from two classes

        :Call:
            >>> cls.combine_optdependencies(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optdict("_optdependencies", parentcls, f)

    # Combine options allowed values
    @classmethod
    def combine_optvals(cls, parentcls=None, f=False):
        r"""Combine *_optvals* from two classes

        :Call:
            >>> cls.combine_optvals(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optdict("_optvals", parentcls, f)
        cls.combine_optdict("_optvalmap", parentcls, f)
        cls.combine_optdict("_optval_converters", parentcls, f)

    # Combine option list subsections
    @classmethod
    def combine_optlists(cls, parentcls=None, f=False):
        r"""Combine *_optlists* from two classes

        Sections present in both *cls._optlists* and
        *parentcls._optlists* are merged.

        :Call:
            >>> cls.combine_optlists(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        cls.combine_optdict("_optlists", parentcls, f)

    # Combine "Tag" mappings
    @classmethod
    def combine_tagmap(cls, parentcls=None, f=False):
        r"""Combine *_tagmap* from two classes

        :Call:
            >>> cls.combine_tagmap(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-03-18 ``@ddalle``: First version
        """
        cls.combine_optdict("_tagmap", parentcls, f)

    # Combine option submap
    @classmethod
    def combine_kw_submap(cls, parentcls=None, f=False):
        r"""Combine *_kw_submap* from two classes

        Sections present in both *cls._kw_submap* and
        *parentcls._kw_submap* are merged.

        :Call:
            >>> cls.combine_kw_submap(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_kw_submap", parentcls, f)

    # Combine suboption aliases
    @classmethod
    def combine_kw_subalias(cls, parentcls=None, f=False):
        r"""Combine *_kw_subalias* from two classes

        Sections present in both *cls._kw_subalias* and
        *parentcls._kw_submap* are merged.

        :Call:
            >>> cls.combine_kw_subalias(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_kw_subalias", parentcls, f)

    # Combine documentation types
    @classmethod
    def combine_rst_types(cls, parentcls=None, f=False):
        r"""Combine *_rst_types* from two classes

        :Call:
            >>> cls.combine_rst_types(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_rst_types", parentcls, f)

    # Combine documentation descriptions
    @classmethod
    def combine_rst_descriptions(cls, parentcls=None, f=False):
        r"""Combine *_rst_descriptions* from two classes

        :Call:
            >>> cls.combine_rst_descriptions(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_rst_descriptions", parentcls, f)

    # Combine global defaults
    @classmethod
    def combine_rc(cls, parentcls=None, f=False):
        r"""Combine *_rc* from two classes

        Options present in both *cls._rc* and
        *parentcls._rc* are merged (recursively).

        :Call:
            >>> cls.combine_rc(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_rc", parentcls, f)

    # Combine global defaults
    @classmethod
    def combine_rc_sections(cls, parentcls=None, f=False):
        r"""Combine *_rc_sections* from two classes

        Sections present in both *cls._rc_sections* and
        *parentcls._rc_sections* are merged (recursively).

        :Call:
            >>> cls.combine_rc_sections(parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | {``False``}
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-12 ``@ddalle``: First version
        """
        cls.combine_optdict("_rc_sections", parentcls, f)
    

    # Combine set attributes
    @classmethod
    def combine_optset(cls, attr, parentcls=None, f=False):
        r"""Combine :class:`set` attribute from two classes

        :Call:
            >>> cls.combine_optset(cls, attr, parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *attr*: :class:`str`
                Name of attribute to combine
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | ``False``
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        # Get current value
        v1 = cls.__dict__.get(attr)
        # Exit if not specifically defined
        if v1 is None:
            if f:
                # Copy value from "dot"
                v1 = copy.copy(getattr(cls, attr))
                # Save a copy
                setattr(cls, attr, v1)
            else:
                return
        # Default parent
        if parentcls is None:
            # Use first basis class
            parentcls = cls.__bases__[0]
        # Get parent value
        v2 = getattr(parentcls, attr)
        # Combine sets
        _combine_set(v1, v2)

    # Combine dict attributes
    @classmethod
    def combine_optdict(cls, attr, parentcls=None, f=False):
        r"""Combine :class:`dict` attribute from two classes

        :Call:
            >>> cls.combine_optdict(cls, attr, parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *attr*: :class:`str`
                Name of attribute to combine
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | ``False``
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        # Get current value
        v1 = cls.__dict__.get(attr)
        # Exit if not specifically defined
        if v1 is None:
            if f:
                # Copy value from "dot"
                v1 = copy.copy(getattr(cls, attr))
                # Save a copy
                setattr(cls, attr, v1)
            else:
                return
        # Default parent
        if parentcls is None:
            # Use first basis class
            parentcls = cls.__bases__[0]
        # Get parent value
        v2 = getattr(parentcls, attr)
        # Combine values (*v1* takes priority)
        _combine_dict(v1, v2)

   # --- Update from Parent: No Recursion ---
    # Combine dict attributes
    @classmethod
    def update_optdict(cls, attr, parentcls=None, f=False):
        r"""Combine :class:`dict` attribute from two classes

        If *attr* is present in both classes, *cls.attr* overrides.

        :Call:
            >>> cls.setdefault_optdict(cls, attr, parentcls=None, f=False)
        :Inputs:
            *cls*: :class:`type`
                :class:`KwargHandler` class or subclass
            *attr*: :class:`str`
                Name of attribute to combine
            *parentcls*: {*cls.__bases__[0]*} | :class:`type`
                Second :class:`KwargHandler` class or subclass
            *f*: ``True`` | ``False``
                Option to combine even if no value in *cls.__dict__*
        :Versions:
            * 2020-02-11 ``@ddalle``: First version
        """
        # Get current value
        v1 = cls.__dict__.get(attr)
        # Exit if not specifically defined
        if (v1 is None) and (not f):
            return
        # Default parent
        if parentcls is None:
            # Use first basis class
            parentcls = cls.__bases__[0]
        # Get parent value
        v2 = getattr(parentcls, attr)
        # Combine values (*v1* takes priority)
        v3 = dict(v2, **v1)
        # Save it
        setattr(cls, attr, v3)

   # --- Class Attribute ---
    @classmethod
    def _getattr_class(cls, attr):
        r"""Get a class attribute with additional rules

        1. If *attr* is in *cls.__dict__*, return it
        2. Otherwise, set *cls.__dict__[attr]* equal to a copy of
           ``getattr(cls, attr)`` and return that value

        The purpose of this procedure is for subclasses of
        :class:`KwargHandler` to extend their own copies of special
        class attributes without starting over.

        :Call:
            >>> cls._getattr_class(attr)
        :Inputs:
            *cls*: :class:`type`
                Options class
            *opt*: :class:`str`
                Name of option to add to list
        :Versions:
            * 2020-01-16 ``@ddalle``: First version
        """
        # Check type
        if not isinstance(attr, str):
            raise TypeError(
                "Attribute name must be string (got %s)" % type(attr))
        # Check if present
        if attr in cls.__dict__:
            # Return it
            return cls.__dict__[attr]
        else:
            # Set it from parent (somewhere in MRO)
            val = getattr(cls, attr)
            # Copy it
            val = copy.deepcopy(val)
            # Save it
            setattr(cls, attr, val)
            # Output
            return val

   # --- Option Addition ---
    # Add an option to the list
    @classmethod
    def _add_option(cls, opt, **kw):
        r"""Add an option to a class's option list

        :Call:
            >>> cls._add_option(opt)
        :Inputs:
            *cls*: :class:`type`
                Options class
            *opt*: :class:`str`
                Name of option to add to list
        :Versions:
            * 2020-01-16 ``@ddalle``: first version
        """
        # Get _optlist attribute
        _optlist = cls._getattr_class("_optlist")
        # Add option to set of option names
        _optlist.add(opt)
        ## Process additional options
        #opttype = kw.get("opttype")
        #aliases = kw.get("aliases")
        #is_list = kw.get("islist")
        #is_ring = kw.get("isring")
        #is_holdlast = kw.get("isholdlast")
        #sections = kw.get("sections")
        #subalias = kw.get("subalias")
        ## Documentation
        #rst_type = kw.get("doctype")
        #rst_desc = kw.get('docdescription")

    # Declare an option's type
    @classmethod
    def _add_opttype(cls, opt, opttype, **kw):
        r"""Declare a particular option's required type

        Checks on *opttype* are not performed.  This method extends the
        parent class's version of *_opttypes* and adds an entry.  The
        *opttype* need not include :class:`NoneType`; keys with values
        of ``None`` are removed before type checks are performed.

        :Call:
            >>> cls._add_opttype(cls, opt, opttype)
        :Inputs:
            *cls*: :class:`type`
                Options class
            *opt*: :class:`str`
                Name of option to add to list
            *opttype*: :class:`type` | :class:`tuple`
                Types to check, second argument to :func:`isinstance`
        :Effects:
            *cls._opttypes*: :class:`dict`
                *opt* key set to *opttype*
        :Versions:
            * 2020-01-16 ``@ddalle``: first version
        """
        # Get _opttype attribute
        _opttypes = cls._getattr_class("_opttypes")
        # Add option to set of option names
        _opttypes[opt] = opttype

   # --- Docstring ---
    # Document a function
    @classmethod
    def _doc_keys(cls, func, sec, indent=12, fmt_key="keys", submap=True):
        r"""Document the keyword list for a function in a class

        :Call:
            >>> cls._doc_keys(func, sec, **kw)
            >>> cls._doc_keys(func, optlist, **kw)
        :Inputs:
            *cls*: :class:`type`
                Class whose *__dict__* has the function to document
            *func*: :class:`str`
                Name of function to document
            *sec*: :class:`str`
                Name of section to get keys from
            *optlist*: :class:`list`\ [:class:`str`]
                Explicit list of keys (like *cls._optlists[sec]*)
            *indent*: {``12``} | :class:`int` >= 0
                Indentation level for parameter name
            *fmt_key*: {``"keys"``} | :class:`str`
                Format key to replace in existing docstring; by default
                this replaces ``"%(keys)s"`` with the RST list
            *submap*: {``True``} | ``False``
                If ``True``, add keys from *cls._kw_submap*
        :Versions:
            * 2020-01-17 ``@ddalle``: First version
        """
        # Get the function
        fn = cls.__dict__[func]
        # Check type
        if not callable(fn):
            raise TypeError(
                "Attr '%s.%s' is not callable" % (cls.__name__, func))
        # Call parent function
        cls._doc_keys_fn(fn, sec, indent, fmt_key, submap)

    # Document a function (any)
    @classmethod
    def _doc_keys_fn(cls, fn, sec, indent=8, fmt_key="keys", submap=True):
        r"""Document the keyword list for a generic function

        :Call:
            >>> cls._doc_keys(fn, sec, **kw)
            >>> cls._doc_keys(fn, optlist, **kw)
        :Inputs:
            *cls*: :class:`type`
                Class with option lists to expand
            *fn*: *callable*
                Function to document
            *sec*: :class:`str`
                Name of section to get keys from
            *optlist*: :class:`list`\ [:class:`str`]
                Explicit list of keys (like *cls._optlists[sec]*)
            *indent*: {``8``} | :class:`int` >= 0
                Indentation level for parameter name
            *fmt_key*: {``"keys"``} | :class:`str`
                Format key to replace in existing docstring; by default
                this replaces ``"%(keys)s"`` with the RST list
            *submap*: {``True``} | ``False``
                If ``True``, add keys from *cls._kw_submap*
        :Versions:
            * 2020-01-17 ``@ddalle``: First version
        """
        # Check type
        if not callable(fn):
            raise TypeError("Function is not callable")
        # Apply text to the docstring
        fn.__doc__ = cls._doc_keys_doc(
            fn.__doc__, sec, indent, fmt_key, submap)

    # Document a string
    @classmethod
    def _doc_keys_doc(cls, doc, sec, indent=8, fmt_key="keys", submap=True):
        r"""Document the keyword list for a generic function

        :Call:
            >>> doc = cls._doc_keys(doc, sec, **kw)
            >>> doc = cls._doc_keys(doc, optlist, **kw)
        :Inputs:
            *cls*: :class:`type`
                Class with option lists to expand
            *doc*: :class:`str`
                Docstring to update
            *sec*: :class:`str`
                Name of section to get keys from
            *optlist*: :class:`list`\ [:class:`str`]
                Explicit list of keys (like *cls._optlists[sec]*)
            *indent*: {``8``} | :class:`int` >= 0
                Indentation level for parameter name
            *fmt_key*: {``"keys"``} | :class:`str`
                Format key to replace in existing docstring; by default
                this replaces ``"%(keys)s"`` with the RST list
            *submap*: {``True``} | ``False``
                If ``True``, add keys from *cls._kw_submap*
        :Outputs:
            *doc*: :class:`str`
                Updated docstring
        :Versions:
            * 2020-01-17 ``@ddalle``: First version
        """
        # Check *sec*
        if isinstance(sec, list):
            # Already a list
            optlist = sec
        else:
            # Get sublist
            optlist = cls._optlists[sec]
        # Check for submap
        if submap:
            # Also map options
            optmap = cls._optmap
            # Get the submap from the class
            kw_submap = cls._kw_submap
            # Loop through parameters
            for opt in list(optlist):
                # Get submap
                opt_submap = kw_submap.get(opt)
                # Check it
                if opt_submap is None:
                    continue
                # Otherwise loop through the submap (dict)
                for subopt in opt_submap:
                    # Check for "."
                    if "." in subopt:
                        # Just look up the parent option
                        subopt, _ = subopt.split(".", 1)
                    # Check if the option is present
                    if subopt not in optlist:
                        optlist.append(subopt)
        else:
            # No option name map
            optmap = {}
        # Create list
        rst_keys = rstutils.rst_param_list(
            optlist,
            cls._rst_types,
            cls._rst_descriptions,
            optmap,
            indent=indent,
            strict=False)
        # Formatting dict
        fmt = {fmt_key: rst_keys}
        # Apply text to the docstring
        for _ in range(10):
            try:
                # Attempt replacement
                doc = doc % fmt
                # If successful, done
                break
            except KeyError as e:
                # Name of missing key
                k = e.args[0]
                # Tell *fmt* to just leave it as is
                fmt[k] = "%%(%s)s" % k
        return doc
  # >


# Combine two mutable values
def _combine_val(v1, v2):
    # Check type
    if isinstance(v1, dict):
        _combine_dict(v1, v2)
    elif isinstance(v1, list):
        _combine_list(v1, v2)
    elif isinstance(v1, set):
        _combine_set(v1, v2)


# Combine two sets
def _combine_set(v1, v2):
    # Check type
    if not isinstance(v1, set):
        raise TypeError("Arg 1 must be 'set', got '%s'" % v1.__class__)
    # Update
    v1.update(v2)


# Combine two lists
def _combine_list(v1, v2):
    # Check type
    if not isinstance(v1, list):
        raise TypeError("Arg 1 must be 'list', got '%s'" % v1.__class__)
    # Loop through *v2*
    for v in v2:
        if v not in v1:
            v1.append(v2)


# Combine two dicts
def _combine_dict(v1, v2):
    # Check type
    if not isinstance(v1, dict):
        raise TypeError("Arg 1 must be 'dict', got '%s'" % v1.__class__)
    if not isinstance(v2, dict):
        raise TypeError("Arg 2 must be 'dict', got '%s'" % v1.__class__)
    # Loop through second dict
    for (k, v) in v2.items():
        # Check presence in first dict
        if k in v1:
            # Recurse
            _combine_val(v1[k], v)
        else:
            # New value
            v1[k] = v
