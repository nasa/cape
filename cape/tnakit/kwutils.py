#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.tnakit.kwutils`: Tools for Processing Keyword Arguments
===================================================================

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


# Map keywords
def map_kw(self, kwmap, **kw):
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
def check_kw(self, kwlist, kwmap, kwdep, mode, **kw):
    r"""Check and map valid keyword names

    :Call:
        >>> kwo = _check_kw(kwlist, kwmap, kwdep, mode, **kw)
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
            self[k] = v
  # >

  # ==================
  # Change Settings
  # ==================
  # <
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
        # Remove anything that's ``None``
        opts = cls.denone(kw)
        # Check validity, apply maps
        opts = check_kw_eltypes(
            self._optlist_check,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies,
            self._warnmode, **opts)
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
                self[k] = v
  # >

  # ============
  # Utilities
  # ============
  # <
    # Remove ``None`` keys
    @staticmethod
    def denone(opts):
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

  # ================
  # Item Selection
  # ================
  # <
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
  # >

  # ===============
  # Subsections
  # ===============
  # <
   # --- Individual Option ---
    # Get individual option
    def get_option(self, opt, parents=None):
        r"""Get value of a specific option, ignoring section

        :Call:
            >>> optval = opts.get_option(opt, parents=None)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *opt*: :class:`str`
                Name of option
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
        optdef = cls._rc.get(opt)
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
        # Check for valid default
        if isinstance(optdef, dict):
            # Apply defaults, but override with explicit values
            optval = dict(optdef, **optval)
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
                subval = self.get_option(fromopt, parents)
            elif ndot == 1:
                # Process later (to set priority)
                continue
            else:
                # Not recursing on dicts of dicts or something
                raise ValueError("Cannot process option '%s'" % fromopt)
            # One more check for ``None``
            if subval is not None:
                # Don't override
                optval.setdefault(subopt, subval)
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
                    subdict = self.get_option(k0, parents | {opt})
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
   # --- Class Attribute ---
    @classmethod
    def _getattr_class(cls, attr):
        r"""Get a class attribute with additional rules

        1. If *attr* is in *cls.__dict__*, return it
        2. Otherwise, set *cls.__dict__[attr]* equal to
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
            # Save it
            cls.__dict__[attr] = val
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
    def _doc_keys(cls, func, sec, fmt_key="keys", submap=True):
        r"""Document the keyword list for a function

        :Call:
            >>> cls._doc_keys(func, sec, fmt_key="keys", submap=True)
            >>> cls._doc_keys(func, optlist, fmt_key="keys")
        :Inputs:
            *cls*: :class:`type`
                Class whose *__dict__* has the function to document
            *func*: :class:`str`
                Name of function to document
            *sec*: :class:`str`
                Name of section to get keys from
            *optlist*: :class:`list`\ [:class:`str`]
                Explicit list of keys (like *cls._optlists[sec]*)
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
            raise TypeError("Attribute '%s' is not callable" % funcname)
        # Check *sec*
        if isinstance(sec, list):
            # Already a list
            optlist = sec
        else:
            # Get sublist
            optlist = cls._optlists[sec]
        # Check for submap
        if submap:
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
        # Create list
        rst_keys = rstutils.rst_param_list(
            optlist,
            cls._rst_types,
            cls._rst_descriptions,
            cls._optmap,
            indent=12,
            strict=False)
        # Apply text to the docstring
        fn.__doc__ = fn.__doc__ % {fmt_key: rst_keys}
  # >
