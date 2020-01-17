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
import difflib

# Local modules
from . import optitem


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
def KwargHandler(dict):
    r"""Class to handle kwargs against preset key lists and types

    :Call:
        >>> opts = KwargHandler(optsdict=None, warnmode=1, **kw)
    :Inputs:
        *opts*: :class:`MPLOpts`
            Options interface
        *optsdict*: {``None``} | :class:`dict`
            Dictionary of previous options (overwritten by *kw*)
        *warnmode*: ``0`` | {``1``} | ``2``
            Warning mode from :mod:`kwutils`
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

    # Options to inherit from elsewhere
    _kw_cascade = {}

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
    _rc_sub = {}
  # >
  
  # ============
  # Config
  # ============
  # <
    # Initialization method
    def __init__(self, optsdict=None, warnmode=1, **kw):
        r"""Initialization method

        :Call:
            >>> opts.__init__(optsdict=None, warnmode=1, **kw)
        :Inputs:
            *opts*: :class:`KwargHandler`
                Options interface
            *optsdict*: {``None``} | :class:`dict`
                Dictionary of previous options (overwritten by *kw*)
            *warnmode*: ``0`` | {``1``} | ``2``
                Warning mode from :mod:`kwutils`
        :Versions:
            * 2019-12-19 ``@ddalle``: First version (plot_mpl.MPLOpts)
        """
        # Get class
        cls = self.__class__
        # Initialize an unfiltered dict
        if isinstance(optsdict, dict):
            # Initialize with dictionary
            optsdict = dict(optsdict, **kw)
        else:
            # Initialize from just keywords
            optsdict = kw
        # Remove anything that's ``None``
        opts = cls.denone(optsdict)

        # Check keywords
        opts = check_kw_eltypes(
            cls._optlist,
            cls._optmap,
            cls._opttypes,
            cls._optdependencies, warnmode, **opts)

        # Copy entries
        for (k, v) in opts.items():
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
                if listtype == 0
                    # Repeat entire list
                    v = optitem.getringel_list(V, i)
                else:
                    # Repeat last value
                    v = optitem.getel_list(v, i)
            else:
                # Get value as a scalar
                if listtype == 0
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
       # --- Select Options ---
        # Create dictionary of all hits for this ection
        kw_sec = {}
        # Loop through option list
        for opt in optlist:
            # Check if present
            if opt not in self:
                continue
            # Get a reference
            optval = self[opt]
            # Check for aliases
            if not isinstance(v, dict):
                # Not alias-able
                kw_sec[opt] = optval
                continue
            # Get aliases
            kw_alias =  cls._kw_subalias.get(opt)
            # Check for aliases
            if kw_alias is None:
                # No aliases for this option
                kw_sec[opt] = optval
                continue
            # Apply aliases
            kw_sec[opt] = {
                kw_alias.get(k, k): v for (k, v) in optval.items()
            }
       # --- Defaults ---
        # Get defaults
        rc = cls._rc.get(sec, {})
        # Apply defaults
        kw_sec = dict(rc, **kw_sec)
       # --- Submaps ---
        # Process any submaps
        # For example "Label" -> "PlotOptions.label"
        # Initialize list of keys to remove
        k_del = set()
        # Loop through all current keys
        for opt, optval in kw_sec.items():
            # Check if a mappable
            if not isinstance(optval, dict):
                continue
            # Get map
            submap = cls._kw_subalias.get(opt)
            # Check for valid map
            if submap is None:
                continue
            # Loop through submap keys
            for (k, kp) in submap.items():
                # Check for mapped option ("Label" in example above)
                if k in self:
                    # Send it to parent key with new name
                    kw_sec[kp] = self[k]
                # Check if in the main list
                if k in kw_sec:
                    # Remove it and send it to parent key (w/ new name)
                    k_del.add(k)
        # Remove any options mapped elsewhere
        for k in k_del:
            kw_sec.pop(k)
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
  # >