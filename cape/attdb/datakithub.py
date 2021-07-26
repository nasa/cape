#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.datakithub`: Hub for importing DataKits by name
=================================================================

This module provides the class :class:`DataKitHub` that provides a tool
to simplify the importing of named "datakits" (from
:mod:`cape.attdb.rdb.DataKit`).  More specifically, it allows users to
create one or more naming conventions for databases/datakits and to
read that data with minimal low-level Python programming.

An instance of the :class:`DataKitHub` class is created by reading a
JSON file that contains the naming conventions, such as the following:

    .. code-block:: python

        from cape.attdb.datakithub import DataKitHub

        # Create an instance
        hub = DataKitHub("datakithub.json")

A simple ``datakithub.json`` file might contain the following:

    .. code-block:: javascript

        {
            "DB-ATT": {
                "repo": "/home/user/datakit/db",
                "type": "module",
                "module_attribute": "db",
                "module_regex": {
                    "DB-ATT-([0-9]+)": "dbatt.db%s",
                },
            }
        }

It will make more sense to explain this content after seeing an example.
Now we can use the :class:`DataKitHub` instance to read databases by
their title, such as ``"DB-ATT-1"`` or ``"DB-ATT-002"``, as long as they
start with ``"DB-ATT"`` or some other string defined in the JSON file.

    .. code-block:: python

        from cape.attdb.datakithub import DataKitHub

        # Create an instance
        hub = DataKitHub("/home/user/datakit/datakithub.json")

        # Read the database "DB-ATT-1"
        db1 = hub.read_db("DB-ATT-1")

        # Read the database "DB-ATT-002"
        db2 = hub.read_db("DB-ATT-002")

This is roughly the same as

    .. code-block:: python

        # Read the database "DB-ATT-1"
        import dbatt.db1
        db1 = dbatt.db1.db

        # Read the database "DB-ATT-002"
        import dbatt.db002
        db2 = dbatt.db002.db

but without having to deal with either ``sys.path`` or the *PYTHONPATH*
environment variable, which can be both tedious and difficult to make
work for multiple users on different types of computers.

Here is a description of the JSON parameters

    *repo*: :class:`str`
        Name of the folder containing the data or modules
    *type*: ``"module"``
        Type of datakit to read
    *module_attribute*: :class:`str` | ``None``
        Name of variable in imported module to use as datakit
    *module_regex*: :class:`dict`\ [:class:`str`]
        Rules for converting a regular expression to module names

"""

# Standard library modules
import importlib
import os
import re
import sys

# CAPE modules
from cape.cfdx.options.util import loadJSONFile

# Version-dependent standard library
if sys.version_info.major > 2:
    # Import the reload() function
    from importlib import reload


# Defaults
DEFAULT_TYPE = "module"


# JSON files
_DEFAULT_JSON = "data/datakithub/datakithub.json"


# DataKit hub class
class DataKitHub(dict):
    r"""Load datakits using only the database name

    :Call:
        >>> hub = DataKitHub(fjson)
    :Inputs:
        *fjson*: {``None``} | :class:`str`
            Path to JSON file with import rules for one or more db names
        *cwd*: {``None``} | :class:`str`
            Path from which to begin search
    :Outputs:
        *hub*: :class:`DataKitHub`
            Instance that implements import rules by name
    :Versions:
        * 2019-02-17 ``@ddalle``: Version 1.0
    """
   # --- DUNDER ---
    # Initialization method
    def __init__(self, fjson=None, cwd=None):
        r"""Initialization method

        :Versions:
            * 2021-02-17 ``@ddalle``: Version 1.0
        """
        # Find best JSON file
        fabs = self._find_dkhubjson(fjson=fjson, cwd=cwd)
        # Initialize fixed attributes
        self.datakit_modules = {}
        self.datakit_groupnames = {}
        # Save folder containing *fjson*
        self.file_json = os.path.basename(fabs)
        self.dir_json = os.path.dirname(fabs)
        self.dir_root = os.path.dirname(self.dir_json)
        # Read the JSON file
        opts = loadJSONFile(fabs)
        # Save it...
        self.update(opts)

    # Locate the best JSON file
    def _find_dkhubjson(self, fjson=None, cwd=None):
        r"""Find the best ``datakithub.json`` file automatically

        :Call:
            >>> fabs = hub._find_dkhubjson(fjson=None, cwd=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance that implements import rules by name
            *fjson*: {``None``} | :class:`str`
                Path to JSON file with import rules
            *cwd*: {``None``} | :class:`str`
                Path from which to begin search
        :Outputs:
            *fabs*: :class:`str`
                Absolute path to such a 
        :Versions:
            * 2021-07-21 ``@ddalle``: Version 1.0
        """
        # Default JSON file pattern
        if fjson is None:
            fjson = _DEFAULT_JSON
        # Use backslashes if necessary
        if os.name == "nt":
            fname = fjson.replace("/", os.sep)
        else:
            # Don't modify on POSIX systems
            fname = fjson
        # Check for absolute path
        if os.path.isabs(fname):
            # Check if it exists
            if os.path.isfile(fname):
                # Found the file!
                return fname
            # Could not find file
            raise SystemError("Could not find file '%s'" % fjson)
        # Absolutize *cwd*
        if cwd is None:
            # Use current path
            path = os.getcwd()
        else:
            # Absolutize *cwd*
            path = os.path.abspath(cwd)
        # Search *path* (and all parents) for *fname*
        while os.path.basename(path):
            # Combine *path* and *fname*
            fabs = os.path.join(path, fname)
            # Check if file exists
            if os.path.isfile(fabs):
                # Found a JSON file!
                return fabs
            elif os.path.isdir(fabs):
                # Found a folder instead ... add "datakithub.json"
                fabs = os.path.join(fabs, "datakithub.json")
                # Check again
                if os.path.isfile(fabs):
                    return fabs
            # Otherwise, move up one level
            path = os.path.dirname(path)
        # If none found, raise an exception
        raise SystemError(
            "File '%s' not found in working directory or any parent" % fname)

   # --- Options ---
    # Find best matching category
    def get_group(self, name):
        r"""Find the first group that matches a datakit *name*

        :Call:
            >>> grp, grpopts = hub.get_group(name)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
        :Outputs:
            *grp*: :class:`str`
                Title of datakit reading group
            *grpopts*: :class:`dict`
                Options for that group
        :Versions:
            * 2021-02-17 ``@ddalle``: Version 1.0
        """
        # Loop through sections
        for grp, grpopts in self.items():
            # Check if *name* matches
            if re.match(grpe, name):
                # Found a match!
                return grp, grpopts

    # Find best matching category
    def get_group_match(self, name, grp=None):
        r"""Find the first group that matches a datakit *name*

        :Call:
            >>> grp = hub.get_group_match(name, grp=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
            *grp*: {``None``} | :class:`str`
                Optional manual override
        :Outputs:
            *grp*: :class:`str`
                Title of datakit reading group
            *grpopts*: :class:`dict`
                Options for that group
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Loop through sections
        for grp, grpopts in self.items():
            # Check if *name* matches
            if re.match(grp, name):
                # Found a match!
                return grp
        # Error if no match
        raise ValueError("No group options for datakit '%s'" % name)

    # Find best matching group (if necessary)
    def _get_group_match(self, name):
        r"""Find the first group that matches a datakit *name*

        :Call:
            >>> grp = hub._get_group_match(name)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
        :Outputs:
            *grp*: :class:`str`
                Title of datakit reading group
            *grpopts*: :class:`dict`
                Options for that group
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Loop through sections
        for grp, grpopts in self.items():
            # Check if *name* matches
            if re.match(grp, name):
                # Found a match!
                return grp

    # Get option from a specific group
    def get_group_opt(self, grp, opt, vdef=None):
        r"""Get the *type* of a given datakit group

        :Call:
            >>> v = hub.get_group_type(grp, opt, vdef=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *grp*: :class:`str`
                Title of datakit reading group
            *opt*: :class:`str`
                Name of option to access
            *vdef*: {``None``} | **any**
                Default value for *opt*
        :Outputs:
            *v*: {*vdef*} | 
                Value of *hub[grp][opt]* or *vdef*
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Get group options
        grpopts = self.get(grp)
        # Check type
        if not isinstance(grpopts, dict):
            return vdef
        # Get type
        return grpopts.get(opt, vdef)

    # Get *type*
    def get_group_type(self, grp):
        r"""Get the *type* of a given datakit group

        :Call:
            >>> typ = hub.get_group_type(grp)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *grp*: :class:`str`
                Title of datakit reading group
        :Outputs:
            *typ*: {``"module"``} | :class:`str`
                DataKit type
        :Versions:
            * 2021-02-17 ``@ddalle``: Version 1.0
            * 2021-02-18 ``@ddalle``: Version 1.1; use ``get_group_opt``
        """
        # Use generic function
        return self.get_group_opt(grp, "type", DEFAULT_TYPE)

   # --- Read DB ---
    # Get datakit by name
    def read_db(self, name, grp=None):
        r"""Read a datakit from its name

        :Call:
            >>> db = hub.load_module(name, grp=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
            *grp*: {``None``} | :class:`str`
                Title of datakit reading group
        :Outputs:
            *db*: :class:`DataKit` | **any**
                Database handle (hopefully a datakit) pointed to by
                options for *grp*
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Get group name if necessary
        grp = self.get_group_match(name, grp=grp)
        # Get type
        typ = self.get_group_type(grp)
        # Filter type
        if typ == "module":
            # Load the module
            mod = self.load_module(name, grp=grp)
            # Exit
            if mod is None:
                return
            # Get attribute option
            attrs = self.get_group_opt(grp, "module_attribute")
            # Convert to list
            if attrs is None:
                # Replace with empty list
                attrs = []
            elif not isinstance(attrs, (list, tuple)):
                # Replace string with singleton list
                attrs = [attrs]
            # Loop through alternates (may be 0 of them)
            for attr in attrs:
                # Try to get attribute
                if attr:
                    # Get "mod.(attr)"
                    db = getattr(mod, attr, None)
                    # Check if we're done
                    if db is not None:
                        # Exit
                        return db
            # Try alternate function(s)
            funcs = self.get_group_opt(grp, "module_function")
            # Convert to list
            if funcs is None:
                # Replace with empty list
                funcs = []
            elif not isinstance(funcs, (list, tuple)):
                # Replace string with singleton list
                funcs = [funcs]
            # Loop through alternates (may be 0 of them)
            for func in funcs:
                # Get "mod.(func)"
                fn = getattr(mod, func, None)
                # Check if callable
                if callable(fn):
                    # Execute the function
                    db = fn()
                    # Use it if not ``None``
                    if db is not None:
                        return db
        else:
            # Unrecognized
            raise ValueError(
                "Could not read database type '%s' for '%s'" % (typ, name))

   # --- Module ---
    # Load module
    def load_module(self, name, grp=None):
        r"""Import a ``"module"`` datakit

        :Call:
            >>> mod = hub.load_module(name, grp=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
            *grp*: {``None``} | :class:`str`
                Title of datakit reading group
        :Outputs:
            *mod*: :class:`module`
                Python module for *name* (if *grp* has valid *type*)
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Check for a previous load
        mod = self.datakit_modules.get(name)
        # Check if it's a module
        if mod is not None:
            # Just return it
            return mod
        # Get group name if necessary
        grp = self.get_group_match(name, grp=grp)
        # Get type
        typ = self.get_group_type(grp)
        # Filter type
        if typ == "module":
            # Get *repo* for system path
            fdir = self._get_module_dir(grp)
        else:
            # Unrecognized
            raise ValueError("Cannot import module for group type '%s'" % typ)
        # Append path if needed
        if fdir and (fdir in sys.path):
            # Note that no path mod needed
            qpath = False
        else:
            # Note it
            qpath = True
            # Add to path
            sys.path.insert(0, fdir)
        # Get the module options
        regex_dict = self.get_group_opt(grp, "module_regex", {})
        # Get a name
        modname = self._get_module_name(name, regex_dict)
        # Check for a match
        if modname is None:
            raise ValueError(
                "No module name found for datakit '%s' from group '%s'"
                % (name, grp))
        # Load the module
        try:
            # Use :mod:`importlib` to load module by string
            mod = importlib.import_module(modname)
            # Save module
            self.datakit_modules[name] = mod
        except Exception:
            # Note failure
            mod = None
            # Status update
            print(
                "Failed to import module '%s' for datakit '%s'"
                % (modname, name))
        # Unload path if needed
        if qpath:
            sys.path.remove(fdir)
        # Output
        return mod

    # Get group path
    def _get_module_dir(self, grp):
        r"""Get absolute path to *repo* for a group

        :Call:
            >>> fabs = hub._get_module_dir(grp)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *grp*: :class:`str`
                Title of datakit reading group
        :Outputs:
            *fabs*: :class:`str` | ``None``
                Absolute path to *repo* option from *grp*
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Get *repo* for system path
        fdir = self.get_group_opt(grp, "repo")
        # Check if it's absolute
        if os.path.isabs(fdir):
            # Already good
            return fdir
        else:
            # Prepend with *dir_root*
            return os.path.join(self.dir_root, fdir)

    # Get module name
    def _get_module_name(self, name, regex_dict):
        r"""Get module name from datakit and regular expression dict

        :Call:
            >>> modname = hub._get_module_name(name, regex_dict)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
            *regex_dict*: :class:`dict`\ [:class:`str`]
                Rules to replace
        :Outputs:
            *modname*: :class:`str` | ``None``
                Full name of module to import (if *name* matches a
                regular expression from the keys of *regex_dict*)
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
        """
        # Check types
        if not isinstance(regex_dict, dict):
            raise TypeError(
                "Regular expressions must be a 'dict' (got '%s')"
                % type(regex_dict).__name__)
        # Loop through regular expressions
        for regex, pattern in regex_dict.items():
            # Check for a match
            if re.match(regex, name):
                # Match found; use a substitution
                return re.sub(regex, pattern, name)

                
