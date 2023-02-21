#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.datakitloader`: Tools for reading DataKits
=============================================================

This class provides the :class:`DataKitLoader`, which takes as input the
module *__name__* and *__file__* to automatically determine a variety of
DataKit parameters.

"""

# Standard library
import fnmatch
import importlib
import json
import os
import re
import sys

# Local modules
from .rdb import DataKit
from ..tnakit import gitutils
from ..tnakit import kwutils
from ..tnakit import shellutils
from ..tnakit import typeutils


# Utility regular expressions
REGEX_INT = re.compile("[0-9]+$")
REGEX_HOST = re.compile("((?P<host>[A-z][A-z0-9.]+):)?(?P<path>[\w./-]+)$")
REGEX_REMOTE = re.compile("((?P<host>[A-z][A-z0-9.]+):)(?P<path>[\w./-]+)$")

# Create types for "strings" based on Python version
if sys.version_info.major == 2:
    # Allow unicode
    STR_TYPE = (str, unicode)
    # Module not found doesn't exist
    IMPORT_ERROR = ImportError
    # Error for no file found
    NOFILE_ERROR = SystemError
else:
    # Just string (which are unicode in Python 3.0+)
    STR_TYPE = str
    # Newer class for import errors
    IMPORT_ERROR = (ModuleNotFoundError, ImportError)
    # Error for no file
    NOFILE_ERROR = FileNotFoundError


# Create class
class DataKitLoader(kwutils.KwargHandler):
    r"""Tool for reading datakits based on module name and file

    :Call:
        >>> dkl = DataKitLoader(name, fname, **kw)
    :Inputs:
        *name*: :class:`str`
            Module name, from *__name__*
        *fname*: :class:`str`
            Absolute path to module file name, from *__file__*
    :Outputs:
        *dkl*: :class:`DataKitLoader`
            Tool for reading datakits for a specific module
    :Versions:
        * 2021-06-25 ``@ddalle``: Version 0.1; Started
    """
  # ==================
  # CLASS ATTRIBUTES
  # ==================
  # <
    # List of options
    _optlist = {
        "DATAKIT_CLS",
        "DB_DIR",
        "DB_DIRS_BY_TYPE",
        "DB_NAME",
        "DB_NAME_REGEX_LIST",
        "DB_NAME_REGEX_GROUPS",
        "DB_NAME_REGEX_INT_GROUPS",
        "DB_NAME_REGEX_STR_GROUPS",
        "DB_NAME_TEMPLATE_LIST",
        "DB_SUFFIXES_BY_TYPE",
        "MODULE_DIR",
        "MODULE_FILE",
        "MODULE_NAME",
        "MODULE_NAME_REGEX_LIST",
        "MODULE_NAME_REGEX_GROUPS",
        "MODULE_NAME_REGEX_INT_GROUPS",
        "MODULE_NAME_REGEX_STR_GROUPS",
        "MODULE_NAME_TEMPLATE_LIST",
    }

    # Types
    _opttypes = {
        "DATAKIT_CLS": type,
        "DB_DIR": str,
        "DB_DIRS_BY_TYPE": (list, tuple),
        "DB_NAME": str,
        "DB_NAME_REGEX_LIST": (list, tuple),
        "DB_NAME_REGEX_GROUPS": dict,
        "DB_NAME_REGEX_INT_GROUPS": (list, tuple, set),
        "DB_NAME_REGEX_STR_GROUPS": (list, tuple, set),
        "DB_NAME_TEMPLATE_LIST": (list, tuple),
        "DB_SUFFIXES_BY_TYPE": dict,
        "MODULE_DIR": str,
        "MODULE_FILE": str,
        "MODULE_NAME": str,
        "MODULE_NAME_REGEX_LIST": (list, tuple),
        "MODULE_NAME_REGEX_GROUPS": dict,
        "MODULE_NAME_REGEX_INT_GROUPS": (list, tuple, set),
        "MODULE_NAME_REGEX_STR_GROUPS": (list, tuple, set),
        "MODULE_NAME_TEMPLATE_LIST": (list, tuple),
    }

    # Default values
    _rc = {
        "DATAKIT_CLS": DataKit,
        "DB_DIR": "db",
        "DB_DIRS_BY_TYPE": {},
        "DB_NAME_REGEX": ".+",
        "DB_NAME_REGEX_GROUPS": {},
        "DB_NAME_REGEX_INT_GROUPS": set(),
        "DB_NAME_REGEX_STR_GROUPS": set(),
        "DB_NAME_TEMPLATE_LIST": ["datakit"],
        "DB_NAME": None,
        "MODULE_NAME_REGEX_LIST": [".+"],
        "MODULE_NAME_REGEX_GROUPS": {},
        "MODULE_NAME_REGEX_INT_GROUPS": set(),
        "MODULE_NAME_REGEX_STR_GROUPS": set(),
        "RAWDATA_DIR": "rawdata",
    }
  # >

  # ===============
  # DUNDER METHODS
  # ===============
  # <
    # Initialization method
    def __init__(self, name=None, fname=None, **kw):
        r"""Initialization method

        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
        """
        # Process keyword options
        kwutils.KwargHandler.__init__(self, **kw)
        # Initialize attributes
        self.rawdata_sources = {}
        self.rawdata_remotes = {}
        self.rawdata_commits = {}
        self.rawdata_sources_commit = {}
        # Default file name
        if fname is None and name is not None:
            # Transform mod1.mod2 -> mod1/mod2
            fname = name.replace(".", os.sep)
            # Append "__init__.py"
            fname = os.path.join(fname, "__init__.py")
        # Absolute fname
        if fname:
            # Absolutize
            fname = os.path.abspath(fname)
            fdir = os.path.dirname(fname)
        else:
            # Null
            fname = None
            fdir = None
        # Use required inputs
        self.setdefault_option("MODULE_NAME", name)
        self.setdefault_option("MODULE_FILE", fname)
        # Set name of folder containing data
        self.set_option("MODULE_DIR", fdir)
        # Get database (datakit) NAME
        self.create_db_name()
  # >

  # ==========================
  # PACKAGE IMPORT/READ
  # ==========================
  # <
   # --- Read datakit from a module/package ---
    def read_db_name(self, dbname=None):
        r"""Read datakit from first available module based on a DB name

        This utilizes the following parameters:

        * *DB_NAME*
        * *DB_NAME_REGEX_LIST*
        * *DB_NAME_REGEX_GROUPS*
        * *MODULE_NAME_TEMPLATE_LIST*

        :Call:
            >>> db = dkl.read_db_name(dbname=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *dbame*: {``None``} | :class:`str`
                Database name parse (default: *DB_NAME*)
        :Outputs:
            *db*: :class:`DataKit`
                Output of :func:`read_db` from module with *DB_NAME*
                equal to *dbname*
        :Versions:
            * 2021-09-10 ``@ddalle``: v1.0
        """
        # Import module
        mod = self.import_db_name(dbname)
        # Check for success
        if mod is None:
            return
        # Otherwise read and return
        return mod.read_db()

   # --- Import a module/package ---
    def import_db_name(self, dbname=None):
        r"""Import first available module based on a DB name

        This utilizes the following parameters:

        * *DB_NAME*
        * *DB_NAME_REGEX_LIST*
        * *DB_NAME_REGEX_GROUPS*
        * *MODULE_NAME_TEMPLATE_LIST*

        :Call:
            >>> mod = dkl.import_db_name(dbname=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *dbame*: {``None``} | :class:`str`
                Database name parse (default: *DB_NAME*)
        :Outputs:
            *mod*: :class:`module`
                Module with *DB_NAME* equal to *dbname*
        :Versions:
            * 2021-07-15 ``@ddalle``: v1.0
        """
        # Get list of candidate module names
        modname_list = self.genr8_modnames(dbname)
        # Loop through candidates
        for modname in modname_list:
            # Attempt to import that module
            try:
                # Import module by string
                mod = importlib.import_module(modname)
            except IMPORT_ERROR:
                # Module doesn't exist; try the next one
                continue
            # Give the user the module
            return mod
        # Otherwise, note that no modules were read
        print("Failed to import module for '%s'" % dbname)
        # Check if any matches were found
        if modname_list:
            print("despite one or more pattern matches:")
            for modname in modname_list:
                print("    %s" % modname)
        else:
            print("No *modname* patterns matched *dbname*")
        # Raise an exception
        raise ImportError("No module found for db '%s'" % dbname)
  # >

  # ==========================
  # MODULE_NAME --> DB_NAME
  # ==========================
  # <
   # --- Create DB names ---
    def make_db_name(self):
        r"""Retrieve or create database name from module name

        This utilizes the following parameters:

        * *MODULE_NAME_REGEX_LIST*
        * *MODULE_NAME_REGEX_GROUPS*
        * *DB_NAME_TEMPLATE_LIST*

        :Call:
            >>> dbname = dkl.make_db_name()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: v1.0
        """
        # Check if present
        if "DB_NAME" in self:
            # Get it
            return self["DB_NAME"]
        # Generate the database name
        dbname = self.genr8_db_name()
        # Save it
        self.set_option("DB_NAME", dbname)

    def create_db_name(self):
        r"""Create and save database name from module name

        This utilizes the following parameters:

        * *MODULE_NAME_REGEX_LIST*
        * *MODULE_NAME_REGEX_GROUPS*
        * *DB_NAME_TEMPLATE_LIST*

        :Call:
            >>> dbname = dkl.create_db_name()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: v1.0
        """
        # Generate the database name
        dbname = self.genr8_db_name()
        # Save it
        self.set_option("DB_NAME", dbname)

    def genr8_db_name(self, modname=None):
        r"""Get database name based on first matching regular expression

        This utilizes the following parameters:

        * *MODULE_NAME*
        * *MODULE_NAME_REGEX_LIST*
        * *MODULE_NAME_REGEX_GROUPS*
        * *DB_NAME_TEMPLATE_LIST*

        :Call:
            >>> dbname = dkl.genr8_db_name(modname=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *modname*: {``None``} | :class:`str`
                Name of module to parse (default: *MODULE_NAME*)
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: v1.0
            * 2021-07-15 ``@ddalle``: Version 1.1; add *modname* arg
        """
        # Get list of regexes
        modname_regexes = self._genr8_modname_regexes()
        # Get format lists
        dbname_templates = self.get_option("DB_NAME_TEMPLATE_LIST")
        # Module name
        if modname is None:
            # Use default; this module
            modname = self.get_option("MODULE_NAME")
        # Check for null module
        if modname is None:
            # Global default
            return "datakit"
        # Loop through regular expressions
        for i, regex in enumerate(modname_regexes):
            # Check for match
            grps = self._genr8_modname_match_groups(regex, modname)
            # Check for match
            if grps is None:
                continue
            # Get the template
            if i >= len(dbname_templates):
                # Not enough templates!
                raise IndexError(
                    ("Module matched regex %i, but only " % (i+1)) + 
                    ("%i templates specified" % len(dbname_templates)))
            # Get template
            dbname_template = dbname_templates[i]
            # Expand all groups
            try:
                # Apply formatting substitutions
                dbname = dbname_template % grps
            except Exception:
                # Missing group or something
                print("Failed to expand DB_NAME_TEMPLATE_LIST %i:" % (i+1))
                print("  template: %s" % dbname_template)
                print("  groups:")
                # Print all groups
                for k, v in grps.items():
                    print("%12s: %s [%s]" % (k, v, type(v).__name__))
                # Raise an exception
                raise KeyError(
                    "Failed to expand DB_NAME_TEMPLATE_LIST %i" % (i+1))
            # Exit loop
            break
        else:
            # No match found; use global default
            dbname = "datakit"
        # Also output it
        return dbname

   # --- Generate module names ---
    def genr8_modnames(self, dbname=None):
        r"""Import first available module based on a DB name

        This utilizes the following parameters:

        * *DB_NAME*
        * *DB_NAME_REGEX_LIST*
        * *DB_NAME_REGEX_GROUPS*
        * *MODULE_NAME_TEMPLATE_LIST*

        :Call:
            >>> modnames = dkl.genr8_modnames(dbname=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *dbame*: {``None``} | :class:`str`
                Database name parse (default: *DB_NAME*)
        :Outputs:
            *modnames*: :class:`list`\ [:class:`str`]
                Candidate module names
        :Versions:
            * 2021-10-22 ``@ddalle``: v1.0
        """
        # Get list of regexes
        dbname_regexes = self._genr8_dbname_regexes()
        # Get format lists (list[list[str]])
        modname_templates = self.get_option("MODULE_NAME_TEMPLATE_LIST")
        # Make list of module names suggested by regexes and *dbname*
        modname_list = []
        # Module name
        if dbname is None:
            # Use default; this datakit
            dbname = self.make_db_name()
        # Loop through regular expressions
        for i, regex in enumerate(dbname_regexes):
            # Check for match
            grps = self._genr8_dbname_match_groups(regex, dbname)
            # Check for match
            if grps is None:
                continue
            # Get the template
            if i >= len(modname_templates):
                # Not enough templates!
                raise IndexError(
                    ("DB name matched regex %i, but only " % (i+1)) + 
                    ("%i templates specified" % len(modname_templates)))
            # Get templates for this regex
            modname_template = modname_templates[i]
            # Expand all groups
            try:
                # Apply formatting substitutions
                modname = modname_template % grps
            except Exception:
                # Missing group or something
                print("Failed to expand MODULE_NAME_TEPLATE_LIST %i:" % (i+1))
                print("  template: %s" % modname_template)
                print("  groups:")
                # Print all groups
                for k, v in grps.items():
                    print("%12s: %s [%s]" % (k, v, type(v).__name__))
                # Raise an exception
                raise KeyError(
                    ("Failed to expand MODULE_NAME_TEPLATE_LIST ") +
                    ("%i (1-based)" % (i+1)))
            # Save candidate module name
            modname_list.append(modname)
        # Export
        return modname_list

   # --- Supporting ---
    def _genr8_modname_match_groups(self, regex, modname):
        r"""Get the match groups if *modname* fully matches *regex*

        :Call:
            >>> groups = dkl._genr8_modname_match_groups(regex, modname)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *regex*: :class:`str`
                Regular expression string
            *modname*: :class:`str`
                Full name of module
        :Outputs:
            *groups*: ``None`` | :class:`dict`
                List of regex strings, converted to :class:`int` if
                possible.

                Upper-case and lower-case groups are also added.  For
                example if there's a match group called ``name``, then
                ``u-name`` is the upper-case version and ``l-name`` is
                the lower-case version.
        :Keys:
            * *MODULE_NAME_REGEX_INT_GROUPS*
            * *MODULE_NAME_REGEX_STR_GROUPS*
        """
        # Attempt to match regex (all of *modname*)
        match = re.match(regex + "$", modname)
        # Check for no match
        if match is None:
            return None
        # Get dictionary from matches
        groupdict = match.groupdict()
        # Names of groups that should always be integers or strings
        igroups = self.get_option("MODULE_NAME_REGEX_INT_GROUPS")
        sgroups = self.get_option("MODULE_NAME_REGEX_STR_GROUPS")
        # Initialize finale groups
        groups = dict(groupdict)
        # Loop through groups that were present
        for grp, val in groupdict.items():
            # Save upper-case and lower-case
            groups["l-" + grp] = val.lower()
            groups["u-" + grp] = val.upper()
            # Check for direct instruction
            if grp in sgroups:
                # Always keep as string
                pass
            elif grp in igroups:
                # Always convert to integer
                try:
                    ival = int(val)
                except Exception:
                    raise ValueError(
                        ("Failed to convert group '%s' with " % grp) +
                        ("with value '%s' to integer" % val))
                # Save integer group
                groups[grp] = ival
                # Save raw value
                groups["s-" + grp] = val
            elif REGEX_INT.match(val):
                # Convertible to integer
                groups[grp] = int(val)
                # Resave raw value
                groups["s-" + grp] = val
        # Output
        return groups

    def _genr8_modname_regexes(self):
        r"""Expand regular expression strings for module name

        This expands things like ``%(group1)s`` to something
        like ``(?P<group1>[1-9][0-9])`` if *group1* is ``"[1-9][0-9]"``.

        :Call:
            >>> regex_list = dkl._genr8_modname_regexes()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *regex_list*: :class:`list`\ [:class:`str`]
                List of regex strings
        :Keys:
            * *MODULE_NAME_REGEX_GROUPS*
            * *MODULE_NAME_REGEX_LIST*
        """
        # Get the regular expressions for each "group"
        grps = self.get_option("MODULE_NAME_REGEX_GROUPS")
        # Add full formatting for regular expression group
        grps_re = {
            k: "(?P<%s>%s)" % (k, v)
            for k, v in grps.items()
        }
        # Get regular expression list
        name_list = self.get_option("MODULE_NAME_REGEX_LIST")
        # Check if it's a list
        if not isinstance(name_list, (list, tuple)):
            # Create a singleton
            name_list = [name_list]
        # Initialize list of expanded regexes
        regex_list = []
        # Loop through raw lists
        for name in name_list:
            # Expand it and append to regular expression list
            regex_list.append(name % grps_re)
        # Output
        return regex_list
  # >

  # ==========================
  # DB_NAME --> MODULE_NAME
  # ==========================
  # <
   # --- Supporting ---
    def _genr8_dbname_match_groups(self, regex, dbname):
        r"""Get the match groups if *modname* fully matches *regex*

        :Call:
            >>> groups = dkl._genr8_dbname_match_groups(regex, dbname)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *regex*: :class:`str`
                Regular expression string
            *dbname*: :class:`str`
                Full database name
        :Outputs:
            *groups*: ``None`` | :class:`dict`
                List of regex strings, converted to :class:`int` if
                possible.

                Upper-case and lower-case groups are also added.  For
                example if there's a match group called ``name``, then
                ``u-name`` is the upper-case version and ``l-name`` is
                the lower-case version.
        :Keys:
            * *DB_NAME_REGEX_INT_GROUPS*
            * *DB_NAME_REGEX_STR_GROUPS*
        """
        # Attempt to match regex (all of *dbname*)
        match = re.match(regex + "$", dbname)
        # Check for no match
        if match is None:
            return None
        # Get dictionary from matches
        groupdict = match.groupdict()
        # Names of groups that should always be integers or strings
        igroups = self.get_option("DB_NAME_REGEX_INT_GROUPS")
        sgroups = self.get_option("DB_NAME_REGEX_STR_GROUPS")
        # Initialize finale groups
        groups = dict(groupdict)
        # Loop through groups that were present
        for grp, val in groupdict.items():
            # Save upper-case and lower-case
            groups["l-" + grp] = val.lower()
            groups["u-" + grp] = val.upper()
            # Check for direct instruction
            if grp in sgroups:
                # Always keep as string
                pass
            elif grp in igroups:
                # Always convert to integer
                try:
                    ival = int(val)
                except Exception:
                    raise ValueError(
                        ("Failed to convert group '%s' with " % grp) +
                        ("with value '%s' to integer" % val))
                # Save integer group
                groups[grp] = ival
                # Save raw value
                groups["s-" + grp] = val
            elif REGEX_INT.match(val):
                # Convertible to integer
                groups[grp] = int(val)
                # Resave raw value
                groups["s-" + grp] = val
        # Output
        return groups

    def _genr8_dbname_regexes(self):
        r"""Expand regular expression strings for database names

        This expands things like ``%(group1)s`` to something
        like ``(?P<group1>[1-9][0-9])`` if *group1* is ``"[1-9][0-9]"``.

        :Call:
            >>> regex_list = dkl._genr8_dbname_regexes()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *regex_list*: :class:`list`\ [:class:`str`]
                Regular expressions to parse database name
        :Keys:
            * *DB_NAME_REGEX_LIST*
            * *DB_NAME_REGEX_GROUPS*
        """
        # Get the regular expressions for each "group"
        grps = self.get_option("DB_NAME_REGEX_GROUPS")
        # Add full formatting for regular expression group
        grps_re = {
            k: "(?P<%s>%s)" % (k, v)
            for k, v in grps.items()
        }
        # Get regular expression list
        name_list = self.get_option("DB_NAME_REGEX_LIST")
        # Check if it's a list
        if not isinstance(name_list, (list, tuple)):
            # Create a singleton
            name_list = [name_list]
        # Initialize list of expanded regexes
        regex_list = []
        # Loop through raw lists
        for name in name_list:
            # Expand it and append to regular expression list
            regex_list.append(name % grps_re)
        # Output
        return regex_list
  # >
  
  # ==================
  # FILE/FOLDER
  # ==================
  # <
   # --- DataKit file names ---
    def get_dbfile(self, fname, ext):
        r"""Get a file name relative to the datakit folder

        :Call:
            >>> fabs = dkl.get_dbfile(fname, ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: ``None`` | :class:`str`
                Name of file relative to *DB_DIRS_BY_TYPE* for *ext*
            *ext*: :class:`str`
                File type
        :Outputs:
            *fabs*: :class:`str`
                Absolute path to file
        :Keys:
            * *MODULE_DIR*
            * *DB_DIR*
            * *DB_DIRS_BY_TYPE*
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Default file name
        if fname is None:
            # Get database name
            dbname = self.make_db_name()
            # Assemble default file name
            fname = "%s.%s" % (dbname, ext)
        # Assert file name
        self._assert_filename(fname)
        # Check for an absolute file name
        self._assert_filename_relative(fname)
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        dbsdir = self.get_option("DB_DIR")
        # Get folder for dbs of this type
        dbtypedir = self.get_dbdir_by_type(ext)
        # Combine directories
        return os.path.join(moddir, dbsdir, dbtypedir, fname)

    def get_dbfiles(self, dbname, ext):
        r"""Get list of datakit filenames for specified type

        :Call:
            >>> fnames = dkl.get_dbfiles(dbname, ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *dbname*: ``None`` | :class:`str`
                Database name (default if ``None``)
            *ext*: :class:`str`
                File type
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                Absolute path to files for datakit
        :Keys:
            * *MODULE_DIR*
            * *DB_DIR*
            * *DB_DIRS_BY_TYPE*
            * *DB_SUFFIXES_BY_TYPE*
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Default database name
        if dbname is None:
            # Get database name
            dbname = self.make_db_name()
        # Get suffixes
        suffixes = self.get_db_suffixes_by_type(ext)
        # Datakit dir
        dbdir = self.get_dbdir(ext)
        # Initialize list of absolute files
        fnames = []
        # Loop through suffixes
        for suffix in suffixes:
            # Construct full file name
            if suffix is None:
                # No suffix
                fname = "%s.%s" % (dbname, ext)
            else:
                # Add a suffix
                fname = "%s-%s.%s" % (dbname, suffix, ext)
            # Save absolute file name
            fnames.append(os.path.join(dbdir, fname))
        # Output
        return fnames

    def get_dbdir(self, ext):
        r"""Get containing folder for specified datakit file type

        :Call:
            >>> fdir = dkl.get_dbdir(ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *ext*: :class:`str`
                File type
        :Outputs:
            *fdir*: :class:`str`
                Absolute folder to *ext* datakit folder
        :Keys:
            * *MODULE_DIR*
            * *DB_DIR*
            * *DB_DIRS_BY_TYPE*
        :See Also:
            * :func:`get_dbdir_by_type`
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        dbsdir = self.get_option("DB_DIR")
        # Get folder for dbs of this type
        dbtypedir = self.get_dbdir_by_type(ext)
        # Combine directories
        return os.path.join(moddir, dbsdir, dbtypedir)

   # --- Raw data files ---
    def get_rawdatafilename(self, fname, dvc=False):
        r"""Get a file name relative to the datakit folder

        :Call:
            >>> fabs = dkl.get_rawdatafilename(fname, dvc=False)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: ``None`` | :class:`str`
                Name of file relative to *DB_DIRS_BY_TYPE* for *ext*
            *dvc*: ``True`` | {``False``}
                Option to pull DVC file where *fabs* doesn't exist
        :Outputs:
            *fabs*: :class:`str`
                Absolute path to raw data file
        :Keys:
            * *MODULE_DIR*
            * *RAWDATA_DIR*
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        rawdir = self.get_option("RAWDATA_DIR")
        # Full path to raw data
        fdir = os.path.join(moddir, rawdir)
        # Return absolute
        fabs = os.path.join(fdir, fname)
        # Option: whether or not to check for DVC files
        if not dvc:
            # Just output in the general case
            return fabs
        # Check if file exists
        if self._check_modfile(fabs):
            # Nominal situation; file exists
            pass
        elif self._check_dvcfile(fabs):
            # Pull the DVC file?
            self.dvc_pull(fabs)
        # Return name of original file regardless
        return fabs

    def get_rawdatadir(self):
        r"""Get absolute path to module's raw data folder

        :Call:
            >>> fdir = dkl.get_rawdatadir()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *fdir*: :class:`str`
                Absolute path to raw data folder
        :Keys:
            * *MODULE_DIR*
            * *RAWDATA_DIR*
        :Versions:
            * 2021-07-08 ``@ddalle``: v1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        rawdir = self.get_option("RAWDATA_DIR")
        # Full path to raw data
        return os.path.join(moddir, rawdir)

   # --- MAT DataKit files ---
    def get_dbfile_mat(self, fname=None):
        return self.get_dbfile(fname, "mat")

    def get_dbfiles_mat(self, dbname=None):
        return self.get_dbfiles(dbname, "mat")

    def get_dbdir_mat(self):
        return self.get_dbdir("mat")
    
   # --- CSV DataKit files ---
    def get_dbfile_csv(self, fname=None):
        return self.get_dbfile(fname, "csv")

    def get_dbfiles_csv(self, dbname=None):
        return self.get_dbfiles(dbname, "csv")

    def get_dbdir_csv(self):
        return self.get_dbdir("csv")

   # --- XLSX datakit files ---
    def get_dbfile_xlsx(self, fname=None):
        return self.get_dbfile(fname, "xlsx")
        
    def get_dbfiles_xlsx(self, dbname=None):
        return self.get_dbfiles(dbname, "xlsx")
        
    def get_dbdir_xlsx(self):
        return self.get_dbdir("xlsx")
        
   # --- DVC files ---
    def dvc_add(self, frel, **kw):
        r"""Add (cache) a file using DVC

        :Call:
            >>> ierr = dkl.dvc_add(frel, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
        :Outputs:
            *ierr*: :class:`int`
                Return code

                * 0: success
                * 512: not a git repo

        :Versions:
            * 2021-09-15 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs = self.get_abspath(frel)
        # Check for DVC flag
        if fabs.endswith(".dvc"):
            # Already a DVC file
            fdvc = fabs
        else:
            # Append .dvc
            fdvc = fabs + ".dvc"
        # Get the folder name
        fdir = os.path.dirname(fdvc)
        # Get the gitdir
        gitdir = gitutils.get_gitdir(fdir)
        # Check for a valid git repo
        if gitdir is None:
            return 512
        # Strip the *gitdir* prefix and .dvc extension
        fcmd = fdvc[len(gitdir):-4].lstrip(os.sep)
        # Shortened file name for pretty STDOUT
        if len(fcmd) > 40:
            fcmdp = "..." + fcmd[-37:]
        else:
            fcmdp = fcmd
        # Initialize command
        cmd = ["lfc", "add", fcmd]
        cmdp = ["lfc", "add", fcmdp]
        # Status update
        print("  > " + " ".join(cmdp))
        # (Try to) execute the pull
        ierr = shellutils.call_q(cmd, cwd=gitdir)
        # Return error code
        return ierr

    def dvc_pull(self, frel, **kw):
        r"""Pull a DVC file

        :Call:
            >>> ierr = dkl.dvc_pull(frel, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
        :Outputs:
            *ierr*: :class:`int`
                Return code

                * 0: success
                * 256: no DVC file
                * 512: not a git repo

        :Versions:
            * 2021-07-19 ``@ddalle``: v1.0
            * 2023-02-21 ``@ddalle``: v2.0; DVC -> LFC
        """
        # Get absolute path
        fabs = self.get_abspath(frel)
        # Check for DVC flag
        if fabs.endswith(".dvc"):
            # Already a DVC file
            fdvc = fabs
        else:
            # Append .dvc
            fdvc = fabs + ".dvc"
            # Check DVC flag
            if not os.path.isfile(fdvc):
                return 256
        # Get the folder name
        fdir = os.path.dirname(fdvc)
        # Get the gitdir
        gitdir = gitutils.get_gitdir(fdir)
        # Check for a valid git repo
        if gitdir is None:
            return 512
        # Strip the *gitdir*
        fcmd = fdvc[len(gitdir):].lstrip(os.sep)
        # Shortened file name for pretty STDOUT
        if len(fcmd) > 45:
            fcmdp = "..." + fcmd[-40:-4]
        else:
            fcmdp = fcmd[:-4]
        # Initialize command
        cmd = ["lfc", "pull", fcmd]
        cmdp = ["lfc", "pull", fcmdp]
        # Other DVC settings
        jobs = kw.get("jobs", kw.get("j", 1))
        remote = kw.get("remote", kw.get("r"))
        # Add other settings
        if jobs:
            cmd.extend(["-j", str(jobs)])
            cmdp.extend(["-j", str(jobs)])
        if remote:
            cmd.extend(["-r", str(remote)])
            cmdp.extend(["-r", str(remote)])
        # Status update
        print("  > " + " ".join(cmdp))
        # (Try to) execute the pull
        ierr = shellutils.call(cmd, cwd=gitdir, stderr=shellutils.PIPE)
        # Return error code
        return ierr

    def dvc_push(self, frel, **kw):
        r"""Push a DVC file

        :Call:
            >>> ierr = dkl.dvc_push(frel, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
        :Outputs:
            *ierr*: :class:`int`
                Return code

                * 0: success
                * 256: no DVC file
                * 512: not a git repo

        :Versions:
            * 2021-09-15 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs = self.get_abspath(frel)
        # Check for DVC flag
        if fabs.endswith(".dvc"):
            # Already a DVC file
            fdvc = fabs
        else:
            # Append .dvc
            fdvc = fabs + ".dvc"
            # Check DVC flag
            if not os.path.isfile(fdvc):
                return 256
        # Get the folder name
        fdir = os.path.dirname(fdvc)
        # Get the gitdir
        gitdir = gitutils.get_gitdir(fdir)
        # Check for a valid git repo
        if gitdir is None:
            return 512
        # Strip the *gitdir*
        fcmd = fdvc[len(gitdir):].lstrip(os.sep)
        # Shortened file name for pretty STDOUT
        if len(fcmd) > 45:
            fcmdp = "..." + fcmd[-40:-4]
        else:
            fcmdp = fcmd[:-4]
        # Initialize command
        cmd = ["lfc", "push", fcmd]
        cmdp = ["lfc", "push", fcmdp]
        # Other DVC settings
        jobs = kw.get("jobs", kw.get("j", 1))
        remote = kw.get("remote", kw.get("r"))
        # Add other settings
        if jobs:
            cmd.extend(["-j", str(jobs)])
            cmdp.extend(["-j", str(jobs)])
        if remote:
            cmd.extend(["-r", str(remote)])
            cmdp.extend(["-r", str(remote)])
        # Status update
        print("  > " + " ".join(cmdp))
        # (Try to) execute the pull
        ierr = shellutils.call_q(cmd, cwd=gitdir)
        # Return error code
        return ierr

    def dvc_status(self, frel, **kw):
        r"""Check status a DVC file

        :Call:
            >>> ierr = dkl.dvc_status(frel, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
        :Outputs:
            *ierr*: :class:`int`
                Return code

                * 0: success
                * 1: out-of-date
                * 256: no DVC file
                * 512: not a git repo

        :Versions:
            * 2021-09-23 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs = self.get_abspath(frel)
        # Check for DVC flag
        if fabs.endswith(".dvc"):
            # Already a DVC file
            fdvc = fabs
        else:
            # Append .dvc
            fdvc = fabs + ".dvc"
            # Check DVC flag
            if not os.path.isfile(fdvc):
                return 256
        # Get the folder name
        fdir = os.path.dirname(fdvc)
        # Get the gitdir
        gitdir = gitutils.get_gitdir(fdir)
        # Check for a valid git repo
        if gitdir is None:
            return 512
        # Strip the *gitdir*
        fcmd = fdvc[len(gitdir):].lstrip(os.sep)
        # Shortened file name for pretty STDOUT
        if len(fcmd) > 43:
            fcmdp = "..." + fcmd[-38:-4]
        else:
            fcmdp = fcmd[:-4]
        # Initialize command
        cmd = ["dvc", "status", fcmd]
        # (Try to) execute the pull
        stdout, _, ierr = shellutils.call_oe(cmd, cwd=gitdir)
        # Check for error
        if ierr:
            return ierr
        # Check if "up to date"
        if len(stdout.strip().split("\n")) > 1:
            # Out-of-date
            return 1
        else:
            # Up-to-date
            return 0

   # --- Raw data update ---
    # Main updater
    def update_rawdata(self, **kw):
        r"""Update raw data using ``rawdata/datakit-sources.json``

        The settings for zero or more "remotes" are read from that JSON
        file in the package's ``rawdata/`` folder. Example contents of
        such a file are shown below:

        .. code-block:: javascript

            {
                "hub": [
                    "/nobackup/user/",
                    "pfe:/nobackupp16/user/git",
                    "linux252:/nobackup/user/git"
                ],
                "remotes": {
                    "origin": {
                        "url": "data/datarepo.git",
                        "type": "git-show",
                        "glob": "aero_STACK*.csv",
                        "regex": [
                            "aero_CORE_no_[a-z]+\.csv",
                            "aero_LSRB_no_[a-z]+\.csv",
                            "aero_RSRB_no_[a-z]+\.csv"
                        ],
                        "commit": null,
                        "branch": "main",
                        "tag": null,
                        "destination": "."
                    }
                }
            }

        :Call:
            >>> dkl.update_rawdata(remote=None, remotes=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``None``} | :class:`str`
                Name of single remote to update
            *remotes*: {``None``} | :class:`list`\ [:class:`str`]
                Name of multiple remotes to update
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
            * 2022-01-18 ``@ddalle``: Version 1.1; remote(s) kwarg
        """
        # User-specified remote(s)
        remote = kw.get("remote")
        if remote is None:
            # Check for list
            remotes = kw.get("remotes", self.get_rawdata_remotelist())
        else:
            # Convert singleton to list
            remotes = [remote]
        # Loop through them
        for remote in remotes:
            self.update_rawdata_remote(remote)

    # Update from one remote
    def update_rawdata_remote(self, remote="origin"):
        r"""Update raw data for one remote

        :Call:
            >>> dkl.update_rawdata_remote(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Get type
        remote_type = self.get_rawdata_opt("type", remote, vdef="git-show")
        # Check type
        if remote_type in {1, "git-show"}:
            # Use git-show to read files
            self._upd8_rawdataremote_git(remote)
        elif remote_type in {2, "rsync"}:
            # Use rsync to copy updated files
            self._upd8_rawdataremote_rsync(remote)
        elif remote_type in {3, "lfc-show"}:
            # Use lfc-show to read files
            self._upd8_rawdataremote_lfc(remote)
        else:
            raise ValueError(
                "Unrecognized raw data remote type '%s'" % remote_type)

    # Update one remote using git-show
    def _upd8_rawdataremote_git(self, remote="origin"):
        r"""Update raw data for one remote using ``git show``

        :Call:
            >>> dkl._upd8_rawdataremote_git(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Status update
        sys.stdout.write("  updating remote '%s' using git-show\n" % remote)
        sys.stdout.flush()
        # Read current commit (already read)
        sha1_current = self.get_rawdata_sourcecommit(remote)
        # Get best available
        url, sha1 = self.get_rawdataremote_git(remote)
        # Check if up-to-date
        if sha1 == sha1_current:
            # Status update
            print("    up-to-date (%s)" % sha1)
            # Terminate (no file transfers needed)
            return
        # Get files
        ls_files = self.get_rawdataremote_gitfiles(remote)
        # Get option regarding destination
        # (Remove subfolders, e.g. folder/file.csv -> file.csv)
        dst_folder = self.get_rawdata_opt("destination", remote)
        # Counters
        n_good = 0
        n_fail = 0
        # Copy each file
        for src in ls_files:
            # Destination folder
            if dst_folder == ".":
                # Just use the "basename"
                dst = os.path.basename(src)
            elif dst_folder:
                # For other non-empty destination folder
                dst = os.path.join(dst_folder, os.path.basename(src))
            else:
                # Copy to file with same path as in git repo
                dst = src
            # Status update
            if src == dst:
                # Same file name as in git repo
                msg = "  copying file '%s'\r" % src
            else:
                # Different git repo name and local name
                msg = "  copying file '%s' -> '%s'\r" % (src, dst)
            # Write the status update
            sys.stdout.write(msg)
            sys.stdout.flush()
            # Copy the file
            ierr = self._upd8_rawdatafile_git(url, src, sha1, dst=dst)
            # Check status
            if ierr:
                # Transfer failed
                n_fail += 1
            else:
                # Transfer succeeded
                n_good += 1
            # Clean up prompt
            sys.stdout.write(" " * len(msg))
            sys.stdout.write("\r")
            sys.stdout.flush()
        # Save the commit
        self.rawdata_sources_commit[remote] = sha1
        # Status update
        if n_fail:
            msg = "  copied %i files (%i failed) from %s" % (
                n_good, n_fail, sha1)
        else:
            msg = "  copied %i files from %s" % (n_good, sha1)
        print(msg)
        # Write it
        self._write_rawdata_commits_json()

    # Update one remote using rsync
    def _upd8_rawdataremote_rsync(self, remote="origin"):
        r"""Update raw data for one remote using ``rsync``

        :Call:
            >>> dkl._upd8_rawdataremote_rsync(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Status update
        sys.stdout.write("updating remote '%s' using rsync\n" % remote)
        sys.stdout.flush()
        # Get best available
        url = self.get_rawdataremote_rsync(remote)
        # Check if no URL
        if url is None:
            return
        # Get files
        ls_files = self.get_rawdataremote_rsyncfiles(remote)
        # Get option regarding destination
        # (Remove subfolders, e.g. folder/file.csv -> file.csv)
        dst_folder = self.get_rawdata_opt("destination", remote)
        # Counters
        n_good = 0
        n_fail = 0
        # Copy each file
        for src in ls_files:
            # Destination folder
            if dst_folder == ".":
                # Just use the "basename"
                dst = os.path.basename(rc)
            elif dst_folder:
                # For other non-empty destination folder
                dst = os.path.join(dst_folder, os.path.basename(src))
            else:
                # Copy to file with same path as in git repo
                dst = src
            # Status update
            if src == dst:
                # Same file name as in git repo
                msg = "  copying file '%s'\r" % src
            else:
                # Different git repo name and local name
                msg = "  copying file '%s' -> '%s'\r" % (src, dst)
            # Write the status update
            sys.stdout.write(msg)
            sys.stdout.flush()
            # Copy the file
            ierr = self._upd8_rawdatafile_rsync(url, src, dst=dst)
            # Check status
            if ierr:
                # Transfer failed
                n_fail += 1
            else:
                # Transfer succeeded
                n_good += 1
            # Clean up prompt
            sys.stdout.write(" " * len(msg))
            sys.stdout.write("\r")
            sys.stdout.flush()
        # Status update
        if n_fail:
            msg = "  copied %i files (%i failed)" % (n_good, n_fail)
        else:
            msg = "  copied %i files" % (n_good)
        print(msg)
        # Write it
        self._write_rawdata_commits_json()

    # Update one remote using lfc-show
    def _upd8_rawdataremote_lfc(self, remote="origin"):
        r"""Update raw data for one remote using ``lfc show``

        :Call:
            >>> dkl._upd8_rawdataremote_lfc(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Versions:
            * 2022-12-22 ``@ddalle``: v1.0; fork _upd8_rawdatafile_git()
        """
        # Status update
        sys.stdout.write("  updating remote '%s' using lfc-show\n" % remote)
        sys.stdout.flush()
        # Read current commit (already read)
        sha1_current = self.get_rawdata_sourcecommit(remote)
        # Get best available
        url, sha1 = self.get_rawdataremote_git(remote)
        # Check if up-to-date
        if sha1 == sha1_current:
            # Status update
            print("    up-to-date (%s)" % sha1)
            # Terminate (no file transfers needed)
            return
        # Get files
        ls_files = self.get_rawdataremote_gitfiles(remote)
        # Get option regarding destination
        # (Remove subfolders, e.g. folder/file.csv -> file.csv)
        dst_folder = self.get_rawdata_opt("destination", remote)
        # Counters
        n_good = 0
        n_fail = 0
        # Copy each file
        for src in ls_files:
            # Strip LFC extensions
            for ext in (".lfc", ".dvc"):
                if src.endswith(ext):
                    src = src[:-len(ext)]
            # Destination folder
            if dst_folder == ".":
                # Just use the "basename"
                dst = os.path.basename(src)
            elif dst_folder:
                # For other non-empty destination folder
                dst = os.path.join(dst_folder, os.path.basename(src))
            else:
                # Copy to file with same path as in git repo
                dst = src
            # Status update
            if src == dst:
                # Same file name as in git repo
                msg = "  copying file '%s'\r" % src
            else:
                # Different git repo name and local name
                msg = "  copying file '%s' -> '%s'\r" % (src, dst)
            # Write the status update
            sys.stdout.write(msg)
            sys.stdout.flush()
            # Copy the file
            ierr = self._upd8_rawdatafile_lfc(url, src, sha1, dst=dst)
            # Check status
            if ierr:
                # Transfer failed
                n_fail += 1
            else:
                # Transfer succeeded
                n_good += 1
            # Clean up prompt
            sys.stdout.write(" " * len(msg))
            sys.stdout.write("\r")
            sys.stdout.flush()
        # Save the commit
        self.rawdata_sources_commit[remote] = sha1
        # Status update
        if n_fail:
            msg = "  copied %i files (%i failed) from %s" % (
                n_good, n_fail, sha1)
        else:
            msg = "  copied %i files from %s" % (n_good, sha1)
        print(msg)
        # Write it
        self._write_rawdata_commits_json()

    # Copy one file from remote repo
    def _upd8_rawdatafile_git(self, url, src, ref, **kw):
        r"""Copy one raw data file from remote using ``git show``

        :Call:
            >>> ierr = dkl._upd8_rawdatafile_git(url, src, ref, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *url*: :class:`str`
                Full path to git remote
            *src*: :class:`str`
                Name of file to copy relative to remote git repo
            *ref*: :class:`str`
                Any valid git reference, usually a SHA-1 hash
            *dst*: {*src*} | :class:`str`
                Name of destination file rel to ``rawdata/``
        :Outputs:
            *ierr*: :class:`int`
                Return code (``0`` for success)
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Name of destination file
        dst = kw.get("dst", src)
        # Check for "/" in output file
        if "/" in dst:
            # Localize path
            fdest = dst.replace("/", os.sep)
            # Prepare folder
            self.prep_dirs_rawdata(fdest)
        else:
            # Use specified path
            fdest = dst
        # Get absolute path to file in rawdata/ file
        fout = self.get_rawdatafilename(fdest)
        # Open file
        f = open(fout, "w")
        # Command to list contents of file
        cmd = ["git", "show", "%s:%s" % (ref, src)]
        # Execute command
        _, _, ierr = self._call(url, cmd, stdout=f, stderr=shellutils.PIPE)
        # Close file
        f.close()
        # Return same return code
        return ierr

    # Copy one file from remote repo
    def _upd8_rawdatafile_lfc(self, url, src, ref, **kw):
        r"""Copy one raw data file from remote using ``lfc show``

        :Call:
            >>> ierr = dkl._upd8_rawdatafile_git(url, src, ref, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *url*: :class:`str`
                Full path to git remote
            *src*: :class:`str`
                Name of file to copy relative to remote git repo
            *ref*: :class:`str`
                Any valid git reference, usually a SHA-1 hash
            *dst*: {*src*} | :class:`str`
                Name of destination file rel to ``rawdata/``
        :Outputs:
            *ierr*: :class:`int`
                Return code (``0`` for success)
        :Versions:
            * 2022-12-22 ``@ddalle``: v1.0; fork _upd8_rawdatafile_git()
        """
        # Name of destination file
        dst = kw.get("dst", src)
        # Check for "/" in output file
        if "/" in dst:
            # Localize path
            fdest = dst.replace("/", os.sep)
            # Prepare folder
            self.prep_dirs_rawdata(fdest)
        else:
            # Use specified path
            fdest = dst
        # Get absolute path to file in rawdata/ file
        fout = self.get_rawdatafilename(fdest)
        # Open file
        fp = open(fout, "wb")
        # Command to list contents of file
        cmd = ["lfc", "show", "-r", ref, src]
        # Execute command
        _, _, ierr = self._call(url, cmd, stdout=fp, stderr=shellutils.PIPE)
        # Close file
        fp.close()
        # Return same return code
        return ierr

    # Copy one file from remote repo
    def _upd8_rawdatafile_rsync(self, url, src, **kw):
        r"""Copy one raw data file from remote using ``rsync``

        :Call:
            >>> ierr = dkl._upd8_rawdatafile_rsync(url, src, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *url*: :class:`str`
                Full path to git remote
            *src*: :class:`str`
                Name of file to copy relative to remote git repo
            *dst*: {*src*} | :class:`str`
                Name of destination file rel to ``rawdata/``
        :Outputs:
            *ierr*: :class:`int`
                Return code (``0`` for success)
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Name of destination file
        dst = kw.get("dst", src)
        # Check for "/" in output file
        if "/" in dst:
            # Localize path
            fdest = dst.replace("/", os.sep)
            # Prepare folder
            self.prep_dirs_rawdata(fdest)
        else:
            # Use specified path
            fdest = dst
        # Full source
        fname = url.rstrip("/") + "/" + src
        # Command to list contents of file
        cmd = ["rsync", "-tuz", fname, fdest]
        # Execute command
        _, _, ierr = shellutils.call_oe(cmd, cwd=self.get_rawdatadir())
        # Return same return code
        return ierr

    # Get list of remote files matching patterns
    def get_rawdataremote_gitfiles(self, remote="origin"):
        r"""List all files in candidate raw data remote source

        :Call:
            >>> fnames = dkl.get_rawdataremote_gitfiles(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of files to be copied from remote repo
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Get list of files
        ls_files = self.list_rawdataremote_git(remote)
        # Get list of globs to copy
        globs = _listify(self.get_rawdata_opt("glob", remote=remote))
        # Get list of file name regular expressions to copy
        regexs = _listify(self.get_rawdata_opt("regex", remote=remote))
        # If no constraints, return all files
        if len(globs) == 0 and len(regexs) == 0:
            return ls_files
        # Otherwise initialize set of files to copy
        file_set = set()
        # Loop through patterns
        for pat in globs:
            # Match git files with pattern
            file_set.update(fnmatch.filter(ls_files, pat))
        # Lop through regexes
        for pat in regexs:
            # Compile
            regex = re.compile(pat)
            # Compare pattern to each file name
            for fname in ls_files:
                # Skip if already included
                if fname in file_set:
                    continue
                # Check for match
                if regex.match(fname + "$"):
                    file_set.add(fname)
        # Output
        return list(file_set)

    # Get list of remote files matching patterns
    def get_rawdataremote_rsyncfiles(self, remote="origin"):
        r"""List all files in candidate remote folder

        :Call:
            >>> fnames = dkl.get_rawdataremote_rsyncfiles(remote)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of files to be copied from remote repo
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Get list of files
        ls_files = self.list_rawdataremote_rsync(remote)
        # Get list of globs to copy
        globs = _listify(self.get_rawdata_opt("glob", remote=remote))
        # Get list of file name regular expressions to copy
        regexs = _listify(self.get_rawdata_opt("regex", remote=remote))
        # If no constraints, return all files
        if len(globs) == 0 and len(regexs) == 0:
            return ls_files
        # Otherwise initialize set of files to copy
        file_set = set()
        # Loop through patterns
        for pat in globs:
            # Match git files with pattern
            file_set.update(fnmatch.filter(ls_files, pat))
        # Lop through regexes
        for pat in regexs:
            # Compile
            regex = re.compile(pat)
            # Compare pattern to each file name
            for fname in ls_files:
                # Skip if already included
                if fname in file_set:
                    continue
                # Check for match
                if regex.match(fname + "$"):
                    file_set.add(fname)
        # Output
        return list(file_set)
        
    # Get full list of files from rawdata source
    def list_rawdataremote_git(self, remote="origin"):
        r"""List all files in candidate raw data remote source

        :Call:
            >>> ls_files = dkl.list_rawdataremote_git(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *ls_files*: :class:`list`\ [:class:`str`]
                List of all files tracked by remote repo
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Get URL and hash
        url, sha1 = self.get_rawdataremote_git(remote)
        # Check for invalid repo
        if url is None:
            return []
        # Status update
        msg = "  getting list of files from remote\r"
        sys.stdout.write(msg)
        sys.stdout.flush()
        # List files
        stdout = self._call_o(
            url, ["git", "ls-tree", "-r", sha1, "--name-only"])
        # Clean up raw data
        sys.stdout.write(" " * len(msg))
        sys.stdout.write("\r")
        sys.stdout.flush()
        # Check validity
        if stdout is None:
            return []
        # Split
        return stdout.strip().split("\n")
        
    # Get full list of files from rawdata source
    def list_rawdataremote_rsync(self, remote="origin"):
        r"""List all files in candidate raw data remote folder

        :Call:
            >>> ls_files = dkl.list_rawdataremote_rsync(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *ls_files*: :class:`list`\ [:class:`str`]
                List of all files in remote source folder
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Get best available
        url = self.get_rawdataremote_rsync(remote)
        # Check for invalid repo
        if url is None:
            return []
        # Status update
        msg = "  getting list of files from remote\r"
        sys.stdout.write(msg)
        sys.stdout.flush()
        # List files
        stdout = self._call_o(url, ["/bin/ls"])
        # Clean up raw data
        sys.stdout.write(" " * len(msg))
        sys.stdout.write("\r")
        sys.stdout.flush()
        # Check validity
        if stdout is None:
            return []
        # Split
        return stdout.strip().split("\n")

    # Get the best rawdata source
    def get_rawdataremote_git(self, remote="origin", f=False):
        r"""Get full URL and SHA-1 hash for raw data source repo

        :Call:
            >>> url, sha1 = dkl.get_rawdataremote_git(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
            *f*: ``True`` | {``False``}
                Option to override *dkl.rawdata_remotes* if present
        :Outputs:
            *url*: ``None`` | :class:`str`
                Full path to valid git repo, if possible
            *sha1*: ``None`` | :class:`str`
                40-character hash of specified commit, if possible
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Check for existing remote
        url = self.rawdata_remotes.get(remote)
        sha1 = self.rawdata_commits.get(remote)
        # Check for early termination
        if url and sha1 and (not f):
            return url, sha1
        # Read settings
        self.read_rawdata_json()
        # Get list of candidates
        url_list = self._get_rawdataremote_urls(remote)
        # Get reference to check
        ref = self.get_rawdata_ref(remote)
        # Prepend to list if *url* is specified
        if url:
            # Remove it if needed
            if url in url_list:
                url_list.remove(url)
            # Put it to front of list
            url_list.insert(0, url)
        # Loop through candidates
        for url in url_list:
            # Status update
            msg = "  trying '%s'" % url
            # Trim if needed
            if len(msg) > 72:
                msg = msg[:49] + "..." + msg[-20:]
            # Show it
            sys.stdout.write(msg + "\r")
            # Get most recent commit if possible
            sha1 = self._get_sha1(url, ref)
            # Clean up prompt
            sys.stdout.write(" " * len(msg))
            sys.stdout.write("\r")
            # Check if successful
            if sha1:
                # Save options
                self.rawdata_remotes[remote] = url
                self.rawdata_commits[remote] = sha1
                # Terminate
                return url, sha1
        # No valid repo found
        return None, None

    # Get commit of current raw data
    def get_rawdata_sourcecommit(self, remote="origin"):
        r"""Get the latest used SHA-1 hash for a remote

        :Call:
            >>> sha1 = dkl.get_rawdata_sourcecommit(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote from which to read *opt*
        :Outputs:
            *sha1*: ``None`` | :class:`str`
                40-character SHA-1 hash if possible from
                ``datakit-sources-commit.json``
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Read JSON file
        self._read_rawdata_commits_json()
        # Get commit for this origin
        return self.rawdata_sources_commit.get(remote)

    # Get the best rawdata source (rsync)
    def get_rawdataremote_rsync(self, remote="origin"):
        r"""Get full URL for ``rsync`` raw data source repo

        If several options are present, this function checks for the
        first with an extant folder.

        :Call:
            >>> url = dkl.get_rawdataremote_rsync(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *url*: ``None`` | :class:`str`
                Full path to valid git repo, if possible
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Check for existing remote
        url = self.rawdata_remotes.get(remote)
        # Check for early termination
        if url:
            return url
        # Read settings
        self.read_rawdata_json()
        # Get list of candidates
        url_list = self._get_rawdataremote_urls(remote)
        # Loop through candidates
        for url in url_list:
            # Status update
            msg = "  trying '%s'" % url
            # Trim if needed
            if len(msg) > 72:
                msg = msg[:49] + "..." + msg[-20:]
            # Show it
            sys.stdout.write(msg + "\r")
            # Get most recent commit if possible
            q = self._isdir(url)
            # Clean up prompt
            sys.stdout.write(" " * len(msg))
            sys.stdout.write("\r")
            # Check if successful
            if q:
                # Save options
                self.rawdata_remotes[remote] = url
                # Terminate
                return url

    # Get raw data settings
    def read_rawdata_json(self, fname="datakit-sources.json", f=False):
        r"""Read ``datakit-sources.json`` from package's raw data folder

        :Call:
            >>> dkl.read_rawdata_json(fname=None, f=False)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: {``"datakit-sources.json"``} | :class:`str`
                Relative or absolute file name (rel. to ``rawdata/``)
            *f*: ``True`` | {``False``}
                Reread even if *dkl.rawdata_sources* is nonempty
        :Effects:
            *dkl.rawdata_sources*: :class:`dict`
                Settings read from JSON file
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Check for reread
        if (not f) and self.rawdata_sources:
            return
        # Get absolute path
        fabs = self.get_rawdatafilename(fname)
        # Check for file
        if not os.path.isfile(fabs):
            return
        # Read the file
        with open(fabs) as f:
            self.rawdata_sources = json.load(f)

    # Get raw data settings
    def _read_rawdata_commits_json(self):
        r"""Read ``datakit-sources-commit.json`` from raw data folder

        :Call:
            >>> dkl._read_rawdata_commits_json()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Effects:
            *dkl.rawdata_sources_commit*: :class:`dict`
                Settings read from JSON file
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs = self.get_rawdatafilename("datakit-sources-commit.json")
        # Check for file
        if not os.path.isfile(fabs):
            return
        # Read the file
        with open(fabs) as fp:
            self.rawdata_sources_commit = json.load(fp)

    # Get raw data settings
    def _write_rawdata_commits_json(self):
        r"""Write ``datakit-sources-commit.json`` in raw data folder

        :Call:
            >>> dkl._write_rawdata_commits_json()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Effects:
            *dkl.rawdata_sources_commit*: :class:`dict`
                Settings read from JSON file
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs = self.get_rawdatafilename("datakit-sources-commit.json")
        # Read the file
        with open(fabs, "w") as fp:
            json.dump(self.rawdata_sources_commit, fp, indent=4)

    # Get git reference for a specified remote
    def get_rawdata_ref(self, remote="origin"):
        r"""Get optional SHA-1 hash, tag, or branch for raw data source

        :Call:
            >>> ref = dkl.get_rawdata_ref(remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote
        :Outputs:
            *ref*: {``"HEAD"``} | :class:`str`
                Valid git reference name
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Try three valid references, with specified preference
        ref = self.get_rawdata_opt("branch", remote=remote, vdef="HEAD")
        ref = self.get_rawdata_opt("tag", remote=remote, vdef=ref)
        ref = self.get_rawdata_opt("commit", remote=remote, vdef=ref)
        # Output
        return ref

    # Get option from rawdata/datakit-sources.json
    def get_rawdata_opt(self, opt, remote="origin", vdef=None):
        r"""Get a ``rawdata/datakit-sources.json`` setting

        :Call:
            >>> v = dkl.get_rawdata_opt(opt, remote="origin")
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *opt*: :class:`str`
                Name of option to read
            *remote*: {``"origin"``} | :class:`str`
                Name of remote from which to read *opt*
            *vdef*: {``None``} | **any**
                Default value if *opt* not present
        :Outputs:
            *v*: {*vdef*} | **any**
                Value from JSON file if possible, else *vdef*
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
            * 2022-01-26 ``@ddalle``: Version 1.1; add substitutions
        """
        # Special case for opt == "hub"
        if opt == "hub":
            return self.rawdata_sources.get("hub", vdef)
        # Format options
        fmt = {
            "remote": remote,
        }
        # Otherwise get the remotes section
        opts = self.rawdata_sources.get("remotes", {})
        # Get options for this remote
        opts_remote = opts.get(remote, {})
        # Get option, using default as needed
        v = opts_remote.get(opt, vdef)
        # Perform substitutions
        if typeutils.isstr(v):
            # Substitute
            v = v % fmt
        elif isinstance(v, (tuple, list)):
            # Substitute each element
            v = v.__class__([vj % fmt for vj in v])
        # Return option with substitutions
        return v

    # Get option from rawdata/datakit-sources.json
    def get_rawdata_remotelist(self):
        r"""Get list of remotes from ``rawdata/datakit-sources.json``

        :Call:
            >>> remotes = dkl.get_rawdata_remotelist()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *remotes*: :class:`list`\ [:class:`str`]
                List of remotes
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Read raw data optsion
        self.read_rawdata_json()
        # Otherwise get the remotes section
        opts = self.rawdata_sources.get("remotes", {})
        # Get remotes, which are the keys of this dict
        if isinstance(opts, dict):
            return list(opts.keys())
        else:
            return []

    # Get list of remote urls to try
    def _get_rawdataremote_urls(self, remote="origin"):
        r"""Get list of candidate URLs for a given remote

        :Call:
            >>> remote_urls = dkl._get_rawdataremote_urls(remote)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *remote*: {``"origin"``} | :class:`str`
                Name of remote from which to read *opt*
        :Outputs:
            *remote_urls*: :class:`list`\ [:class:`str`]
                List of candidate URLs for *remote*
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Get hubs
        hubs = _listify(self.get_rawdata_opt("hub"))
        # Get URLs
        urls = _listify(self.get_rawdata_opt("url", remote))
        # Initialize list
        remote_urls = []
        # Loop through URLs
        for url in urls:
            # Match against full URL with remote host
            match = REGEX_REMOTE.match(url)
            # Check if it's absolute
            if match:
                # Absolute path with SSH host
                remote_urls.append(url)
                continue
            elif os.path.isabs(url):
                # Already an absolute path
                remote_urls.append(url)
                continue
            # Got a relative path; prepend it with each hub
            for hub in hubs:
                remote_urls.append(hub + "/" + url)
        # Output
        return remote_urls
        

    # Check most recent commit
    def _get_sha1(self, fgit, ref=None):
        r"""Get the SHA-1 hash of specified ref from a git repo

        :Call:
            >>> commit = dkl._get_sha1(fgit, ref=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fgit*: :class:`str`
                URL to a (candidate) git repo
            *ref*: {``"HEAD"``} | :class:`str`
                Any valid git reference; branch, tag, or SHA-1 hash
        :Outputs:
            *commit*: ``None`` | :class:`str`
                SHA-1 hash of commit from *ref* if *fgit* is a repo
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Default ref
        if ref is None:
            ref = "HEAD"
        # Call git command
        return self._call_o(fgit, ["git", "rev-parse", ref])

    # Run a git command remotely or locally
    def _call_o(self, fgit, cmd, **kw):
        r"""Run a command locally or remotely and capture STDOUT

        :Call:
            >>> stdout = dkl._call_o(fgit, cmd, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fgit*: :class:`str`
                URL to a (candidate) git repo
            *cmd*: :class:`list`\ [:class:`str`]
                Subprocess-style command to run
            *kw*: :class:`dict`
                Options passed to :func:`shellutils.call_oe`
        :Outputs:
            *stdout*: ``None`` | :class:`str`
                Decoded STDOUT if command exited without error
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Default options
        kw.setdefault("stdout", shellutils.PIPE)
        kw.setdefault("stderr", shellutils.PIPE)
        # Call command
        stdout, _, ierr = self._call(fgit, cmd, **kw)
        # Check for errors
        if ierr:
            return
        # Return the output
        if stdout is None:
            return
        else:
            return stdout.strip()

    # Run a git command remotely or locally
    def _call(self, fgit, cmd, **kw):
        r"""Run a command locally or remotely and capture STDOUT

        :Call:
            >>> out, err, ierr = dkl._call_o(fgit, cmd, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fgit*: :class:`str`
                URL to a (candidate) git repo
            *cmd*: :class:`list`\ [:class:`str`]
                Subprocess-style command to run
            *kw*: :class:`dict`
                Options passed to :func:`shellutils.call_oe`
        :Outputs:
            *stdout*: ``None`` | :class:`str`
                Decoded STDOUT if command exited without error
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Parse for remotes
        match = REGEX_HOST.match(fgit)
        # Check for bad match
        if match is None:
            raise ValueError("Unable to parse remote repo '%s'" % fgit)
        # Get groups
        grps = match.groupdict()
        # Set options
        kw.setdefault("cwd", grps["path"])
        kw.setdefault("host", grps["host"])
        # Get most recent commit
        return shellutils._call(cmd, **kw)

    # Check if a (remote) folder exists
    def _isdir(self, url):
        r"""Check if a local/remote folder exists

        :Call:
            >>> q = dkl._isdir(url)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *url*: :class:`str`
                URL to a (candidate) local/remote folder
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *url* is an extant folder
        :Versions:
            * 2021-09-02 ``@ddalle``: v1.0
        """
        # Parse for remotes
        match = REGEX_HOST.match(url)
        # Check for bad match
        if match is None:
            raise ValueError("Unable to parse remote folder '%s'" % url)
        # Get groups
        host = match.group("host")
        path = match.group("path")
        # Check for remote host
        if host:
            # Use SSH to access remote host
            cmd = ["ssh", "-q", host, "test", "-d", path]
            # Call command
            ierr = shellutils.call_q(cmd)
            # Return code is 0 if folder exists (and is a folder)
            return ierr == 0
        else:
            # Just check for folder locally
            return os.path.isdir(os.path.realpath(path))

   # --- Generic file names ---
    def get_abspath(self, frel):
        r"""Get the full filename from path relative to *MODULE_DIR*

        :Call:
            >>> fabs = dkl.get_abspath(frel)
            >>> fabs = dkl.get_abspath(fabs)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
            *fabs*: :class:`str`
                Existing absolute path
        :Keys:
            * *MODULE_DIR*
        :Outputs:
            *fabs*: :class:`str`
                Absolute path to file
        :Versions:
            * 2021-07-05 ``@ddalle``: v1.0
        """
        # Check file name
        self._assert_filename(frel)
        # Check for absolute file
        if os.path.isabs(frel):
            # Already absolute
            return frel
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        # Return full path to file name
        return os.path.join(moddir, frel)

   # --- Create folder ---
    def prep_dirs(self, frel):
        r"""Prepare folders needed for file if needed

        Any folders in *frel* that don't exist will be created. For
        example ``"db/csv/datakit.csv"`` will create the folders ``db/``
        and ``db/csv/`` if they don't already exist.

        :Call:
            >>> dkl.prep_dirs(frel)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to *MODULE_DIR*
            *fabs*: :class:`str`
                Existing absolute path
        :Keys:
            * *MODULE_DIR*
        :See also:
            * :func:`DataKitLoader.get_abspath`
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Get absolute file path
        fabs = self.get_abspath(frel)
        # Get just the folder containing *fabs*
        fdir = os.path.dirname(fabs)
        # Folders to create
        fdirs_new = []
        # Loop through folders in reverse order
        while fdir:
            # Check if folder exists
            if os.path.isdir(fdir):
                break
            # Split off the last folder
            fdir, fdir_last = os.path.split(fdir)
            # Create the last folder
            fdirs_new.insert(0, fdir_last)
        else:
            # Didn't find any folders!
            raise ValueError("Cannot create absolute path '%s'" % fabs)
        # Loop through folders that need to be created
        for fdir_new in fdirs_new:
            # Append folder name
            fdir = os.path.join(fdir, fdir_new)
            # Check if it was created since the last check
            if os.path.isdir(fdir):
                continue
            # Create the folder
            os.mkdir(fdir)

    def prep_dirs_rawdata(self, frel):
        r"""Prepare folders relative to ``rawdata/`` folder

        :Call:
            >>> dkl.prep_dirs_rawdata(frel)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *frel*: :class:`str`
                Name of file relative to ``rawdata/`` folder
            *fabs*: :class:`str`
                Existing absolute path
        :Keys:
            * *MODULE_DIR*
        :See also:
            * :func:`DataKitLoader.prep_dirs`
            * :func:`DataKitLoader.get_abspath`
        :Versions:
            * 2021-09-01 ``@ddalle``: v1.0
        """
        # Check for absolute path
        if os.path.isabs(frel):
            # Don't prepend to already-absolute path
            self.prep_dirs(frel)
        else:
            # Prepend "rawdata" to relative path
            self.prep_dirs(os.path.join("rawdata", frel))

   # --- File checks ---
    def check_file(self, fname, f=False, dvc=True):
        r"""Check if a file exists OR a ``.dvc`` version

        * If *f* is ``True``, this returns ``False`` always
        * If *fabs* exists, this returns ``True``
        * If *fabs* plus ``.dvc`` exists, it also returns ``True``

        :Call:
            >>> q = dkl.check_file(fname, f=False, dvc=True)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file [optionally relative to *MODULE_DIR*]
            *f*: ``True`` | {``False``}
                Force-overwrite option; always returns ``False``
            *dvc*: {``True``} | ``False``
                Option to check for ``.dvc`` extension
        :Keys:
            * *MODULE_DIR*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fname* or DVC file exists
        :Versions:
            * 2021-07-19 ``@ddalle``: v1.0
        """
        # Check for force-overwrite option
        if f:
            # File does "not" exist (even if it does)
            return False
        # Get absolute path
        fabs  = self.get_abspath(fname)
        # Check if it exists
        if self._check_modfile(fabs):
            return True
        # Process option: whether or not to check for DVC file
        if not dvc:
            return False
        # Check if it exists
        if self._check_dvcfile(fabs):
            return True
        # No versions of file exist
        return False

    def check_modfile(self, fname):
        r"""Check if a file exists OR a ``.dvc`` version

        :Call:
            >>> q = dkl.check_modfile(fname)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file [optionally relative to *MODULE_DIR*]
        :Keys:
            * *MODULE_DIR*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fname* or DVC file exists
        :Versions:
            * 2021-07-19 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs  = self.get_abspath(fname)
        # Check if it exists
        return self._check_modfile(fabs)

    def check_dvcfile(self, fname, f=False):
        r"""Check if a file exists with appended``.dvc`` extension

        :Call:
            >>> q = dkl.check_dvcfile(fname)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file [optionally relative to *MODULE_DIR*]
        :Keys:
            * *MODULE_DIR*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fname* or DVC file exists
        :Versions:
            * 2021-07-19 ``@ddalle``: v1.0
        """
        # Get absolute path
        fabs  = self.get_abspath(fname)
        # Check if it exists
        return self._check_dvcile(fabs)

    def _check_modfile(self, fabs):
        # Check if it exists
        if os.path.isfile(fabs):
            # File exists
            return True
        elif os.path.isdir(fabs):
            # Problem!
            raise SystemError("Requested file '%s' is a folder!" % fabs)
        else:
            # File does not exist
            return False

    def _check_dvcfile(self, fabs):
        # Add the DVC suffix
        fdvc = fabs + ".dvc"
        # Check if it exists
        if os.path.isfile(fdvc):
            # File exists
            return True
        elif os.path.isdir(fdvc):
            # Problem!
            raise SystemError("Requested file '%s' is a folder!" % fdvc)
        else:
            # File does not exist
            return False

   # --- Python checks ---
    def _assert_filename(self, fname, name=None):
        r"""Assert type for a file name

        :Call:
            >>> dkl._assert_filename(fname, name=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of a file
            *name*: {``None``} | :class:`str`
                Optional name to use in error messsage
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Check type
        if not isinstance(fname, STR_TYPE):
            # Check for a variable name
            if name:
                raise TypeError(
                    "File name *%s* is '%s'; expected 'str'"
                    % (name, type(fname).__name__))
            else:
                raise TypeError(
                    "File name is '%s'; expected 'str'"
                    % type(fname).__name__)

    def _assert_filename_relative(self, fname, name=None):
        r"""Assert that a file name is not absolute

        :Call:
            >>> dkl._assert_filename_relative(fname, name=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of a file
            *name*: {``None``} | :class:`str`
                Optional name to use in error messsage
        :Versions:
            * 2021-07-07 ``@ddalle``: v1.0
        """
        # Check type
        if os.path.isabs(fname):
            # Check for a variable name
            if name:
                raise TypeError(
                    "File name *%s* is absolute; expected relative" % name)
            else:
                raise TypeError("File name is absolute; expected relative")
  # >

  # ==================
  # DATAKIT MAIN
  # ==================
  # <
   # --- Combined readers ---
    def read_db_mat(self, cls=None, **kw):
        r"""Read a datakit using ``.mat`` file type

        :Call:
            >>> db = dkl.read_db_mat(fname, cls=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-07-03 ``@ddalle``: v1.0
        """
        # Get full list of file names
        fnames = self.get_db_filenames_by_type("mat")
        # Combine option
        kw["cls"] = cls
        # Read those files
        for j, fname in enumerate(fnames):
            # Read with default options
            if j == 0:
                # Read initial database
                db = self.read_dbfile_mat(fname, **kw)
            else:
                # Get absolute path
                fmat = self.get_dbfile_mat(fname)
                # Use existing database
                db.read_mat(fmat)
        # Output
        return db

    def read_db_csv(self, cls=None, **kw):
        r"""Read a datakit using ``.csv`` file type

        :Call:
            >>> db = dkl.read_db_csv(fname, cls=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-07-03 ``@ddalle``: v1.0
        """
        # Get full list of file names
        fnames = self.get_db_filenames_by_type("csv")
        # Combine option
        kw["cls"] = cls
        # Read those files
        for j, fname in enumerate(fnames):
            # Read with default options
            if j == 0:
                # Read initial database
                db = self.read_dbfile_csv(fname, **kw)
            else:
                # Get absolute path
                fcsv = self.get_dbfile_csv(fname)
                # Use existing database
                db.read_csv(fcsv)
        # Output
        return db

   # --- Combined writers ---
    def write_db_csv(self, readfunc, f=True, db=None, **kw):
        r"""Write (all) canonical db CSV file(s)

        :Call:
            >>> db = dkl.write_db_csv(readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *cols*: {``None``} | :class:`list`
                If *dkl* has more than one file, *cols* must be a list
                of lists specifying which columns to write to each file
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2021-09-10 ``@ddalle``: v1.0
        """
        # File name for MAT
        fnames = self.get_dbfiles_csv()
        # Check for multiple
        if len(fnames) > 1:
            # Get column lists
            cols = kw.pop("cols", None)
            # Check
            if cols is None:
                raise ValueError(
                    ("Cannot write multiple CSV files w/o 'cols' kwarg,") +
                    ("a list of columns to write for each CSV file"))
        else:
            cols = kw.pop("cols", None)
        # Loop through files
        for j, fname in enumerate(fnames):
            # Get list of cols if needed
            if len(fnames) > 1:
                # Write columns for file *j*
                kw["cols"] = cols[j]
            else:
                # Write main list
                kw["cols"] = cols
            # Write file if needed
            db = self.write_dbfile_csv(fname, readfunc, f=f, db=db, **kw)
        # Return *db* in case read during process
        return db

    def write_db_mat(self, readfunc, f=True, db=None, **kw):
        r"""Write (all) canonical db MAT file(s)

        :Call:
            >>> db = dkl.write_db_mat(readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *cols*: {``None``} | :class:`list`
                If *dkl* has more than one file, *cols* must be a list
                of lists specifying which columns to write to each file
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2021-09-10 ``@ddalle``: v1.0
        """
        # File name for MAT
        fmats = self.get_dbfiles_mat()
        # Check for multiple
        if len(fmats) > 1:
            # Get column lists
            cols = kw.pop("cols", None)
            # Check
            if cols is None:
                raise ValueError(
                    ("Cannot write multiple MAT files w/o 'cols' kwarg,") +
                    ("a list of columns to write for each MAT file"))
        else:
            cols = kw.pop("cols", None)
        # Loop through files
        for j, fmat in enumerate(fmats):
            # Get list of cols if needed
            if len(fmats) > 1:
                # Write columns for file *j*
                kw["cols"] = cols[j]
            else:
                # Write main list
                kw["cols"] = cols
            # Write file if needed
            db = self.write_dbfile_mat(fmat, readfunc, f=f, db=db, **kw)
        # Return *db* in case read during process
        return db

    def write_db_xlsx(self, readfunc, f=True, db=None, **kw):
        r"""Write (all) canonical db XLSX file(s)

        :Call:
            >>> db = dkl.write_db_xlsx(readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *cols*: {``None``} | :class:`list`
                If *dkl* has more than one file, *cols* must be a list
                of lists specifying which columns to write to each file
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2022-12-14 ``@ddalle``: v1.0
        """
        # File name for XLSX
        fxlss = self.get_dbfiles_xlsx()
        # Check for multiple
        if len(fxlss) > 1:
            raise ValueError("Got %i XLS file names; expected 1" % len(fxlss))
        # Unpack file name
        fxls, = fxlss
        # Write file
        db = self.write_dbfile_xlsx(fxls, readfunc, f=f, db=db, **kw)
        # Return *db* in case read during process
        return db

   # --- Individual file writers ---
    def write_dbfile_csv(self, fcsv, readfunc, f=True, db=None, **kw):
        r"""Write a canonical db CSV file

        :Call:
            >>> db = dkl.write_dbfile_csv(fcsv, readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fscv*: :class:`str`
                Name of file to write
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2021-09-10 ``@ddalle``: v1.0
            * 2021-09-15 ``@ddalle``: Version 1.1; check for DVC stub
            * 2021-09-15 ``@ddalle``: Version 1.2; add *dvc* option
        """
        # DVC option
        dvc = kw.get("dvc", False)
        # Get DVC file name
        if fcsv.endswith(".dvc"):
            # Already a DVC stub
            fdvc = fcsv
        else:
            # Append ".dvc" extension
            fdvc = fcsv + ".dvc"
        # Check if it exists
        if f or not (os.path.isfile(fcsv) or os.path.isfile(fdvc)):
            # Read datakit from source
            if db is None:
                db = readfunc()
            # Create folders as needed
            self.prep_dirs(fcsv)
            # Write it
            db.write_csv(fcsv, **kw)
            # Process DVC
            if dvc or os.path.isfile(fdvc):
                # Add the file
                ierr = self.dvc_add(fcsv)
                if ierr:
                    print(
                        "Failed to dvc-add file '%s'"
                        % os.path.basename(fcsv))
                    return db
                # Push the file
                ierr = self.dvc_push(fcsv)
                if ierr:
                    print(
                        "Failed to dvc-push file '%s'"
                        % os.path.basename(fcsv))
        # Return *db* in case it was read during process
        return db

    def write_dbfile_mat(self, fmat, readfunc, f=True, db=None, **kw):
        r"""Write a canonical db MAT file

        :Call:
            >>> db = dkl.write_dbfile_mat(fmat, readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fmat*: :class:`str`
                Name of file to write
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2021-09-10 ``@ddalle``: v1.0
            * 2021-09-15 ``@ddalle``: Version 1.1; check for DVC stub
            * 2021-09-15 ``@ddalle``: Version 1.2; add *dvc* option
        """
        # DVC option
        dvc = kw.get("dvc", False)
        # Get DVC file name
        if fmat.endswith(".dvc"):
            # Already a DVC stub
            fdvc = fmat
        else:
            # Append ".dvc" extension
            fdvc = fmat + ".dvc"
        # Check if it exists
        if f or not (os.path.isfile(fmat) or os.path.isfile(fdvc)):
            # Read datakit from source
            if db is None:
                db = readfunc()
            # Create folders as needed
            self.prep_dirs(fmat)
            # Write it
            db.write_mat(fmat, **kw)
            # Process DVC
            if dvc or os.path.isfile(fdvc):
                # Add the file
                ierr = self.dvc_add(fmat)
                if ierr:
                    print(
                        "Failed to dvc-add file '%s'"
                        % os.path.basename(fmat))
                    return db
                # Push the file
                ierr = self.dvc_push(fmat)
                if ierr:
                    print(
                        "Failed to dvc-push file '%s'"
                        % os.path.basename(fmat))
        # Return *db* in case it was read during process
        return db

    def write_dbfile_xlsx(self, fxls, readfunc, f=True, db=None, **kw):
        r"""Write a canonical db XLSX file

        :Call:
            >>> db = dkl.write_dbfile_xlsx(fmat, readfunc, f=True, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fxlsx*: :class:`str`
                Name of file to write
            *readfunc*: **callable**
                Function to read source datakit if needed
            *f*: {``True``} | ``False``
                Overwrite *fmat* if it exists
            *db*: {``None``} | :class:`DataKit`
                Existing source datakit to write
            *dvc*: ``True`` | {``False``}
                Option to add and push data file using ``dvc``
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                If source datakit is read during execution, return it
                to be used in other write functions
        :Versions:
            * 2022-12-14 ``@ddalle``: v1.0
        """
        # DVC option
        dvc = kw.pop("dvc", False)
        # Get DVC file name
        if fxls.endswith(".dvc"):
            # Already a DVC stub
            fdvc = fxls
        else:
            # Append ".dvc" extension
            fdvc = fxls + ".dvc"
        # Check if it exists
        if f or not (os.path.isfile(fxls) or os.path.isfile(fdvc)):
            # Read datakit from source
            if db is None:
                db = readfunc()
            # Create folders as needed
            self.prep_dirs(fxls)
            # Write it
            db.write_xls(fxls, **kw)
            # Process DVC
            if dvc or os.path.isfile(fdvc):
                # Add the file
                ierr = self.dvc_add(fxls)
                if ierr:
                    print(
                        "Failed to dvc-add file '%s'"
                        % os.path.basename(fxls))
                    return db
                # Push the file
                ierr = self.dvc_push(fxls)
                if ierr:
                    print(
                        "Failed to dvc-push file '%s'"
                        % os.path.basename(fxls))
        # Return *db* in case it was read during process
        return db

   # --- Individual file readers ---
    def read_dbfile_mat(self, fname, **kw):
        r"""Read a ``.mat`` file from *DB_DIR*

        :Call:
            >>> db = dkl.read_dbfile_mat(fname, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``"mat"``} | ``None`` | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
        """
        # Set default file type
        kw.setdefault("ftype", "mat")
        # Read from db/ folder
        return self.read_dbfile(fname, "mat", **kw)

    def read_dbfile_csv(self, fname, **kw):
        r"""Read a ``.mat`` file from *DB_DIR*

        :Call:
            >>> db = dkl.read_dbfile_mat(fname, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``"mat"``} | ``None`` | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
        """
        # Set default file descriptor
        kw.setdefault("ftype", "csv")
        # Read from db/ folder
        return self.read_dbfile(fname, "csv", **kw)

    def read_dbfile_csv_rbf(self, fname, **kw):
        r"""Read a ``.mat`` file from *DB_DIR*

        :Call:
            >>> db = dkl.read_dbfile_mat(fname, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``"mat"``} | ``None`` | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
        """
        # Set default file descriptor
        kw.setdefault("ftype", "csv")
        # Read from db/ folder
        return self.read_dbfile(fname, "csv", **kw)


    def read_dbfile(self, fname, ext, **kw):
        r"""Read a databook file from *DB_DIR*

        :Call:
            >>> db = dkl.read_dbfile_mat(self, ext, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: ``None`` | :class:`str`
                Name of file to read from raw data folder
            *ext*: :class:`str`
                Database file type
            *ftype*: {``"mat"``} | ``None`` | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
        :Keys:
            * *MODULE_DIR*
            * *DB_DIR*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
        """
        # Absolute file name
        fabs = self.get_dbfile(fname, ext)
        # Read that file
        return self._read_dbfile(fabs, **kw)
        
    def read_rawdatafile(self, fname, ftype=None, cls=None, **kw):
        r"""Read a file from the *RAW_DATA* folder

        :Call:
            >>> db = dkl.read_rawdatafile(fname, ftype=None, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :See Also:
            * :func:`get_rawdatafilename`
        :Versions:
            * 2021-06-25 ``@ddalle``: v1.0
            * 2021-07-07 ``@ddalle``: Version 1.1
                - use :func:`get_rawdatafilename`
        """
        # Absolute file name
        fabs = self.get_rawdatafilename(fname)
        # Read that file
        return self._read_dbfile(fabs, ftype=ftype, cls=cls, **kw)

    def _read_dbfile(self, fabs, ftype=None, cls=None, **kw):
        r"""Read a file using specified DataKit class

        :Call:
            >>> db = dkl._read_dbfile(fabs, ftype=None, cls=None, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
            *dvc*: {``True``} | ``False``
                Option to pull DVC file where *fabs* doesn't exist
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-28 ``@ddalle``: v1.0
            * 2021-09-23 ``@ddalle``: Version 1.1; check ``dvc status``
        """
        # Default class
        if cls is None:
            cls = self.get_option("DATAKIT_CLS")
        # Option: whether or not to check for DVC files
        dvc = kw.get("dvc", True)
        # Check for DVC file
        if dvc and self._check_dvcfile(fabs):
            # Name of DVC file
            fdvc = fabs + ".dvc"
            # Check status
            if not os.path.isfile(fabs):
                # No main file; just pull
                self.dvc_pull(fabs, **kw)
            elif os.path.getmtime(fabs) > os.path.getmtime(fdvc):
                # No reason to check status
                pass
            elif self.dvc_status(fabs):
                # Pull it
                self.dvc_pull(fabs, **kw)
        # Check if file exists
        if not self._check_modfile(fabs):
            # No such file
            raise NOFILE_ERROR("No file '%s' found" % fabs)
        # Check for user-specified file type
        if ftype is None:
            # Let *cls* determine the file type
            return cls(fabs, **kw)
        else:
            # Set additional keyword arg
            kw[ftype] = fabs
            # Read the file using *ftype* kwarg
            return cls(**kw)
        

   # --- Read/write attributes ---
    def get_dbdir_by_type(self, ext):
        r"""Get datakit directory for given file type

        :Call:
            >>> dkl.get_db_dir_by_type(ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *ext*: :class:`str`
                File extension type
        :Keys:
            * *MODULE_DIR*
            * *DB_DIR*
            * *DB_DIRS_BY_TYPE*
        :Outputs:
            *fdir*: :class:`str`
                Absolute path to *ext* datakit folder
        :Versions:
            * 2021-06-29 ``@ddalle``: v1.0
        """
        # Dictionary of db folders for each file format
        dbtypedirs = self.get_option("DB_DIRS_BY_TYPE", {})
        # Get option for specified file type
        return dbtypedirs.get(ext, ext)

    def get_db_suffixes_by_type(self, ext):
        r"""Get list of suffixes for given data file type

        :Call:
            >>> suffixes = dkl.get_db_suffixes_by_type(ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *ext*: :class:`str`
                File extension type
        :Keys:
            * *DB_SUFFIXES_BY_TYPE*
        :Outputs:
            *suffixes*: :class:`list`\ [:class:`str` | ``None``]
                List of additional suffixes (if any) for *ext* type
        :Versions:
            * 2021-07-01 ``@ddalle``: v1.0
        """
        # Dictionary of db suffixes for each file format
        suffixdict = self.get_option("DB_SUFFIXES_BY_TYPE", {})
        # Get suffixes for this type
        suffixes = suffixdict.get(ext)
        # Check for any
        if suffixes is None:
            # Return list of just ``None``
            return [None]
        elif not suffixes:
            # Use list of ``None`` for any empty suffixes
            return [None]
        elif isinstance(suffixes, (list, tuple)):
            # Already a list
            return suffixes
        else:
            # Convert single suffix to list
            return [suffixes]
        
    def get_db_filenames_by_type(self, ext):
        r"""Get list of file names for a given data file type

        :Call:
            >>> fnames = dkl.get_db_filenames_by_type(ext)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *ext*: :class:`str`
                File extension type
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of datakit file names; one for each suffix
        :Versions:
            * 2021-07-01 ``@ddalle``: v1.0
        """
        # Get database name
        dbname = self.make_db_name()
        # Get list of suffixes for database files
        suffixes = self.get_db_suffixes_by_type(ext)
        # Initialize list of files
        fnames = []
        # Loop through suffixes
        for suffix in suffixes:
            # Construct full file name
            if suffix is None:
                # No suffix
                fname = "%s.%s" % (dbname, ext)
            else:
                # Add a suffix
                fname = "%s-%s.%s" % (dbname, suffix, ext)
            # Save absolute file name
            fnames.append(fname)
        # Output
        return fnames
  # >


# Convert to list
def _listify(v):
    r"""Convert scalar or ``None`` to :class:`list`

    :Call:
        >>> V = _listify(V)
        >>> V = _listify(v)
        >>> V = _listify(vnone)
    :Inputs:
        *V*: :class:`list` | :class:`tuple`
            Preexisting :class:`list` or :class:`tuple`
        *vnone*: ``None``
            Empty input of ``None``
        *v*: **scalar**
            Anything else
    :Outputs:
        *V*: :class:`list`
            Enforced list, either

            * ``V`` --> ``V``
            * ``V`` --> ``list(V)``
            * ``None`` --> ``[]``
            * ``v`` --> ``[v]``
    :Versions:
        * 2021-08-18 ``@ddalle``: v1.0
    """
    # Check type
    if isinstance(v, list):
        # Return it
        return v
    elif isinstance(v, tuple):
        # Just convert it
        return list(v)
    elif v is None:
        # Empty list
        return []
    else:
        # Create singleton
        return [v]

