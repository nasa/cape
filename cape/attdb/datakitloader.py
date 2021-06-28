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
import os
import re

# Local modules
from .rdb import DataKit
from ..tnakit import kwutils


# Utility regular expressions
REGEX_INT = re.compile("[0-9]+")


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
        "DB_NAME_TEMPLATE_LIST",
        "DB_SUFFIXES_BY_TYPE",
        "MODULE_DIR",
        "MODULE_FILE",
        "MODULE_NAME",
        "MODULE_NAME_REGEX_LIST",
        "MODULE_NAME_REGEX_GROUPS",
        "MODULE_NAME_REGEX_INT_GROUPS",
        "MODULE_NAME_REGEX_STR_GROUPS"
    }

    # Types
    _opttypes = {
        "DATAKIT_CLS": type,
        "DB_DIR": str,
        "DB_DIRS_BY_TYPE": (list, tuple),
        "DB_NAME": str,
        "DB_NAME_TEMPLATE_LIST": (list, tuple),
        "DB_SUFFIXES_BY_TYPE": dict,
        "MODULE_DIR": str,
        "MODULE_FILE": str,
        "MODULE_NAME": str,
        "MODULE_NAME_REGEX_LIST": (list, tuple),
        "MODULE_NAME_REGEX_GROUPS": dict,
        "MODULE_NAME_REGEX_INT_GROUPS": (list, tuple, set),
        "MODULE_NAME_REGEX_STR_GROUPS": (list, tuple, set),
    }

    # Default values
    _rc = {
        "DATAKIT_CLS": DataKit,
        "DB_DIR": "db",
        "DB_DIRS_BY_TYPE": {},
        "DB_NAME_FORMATS": ["datakit"],
        "DB_NAME": None,
        "MODULE_NAME_REGEX_LIST": [".+"],
        "MODULE_NAME_REGEX_GROUPS": {},
        "MODULE_NAME_REGEX_INT_GROUPS": set(),
        "MODULE_NAME_REGEX_STR_GROUPS": set(),
        "RAWDATA_DIR": "rawdata",
    }
  # >

  # ===============
  # DUNDER METHOD
  # ===============
  # <
    def __init__(self, name, fname, **kw):
        r"""Initialization method

        :Versions:
            * 2021-06-25 ``@ddalle``: Version 1.0
        """
        # Process keyword options
        kwutils.KwargHandler.__init__(self, **kw)
        # Use required inputs
        self.setdefault_option("MODULE_NAME", name)
        self.setdefault_option("MODULE_FILE", os.path.abspath(fname))
        # Set name of folder containing data
        self.set_option("MODULE_DIR", os.path.dirname(self["MODULE_FILE"]))
        # Get database (datakit) NAME
        self.create_db_name()
  # >

  # =================
  # DATAKIT NAME
  # =================
  # <
    def make_db_name(self):
        r"""Retrieve or create database name from module name

        This utilizes the following parameters:

        * *MODULE_NAME_REGEX_LIST*
        * *MODULE_NAME_REGEX_GROUPS*
        * *DB_NAME_FORMATS*

        :Call:
            >>> dbname = dkl.make_db_name()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
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
        * *DB_NAME_FORMATS*

        :Call:
            >>> dbname = dkl.create_db_name()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
        """
        # Generate the database name
        dbname = self.genr8_db_name()
        # Save it
        self.set_option("DB_NAME", dbname)

    def genr8_db_name(self):
        r"""Get database name based on first matching regular expression

        This utilizes the following parameters:

        * *MODULE_NAME_REGEX_LIST*
        * *MODULE_NAME_REGEX_GROUPS*
        * *DB_NAME_FORMATS*

        :Call:
            >>> dbname = dkl.genr8_db_name()
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
        :Outputs:
            *dbname*: :class:`str`
                Prescribed datakit name
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
        """
        # Get list of regexes
        modname_regexes = self._genr8_modname_regexes()
        # Get format lists
        dbname_templates = self.get_option("DB_NAME_TEMPLATE_LIST")
        # Module name
        modname = self.get_option("MODULE_NAME")
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
                print("Failed to expand DB_NAME_TEPLATE_LIST %i:" % (i+1))
                print("  template: %s" % dbname_template)
                print("  groups:")
                # Print all groups
                for k, v in grps.items():
                    print("%12s: %s [%s]" % (k, v, type(v).__name__))
                # Raise an exception
                raise KeyError(
                    "Failed to expand DB_NAME_TEPLATE_LIST %i" % (i+1))
            # Exit loop
            break
        else:
            # No match found; use global default
            dbname = "datakit"
        # Also output it
        return dbname
  # >
  
  # ==================
  # FILE?FOLDER
  # ==================
  # <
    def _mkdirs(*fdirs):
        r"""Ensure folders exist

        :Call:
            >>> dkl._mkdirs(fdir1, fdir2, ...)
        :Inputs:
            *fdir1*: :class:`str`
                Ensure folder named *fdir1* exists in *MODULE_DIR*
            *fdir2*: :class:`str`
                Ensure folder ``os.path.join(fdir1, fdir2)`` exists
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
        """
        # Get module dir
        moddir = self.get_option("MODULE_DIR")
        # Initialize dir name ad *moddir* level
        fdir = moddir
        # Loop through relative folder names (cumulatively)
        for fdiri in fdirs:
            # Combine path
            fdir = os.path.join(fdir, fdiri)
            # Test if folder exists
            if os.path.isdir(fdir):
                # Already exists; go to next one
                continue
            elif os.path.isfile(fdir):
                # File with same name exists!
                raise ValueError(
                    "Cannot create folder '%s'; file with same name exists"
                    % fdir)
            else:
                # Create folder
                os.mkdir(fdir)
  # >

  # ==================
  # DATAKIT MAIN
  # ==================
    def get_db_dir_by_type(self, ext):
        # Top level dir
        moddir = self.get_option("MODULE_DIR")
        # Name for folder containing all datakits
        dbdir = self.get_option("DB_DIR")
        # Dictionary of db folders for each file format
        typdirs = self.get_option("DB_DIRS_BY_TYPE", {})
        # Get option for specified file type
        typdir = typdirs.get(ext, ext)
        # Output
        return os.path.join(moddir, dbdir, typdir)

    def get_db_suffixes_by_type(self, ext):
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
        
    def get_db_files_by_type(self, ext):
        # Full path to raw data
        fdir = self.get_db_dir_by_type(ext)
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
            fnames.append(os.path.join(fdir, fname))
        # Output
        return fnames
        
        
    def read_db_mat(self, ftype=None, cls=None, **kw):
        r"""Read a file from the RAW_DATA folder

        :Call:
            >>> db = dkl.read_rawdata(fname, ftype=None, cls=None, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl.DATAKIT_CLS*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: Version 1.0
        """
        # File name
        fabs = self.get_db_filename_by_type("mat")
        # Read that file
        return self._read_db(fabs, ftype=None, cls=None, **kw)
        
    def read_rawdata(self, fname, ftype=None, cls=None, **kw):
        r"""Read a file from the RAW_DATA folder

        :Call:
            >>> db = dkl.read_rawdata(fname, ftype=None, cls=None, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl.DATAKIT_CLS*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-25 ``@ddalle``: Version 1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        rawdir = self.get_option("RAWDATA_DIR")
        # Full path to raw data
        fdir = os.path.join(moddir, rawdir)
        # File name
        fabs = os.path.join(fdir, fname)
        # Read that file
        return self._read_db(fabs, ftype=None, cls=None, **kw)

    def _genr8_modname_match_groups(self, regex, modname):
        # Attempt to match regex (all of *modname*)
        match = re.fullmatch(regex, modname)
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
            elif REGEX_INT.fullmatch(val):
                # Convertible to integer
                groups[grp] = int(val)
                # Resave raw value
                groups["s-" + grp] = val
        # Output
        return groups

    def _genr8_modname_regexes(self):
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

    def _read_db(self, fabs, ftype=None, cls=None, **kw):
        r"""Read a file using specified DataKit class

        :Call:
            >>> db = dkl._read_db(fabs, ftype=None, cls=None, **kw)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl.DATAKIT_CLS*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
        """
        # Default class
        if cls is None:
            cls = self.get_option("DATAKIT_CLS")
        # Check for user-specified file type
        if ftype is None:
            # Let *cls* determine the file type
            return cls(fabs, **kw)
        else:
            # Set additional keyword arg
            kw[ftype] = fabs
            # Read the file using *ftype* kwarg
            return cls(**kw)

