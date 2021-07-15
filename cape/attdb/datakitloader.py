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
import sys

# Local modules
from .rdb import DataKit
from ..tnakit import kwutils


# Utility regular expressions
REGEX_INT = re.compile("[0-9]+")

# Create types for "strings" based on Python version
if sys.version_info.major == 2:
    # Allow unicode
    STR_TYPE = (str, unicode)
else:
    # Just string (which are unicode in Python 3.0+)
    STR_TYPE = str


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
        "DB_NAME_REGEX",
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
        "DB_NAME_REGEX": str,
        "DB_NAME_REGEX_GROUPS": dict,
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
  # DUNDER METHOD
  # ===============
  # <
    # Initialization method
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
   # --- Create names ---
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
        r"""Expand regular expression strings for module name

        This expands things like ``%(group1)s`` to something
        like ``?P<group1>[1-9][0-9]``.

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
            * 2021-07-07 ``@ddalle``: Version 1.0
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
            * 2021-07-07 ``@ddalle``: Version 1.0
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
            * 2021-07-07 ``@ddalle``: Version 1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        dbsdir = self.get_option("DB_DIR")
        # Get folder for dbs of this type
        dbtypedir = self.get_dbdir_by_type(ext)
        # Combine directories
        return os.path.join(moddir, dbsdir, dbtypedir)

   # --- Raw data files ---
    def get_rawdatafilename(self, fname):
        r"""Get a file name relative to the datakit folder

        :Call:
            >>> fabs = dkl.get_rawdatafilename(fname)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *fname*: ``None`` | :class:`str`
                Name of file relative to *DB_DIRS_BY_TYPE* for *ext*
        :Outputs:
            *fabs*: :class:`str`
                Absolute path to raw data file
        :Keys:
            * *MODULE_DIR*
            * *RAWDATA_DIR*
        :Versions:
            * 2021-07-07 ``@ddalle``: Version 1.0
        """
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        rawdir = self.get_option("RAWDATA_DIR")
        # Full path to raw data
        fdir = os.path.join(moddir, rawdir)
        # Return absolute
        return os.path.join(fdir, fname)

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
            * 2021-07-08 ``@ddalle``: Version 1.0
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
            * 2021-07-05 ``@ddalle``: Version 1.0
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
        return os.path.join(fdir, frel)

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
            * 2021-07-07 ``@ddalle``: Version 1.0
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

   # --- Checks ---
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
            * 2021-07-07 ``@ddalle``: Version 1.0
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
            * 2021-07-07 ``@ddalle``: Version 1.0
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
            >>> db = dkl.read_rawdata(fname, cls=None)
        :Inputs:
            *dkl*: :class:`DataKitLoader`
                Tool for reading datakits for a specific module
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
        :Outputs:
            *db*: *dkl["DATAKIT_CLS"]* | *cls*
                DataKit instance read from *fname*
        :Versions:
            * 2021-07-03 ``@ddalle``: Version 1.0
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
                # Use existing database
                db.read_mat(fname)
        # Output
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
            * 2021-06-25 ``@ddalle``: Version 1.0
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
            * 2021-06-25 ``@ddalle``: Version 1.0
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
            * 2021-06-25 ``@ddalle``: Version 1.0
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
            * 2021-06-25 ``@ddalle``: Version 1.0
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
            * 2021-06-29 ``@ddalle``: Version 1.0
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
            * 2021-07-01 ``@ddalle``: Version 1.0
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
            * 2021-07-01 ``@ddalle``: Version 1.0
        """
        # Full path to raw data
        fdir = self.get_dbdir_by_type(ext)
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

