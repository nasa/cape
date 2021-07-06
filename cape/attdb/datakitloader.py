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
from ..tnakit import typeutls


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
   # --- File names and prep ---
    def genr8_abs_filename(self, frel):
        r"""Generate full filename from path relative to *MODULE_DIR*

        :Call:
            >>> fabs = dkl.genr8_abs_filename(frel)
            >>> fabs = dkl.genr8_abs_filename(fabs)
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
        if not typeutils.isstr(frel):
            raise TypeError(
                "Relative file name *frel* must be 'str'; got '%s'"
                % type(frel).__name__)
        # Check for absolute file
        if os.path.isabs(frel):
            # Already absolute
            return frel
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        # Return full path to file name
        return os.path.join(fdir, frel)

    def prep_module_filename(self, frel):
        # Check file name
        if not typeutils.isstr(frel):
            raise TypeError(
                "Relative file name *frel* must be 'str'; got '%s'"
                % type(frel).__name__)
        # Check for absolute file
        if os.path.isabs(frel):
            raise ValueError(
                "Relative file name *frel* '%s' is absolute" % frel)
        # Create list of individual dirs
        fdirs = frel.split(os.sep)
        # Create folders
        self.mkdirs(fdirs)
        # Return absolute file name
        return self.genr8_abs_filename(frel)

   # --- Supporting ---
    def mkdirs(self, fdirs):
        r"""Ensure folders exist

        :Call:
            >>> dkl.mkdirs(fdirs)
        :Inputs:
            *fdirs*: :class:`list`\ [:class:`str`]
                List of folder names relative to *MODULE_DIR*
        :Versions:
            * 2021-06-28 ``@ddalle``: Version 1.0
            * 2021-07-05 ``@ddalle``: Version 2.0; list of dirs
        """
        # Get module dir
        moddir = self.get_option("MODULE_DIR")
        # Initialize dir name ad *moddir* level
        fdir = moddir
        # Loop through relative folder names (cumulatively)
        for fdiri in fdirs:
            # Skip empty folders
            if not fdiri:
                continue
            # Combine path cumulatively
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
            *fname*: :class:`str`
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
        # Get top-level and relative raw-data folder
        moddir = self.get_option("MODULE_DIR")
        dbsdir = self.get_option("DB_DIR")
        # Get folder for dbs of this type
        dbtypedir = self.get_db_typedir(ext)
        # Full path to raw data
        fdir = os.path.join(moddir, dbsdir, dbtypedir)
        # File name
        fabs = os.path.join(fdir, fname)
        # Read that file
        return self._read_dbfile(fabs, **kw)
        
    def read_rawdata(self, fname, ftype=None, cls=None, **kw):
        r"""Read a file from the *RAW_DATA* folder

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
                Class to read *fname* other than *dkl["DATAKIT_CLS"]*
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

   # --- Write individual files ---
    def _write_dbfile(self, db, frel, ext=None, **kw):
        r"""Write a datakit file to specified relative file name

        :Call:
            >>> dkl._write_dbfile(db, frel, ext=None, **kw)
        :Inputs:
            *frel*: :class:`str`
                File name to write relative to *MODULE_DIR*
            *db*: ``None`` | :class:`DataKit`
                Instance of DataKit to write from
            *ext*: {``None``} | :class:`str`
                File type, or guess from *frel* extension
            *func*: {``None``} | **callable**
                Function/method to use for writing; overrides *ext*
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2021-07-05 ``@ddalle``: Version 1.0
        """
        # Get function option
        fn = kw.get("func")
        # Default extension
        if ext is None:
            # Get extension from file name
            ext = frel.split(".")[-1]
        # Check *ext* type
        if not typeutils.isstr(ext):
            raise ValueError(
                "Input *ext* must be 'str'; got '%s'" % type(ext).__name__)
        # Normalize *ext* variable
        _ext = ext.lower().replace("_", "-")
        # Get function
        if fn is not None:
            # Function already found
            pass
        elif _ext == "csv":
            # Use CSV writer
            fn = db.write_csv
        elif _ext == "csv-dense":
            # Use simple CSV writer
            fn = db.write_csv_dense
        elif _ext == "mat":
            # Use MAT file
            fn = db.write_mat
        elif _ext == "rbf-csv":
            # Write special RBF CSV file
            fn = db.write_rbf_csv
        elif _ext == "tsv":
            # Write tab-separated values
            fn = db.write_tsv
        elif _ext == "tsv-dense":
            # Write simple TSV
            fn = db.write_tsv_dense
        elif _ext in {"xls", "xlsx"}:
            # Write Excel spreadsheet
            fn = db.write_xls
        else:
            raise ValueError("Unknown file type '%s'" % ext)
        # Normalize file name
        if os.path.isabs(frel):
            # Already absolute ... do nothing
            fabs = frel
        else:
            # Get folder portion
            fdir = os.path.dirname(frel)
            # Base dir
            moddir = self.get_option("MODULE_DIR")
            # Absolutize
            fabs = os.path(moddir, frel.replace("/", os.sep))
            # Create folders as needed
            fdirs = fdir.split(os.sep)
            self._mkdirs(*fdirs)
        # Write the file
        fn(fabs, **kw)
        

   # --- Read/write attributes ---
    def get_db_typedir(self, ext):
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
            fnames.append(fname)
        # Output
        return fnames
  # >

