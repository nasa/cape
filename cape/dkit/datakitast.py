r"""
:mod:`cape.dkit.datakitast`: DataKit assistant module
=========================================================

This module provides the class :class:`DataKitAssistant`, which provides
useful tools for a DataKit module. It helps identify a DataKit module
by its module name, manages subfolders like ``rawdata/``, and helps read
other DataKits in the same repository (DataKit collection).
"""

# Standard library
import inspect
import os
import re
import sys
from typing import Optional

# Third-party

# Local imports
from .rdb import DataKit
from ..optdict import OptionsDict


# Utility regular expressions
REGEX_INT = re.compile("[0-9]+$")
REGEX_HOST = re.compile(r"((?P<host>[A-z][A-z0-9.]+):)?(?P<path>[\w./-]+)$")
REGEX_REMOTE = re.compile(r"((?P<host>[A-z][A-z0-9.]+):)(?P<path>[\w./-]+)$")


# Primary class
class DataKitAssistant(OptionsDict):
  # === Class attributes ===
    # Attributes
    __slots__ = (
        "rawdata_sources",
        "rawdata_remotes",
        "rawdata_commits",
        "rawdata_sources_commit",
    )

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
        "DB_DIRS_BY_TYPE": str,
        "DB_NAME": str,
        "DB_NAME_REGEX_LIST": str,
        "DB_NAME_REGEX_GROUPS": dict,
        "DB_NAME_REGEX_INT_GROUPS": str,
        "DB_NAME_REGEX_STR_GROUPS": str,
        "DB_NAME_TEMPLATE_LIST": str,
        "DB_SUFFIXES_BY_TYPE": dict,
        "MODULE_DIR": str,
        "MODULE_FILE": str,
        "MODULE_NAME": str,
        "MODULE_NAME_REGEX_LIST": str,
        "MODULE_NAME_REGEX_GROUPS": dict,
        "MODULE_NAME_REGEX_INT_GROUPS": str,
        "MODULE_NAME_REGEX_STR_GROUPS": str,
        "MODULE_NAME_TEMPLATE_LIST": str,
    }

    # Required lists
    _optlistdepth = {
        "DB_DIRS_BY_TYPE": 1,
        "DB_NAME_REGEX_LIST": 1,
        "DB_NAME_REGEX_INT_GROUPS": 1,
        "DB_NAME_REGEX_STR_GROUPS": 1,
        "DB_NAME_TEMPLATE_LIST": 1,
        "MODULE_NAME_REGEX_LIST": 1,
        "MODULE_NAME_REGEX_INT_GROUPS": 1,
        "MODULE_NAME_REGEX_STR_GROUPS": 1,
        "MODULE_NAME_TEMPLATE_LIST": 1,
    }

    # Default values
    _rc = {
        "DATAKIT_CLS": DataKit,
        "DB_DIR": "db",
        "DB_DIRS_BY_TYPE": [],
        "DB_NAME_REGEX": ".+",
        "DB_NAME_REGEX_GROUPS": {},
        "DB_NAME_REGEX_INT_GROUPS": (),
        "DB_NAME_REGEX_STR_GROUPS": (),
        "DB_NAME_TEMPLATE_LIST": ["datakit"],
        "DB_NAME": None,
        "MODULE_NAME_REGEX_LIST": [".+"],
        "MODULE_NAME_REGEX_GROUPS": {},
        "MODULE_NAME_REGEX_INT_GROUPS": (),
        "MODULE_NAME_REGEX_STR_GROUPS": (),
        "RAWDATA_DIR": "rawdata",
    }

  # === __dunder__ ===
    def __init__(
            self,
            name: Optional[str] = None,
            fname: Optional[str] = None, **kw):
        # Initialize attributes
        self.rawdata_sources = {}
        self.rawdata_remotes = {}
        self.rawdata_commits = {}
        self.rawdata_sources_commit = {}
        # Get name of calling function
        caller_frame = sys._getframe(1)
        caller_func = caller_frame.f_code
        # Get module file name
        modfile = caller_func.co_filename
        # Get module of calling function
        mod = inspect.getmodule(caller_frame)
        modname = mod.__name__
        # Apply defaults
        name = modname if name is None else name
        fname = modfile if fname is None else fname
        # Calling folder
        fdir = os.path.dirname(fname)
        # Process options
        OptionsDict.__init__(self, **kw)
        # Set options
        self.set_opt("MODULE_NAME", name)
        self.set_opt("MODULE_FILE", fname)
        self.set_opt("MODULE_DIR", fdir)
        # Get database (datakit) NAME
        self.create_db_name()

    def __str__(self) -> str:
        # Name of class
        clsname = self.__class__.__name__
        # Get base of module name
        name = self.get_opt("MODULE_NAME", vdef="datakit").rsplit('.', 1)[-1]
        # Get database name
        dbname = self.get_opt("DB_NAME")
        # Form the string
        return f"{clsname}({name}, '{dbname}')"

    def __repr__(self) -> str:
        return self.__str__()

  # === MODULE_NAME --> DB_NAME ===
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
        self.set_opt("DB_NAME", dbname)

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
        self.set_opt("DB_NAME", dbname)

    def genr8_db_name(self, modname: Optional[str] = None) -> str:
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
            * 2021-07-15 ``@ddalle``: v1.1; add *modname* arg
        """
        # Get list of regexes
        modname_regexes = self._genr8_modname_regexes()
        # Get format lists
        dbname_templates = self.get_opt("DB_NAME_TEMPLATE_LIST")
        # Module name
        if modname is None:
            # Use default; this module
            modname = self.get_opt("MODULE_NAME")
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
    def genr8_modnames(self, dbname: Optional[str] = None) -> list:
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
        modname_templates = self.get_opt("MODULE_NAME_TEMPLATE_LIST")
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
    def _genr8_modname_match_groups(self, regex: str, modname: str) -> dict:
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
        try:
            match = re.match(regex + "$", modname)
        except Exception:
            print(f"  Invalid regular expression '{regex}'")
            return None
        # Check for no match
        if match is None:
            return None
        # Get dictionary from matches
        groupdict = match.groupdict()
        # Names of groups that should always be integers or strings
        igroups = self.get_opt("MODULE_NAME_REGEX_INT_GROUPS")
        sgroups = self.get_opt("MODULE_NAME_REGEX_STR_GROUPS")
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

    def _genr8_modname_regexes(self) -> list:
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
        grps = self.get_opt("MODULE_NAME_REGEX_GROUPS")
        # Add full formatting for regular expression group
        grps_re = {
            k: "(?P<%s>%s)" % (k, v)
            for k, v in grps.items()
        }
        # Get regular expression list
        name_list = self.get_opt("MODULE_NAME_REGEX_LIST")
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

