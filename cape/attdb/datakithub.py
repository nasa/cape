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
        hub = DataKitHub()

This will look for a file

    ``data/datakithub/datakithub.json``

in the current folder and each parent folder.

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
    *module_attribute*: :class:`str` | :class:`list` | ``None``
        Name of variable(s) in imported module to use as datakit
    *module_function*: :class:`str` | :class:`list` | ``None``
        Name of function(s) from imported module that return datakit
    *module_regex*: :class:`dict`\ [:class:`str`]
        Rules for converting a regular expression to module names

"""

# Standard library modules
import importlib
import os
import re
import sys

# CAPE modules
from ..cfdx.options.util import loadJSONFile

# Version-dependent standard library
if sys.version_info.major > 2:
    # Import the reload() function
    from importlib import reload


# Defaults
DEFAULT_REPO = None
DEFAULT_TYPE = "module"
DEFAULT_ATTRIBUTE = None
DEFAULT_FUNCTION = ["read_db"]
# Combined efaults
DEFAULT_SECTION = {
    "repo": DEFAULT_REPO,
    "type": DEFAULT_TYPE,
    "module_attribute": DEFAULT_ATTRIBUTE,
    "module_function": DEFAULT_FUNCTION,
}

# Error codes
ERROR_NOMATCH_SECTION = 1
ERROR_NOMATCH_DBNAME = 2
ERROR_IMPORT_ERROR = 3
ERROR_NO_ATTRIBUTE = 4
ERROR_NO_FUNCTION = 5
ERROR_FUNCTION_CALL = 6
ERROR_FUNCTION_NONE = 7

# Error messages
ERRMSG_NOMATCH_SECTION = "Section '%s' does not match db name (ierr=1)"
ERRMSG_NOMATCH_DBNAME =  "  Regex '%s' does not match db name (ierr=2)"
ERRMSG_IMPORT_ERROR =    "    Import module '%s' failed (ierr=3)"
ERRMSG_NO_ATTRIBUTE =    "    No attribute '%s.%s' (ierr=4)"
ERRMSG_NO_FUNCTION =     "    No function '%s.%s()' (ierr=5)"
ERRMSG_FUNCTION_CALL =   "    Exception while calling '%s.%s()' (ierr=6)"
ERRMSG_FUNCTION_NONE =   "    Function '%s.%s()' returned None (ierr=7)"
# Success messages
MSG_SECTION = "Section '%s'"
MSG_DBNAME = "  Regex '%s' matches"
MSG_IMPORT = "    Import module '%s' succeeded"
MSG_READDB_ATTR = "    Datakit from attribute '%s.%s'"
MSG_READDB_FUNC = "    Datakit from function '%s.%s()'"


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
        * 2021-08-19 ``@ddalle``: Version 2.0
            - simpler search for JSON file
            - similar to how git finds ``.git`` folder
            - better regular expression support
            - can try multiple sections if one matches but fails
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
        # Initialize dictionary of known groups
        self.regex_groups = {}
        # Initialize fixed attributes
        self.datakit_modules = {}
        # Save folder containing *fjson*
        self.file_json = os.path.basename(fabs)
        self.dir_json = os.path.dirname(fabs)
        self.dir_root = os.path.dirname(self.dir_json)
        # Read the JSON file
        opts = loadJSONFile(fabs)
        # Save regex groups definition
        self.regex_groups = opts.pop(".regex_groups", {})
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
    def get_section(self, sec):
        r"""Get options for specified module section

        :Call:
            >>> secopts = hub.get_section(sec)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *sec*: :class:`str`
                Name of datakit section
        :Outputs:
            *secopts*: :class:`dict`
                Options for *sec* loaded in *hub[sec]*
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Get options from dict
        return self.get(sec, {})

    # Get option from a specific section
    def get_section_opt(self, sec, opt, vdef=None):
        r"""Get the *type* of a given datakit group

        :Call:
            >>> v = hub.get_section_opt(grp, opt, vdef=None)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *sec*: :class:`str`
                Name of datakit section
            *opt*: :class:`str`
                Name of option to access
            *vdef*: {``None``} | **any**
                Default value for *opt*
        :Outputs:
            *v*: {*vdef*} | 
                Value of *hub[grp][opt]* or *vdef*
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
            * 2021-08-18 ``@ddalle``: Version 1.1
                - was :func:`get_group_opt`
                - add module-level defaults
        """
        # Get group options
        secopts = self.get_section(sec)
        # Process default
        if vdef is None:
            # Use global default
            vdef = DEFAULT_SECTION.get(opt)
        # Check type
        if not isinstance(secopts, dict):
            # Use default if seciton is not a dict
            return vdef
        # Get type
        return secopts.get(opt, vdef)

    # Get repo
    def get_section_type(self, sec):
        r"""Get *type* option for section

        :Call:
            >>> sectype = hub.get_section_type(sec)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *sec*: :class:`str`
                Name of datakit section
        :Outputs:
            *sectype*: :class:`str`
                Name of folder to add to path
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        return self.get_section_opt("type")

    # Get repo
    def get_section_repo(self, sec):
        r"""Get *repo* option for section

        :Call:
            >>> repo = hub.get_section_repo(sec)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *sec*: :class:`str`
                Name of datakit section
        :Outputs:
            *repo*: ``None`` | :class:`dict`
                Name of folder to add to path
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        return self.get_section_opt("repo")

   # --- Read DB ---
    # Get datakit by name
    def read_db(self, dbname, **kw):
        r"""Read a datakit based on DB name

        :Call:
            >>> db = hub.read_db(dbname, **kw)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                Data interface if successful
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
            * 2021-08-19 ``@ddalle``: Version 2.0
                - better regex and fallback support
                - verbosity options
                - calls :func:`read_dbname`
        """
        # Use read_dbname()
        return self.read_dbname(dbname, **kw)

    # Read a database by name
    def read_dbname(self, dbname, **kw):
        r"""Read a datakit based on DB name

        :Call:
            >>> db = hub.read_dbname(dbname, **kw)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *db*: ``None`` | :class:`DataKit`
                Data interface if successful
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Initialize error
        ierr = 0
        # Loop through sections
        for sec in self:
            # Attempt to read
            jerr, db = self._read_dbname_section(dbname, sec, **kw)
            # Check status
            if jerr == 0:
                # Got it
                return db
            # Keep track of best step
            ierr = max(ierr, jerr)

    # Try all candidates from a section
    def _read_dbname_section(self, dbname, sec, **kw):
        r"""Try all candidate modules from a given section

        :Call:
            >>> ierr, db = hub._read_dbname_section(dbname, sec, **kw)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *ierr*: ``0`` | :class:`int`
                Exit code

                * ``0``: success
                * ``1``: *dbname* doesn't match *sec*
                * ``2``: *dbname* doesn't match *regex*
                * ``3``: couldn't import module
                * ``4``: unable to find attribute from module
                * ``5``: unable to find function from module
                * ``6``: error during valid *module_function* execution
                * ``7``: *module_function* returned ``None``

            *db*: ``None`` | :class:`DataKit`
                Data interface if successful
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Verbosity flags
        v = kw.get("verbose", kw.get("v", False))
        vv = kw.get("veryverbose", kw.get("vv", False))
        vvv = kw.get("veryveryverbose", kw.get("vvv", False))
        # Verbosity cascade
        vv = vv or vvv
        v = v or vv
        # Check valid section
        if self.match_section(sec, dbname) is None:
            # Verbosity option
            if vvv:
                print(ERRMSG_NOMATCH_SECTION % sec)
            # Early exit
            return ERROR_NOMATCH_SECTION, None
        # Matching section status update
        if v:
            print(MSG_SECTION % sec)
        # Otherwise get candidate module name rules
        module_regex_dict = self.get_section_opt(sec, "module_regex", {})
        # Initial error (in case "module_regex" is empty)
        ierr = ERROR_NOMATCH_DBNAME
        # Loop through candidates
        for regex, templates in module_regex_dict.items():
            # Ensure we have a list of templates
            modname_template_list = _listify(templates)
            # Loop through module templates
            for template in modname_template_list:
                # Attempt to load it
                jerr, db = self._read_dbname(
                    dbname, sec, regex, template, **kw)
                # Check for success
                if jerr == 0:
                    # Success
                    return jerr, db
                # Update error code (higher means it got farther)
                ierr = max(ierr, jerr)
        # If reaching this point, return the best error code
        return ierr, None

    # Read a DB from a loaded module
    def _read_dbname(self, dbname, sec, regex, template, **kw):
        r"""Read a datakit based on DB name from specified section

        :Call:
            >>> ierr, db = hub._read_dbname(dbname, sec, regex, fmt)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
            *regex*: :class:`str`
                Regular expression template for database names
            *fmt*: :class:`str`
                Template for module name based on *regex* match groups
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *ierr*: ``0`` | :class:`int`
                Exit code

                * ``0``: success
                * ``1``: *dbname* doesn't match *sec*
                * ``2``: *dbname* doesn't match *regex*
                * ``3``: couldn't import module
                * ``4``: unable to find attribute from module
                * ``5``: unable to find function from module
                * ``6``: error during valid *module_function* execution
                * ``7``: *module_function* returned ``None``

            *db*: ``None`` | :class:`DataKit`
                Data interface if successful
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Attempt to import the module
        ierr, mod = self._import_dbname(dbname, sec, regex, template, **kw)
        # Check for errors
        if ierr:
            # Use existing error code
            return ierr, None
        # Verbosity flags
        v = kw.get("verbose", kw.get("v", False))
        vv = kw.get("veryverbose", kw.get("vv", False))
        vvv = kw.get("veryveryverbose", kw.get("vvv", False))
        # Verbosity cascade
        vv = vv or vvv
        v = v or vv
        # Get candidate db attributes and functions
        mod_attrs = self.get_section_opt(sec, "module_attribute")
        mod_funcs = self.get_section_opt(sec, "module_function")
        # Convert to lists
        mod_attrs = _listify(mod_attrs)
        mod_funcs = _listify(mod_funcs)
        # Loop through attributes (if any)
        for attr in mod_attrs:
            # Attempt to get it
            db = getattr(mod, attr, None)
            # Check for ... something
            if db is None:
                # Status message
                if vv:
                    print(ERRMSG_NO_ATTRIBUTE % (mod.__name__, attr))
            else:
                # Status message
                if v:
                    print(MSG_READDB_ATTR % (mod.__name__, attr))
                # Return it
                return 0, db
        # If reaching this point, no *module_attribute* worked
        ierr = ERROR_NO_ATTRIBUTE
        # Loop through functions (if any)
        for func in mod_funcs:
            # Get the function
            fn = getattr(mod, func, None)
            # Check for function
            if fn is None:
                # Status message
                if vv:
                    print(ERRMSG_NO_FUNCTION % (mod.__name__, func))
                # Update error code
                ierr = max(ierr, ERROR_NO_FUNCTION)
                # Try next function (if any)
                continue
            # Try to call the function
            try:
                db = fn()
            except Exception:
                # Status message
                if vv:
                    print(ERRMSG_FUNCTION_CALL % (mod.__name__, func))
                # Failure during execution
                ierr = max(ierr, ERROR_FUNCTION_CALL)
                # Try the next function (if any)
                continue
            # Check for output
            if db is None:
                # Status message
                if vv:
                    print(ERRMSG_FUNCTION_NONE % (mod.__name__, func))
                # Function worked but returned empty result
                ierr = max(ierr, ERROR_FUNCTION_NONE)
            else:
                # Success message
                if v:
                    print(MSG_READDB_FUNC % (mod.__name__, func))
                # Got valid datakit
                return 0, db
        # If this module failed, remove it from the cache
        # (Otherwise future attempts will just go back to this module)
        self.datakit_modules.pop(dbname, None)
        # If reaching this point, return current error
        return ierr, None

   # --- Module ---
    # Import a module by name
    def import_module(self, dbname, **kw):
        r"""Import a datakit module based on DB name

        :Call:
            >>> mod = hub.import_module(dbname, **kw)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *mod*: ``None`` | :class:`module`
                Imported module if possible
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
            * 2021-08-19 ``@ddalle``: Version 2.0
                - forked from :func:`load_module`
                - better regular expression support
                - better fallback if more than one section matches
        """
        # Use :func:`import_dbname`
        return self.import_dbname(dbname, **kw)

    # Import a module by name
    def import_dbname(self, dbname, **kw):
        r"""Import a datakit module based on DB name

        :Call:
            >>> mod = hub.import_dbname(dbname, **kw)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *mod*: ``None`` | :class:`module`
                Imported module if possible
        :Versions:
            * 2021-02-18 ``@ddalle``: Version 1.0
            * 2021-08-19 ``@ddalle``: Version 2.0
                - forked from :func:`load_module`
                - better regular expression support
                - better fallback if more than one section matches
        """
        # Initialize error
        ierr = 0
        # Loop through sections
        for sec in self:
            # Attempt to read
            jerr, db = self._import_dbname_section(dbname, sec, **kw)
            # Check status
            if jerr == 0:
                # Got it
                return db
            # Keep track of best step
            ierr = max(ierr, jerr)

    # Try all candidates from a section
    def _import_dbname_section(self, dbname, sec, **kw):
        r"""Try all candidate modules from a given section

        :Call:
            >>> ierr, mod = hub._import_dbname_section(dbname, sec)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *ierr*: ``0`` | :class:`int`
                Exit code

                * ``0``: success
                * ``1``: *dbname* doesn't match *sec*
                * ``2``: *dbname* doesn't match *regex*
                * ``3``: couldn't import module

            *mod*: ``None`` | :class:`module`
                Imported module if possible
        :Versions:
            * 2021-08-19 ``@ddalle``: Version 1.0
        """
        # Verbosity flags
        v = kw.get("verbose", kw.get("v", False))
        vv = kw.get("veryverbose", kw.get("vv", False))
        vvv = kw.get("veryveryverbose", kw.get("vvv", False))
        # Verbosity cascade
        vv = vv or vvv
        v = v or vv
        # Check valid section
        if self.match_section(sec, dbname) is None:
            # Verbosity option
            if vvv:
                print(ERRMSG_NOMATCH_SECTION % sec)
            # Early exit
            return ERROR_NOMATCH_SECTION, None
        # Matching section status update
        if v:
            print(MSG_SECTION % sec)
        # Otherwise get candidate module name rules
        module_regex_dict = self.get_section_opt(sec, "module_regex", {})
        # Initial error (in case "module_regex" is empty)
        ierr = 0
        # Loop through candidates
        for regex, templates in module_regex_dict.items():
            # Ensure we have a list of templates
            modname_template_list = _listify(templates)
            # Loop through module templates
            for template in modname_template_list:
                # Attempt to load it
                jerr, mod = self._import_dbname(
                    dbname, sec, regex, template, **kw)
                # Check for success
                if jerr == 0:
                    # Success
                    return jerr, mod
                # Update error code (higher means it got farther)
                ierr = max(ierr, jerr)
        # If reaching this point, return the best error code
        return ierr, None
        
    # Load a module from name and section
    def _import_dbname(self, dbname, sec, regex, template, **kw):
        r"""Import a datakit by DB name from section and modname regex

        :Call:
            >>> ierr, mod = hub._import_dbname(dbname, sec, regex, fmt)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
            *regex*: :class:`str`
                Regular expression template for database names
            *fmt*: :class:`str`
                Template for module name based on *regex* match groups
        :Keyword Arguments:
            *v*, *verbose*: ``True`` | {``False``}
                Option to report results of matching modules
            *vv*, *veryverbose*: ``True`` | {``False``}
                Option to report all attempts in matching sections
            *vvv*, *veryveryverbose*: ``True`` | {``False``}
                Option to report all attempts
        :Outputs:
            *ierr*: ``0`` | :class:`int`
                Exit code

                * ``0``: success
                * ``1``: *dbname* doesn't match *sec*
                * ``2``: *dbname* doesn't match *regex*
                * ``3``: couldn't import module

            *mod*: ``None`` | :class:`module`
                Imported module if possible
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Verbosity flags
        v = kw.get("verbose", kw.get("v", False))
        vv = kw.get("veryverbose", kw.get("vv", False))
        vvv = kw.get("veryveryverbose", kw.get("vvv", False))
        # Verbosity cascade
        vv = vv or vvv
        v = v or vv
        # Check valid section
        if self.match_section(sec, dbname) is None:
            # Verbosity option
            if vvv:
                print(ERRMSG_NOMATCH_SECTION % (sec))
            # Early exit
            return ERROR_NOMATCH_SECTION, None
        # Check if *dbname* matches *regex*
        modname = self.genr8_modname(dbname, regex, template)
        # Exit if no match
        if modname is None:
            # Verbosity option
            if vv:
                print(ERRMSG_NOMATCH_DBNAME % regex)
            # Early exit
            return ERROR_NOMATCH_DBNAME, None
        # Status update for match
        if v:
            print(MSG_DBNAME % regex)
        # Check for a previous load
        mod = self.datakit_modules.get(dbname)
        # Check if it's a module
        if mod is not None:
            # Verbosity option
            if v:
                print(MSG_IMPORT % modname)
            # Just return it
            return 0, mod
        # Get repo to add to path if necessary
        repo = self.genr8_modpath(dbname, sec)
        # Add to path if necessary
        if repo is None:
            # No required special path
            pass
        elif repo in sys.path:
            # Requested path already present
            pass
        else:
            # Add to front of path
            sys.path.insert(0, repo)
        # Attempt to import the module
        try:
            # Import module by name
            mod = importlib.import_module(modname)
            # Save the new module
            self.datakit_modules[dbname] = mod
            # Success
            ierr = 0
            # Status message
            if v:
                print(MSG_IMPORT % modname)
        except Exception:
            # No module read
            mod = None
            ierr = ERROR_IMPORT_ERROR
            # Status message
            if vv:
                print(ERRMSG_IMPORT_ERROR % modname)
        # Output
        return ierr, mod

   # --- Module Name ---
    # Expand module name based on template
    def genr8_modname(self, dbname, regex, template):
        r"""Determine module name from DB name, regex, and template
    
        :Call:
            >>> modname = hub.genr8_modname(dbname, regex, template)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
            *regex*: :class:`str`
                Regular expression template for database names
            *template*: :class:`str`
                Template for module name based on *regex* match groups
        :Outputs:
            *modname*: ``None`` | :class:`str`
                Name of module according to regex and template
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Attempt to match *dbname* to the regex (with expansions)
        grpdict = self.fullmatch(regex, dbname)
        # Check for match
        if grpdict is None:
            # No match
            return
        # Preporocess the template (e.g. \1 -> %(1)s)
        fmt = prepare_template(template)
        # Otherwise expand the template using matched groups
        return fmt % grpdict

    # Get path for specified database name
    def genr8_modpath(self, dbname, sec):
        r"""Generate $PYTHONPATH for given database name (if any)

        :Call:
            >>> modpath = hub.genr8_modpath(dbname, sec)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *dbname*: :class:`str`
                Database name for one datakit
            *sec*: :class:`str`
                Regular expression template for section of datakits
            *template*: :class:`str`
                Template for module name based on *regex* match groups
        :Outputs:
            *modpath*: ``None`` | :class:`str`
                Path to module if not in existing ``$PYTHONPATH``
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Match the database name to the section name
        grpdict = self.match(sec, dbname)
        # Check for non-matching database
        if grpdict is None:
            return
        # Get repo option
        repo = self.get_section_opt(sec, "repo")
        # Expand template
        fmt = prepare_template(repo)
        # Use regular expression groups
        path = fmt % grpdict
        # Expand to absolute path
        return self.abspath(path)

   # --- Regular expressions ---
    # Get dict of group regular expressions
    def get_regex_groups(self):
        r"""Get expanded regular expressions from *hub.regex_groups*

        :Call:
            >>> regex_dict = hub.get_regex_groups()
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *regex_template*: :class:`str`
                Raw template with ``<grp>`` or ``%(grp)s`` groups
        :Outputs:
            *regex_dict*: :class:`dict`\ [:class:`str`]
                Expanded regex with ``(?P<grp>...)`` for each group
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Initialize output
        regex_dict = {}
        # Loop through defined groups
        for grp, pattern in self.regex_groups.items():
            # Form full pattern with ?P<> notation
            regex = "(?P<%s>%s)" % (grp, pattern)
            # Save it
            regex_dict[grp] = regex
        # Output
        return regex_dict

    # Expand a regular expression
    def expand_regex(self, regex_template):
        r"""Expand a regular expression template

        Use defined groups from *hub.regex_groups*

        :Call:
            >>> regex = hub.expand_regex(regex_template)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *regex_template*: :class:`str`
                Raw template with ``<grp>`` or ``%(grp)s`` groups
        :Outputs:
            *regex*: :class:`str`
                Expanded regex with ``(?P<grp>...)`` filled in
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Expand <grp> shorthand (but not ?P<grp>)
        #     <grp>    --> %(grp)s
        #     ?P<grp>  --> ?P<grp>
        template = re.sub(r"(?<!\?P)<(\w+)>", r"%(\1)s", regex_template)
        # Get regular expression for each group
        regex_dict = self.get_regex_groups()
        # Expand template
        return template % regex_dict

    # Check if a database name matches a section
    def match(self, regex_template, dbname):
        r"""Match a regular expression template to a target string

        :Call:
            >>> groupdict = hub.match(regex_template, dbname)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *regex_template*: :class:`str`
                Regular expression template for section of datakits
            *dbname*: :class:`str`
                Database name for one datakit
        :Outputs:
            *groupdict*: ``None`` | :class:`dict`\ [:class:`str`]
                Augmented :class:`dict` of groups from regex
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Expand the regular expression template
        regex = self.expand_regex(regex_template)
        # Process regular expression
        match = re.match(regex, dbname)
        # Process augmented groups
        return self._process_groupdict(match)

    # Check if a database name matches a section
    def fullmatch(self, regex_template, dbname):
        r"""Match a full string (usually DB name) to a regex template

        :Call:
            >>> groupdict = hub.match(regex_template, dbname)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *regex_template*: :class:`str`
                Regular expression template for section of datakits
            *dbname*: :class:`str`
                Database name for one datakit
        :Outputs:
            *groupdict*: ``None`` | :class:`dict`\ [:class:`str`]
                Augmented :class:`dict` of groups from regex
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Expand the regular expression template
        regex = self.expand_regex(regex_template)
        # Process regular expression
        match = re.fullmatch(regex, dbname)
        # Process augmented groups
        return self._process_groupdict(match)

    # Get dictionary of match groups from match object
    def _process_groupdict(self, match):
        r"""Convert :class:`re.Match` to augmented group :class:`dict`

        :Call:
            >>> groupdict = hub._process_groupdict(match)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *match*: :class:`re.Match`
                Output from :func:`re.match` or similar
        :Outputs:
            *groupdict*: ``None`` | :class:`dict`\ [:class:`str`]
                Augmented :class:`dict` of groups from regex
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Check for match
        if match is None:
            return
        # Initialize augmented dictionary of matches
        groupdict = {}
        # Loop through original matches
        for grp, txt in match.groupdict().items():
            # Save raw text
            groupdict["s-" + grp] = txt
            # Save upper- and lower-case versions
            groupdict["l-" + grp] = txt.lower()
            groupdict["u-" + grp] = txt.upper()
            # Attempt to save as integer
            try:
                # Convert to integer
                groupdict[grp] = int(txt)
            except Exception:
                # Couldn't be converted to integer; use string
                groupdict[grp] = txt
        # Loop through groups in order
        for j, txt in enumerate(match.groups()):
            # Create a group "name," which is really just the index
            grp = str(j + 1)
            # Save raw text
            groupdict["s-" + grp] = txt
            # Save upper- and lower-case versions
            groupdict["l-" + grp] = txt.lower()
            groupdict["u-" + grp] = txt.lower()
            # Attempt to save as integer
            try:
                # Convert to integer
                groupdict[grp] = int(txt)
            except Exception:
                # Couldn't be converted to integer; use string
                groupdict[grp] = txt
        # Output
        return groupdict

    # Check if a database name matches a section
    def match_section(self, sec, dbname):
        r"""Check if a database name matches a given section

        :Call:
            >>> groupdict = hub.match_section(section, dbname)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *section*: :class:`str`
                Regular expression template for section of datakits
            *dbname*: :class:`str`
                Database name for one datakit
        :Outputs:
            *groupdict*: ``None`` | :class:`dict`\ [:class:`str`]
                Augmented :class:`dict` of groups from regex
        :Versions:
            * 2021-08-17 ``@ddalle``: Version 1.0
        """
        # Get augmented matches
        groupdict = self.match(sec, dbname)
        # Make non empty
        if isinstance(groupdict, dict) and len(groupdict) == 0:
            # Add a "whole"
            groupdict["1"] = sec
        # Output
        return groupdict

   # --- Path/OS utils ---
    # Expand path
    def abspath(self, path):
        r"""Expand absolute path to a relative path

        :Call:
            >>> abspath = hub.abspath(path)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *path*: :class:`str`
                Path to some file, relative or absolute
        :Outputs:
            *abspath*: ``None`` | :class:`str`
                Absolute path to *path*
        :Versions:
            * 2021-08-18 ``@ddalle``: Version 1.0
        """
        # Check for empty path
        if path is None:
            return
        # Expand any tildes (~)
        relpath = os.path.expanduser(path)
        # Check for absolute path
        if os.path.isabs(relpath):
            # Use it as is; already absolute
            return relpath
        # Otherwise append *dir_root*
        abspath = os.path.join(self.dir_root, relpath)
        # Follow links if necessary
        return os.path.realpath(abspath)
# class DataKitHub


# Expand a template (\\g<grp> -> %(grp)s)
def prepare_template(template):
    r"""Expand a string template with some substitutions

    The substitutions made include:

        * ``r"\g<grp>"`` --> ``"%(grp)s"``
        * ``r"\l\g<grp>"`` --> ``"%(l-grp)s"``
        * ``r"\u\1"`` --> ``"%(u-1)s"``
        * ``r"\1"`` --> ``"%(1)s"``

    :Call:
        >>> fmt = prepare_template(template)
    :Inputs:
        *template*: :class:`str`
            Initial template, mixing :class:`dict` string expansion and
            :func:`re.sub` syntax
    :Outputs:
        *fmt*: :class:`str`
            Template ready for standard string expansion, for example
            using ``fmt % grpdict`` where *grpdict* is a :class:`dict`
    :Versions:
        * 2021-08-18 ``@ddalle``: Version 1.0
    """
    # Substitute modified re.sub() groups like \l\g<grp>
    fmt1 = re.sub(r"\\([A-z])\\g<(\w+)>", r"%(\1-\2)s", template)
    # Substitute plain groups like \g<grp>
    fmt2 = re.sub(r"\\g<(\w+)>", r"%(\1)s", fmt1)
    # Substitute modified numbered groups like \u\1
    fmt3 = re.sub(r"\\([A-z])\\([1-9][0-9]*)", r"%(\1-\2)s", fmt2)
    # Substitute regular numbered groups like \2
    return re.sub(r"\\([1-9][0-9]*)", r"%(\1)s", fmt3)


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
        * 2021-08-18 ``@ddalle``: Version 1.0
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

