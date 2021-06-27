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
   # ==========
    # List of options
    _optlist = {
        "DATAKIT_CLS",
        "DB_DIR",
        "DB_DIRS_BY_TYPE",
        "DB_NAME",
        "DB_NAME_TEMPLATES",
        "MODULE_DIR",
        "MODULE_FILE",
        "MODULE_NAME",
        "MODULE_NAME_REGEX_LIST",
        "MODULE_NAME_REGEX_GROUPS"
    }

    # Types
    _opttypes = {
        "DATAKIT_CLS": type,
        "DB_DIR": str,
        "DB_DIRS_BY_TYPE": (list, tuple),
        "DB_NAME": str,
        "DB_NAME_TEMPLATES": (list, tuple),
        "MODULE_DIR": str,
        "MODULE_FILE": str,
        "MODULE_NAME": str,
        "MODULE_NAME_REGEX_LIST": (list, tuple),
        "MODULE_NAME_REGEX_GROUPS": dict,
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
        "RAWDATA_DIR": "rawdata",
    }

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
        
    def read_rawdata(self, fname, ftype=None, cls=None, **kw):
        r"""Read a file from the RAW_DATA folder

        :Call:
            >>> db = dkl.read_rawdata(fname, ftype=None, cls=None, **kw)
        :Inputs:
            *fname*: :class:`str`
                Name of file to read from raw data folder
            *ftype*: {``None``} | :class:`str`
                Optional specifier to predetermine file type
            *cls*: {``None``} | :class:`type`
                Class to read *fname* other than *dkl.DATAKIT_CLS*
            *kw*: :class:`dict`
                Additional keyword arguments passed to *cls*
        :Outputs:
            *db*: *dkl.DATAKIT_CLS* | *cls*
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

