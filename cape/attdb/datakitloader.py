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


# Default options
rc = {
    "DATAKIT_CLS": DataKit,
    "DB_DIR": "db",
    "DB_DIRS_BY_TYPE": {},
    "DB_NAME_FORMATS": ["datakit"],
    "DB_NAME": None,
    "MODULE_NAME_REGEX": [".+"],
    "MODULE_NAME_REGEX_GROUPS": = {},
    "RAWDATA_DIR": "rawdata",
}


# Get default value from *rc*
def getrc(kw, key, mode=0):
    r"""Get option from *kw*, using default value from *rc*

    :Call:
        >>> v = getrc(kw, key, mode=0)
    :Inputs:
        *kw*: :class:`dict`
            Keyword arguments from some function
        *key*: :class:`str`
            Name of parameter to inspect
        *mode*: {``0``} | ``1``
            If ``0``, ignore ``kw[key]`` if it is ``None``
    :Outputs:
        *v*: **any**
            Either *kw[key]* or *rc[key]*
    :Versions:
        * 2021-06-26 ``@ddalle``: Version 1.0
    """
    # Check mode
    if mode == 0:
        # Get value from *kw*
        v1 = kw.get(key)
        # Don't use ``None`` from *kw*
        if v1 is not None:
            # Use it
            return v1
    elif mode == 1:
        # Use value from *kw* even if ``None``
        if key in kw:
            return kw[key]
    else:
        # Invalid mode
        raise ValueError("Invalid mode '%s'; must be 0 or 1" % mode)
    # Check for option
    if key in rc:
        # Return default
        return rc[key]
    else:
        # No default value
        raise KeyError("No default value for key '%s'" % key)


# Create class
class DataKitLoader(object):
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
    def __init__(self, name, fname, **kw):
        r"""Initialization method

        :Versions:
            * 2021-06-25 ``@ddalle``: Version 1.0
        """
        # Save module name and file
        self.MODULE_NAME = name
        self.MODULE_FILE = os.path.abspath(fname)
        # Containing folder
        self.MODULE_DIR = os.path.dirname(self.MODULE_FILE)
        # Raw data folder
        self.RAWDATA_DIR = getrc(kw, "RAWDATA_DIR")
        # Processed data folder
        self.DB_DIR = getrc(kw, "DB_DIR")
        self.DB_DIRS_BY_TYPE = getrc(kw, "DB_DIRS_BY_TYPE")
        # Datakit class (default)
        self.DATAKIT_CLS = getrc(kw, "DATAKIT_CLS")
        # Regular expression items
        self.MODULE_NAME_REGEX = getrc(kw, "MODULE_NAME_REGEX")
        self.MODULE_NAME_REGEX_GROUPS = getrc(kw, "MODULE_NAME_REGEX_GROUPS")
        #

    def _genr8_modname_regexes(self):
        # Get the regular expressions for each "group"
        grps = dict(self.MODULE_NAME_REGEX_GROUPS)
        # Add full formatting for regular expression group
        grps_re = {
            k: "(?P<%s>%s)" % (k, v)
            for k, v in grps.items()
        }
        # Get regular expression list
        regex_list = self.MODULE_NAME_REGEX
        # Check if it's a list
        if not isinstance(regex_list, (list, tuple)):
            # Create a singleton
            regex_list = [regex_list]
        # Output
        return grps
        
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
        # Get aw data folder
        fdir = os.path.join(self.MODULE_DIR, self.RAWDATA_DIR)
        # File name
        fabs = os.path.join(fdir, fname)
        # Default class
        if cls is None:
            cls = self.DATAKIT_CLS
        # Check for user-specified file type
        if ftype is None:
            # Let *cls* determine the file type
            return cls(fabs, **kw)
        else:
            # Set additional keyword arg
            kw[ftype] = fabs
            # Read the file using *ftype* kwarg
            return cls(**kw)

