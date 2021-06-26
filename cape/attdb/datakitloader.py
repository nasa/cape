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

# Local modules
from .rdb import DataKit


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
    # Initialization method
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
        self.RAWDATA_DIR = kw.get("RAWDATA_DIR", "rawdata")
        # Processed data folder
        self.DB_DIR = kw.get("DB_DIR", "db")
        self.DB_DIRS_BY_TYPE = {}
        # Datakit class (default)
        self.DATAKIT_CLS = kw.get("DATAKIT_CLS", DataKit)

  # ===========
  # RAW DATA
  # ===========
  # <
    # Read a given file from raw data folder
    def read_rawdata(self, fname, cls=None):
        # Get aw data folder
        fdir = os.path.join(self.MODULE_DIR, self.RAWDATA_DIR)
        # File name
        fabs = os.path.join(fdir, fname)
        # Default class
        if cls is None:
            cls = self.DATAKIT_CLS
        # Read the file
        return cls(fabs)

