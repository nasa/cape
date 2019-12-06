#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.datafile`: Data container for data from files
=====================================================================

This module provides a class :class:`DataContainer` that holds data
read from one of several formats.  Reading the data into this container
class avoids issues with name conflicts that might arise from directly
inheriting from several different file-reading classes.  For example,
some file formats support data types that are not supported by other
file formats.

Each of the file-reading classes inherit from :class:`BaseFile`, but
the :class:`DataContainer` class defined in this module only inherits
directly from :class:`dict`.  However, the options processed by the
appropriate file-reader class are saved as *db.opts* if *db* is an
instance of :class:`DataContainer`.

"""

# Standard library


# Numerics


# CAPE modules
import cape.tnakit.typeutils as typeutils


# Local modules
from .csv      import CSVFile, CSVSimple
from .textdata import TextDataFile


# Data container class
class DataConainter(dict):
    
    # Read data from a CSV instance
    @classmethod
    def from_csv(cls, fname, **kw):
        # Create a new instance
        self = cls()
        # Check input type
        if isinstance(fname, CSVFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = CSVFile(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the options
        self.copy_options(dbf.opts)
        # Output
        return self


    # Read data from an arbitrary-text data instance
        
  # ==================
  # Options
  # ==================
  # <
   # --- Copy/Link ---
    # Link options
    def copy_options(self, opts):
        r"""Copy a database's options

        :Call:
            >>> db.copy_options(dbsrc)
        :Inputs:
            *opts*: :class:`dict`
                Options dictionary
        :Effects:
            *db.opts*: :class:`dict`
                Options merged with or copied from *opts*
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Check input
        if not isinstance(opts, dict):
            raise TypeError("Options input must be dict-type")
        # Get options
        dbopts = self.__dict__.setdefault("opts", {})
        # Merge
        self.opts = dict(dbopts, **opts)
  # >

  # ==================
  # Data
  # ==================
  # <
   # --- Save ---
    # Save a column
    def save_col(self, col, V):
        r"""Save a column to database
        
        :Call:
            >>> db.save_col(col, V)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.datafile.DataContainer`
                Data container
            *col*: :class:`str`
                Name of column to save
            *V*: :class:`any`
                Value to save for "column"
        :Effects:
            *db.cols*: 
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Safely get columns list
        cols = self.__dict__.setdefault("cols", [])
        # Check if present
        if col not in cols:
            cols.append(col)
        # Save the data (don't copy it)
        self[col] = V

   # --- Copy/Link ---
    # Link data
    def link_data(self, dbsrc, cols=None):
        # Default columns
        if cols is None:
            cols = dbsrc.cols
        # Check input types
        if not isinstance(cols, list):
            # Column list must be a list
            raise TypeError("Column list must be a list, got '%s'"
                % cols.__class__.__name__)
        elif not isinstance(dbsrc, dict):
            # Source must be a dictionary
            raise TypeError("Source data must be a dict")
        # Loop through columns
        for col in cols:
            # Check type
            if not typeutils.isstr(col):
                raise TypeError("Column names must be strings")
            # Check if data is present
            if col not in dbsrc:
                raise KeyError("No column '%s'" % col)
            # Save the data
            self.save_col(col, dbsrc[col])
  # >

  # =================
  # Writers
  # =================
  # <
   # --- CSV ---
    # Create a CSV file
    def get_CSVFile(self, cols=None):
        r"""Turn data container into CSV file interface

        :Call:
            >>> dbcsv = db.get_CSVFile(cols=None)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.datafile.DataContainer`
                Data container
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to include (default *db.cols*)
        :Outputs:
            *dbcsv*: :class:`cape.attdb.ftypes.CSVFile`
                CSV file interface
        :Versions:
            * 2019-12-06 ``@ddalle``: First version
        """
        # Default column list
        if cols is None:
            cols = self.cols
        # Get options interface
        opts = self.__dict__.get("opts", {})
        # Get relevant options
        kw = {}
        for (k, vdef) in CSVFile._DefaultOpts.items():
            # Set property
            kw[k] = opts.get(k, vdef)
        # Set values
        kw["Values"] = {col: self[col] for col in cols}
        # Explicit column list
        kw["cols"] = cols
        # Create instance
        return CSVFile(**kw)
        

    # Write dense CSV file
    def write_csv_dense(self, fname, cols=None):
        r""""Write dense CSV file
        
        """
        pass
    
  # >

