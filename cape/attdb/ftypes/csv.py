#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.csv`: Comma-separated value read/write
===============================================================

This module contains a basic interface in the spirit of
:mod:`cape.attdb.ftypes` for standard comma-separated value files.  It
creates a class, :class:`CSVFile` that does not rely on the popular
:func:`numpy.loadtxt` function.

If possible, the column names (which become keys in the
:class:`dict`-like class) are read from the header row.  If the file
begins with multiple comment lines, the column names are read from the
final comment before the beginning of data.
"""

# Third-party modules
import numpy as np

# Local modules
from .basefile import BaseFile

# Class for handling data from CSV files
class CSVFile(BaseFile):
    
    
    # Reader
    def read_csv(self, fname):
        # Open file
        f = open(fname, 'r')
        # Process column names
        self.read_header()
        # Loop through lines
        
        # Close file
        f.close()
    
    # Read initial comments
    def read_header(self, f, comment='#', delimiter=","):
        r"""Read column names from beginning of open file
        
        :Class:
            >>> db.read_header(f, comment='#', delimiter=",")
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
            *comment*: {``'#'``} | :class:`str`
                Character or string to begin a comment line
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Go to beginning of file
        f.seek(0)
        # Save current position
        pos = f.tell()
        # Read lines until one is not a comment
        line = f.readline()
        # Check for empty comment char
        if not comment:
            # Use line as is
            coltxts = line.strip(comment).split(delimiter)
            # Strip spaces
            self.cols = [col.strip() for col in coltxts]
        # Check for valid header
        if not line.startswith(comment):
            self.cols = []
        # Loop until line is not a comment
        while line.startswith(comment):
            # Check for empty line
            if len(line.lstrip(comment).strip()) > 0:
                # Save current line as candidate header
                header = line
            # Remember position
            pos = f.tell()
            # Read next line
            line = f.readline()
        # Go back to last position before comment
        f.seek(pos)
        # Create columns
        coltxts = header.lstrip(comment).split(delimiter)
        # Strip any whitespace
        self.cols = [col.strip() for col in coltxts]
        
        
