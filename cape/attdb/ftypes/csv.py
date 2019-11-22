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

# Standard library
import re

# Third-party modules
import numpy as np

# CAPE modules
import cape.tnakit.typeutils as typeutils

# Local modules
from .basefile import BaseFile


# Regular expressions
regex_numeric = re.compile("\d")
regex_alpha   = re.compile("[A-z_]")

# Class for handling data from CSV files
class CSVFile(BaseFile):
    r"""Class for reading CSV files
    
    :Call:
        >>> db = CSVFile(fname, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
    :See also:
        * :class:`cape.attdb.ftypes.basefile.BaseFile`
    :Versions:
        * 2019-11-12 ``@ddalle``: First version
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Save file name
        self.fname = fname
        # Process definitions
        kw = self.process_col_types(**kw)
        
        # Read file if appropriate
        if fname and typeutiles.isstr(fname):
            # Read valid file
            self.read_csv(fname)
        
        
        
  # >
    
    
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
    def read_csv_header(self, f):
        r"""Read column names from beginning of open file
        
        :Class:
            >>> db.read_header(f)
        :Inputs:
            *db*: :class:`cape.attdb.ftypes.csv.CSVFile`
                CSV file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: First version
        """
        # Set header flags
        self._csv_header_once = False
        self._csv_header_complete = False
        # Read until header_complete flag set
        while not self._csv_header_complete:
            self.read_csv_headerline(f)
        # Remove flags
        del self._csv_header_once
        del self._csv_header_complete
        
        
    # Read a line as if it were a header
    def read_csv_headerline(self, f):
        # Check if header has already been processed
        if self._csv_header_complete:
            return
        # Save current position
        pos = f.tell()
        # Read line
        line = f.readline()
        # Check if it starts with a comment
        if line == "":
            # End of file
            self._csv_header_complete = True
            return
        elif line.startswith("#"):
            # Remove comment
            line = line.lstrip("#")
            # Check for empty comment
            if line.strip() == "":
                # Don't process and don't set any flags
                return
            # Strip comment char and split line into columns
            cols = [col.strip() for col in line.split(",")]
            # Marker that header has been read
            self._csv_header_once = True
        elif not self._csv_header_once:
            # Check for empty line
            if line.strip() == "":
                # Return without setting any flags
                return
            # Split line into columns without strip
            cols = [col.strip() for col in line.split(",")]
            # Marker that header has been read
            self._csv_header_once = True
            # Check valid names of each column
            for col in cols:
                # If it begins with a number, it's probably a data row
                if not regex_alpha.match(col):
                    # Marker for no header
                    self._csv_header_complete = True
                    # Return file to previous position
                    f.seek(pos)
                    # Exit
                    return
        else:
            # Non-comment row following comment: data
            f.seek(pos)
            # Mark completion of header
            self._csv_header_complete = True
        # Save column names if reaching this point
        self.cols = cols
        # Output column names for kicks
        return cols
            
