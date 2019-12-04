#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes`: Data file type interfaces
====================================================

This module contains containers for reading and writing a variety of
file formats that contain tabular data.

Each data file type interface is a subclass of the template class
:class:`cape.attdb.ftypes.basefile.BaseFile`, which is imported
directly to this module.

"""

# Import basic file type
from .basefile import BaseFile
from .csv      import CSVFile, CSVSimple
from .textdata import TextDataFile

# Create a more generic class
class DataFile(CSVFile, CSVSimple, TextDataFile):
    r"""Combined data file interface
    
    :Call:
        >>> db = DBResponseNull(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
    :Outputs:
        *db*: :class:`cape.attdb.ftypes.DataFile`
            Generic database
    :Versions:
        * 2019-12-04 ``@ddalle``: First version
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method
        
        :Versions:
            * 2019-12-04 ``@ddalle``: First version
        """
        # Initialize options
        self.opts = {}
        self.cols = []
        self.n = 0
        
        # Initialize file names for each potential type
        fcsv  = None
        fcsvs = None
        ftdat = None
        
        # Process file name input
        if fname is not None:
            # Get file extension
            ext = fname.split(".")[-1]
            # Filter it
            if ext == "csv":
                # Assume it's a CSV file
                fcsv = fname
            else:
                # Fall back to generic text
                ftdat = fname
        
        # Get option for each base CSV file
        fcsv = kw.pop("csv", fcsv)
        # Check for valid option
        if fcsv is not None:
            # Read CSV file
            CSVFile.__init__(self, fcsv, **kw)
            # Exit
            return
            
        # Get option for simple CSV file
        fcsvs = kw.pop("simplecsv", fcsvs)
        # Check for valid file name
        if fcsvs is not None:
            # Read simple CSV file
            CSVSimple.__init__(self, fcsvs, **kw)
            # Exit
            return

        # Get option for text data file
        ftdat = kw.pop("textdata", ftdat)
        # Check for valid file name
        if ftdat is not None:
            # Read generic text data file
            TextDataFile.__init__(self, ftdat, **kw)
            return
            
        # If reaching this point, process other inputs
        kw = self.process_col_defns(**kw)
        # Check for value overrides
        kw = self.process_kw_values(**kw)
        # Warn about any unused inputs
        self.warn_kwargs(kw)
    
   # >

