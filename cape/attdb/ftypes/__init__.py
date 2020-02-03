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
from .basedata import BaseData
from .basefile import BaseFile
#from .csv import CSVFile, CSVSimple
#from .mat import MATFile
#from .xls import XLSFile
#from .textdata import TextDataFile
