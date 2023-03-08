# -*- coding: utf-8 -*-
r"""
This module contains containers for reading and writing a variety of
file formats that contain tabular data.

Each data file type interface is a subclass of the template class
:class:`cape.attdb.ftypes.basefile.BaseFile`, which is imported
directly to this module.

"""

# Import basic file type
from .basedata import BaseData, BaseDataDefn, BaseDataOpts
from .basefile import BaseFile, BaseFileDefn, BaseFileOpts
from .csvfile import CSVFile, CSVSimple
from .tsvfile import TSVFile, TSVSimple
from .matfile import MATFile
from .xlsfile import XLSFile
from .textdata import TextDataFile
