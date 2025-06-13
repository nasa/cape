# -*- coding: utf-8 -*-
r"""
:mod:`cape.dkit.datakitloader`: DataKit collection tools
=======================================================================

This class provides the :class:`DataKitLoader`, which takes as input the
module *__name__* and *__file__* to automatically determine a variety of
DataKit parameters.

"""

# Standard library

# Local modules
from .datakitast import DataKitAssistant


# Create class
class DataKitLoader(DataKitAssistant):
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
    """
    pass

