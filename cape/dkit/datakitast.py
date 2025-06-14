r"""
:mod:`cape.dkit.datakitast`: DataKit assistant module
=========================================================

This module provides the class :class:`DataKitAssistant`, which provides
useful tools for a DataKit module. It helps identify a DataKit module
by its module name, manages subfolders like ``rawdata/``, and helps read
other DataKits in the same repository (DataKit collection).
"""

# Standard library

# Third-party

# Local imports
from .datakitloader import DataKitLoader


# Primary class
class DataKitAssistant(DataKitLoader):
    r"""Tool for reading datakits based on module name and file

    :Call:
        >>> ast = DataKitAssistant(DATAKIT_CLS=None, **kw)
    :Inputs:
        *DATAKIT_CLS*: {``None``} | :class:`type`
            Optional subclass of :class:`DataKit` to use for this pkg
    :Outputs:
        *ast*: :class:`DataKitAssistant`
            Assistant for finding file paths and related DataKits
    """
    # Attributes
    __slots__ = ()
