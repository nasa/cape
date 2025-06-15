r"""
:mod:`cape.dkit.datakitast`: DataKit assistant module
=========================================================

This module provides the class :class:`DataKitAssistant`, which provides
useful tools for a DataKit module. It helps identify a DataKit module
by its module name, manages subfolders like ``rawdata/``, and helps read
other DataKits in the same repository (DataKit collection).
"""

# Standard library
from typing import Optional

# Third-party

# Local imports
from . import quickstart
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

    # Quickstart
    def quickstart(
            self,
            dbname: str,
            reqs: Optional[list] = None,
            template: Optional[str] = None,
            metaopts: Optional[dict] = None,
            title: Optional[str] = None, **kw):
        # Get parent directory
        rootdir = self.get_rootdir()
        # Get full datakit name
        fulldbname = self.resolve_dbname(dbname)
        # Get module names
        modnames = self.genr8_modnames(fulldbname)
        modname = modnames[0]
        # Default requirements
        reqs = [self.get_opt("DB_NAME")] if reqs is None else reqs
        # Initialize quickstarter
        starter = quickstart.DataKitQuickStarter(
            modname,
            where=rootdir,
            r=reqs, template=template,
            title=title, **kw)
        # Apply metadata options
        starter.opts.setdefault("meta", {})
        starter.opts["meta"].update(metaopts)
        # Run command
        starter.quickstart()
