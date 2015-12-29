"""
Interface to FUN3D run control options
======================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Import options-specific utilities
from .util import rc0, getel, odict

# Import template module
import cape.options.runControl

# Class for Report settings
class RunControl(cape.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2015-12-29 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
# class RunControl


