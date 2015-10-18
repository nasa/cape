"""
Interface to FUN3D run control options
======================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""




# Import options-specific utilities
from .util import rc0, getel

# Import template module
import cape.options.RunControl

# Class for Report settings
class RunControl(cape.options.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
# class RunControl


