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
        >>> opts = RunControl(**kw)
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    # Function to get prefix
    def get_Prefix(self, i=None):
        """Get the project rootname, or file prefix
        
        :Call:
            >>> fpre = opts.get_Prefix()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fpre*: :class:`str`
                Name of OVERFLOW prefix
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_key("Prefix", i, "project_rootname")
    
    # Function to get OVERFLOW command
    def get_overruncmd(self, i=None):
        """Get the name of the OVERFLOW binary to run
        
        :Call:
            >>> fcmd = opts.get_overruncmd(i)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fcmd*: :class:`cmd`
                Name of the command-line function to use
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
# class RunControl


