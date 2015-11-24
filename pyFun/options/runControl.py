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

# Class for `nodet` imputs
class nodet(odict):
    """Class for ``nodet`` command-line inputs"""
    
    # Animation frequency
    def get_nodet_animation_freq(self, i=None):
        """Get animation frequency command-line option
        
        :Call:
            >>> f = opts.get_nodet_animation_freq(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *f*: :class:`int`
                Animation frequency; when ``nodet`` outputs are written
        :Versions:
            * 2015-11-24 ``@ddalle``: First version
        """
        return self.get_key('nodet_animation_freq', i)
        
    # Set animation frequency
    def set_nodet_animation_freq(self, f=rc0('nodet_animation_freq'), i=None):
        """Set animation frequency command-line option
        
        :Call:
            >>> opts.set_nodet_animation_freq(f, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *f*: :class:`int`
                Animation frequency; when ``nodet`` outputs are written
            *i*: :class:`int`
                Run index
        :Versions:
            * 2015-11-24 ``@ddalle``: First version
        """
        self.set_key('nodet_animation_freq', f, i)

# Class for Report settings
class RunControl(cape.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
# class RunControl


