"""
Interface to FUN3D namelist options
===================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Ipmort options-specific utilities
from util import rc0, odict, getel

# Class for namelist settings
class Fun3d(odict):
    """Dictionary-based interface for FUN3D namelists"""
    
    # Get the project namelist
    def get_project(self, i=None):
        """Return the ``project`` namelist
        
        :Call:
            >>> d = opts.get_project(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *d*: :class:`pyFun.options.odict`
                Project namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Get the value
        d = getel(self, 'project', i) 
        # Check for None
        if d is None:
            # Return empty dict
            return odict()
        else:
            # Convert dictionary to odict
            return odict(**d)
    
    # Get the project namelist
    def get_raw_grid(self, i=None):
        """Return the ``raw_grid`` namelist
        
        :Call:
            >>> d = opts.get_raw_grid(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *d*: :class:`pyFun.options.odict`
                Grid namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Get the value
        d = getel(self, 'raw_grid', i) 
        # Check for None
        if d is None:
            # Return empty dict
            return odict()
        else:
            # Convert dictionary to odict
            return odict(**d)
            
    # Get rootname
    def get_project_rootname(self, i=None):
        """Return the project root name
        
        :Call:
            >>> rname = opts.get_project_rootname(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *rname*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Get the namelist
        d = self.get_project(i)
        # Get the value.
        return d.get_key('project_rootname', i)
        
    # Grid format
    def get_grid_format(self, i=None):
        """Return the grid format
        
        :Call:
            >>> fmat = opts.get_grid_format(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *fmat*: :class:`str`
                Grid format
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Get the raw_grid namelist
        d = self.get_raw_grid(i)
        # Get the value.
        return d.get_key('grid_format', i)
        
        
    # Reduce to a single run sequence
    def select_namelist(self, i=0):
        """Reduce namelist options to a single instance (i.e. sample lists)
        
        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *d*: :class:`pyFun.options.odict`
                Project namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Initialize output
        d = {}
        # Loop through keys
        for sec in self:
            # Get the list
            L = getel(self, sec, i)
            # Initialize this list.
            d[sec] = {}
            # Loop through subkeys
            for k in L:
                # Select the key and assign it.
                d[sec][k] = getel(L, k, i)
        # Output
        return d
            
        
        
