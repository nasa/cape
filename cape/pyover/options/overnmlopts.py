"""
:mod:`cape.pyover.options.overnml`: OVERFLOW namelist options
==============================================================

This module provides a class to mirror the Fortran namelist capability.  The
module provides a class, :class:`pyOver.options.overnml.OverNml`, which
interprets the settings of the ``"Overflow"`` section of the master JSON file.
These settings are then applied to the main OVERFLOW input file, the
``overflow.inp`` namelist.

At this time, nonunique section names are not allowed.  To modify parameters in
repeated sections, use the options in the ``"Grids"`` section using
:class:`pyOver.options.gridSystem.GridSystem`.

:See also:
    * :mod:`cape.pyover.options.gridSystem`
    * :mod:`cape.pyover.overNamelist`
    * :mod:`cape.pyover.cntl`
    * :mod:`cape.namelist2`
"""

# Ipmort options-specific utilities
from ...optdict import OptionsDict


# Class for namelist settings
class OverNmlOpts(OptionsDict):
    """Dictionary-based interface for OVERFLOW namelists"""
    
    # Get the GLOBAL namelist
    def get_GLOBAL(self, i=None):
        """Return the ``GLOBAL`` namelist
        
        :Call:
            >>> d = opts.get_GLOBAL(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *d*: :class:`pyOver.options.odict`
                GLOBAL namelist
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value
        d = getel(self.get('GLOBAL'), i) 
        # Check for None
        if d is None:
            # Return empty dict
            return odict()
        else:
            # Convert dictionary to odict
            return odict(**d)
    
    # Get the FLOINP namelist
    def get_FLOINP(self, i=None):
        """Return the ``FLOINP`` namelist
        
        :Call:
            >>> d = opts.get_raw_grid(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *d*: :class:`pyOver.options.odict`
                Flow inputs namelist
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value
        d = getel(self.get('FLOINP'), i) 
        # Check for None
        if d is None:
            # Return empty dict
            return odict()
        else:
            # Convert dictionary to odict
            return odict(**d)
        
    # Reduce to a single run sequence
    def select_namelist(self, i=0):
        """Reduce namelist options to a single instance (i.e. sample lists)
        
        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *d*: :class:`pyOver.options.odict`
                Project namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
            * 2016-02-01 ``@ddalle``: Copied from pyFun
        """
        # Initialize output
        d = {}
        # Loop through keys
        for sec in self:
            # Get the list
            L = getel(self[sec], i)
            # Initialize this list.
            d[sec] = {}
            # Loop through subkeys
            for k in L:
                # Select the key and assign it.
                d[sec][k] = getel(L[k], i)
        # Output
        return d
        
    # Get value by name
    def get_namelist_var(self, sec, key, i=None):
        """Select a namelist key from a specified section
        
        Roughly, this returns ``opts[sec][key]``.
        
        :Call:
            >>> val = opts.get_namelist_var(sec, key, i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value from JSON options
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2016-02-01 ``@ddalle``: Copied from pyFun
        """
        # Check for namelist
        if sec not in self: return None
        # Select the namelist
        d = getel(self.get(sec), i)
        # Select the value.
        return getel(d.get(key), i)
        
        
