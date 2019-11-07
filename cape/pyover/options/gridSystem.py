"""
:mod:`cape.pyover.options.gridSystem`: OVERFLOW grid namelist options
======================================================================

This module provides a class to alter namelist settings for each grid in an
Overflow namelist.  This modifies the repeated sections (such as ``GRDNAM``,
``NITERS``, ``METPRM``, ``TIMACU``, etc.) in the ``overflow.inp`` input file.

Users can use the ``"ALL"`` dictionary of settings to apply settings to every
grid in the system.  Any other dictionary in the top level applies to a grid by
the name of that key.  An example follows.

    .. code-block:: javascript
        
        "Grids": {
            "ALL": {
                "TIMACU": {
                    "ITIME": 3,
                    "CFLMIN": [0.01, 0.025, 0.05, 0.05],
                    "CFLMAX": [0.25, 0.50,  1.00, 1.00]
                }
            },
            "Fuselage": {
                "TIMACU": {
                    "CFLMAX": [0.2, 0.4, 0.9, 1.0]
                }
            }
        }

This example sets the CFL number for each grid (and also sets the *ITIME*
setting to 3).  Then it finds the grid called ``"Fuselage"`` and changes the
max CFL number to slightly lower values for each phase.  The :class:`list`
input tells pyOver to set the max CFL number to ``0.25`` for ``run.01.inp``,
``0.50`` for ``run.02.inp``, etc.  If there are more phases than entries in the
:class:`list`, the last value is repeated as necessary.

For other namelist settings that do not refer to grids, see
:class:`pyOver.options.overnml.OverNml`.

:See also:
    * :mod:`cape.pyover.options.overnml`
    * :mod:`cape.pyover.overNamelist`
    * :mod:`cape.pyover.cntl`
    * :mod:`cape.filecntl.namelist2`
"""

# Ipmort options-specific utilities
from .util import rc0, odict, getel

# Class for namelist settings
class GridSystemNml(odict):
    """Dictionary-based interface for OVERFLOW namelist grid system options"""
    
    # Get the ALL namelist
    def get_ALL(self, i=None):
        """Return the ``ALL`` namelist of settings applied to all grids
        
        :Call:
            >>> d = opts.get_ALL(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *d*: :class:`pyOver.options.odict`
                ALL namelist
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_GridByName('ALL', i)
        
    # Select a grid
    def get_GridByName(self, grdnam, i=None):
        """Return a dictionary of options for a specific grid
        
        :Call:
            >>> d = opts.get_GridByName(grdnam, i=None)
        :Inputs:
            *opts*: :class;`pyOver.options.Options`
                Options interface
            *grdnam*: :class:`str` | :class:`int`
                Name or number of grid to alter
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *d*: :class:`pyOver.options.odict`
                Dictionary of options for grid *gridnam*
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value
        kw = getel(self.get(grdnam), i)
        # Initialize output
        d = odict()
        # Loop through fields and keys of *d*
        for sec in kw:
            # Initialize section
            d[sec] = {}
            # Loop through keys
            for k in kw[sec]:
                # Select for the requested phase number
                d[sec][k] = getel(kw[sec][k], i)
        # Output
        return d
        
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
    def get_grid_var(self, grdnam, sec, key, i=None):
        """Select a namelist key from a specified section
        
        Roughly, this returns ``opts[sec][key]``.
        
        :Call:
            >>> val = opts.get_grid_var0(sec, key, i=None)
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
            * 2016-02-01 ``@ddalle``: First version
        """
        # Check for grid
        if grdnam not in self: return None
        # Check for namelist
        if sec not in self[grdnam]: return None
        # Select the namelist
        d = getel(self, sec, i)
        # Select the value.
        return getel(d.get(key), i)
        
    # Get value by grid defaulting to ALL
    def get_GridKey(self, grdnam, sec, key, i=None):
        """Select a grid option
        
        If the option is not found for grid *gridnam*, default to the value in
        the ``"ALL"`` section
        
        :Call:
            >>> val = opts.get_GridKey(grdnam, sec, key, i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value directly.
        val = self.get_grid_var(grdnam, sec, key, i)
        # Check if ``None``
        if val is None:
            # Try the default option
            return self.get_grid_var('ALL', sec, key, i)
# class GridSystemNml
        
