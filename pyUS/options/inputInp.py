"""
:mod:`pyUS.options.inputInp`: US3D ``input.inp`` options 
=========================================================

This module provides a class to interpret JSON options that are converted to
``input.inp`` settings for FUN3D.  The
module provides a class, :class:`pyUS.options.inputInp.InputInpOpts`, which
interprets the settings of the ``"US3D"`` section of the master JSON file.
These settings are then applied to the main OVERFLOW input file, the
``input.inp`` input file, which is in a format specific to US3D.

An example JSON setting is shown below.

    .. code-block:: javascript
    
        "US3D": {
            "CFD_SOLVER": {
                "nstop": 30000,
                "ivisc": 11,
                "cfl": [1.1, 5.1, 10.1]
            },
            "CFD_SOLVER_OPTS": {
                "chem_vibr_diso": 1.0
            },
            "MANAGE": {
                "flag": 4
            }
        }
        
This will cause or partially cause changes to the sections ``[CFD_SOVER]``,
``[CFD_SOLVER_OPTS]``, and ``[MANAGE]`` in the ``input.00.inp`` file.
        
The edits to ``input.02.inp`` are from the third entries of each list, so the
*cfl* parameter would be ``10.1`` instead of ``1.1``.
            
Each setting and section in the ``"US"`` section may be either present in
the template ``input.inp`` file or missing.  It will be either edited or added
as appropriate, even if the specified section does not exist.

:See also:
    * :mod:`pyUS.inputInp`
    * :mod:`pyUS.us3d`
    * :mod:`cape.namelist`
"""

# Ipmort options-specific utilities
from .util import rc0, odict, getel, setel

# Class for namelist settings
class InputInpOpts(odict):
    """Dictionary-based interface for US3D input file ``input.inp``"""
    
   # ---------
   # General
   # ---------
   # <
    # Get the settings for a section namelist
    def get_section(self, sec, j=None):
        """Return a settings :class:`dict` for an ``input.inp`` section
        
        :Call:
            >>> d = opts.get_section(sec, j=None)
        :Inputs:
            *opts*: :pyUS.options.Options`
                Options interface
            *sec*: :class:`str`
                Name of appropriate ``input.inp`` section
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *d*: :class:`cape.options.odict`
                Settings for that section
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        # Get the value
        d = getel(self.get(sec), j) 
        # Check for None
        if d is None:
            # Return empty dict
            return odict()
        else:
            # Convert dictionary to odict
            return odict(**d)
    
    # Reduce entire options dict to a single phase
    def select_InputInp(self, j=0):
        """Reduce ``input.inp`` options to options for a single phase
        
        :Call:
            >>> d = opts.select_InputInp(j=0)
        :Inputs:
            *opts*: :class:`pyUS.options.Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *d*: :class:`cape.options.odict`
                Project settings for phase *j*
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        # Initialize output
        d = odict()
        # Loop through keys
        for sec in self:
            # Get the list
            L = getel(self[sec], j)
            # Initialize this list.
            d[sec] = {}
            # Loop through subkeys
            for k in L:
                # Select the key and assign it.
                d[sec][k] = getel(L[k], j)
        # Output
        return d
        
    # Get value by name
    def get_InputInp_key(self, sec, key, j=None):
        """Select an ``input.inp`` setting from a specified section
        
        Roughly, this returns ``opts[sec][key]``.
        
        :Call:
            >>> val = opts.get_namelist_key(sec, key, j=None)
        :Inputs:
            *opts*: :class:`pyUS.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *val*: JSON-type
                Value from JSON options
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From :func:`get_namelist_var`
        """
        # Check for namelist
        if sec not in self:
            return None
        # Select the namelist
        d = getel(self[sec], j)
        # Select the value.
        return getel(d.get(key), j)
        
    # Set value by name
    def set_InputInp_key(self, sec, key, val, j=None):
        """Set a namelist key for a specified phase or phases
        
        Roughly, this sets ``opts["US3D"][sec][key]`` or
        ``opts["US3D"][sec][key][i]`` equal to *val*
        
        :Call:
            >>> opts.set_InputInp_key(sec, key, val, j=None)
        :Inputs:
            *opts*: :class:`pyUS.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *val*: JSON-type
                Value for JSON options
            *j*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2017-04-05 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From :func:`set_namelist_var`
        """
        # Initialize section
        if sec not in self:
            self[sec] = {}
        # Initialize key
        if key not in self[sec]:
            self[sec][key] = None
        # Set value
        self[sec][key] = setel(self[sec][key], j, val)
   # >
   
   # -----------
   # CFD_SOLVER
   # -----------
   # <
    # Get entire section for CFD_SOLVER
    def get_CFDSOLVER(self, j=None):
        """Return a ``"CFD_SOLVER"`` settings :class:`dict`
        
        :Call:
            >>> d = opts.get_CFDSOLVER(sec, j=None)
        :Inputs:
            *opts*: :pyUS.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *d*: :class:`cape.options.odict`
                Settings for that section
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        return self.get_section("CFD_SOLVER", j=j)
    
    # Get an options from the "CFD_SOLVER" section
    def get_CFDSOLVER_key(self, k, j=None):
        """Return a named parameter from the *CFD_SOLVER* section
        
        :Call:
            >>> v = opts.get_CFDSOLVER_key(k, j=None)
        :Inputs:
            *opts*: :pyUS.options.Options`
                Options interface
            *k*: :class:`str`
                Name of appropriate key from ``input.inp`` section
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *v*: :class:`int` | :class:`float` | :class:`list`
                Value for (numeric) parameter
        :Versions:
            * 2019-06-27 ``@ddalle``: First version
        """
        # Get section
        d = self.get_section("CFD_SOLVER", j=j)
        # Check for empty setting
        if d is None:
            return None
        # Get key from that section
        return d.get_key(k, j)
    
    
   # >
        
