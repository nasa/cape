"""
:mod:`cape.pyfun.options.fun3dnml`: FUN3D namelist options
===========================================================

This module provides a class to interpret JSON options that are converted to
Fortran namelist format for FUN3D.  The
module provides a class, :class:`pyFun.options.fun3dnml.Fun3DNml`, which
interprets the settings of the ``"Fun3D"`` section of the master JSON file.
These settings are then applied to the main OVERFLOW input file, the
``fun3d.nml`` namelist.

An example JSON setting is shown below.

    .. code-block:: javascript
    
        "Fun3D": {
            "nonlinear_solver_parameters": {
                "schedule_cfl": [[1.0, 5.0], [5.0, 20.0], [20.0, 20.0]],
                "time_accuracy": ["steady", "steady", "2ndorder"],
                "time_step_nondim": 2.0,
                "subiterations": 5
            },
            "boundary_output_variables": {
                "boundary_list": "7-52",
                "turres1": true,
                "p_tavg": [false, false, true]
            }
        }
        
This will cause the following settings to be applied to ``fun3d.00.nml``.

    .. code-block:: none
    
        &nonlinear_solver_parameters
            schedule_cfl = 1.0 5.0
            time_accuracy = 'steady'
            time_step_nondim = 2.0
            subiterations = 5
        /
        &boundary_output_variables
            boundary_list = '7-52'
            turres1 = .true.
            p_tavg = .false.
        /
        
The edits to ``fun3d.02.nml`` are from the third entries of each list:

    .. code-block:: none
    
        &nonlinear_solver_parameters
            schedule_cfl = 20.0 20.0
            time_accuracy = '2ndorder'
            time_step_nondim = 2.0
            subiterations = 5
        /
        &boundary_output_variables
            boundary_list = '7-52'
            turres1 = .true.
            p_tavg = .true.
        /
            
Each setting and section in the ``"Fun3D"`` section may be either present in
the template namelist or missing.  It will be either edited or added as
appropriate, even if the specified section does not exist.

:See also:
    * :mod:`cape.pyfun.namelist`
    * :mod:`cape.pyfun.cntl`
    * :mod:`cape.namelist`
"""

# Ipmort options-specific utilities
from .util import rc0, odict, getel, setel

# Class for namelist settings
class Fun3DNml(odict):
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
        d = getel(self.get('project'), i) 
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
        d = getel(self.get('raw_grid'), i) 
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
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *i*: :class:`int` | ``None``
                Run sequence index
        :Outputs:
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value from JSON options
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Check for namelist
        if sec not in self: return None
        # Select the namelist
        d = getel(self[sec], i)
        # Select the value.
        return getel(d.get(key), i)
        
    # Set value by name
    def set_namelist_var(self, sec, key, val, i=None):
        """Set a namelist key for a specified phase or phases
        
        Roughly, this sets ``opts["Fun3D"][sec][key]`` or
        ``opts["Fun3D"][sec][key][i]`` equal to *val*
        
        :Call:
            >>> opts.set_namelist_var(sec, key, val, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value from JSON options
            *i*: :class:`int` | ``None``
                Run sequence index
        :Versions:
            * 2017-04-05 ``@ddalle``: First version
        """
        # Initialize section
        if sec not in self: self[sec] = {}
        # Initialize key
        if key not in self[sec]: self[sec][key] = None
        # Set value
        self[sec][key] = setel(self[sec][key], i, val)
        
