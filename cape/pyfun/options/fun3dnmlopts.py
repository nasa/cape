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
    * :mod:`cape.filecntl.namelist`
"""

# Local imports
from ...optdict import OptionsDict
from ...optdict.optitem import setel


# Class for namelist settings
class Fun3DNmlOpts(OptionsDict):
    r"""Dictionary-based interface for FUN3D namelists"""

    # Get the project namelist
    def get_project(self):
        r"""Return the ``project`` namelist

        :Call:
            >>> d = opts.get_project(i=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Outputs:
            *d*: :class:`pyFun.options.odict`
                Project namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-13 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        # Get section
        return OptionsDict(self.get_opt("project", vdef={}))

    # Get the project namelist
    def get_raw_grid(self):
        r"""Return the ``raw_grid`` namelist

        :Call:
            >>> d = opts.get_raw_grid(i=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Outputs:
            *d*: :class:`pyFun.options.odict`
                Grid namelist
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-13 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        # Get section
        return OptionsDict(self.get_opt("raw_grid", vdef={}))

    # Get rootname
    def get_project_rootname(self, j=None, **kw):
        r"""Return the project root name

        :Call:
            >>> rname = opts.get_project_rootname(j=None, **kw)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *rname*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-13 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        return self.get_subopt("project", 'project_rootname', j=j, **kw)

    # Grid format
    def get_grid_format(self, j=None, **kw):
        r"""Return the grid format

        :Call:
            >>> fmat = opts.get_grid_format(i=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *fmat*: :class:`str`
                Grid format
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-13 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        return self.get_subopt("raw_grid", "grid_format", j=j, **kw)

    # Reduce to a single run sequence
    def select_namelist(self, j=0, **kw):
        r"""Sample namelist at particular conditions

        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase index
        :Outputs:
            *d*: :class:`dict`
                Namelist sampled for phase and case indices
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; ``OptionsDict`` tools
        """
        # Sample list -> scalar, evaluate @expr, etc.
        return self.sample_dict(self, j=j, **kw)

    # Get value by name
    def get_namelist_var(self, sec, key, j=None, **kw):
        r"""Select a namelist key from a specified section

        Roughly, this returns ``opts[sec][key]``.

        :Call:
            >>> val = opts.get_namelist_var(sec, key, j=None, **kw)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *val*: :class:`object`
                Value from JSON options
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; ``OptionsDict`` tools
        """
        # Return subsection options
        return self.get_subopt(sec, key, j=j, **kw)

    # Set value by name
    def set_namelist_var(self, sec, key, val, j=None):
        r"""Set a namelist key for a specified phase or phases

        Roughly, this sets ``opts["Fun3D"][sec][key]`` or
        ``opts["Fun3D"][sec][key][i]`` equal to *val*

        :Call:
            >>> opts.set_namelist_var(sec, key, val, i=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *sec*: :class:`str`
                Section name
            *key*: :class:`str`
                Variable name
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value from JSON options
            *j*: {``None``} | :class:`int`
                Phase index
        :Versions:
            * 2017-04-05 ``@ddalle``: v1.0
        """
        # Initialize section
        secopts = self.setdefault(sec, {})
        # Initialize key
        v0 = secopts.setdefault(key, None)
        # Set value
        self[sec][key] = setel(v0, val, j=j)


# Class for "Dual" namelist settings
class DualFun3DNmlOpts(OptionsDict):
    r"""Dictionary-based interface for FUN3D namelists"""

    # Reduce to a single run sequence
    def select_dual_namelist(self, j=0, **kw):
        r"""Sample "dual" namelist at particular conditions

        :Call:
            >>> d = opts.select_dual_namelist(i)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase index
        :Outputs:
            *d*: :class:`dict`
                Namelist sampled for phase and case indices
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; ``OptionsDict`` tools
            * 2023-05-18 ``@ddalle``: v2.1; fork get_namelist()
        """
        # Sample list -> scalar, evaluate @expr, etc.
        return self.sample_dict(self, j=j, **kw)


# Class for "moving_body" namelist settings
class MovingBodyFun3DNmlOpts(OptionsDict):
    r"""Dictionary-based interface for FUN3D namelists"""

    # Reduce to a single run sequence
    def select_moving_body_input(self, j=0, **kw):
        r"""Sample "dual" namelist at particular conditions

        :Call:
            >>> d = opts.select_moving_body_input(i)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase index
        :Outputs:
            *d*: :class:`dict`
                Namelist sampled for phase and case indices
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; ``OptionsDict`` tools
            * 2023-05-18 ``@ddalle``: v2.1; fork get_namelist()
        """
        # Sample list -> scalar, evaluate @expr, etc.
        return self.sample_dict(self, j=j, **kw)
