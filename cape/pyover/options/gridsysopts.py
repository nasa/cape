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
    * :mod:`cape.pyover.options.overnmlopts`
    * :mod:`cape.pyover.overNamelist`
    * :mod:`cape.pyover.cntl`
    * :mod:`cape.filecntl.namelist2`
"""

# Local imports
from ...optdict import OptionsDict


# Class for grid namelist settings
class GridSystemNmlOpts(OptionsDict):
    r"""Interface to OVERFLOW namelist grid system options"""

    # Get the ALL namelist
    def get_ALL(self, j=None, **kw):
        r"""Return the ``ALL`` namelist of settings applied to all grids

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
            * 2016-02-01 ``@ddalle``: v1.0
        """
        return self.get_GridByName('ALL', j, **kw)

    # Select a grid
    def get_GridByName(self, grdnam, j=None, **kw):
        r"""Return a dictionary of options for a specific grid

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
            *d*: :class:`OptionsDict`
                Dictionary of options for grid *gridnam*
        :Versions:
            * 2016-02-01 ``@ddalle``: v1.0
        """
        # Dont' force sampling
        kw.setdefault("f", False)
        # Sample dictionary w/o single-phase req
        return self.sample_dict(self.get(grdnam, {}), j=j, **kw)

