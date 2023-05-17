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

    # Reduce to a single run sequence
    def select_namelist(self, j=0, **kw):
        r"""Sample namelist at particular conditions

        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *d*: :class:`dict`
                Project namelist
        :Versions:
            * 2016-02-01 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        # Sample list -> scalar, evalue @expr, etc.
        return self.sample_dict(self, j=j, **kw)

    # Get value by name
    def get_namelist_var(self, sec, key, j=None, **kw):
        r"""Select a namelist key from a specified section

        Roughly, this returns ``opts[sec][key]``.

        :Call:
            >>> val = opts.get_namelist_var(sec, key, j=None, **kw)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
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
            * 2016-02-01 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; use ``OptionsDict``
        """
        # Get subsection options
        return self.get_subopt(sec, key, j=j, **kw)

