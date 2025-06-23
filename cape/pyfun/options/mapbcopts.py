"""
:mod:`cape.pyfun.options.mapbcopts`: MapBC interface options
=============================================================

This module provides options for users to modify a FUN3D ``.mapbc`` file
programmatically. One reason this might be useful is to simplify the
process of changing the boundary condition from one phase to another,
for example turning thrust on only after the first phase is complete.
"""

# Local imports
from ...optdict import INT_TYPES, OptionsDict


# Class for FUN3D mesh settings
class MapBCOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Option types
    _opttypes = {
        "_default_": INT_TYPES,
    }
