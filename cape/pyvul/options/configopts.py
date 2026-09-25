r"""
:mod:`cape.pyvul.options.configopts`: VULCAN surface config options
===================================================================

This module provides options for defining some aspects of the surface
configuration for VULCAN-CFD. It is mostly the same as

    :mod:`cape.cfdx.options.configopts`

The new option ``"FarfieldComponents"`` lists which boundary condition
groups (usually ``FIX_IN`` types) receive the freestream flow state.
If the list is empty, all ``FIX_IN`` groups get the freestream state.

:See Also:
    * :mod:`cape.cfdx.options.configopts`
    * :mod:`cape.pyvul.inpfile`
"""

# Local imports
from ...cfdx.options import configopts


# Class for "Config" section
class ConfigOpts(configopts.ConfigOpts):
    r"""Options class for VULCAN-CFD configuration

    :Call:
        >>> opts = ConfigOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of configuration
    :Outputs:
        *opts*: :class:`cape.pyvul.options.configopts.ConfigOpts`
            VULCAN component configuration option interface
    """
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "FarfieldComponents",
    }

    # Aliases
    _optmap = {
        "FarfieldBCs": "FarfieldComponents",
    }

    # Types
    _opttypes = {
        "FarfieldComponents": str,
    }

    # Items required to be a list
    _optlistdepth = {
        "FarfieldComponents": 1,
    }

    # Defaults
    _rc = {
        "FarfieldComponents": [],
    }

    # Descriptions
    _rst_descriptions = {
        "FarfieldComponents": "BC groups that receive freestream state",
    }


# Add properties
ConfigOpts.add_properties(ConfigOpts._raw_optlist)
