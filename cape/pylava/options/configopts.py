r"""
:mod:`cape.pylava.options.configopts`: LAVA surface config opts
================================================================

This module provides options for defining some aspects of the surface
configuration for LAVA. It is mostly the same as

    :mod:`cape.cfdx.options.configopts`

The ``"Config"`` section defines which components are requested by
LAVA for iterative force & moment history reporting. For LAVA-
Cartesian, these are written as ``group_N`` subsections of the
``output.loads`` section of the ``run.inputs`` file.

:See Also:
    * :mod:`cape.cfdx.options.configopts`
    * :mod:`cape.config`
"""

# Local imports
from ...cfdx.options import configopts


# Class for "Config" section
class ConfigOpts(configopts.ConfigOpts):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "KeepTemplateComponents",
    }

    # Aliases
    _optmap = {
        "KeepComponents": "KeepTemplateComponents",
        "KeepGroups": "KeepTemplateComponents",
    }

    # Defaults
    _rc = {
        "KeepTemplateComponents": False,
    }

    # Descriptions
    _rst_descriptions = {
        "KeepTemplateComponents": "add to template load groups",
    }


# Add properties
ConfigOpts.add_properties(ConfigOpts._raw_optlist)
