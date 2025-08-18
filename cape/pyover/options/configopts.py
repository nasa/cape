r"""
:mod:`cape.pyover.options.configopts`: OVERFLOW mesh config opts
==================================================================

This module provides options for defining some aspects of the mesh
configuration for OVERFLOW. It is mostly the same as

    :mod:`cape.cfdx.options.configopts`
"""


# Local imports
from ...cfdx.options import configopts


# Class for "Config" section
class ConfigOpts(configopts.ConfigOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Mixsur",
        "Splitmq",
        "TriqMethod",
        "Usurp",
    }

    # Aliases
    _optmap = {
        "MixsurI": "Mixsur",
        "SplitmqI": "Splitmq",
        "SplitmxI": "Splitmx",
        "TriMethod": "TriqMethod",
        "UsurpI": "Usurp",
        "mixsur": "Mixsur",
        "splitmq": "Splitmq",
        "splitmx": "Splitmx",
        "usurp": "Usurp",
    }

    # Types
    _opttypes = {
        "mixsur": str,
        "splitmq": str,
        "usurp": str,
    }

    # Defaults
    _rc = {
        "Mixsur": "mixsur.i",
        "TriqMethod": "mixsur",
        "Usurp": "mixsur.i",
    }

    # Descriptions
    _rst_descriptions = {
        "Mixsur": "input file for ``mixsur``, ``overint``, or ``usurp``",
        "Splitmq": "input file for ``splitmq``",
        "Splitmx": "input file for ``splitmx``",
        "TriqMethod": "method to use to create ``grid.i.triq``",
        "Usurp": "input file for ``usurp``",
    }


# Add properties
ConfigOpts.add_properties(ConfigOpts._optlist, prefix="Config")
