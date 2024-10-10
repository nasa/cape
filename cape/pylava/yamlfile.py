r"""
:mod:`cape.pylava.yamlfile`: Custom class for LAVA input YAML files
---------------------------------------------------------------------

This module provides the class :class:`RunYAMLFile` that interacts with
the YAML (Yet Another Markup Language) files used to control settings
for the various LAVA solvers.
"""

# Standard library

# Local imports
from ..optdict import OptionsDict


# Primary class
class RunYAMLFile(OptionsDict):
    r"""Custon YAML file control class

    :Call:
        >>> obj = RunYAMLFile(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of file to read
    :Outputs:
        *obj*: :class:`RunYAMLFile`
            YAML file interface instance
    :Versions:
        * 2024-10-10 ``@ddalle``: v1.0
    """
    pass
