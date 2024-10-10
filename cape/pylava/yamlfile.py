r"""
:mod:`cape.pylava.yamlfile`: Custom class for LAVA input YAML files
---------------------------------------------------------------------

This module provides the class :class:`RunYAMLFile` that interacts with
the YAML (Yet Another Markup Language) files used to control settings
for the various LAVA solvers.
"""

# Standard library
from typing import Optional

# Third-party
import yaml

# Local imports


# Primary class
class RunYAMLFile(dict):
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
    # Initialization
    def __init__(self, fname: Optional[str] = None):
        ...

    # Read YAML file
    def read_yaml(self, fname: str):
        # Read file
        with open(fname, 'r') as fp:
            raw = yaml.safe_load(fp)
        # Save settings
        self.apply(raw)

    # Save group of settings
    def save_dict(self, raw: dict):
        ...
