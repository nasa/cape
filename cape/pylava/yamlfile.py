r"""
:mod:`cape.pylava.yamlfile`: Custom class for LAVA input YAML files
---------------------------------------------------------------------

This module provides the class :class:`RunYAMLFile` that interacts with
the YAML (Yet Another Markup Language) files used to control settings
for the various LAVA solvers.
"""

# Standard library
from typing import Any

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
    # Class atributes
    __slots__ = ()
    _ignore_case = True
    _lower_case = True

    # Get a section
    def make_section(self, sec: str) -> OptionsDict:
        r"""Get or create a subsection as on :class:`OptionsDict`

        :Call:
            >>> d = opts.make_section(sec)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *sec*: :class:`str`
                Name of section
        :Outputs:
            *d*: :class:`OptionsDict`
                Contents of that section; added to *opts*
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Check if section is present
        if sec in self:
            # Get it
            d = self[sec]
        else:
            d = dict()
        # Check type
        if isinstance(d, OptionsDict):
            # Already correct type
            return d
        else:
            # Convert dict -> OptionsDict
            opts = OptionsDict(d)
            self[sec] = opts
            # Output
            return opts

    # Get general parameter
    def get_lava_subopt(self, sec: str, opt: str) -> Any:
        # Get/Create the section
        secopts = self.make_section(sec.lower())
        # Get options
        return secopts.get_opt(opt.lower())

    # Set generic parameter
    def set_lava_subopt(self, sec: str, opt: str, v: Any):
        # Get/create the section'
        secopts = self.make_section(sec.lower())
        # Save value
        secopts[opt.lower()] = v

    # Get angle of attack
    def get_alpha(self) -> float:
        return self.get_lava_subopt("referenceconditions", "alpha")

    # Set angle of attack
    def set_alpha(self, alpha: float):
        r"""Set the angle of attack

        :Call:
            >>> opts.set_alpha(alpha)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_lava_subopt("referenceconditions", "alpha")
