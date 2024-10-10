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

    def get_refcond(self, opt: str) -> Any:
        # Fixed section name
        sec = "referenceconditions"
        # Use generic method
        return self.get_lava_subopt(sec, opt)

    def set_refcond(self, opt: str, v: Any):
        # Fixed section name
        sec = "referenceconditions"
        # Use generic method
        self.set_lava_subopt(sec, opt, v)

    def get_alpha(self) -> float:
        r"""Get the angle of attack

        :Call:
            >>> alpha = opts.get_alpha()
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
        :Outputs:
            *alpha*: :class:`float`
                Angle of attack [deg]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        return self.get_refcond("alpha")

    def set_alpha(self, alpha: float):
        r"""Set the angle of attack

        :Call:
            >>> opts.set_alpha(alpha)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *alpha*: :class:`float`
                Angle of attack [deg]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_refcond("alpha", alpha)

    def get_gamma(self) -> float:
        r"""Get the ratio of specific heats (*gamma*)

        :Call:
            >>> gam = opts.get_gamma()
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
        :Outputs:
            *gam*: :class:`float`
                Ratio of specific heats
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        return self.get_refcond("gamma")

    def set_gamma(self, gam: float):
        r"""Set the ratio of specific heats (*gamma*)

        :Call:
            >>> opts.get_gam(gam)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *gam*: :class:`float`
                Ratio of specific heats
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_refcond("gamma", gam)

    def get_pressure(self) -> float:
        r"""Get the freestream/farfield pressure [Pa]

        :Call:
            >>> p = opts.get_pressure()
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
        :Outputs:
            *p*: :class:`float`
                Static pressure [Pa]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        return self.get_refcond("pressure")

    def set_pressure(self, p: float):
        r"""Set the freestream/farfield pressure [Pa]

        :Call:
            >>> opts.set_pressure(p)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *p*: :class:`float`
                Static pressure [Pa]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_refcond("pressure", p)

    def get_temperature(self) -> float:
        r"""Get the freestream/farfield temperature [K]

        :Call:
            >>> tinf = opts.get_temperature()
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
        :Outputs:
            *tinf*: :class:`float`
                Static temperature [K]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        return self.get_refcond("temperature")

    def set_temperature(self, tinf: float):
        r"""Get the freestream/farfield temperature [K]

        :Call:
            >>> opts.set_temperature(tinf)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *tinf*: :class:`float`
                Static temperature [K]
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_refcond("temperature", tinf)

    def get_umag(self) -> float:
        r"""get the velocity magnitude [m/s]

        :Call:
            >>> u = opts.get_umag()
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
        :Outputs:
            *u*: :class:`float`
                Velocity magnitude
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        return self.get_refcond("umag")

    def set_umag(self, u: float):
        r"""Set the velocity magnitude [m/s]

        :Call:
            >>> opts.set_umag(u)
        :Inputs:
            *opts*: :class:`RunYAMLFile`
                LAVA YAML file interface
            *u*: :class:`float`
                Velocity magnitude
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.set_refcond("umag", u)
