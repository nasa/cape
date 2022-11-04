r"""
:mod:`cape.pyfun.options.configopts`: FUn3D surface config opts
===============================================================

This module provides options for defining some aspects of the surface
configuration for FUN3D. It is mostly the same as

    :mod:`cape.cfdx.options.configopts`

The ``"Config"`` section defines which components are requested by
FUN3D for iterative force & moment history reporting. For the moment
histories, this section also specifies the moment reference points
(moment center in FUN3D nomenclature) for each component.

:See Also:
    * :mod:`cape.cfdx.options.configopts`
    * :mod:`cape.config`
"""


# Standard library
from os.path import join

# Local imports
from ...cfdx.options import configopts


# Class for PBS settings
class ConfigOpts(configopts.ConfigOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Inputs",
        "KineticDataFile",
        "MovingBodyInputFile",
        "RubberDataFile",
        "SpeciesThermoDataFile",
        "TDataFile",
    }

    # Aliases
    _optmap = {
        "KineticData": "KineticDataFile",
        "MovingBodyInput": "MovingBodyInputFile",
        "RubberData": "RubberDataFile",
        "SpeciesThermoData": "SpeciesThermoDataFile",
        "TData": "TDataFile",
        "kinetic_data": "KineticDataFile",
        "moving_body.input": "MovingBodyInputFile",
        "rubber.data": "RubberDataFile",
        "species_thermo_data": "SpeciesThermoDataFile",
        "tdata": "TDataFile",
    }

    # Types
    _opttypes = {
        "Inputs": dict,
        "KineticDataFile": str,
        "MovingBodyInputFile": str,
        "RubberDataFile": str,
        "SpeciesThermoDataFile": str,
        "TDataFile": str,
    }

    # Defaults
    _rc = {
        "KineticDataFile": join("inputs", "kinetic_data"),
        "MovingBodyInputFile": join("inputs", "moving_body.input"),
        "RubberDataFile": join("inputs", "rubber.data"),
        "SpeciesThermoDataFile": join("inputs", "species_thermo_data"),
        "TDataFile": join("inputs", "tdata"),
    }

    # Descriptions
    _rst_descriptions = {
        "Inputs": "dictionary of component indices for named comps",
        "KineticDataFile": "template ``kinetic_data`` file",
        "MovingBodyInputFile": "template ``moving_body.input`` file",
        "RubberDataFile": "template for ``rubber.data`` file",
        "SpeciesThermoDataFile": "template ``species_thermo_data`` file",
        "TDataFile": "template for ``tdata`` file",
    }

   # ------------------
   # Component Mapping
   # ------------------
   # [
    # Get inputs for a particular component
    def get_ConfigInput(self, comp):
        r"""Return the input for a particular component

        :Call:
            >>> inp = opts.get_ConfigInput(comp)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Outputs:
            *inp*: :class:`str` | :class:`list`\ [:class:`int`]
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: Version 1.0
        """
        # Get the inputs.
        conf_inp = self.get_opt("Inputs", vdef={})
        # Get the definitions
        return conf_inp.get(comp)

    # Set inputs for a particular component
    def set_ConfigInput(self, comp, inp):
        r"""Set the input for a particular component

        :Call:
            >>> opts.set_ConfigInput(comp, nip)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *inp*: :class:`str` | :class:`list`\ [:class:`int`]
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: Version 1.0
        """
        # Ensure the field exists.
        self.setdefault("Inputs", {})
        # Set the value.
        self["Inputs"][comp] = inp
   # ]


# Add properties
ConfigOpts.add_properties(ConfigOpts._optlist)
