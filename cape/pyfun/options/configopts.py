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
from ...optdict import INT_TYPES
from ...optdict.optitem import getel
from ...cfdx.options import configopts


# Class for "Config" section
class ConfigOpts(configopts.ConfigOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Inputs",
        "KeepTemplateComponents",
        "KineticDataFile",
        "MovingBodyInputFile",
        "MovingBodyDefns",
        "RubberDataFile",
        "SpeciesThermoDataFile",
        "TDataFile",
    }

    # Aliases
    _optmap = {
        "KeepComponents": "KeepTemplateComponents",
        "KeepInputs": "KeepTemplateComponents",
        "KineticData": "KineticDataFile",
        "MovingBodyBodies": "MovingBodyDefns",
        "MovingBodyDefinitions": "MovingBodyDefns",
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
        "MovingBodyDefns": INT_TYPES + (str,),
        "MovingBodyInputFile": str,
        "RubberDataFile": str,
        "SpeciesThermoDataFile": str,
        "TDataFile": str,
    }

    # Items required to be a list
    _optlistdepth = {
        "MovingBodyDefns": 1,
    }

    # Defaults
    _rc = {
        "KeepTemplateComponents": False,
        "KineticDataFile": join("inputs", "kinetic_data"),
        "MovingBodyInputFile": join("inputs", "moving_body.input"),
        "RubberDataFile": join("inputs", "rubber.data"),
        "SpeciesThermoDataFile": join("inputs", "species_thermo_data"),
        "TDataFile": join("inputs", "tdata"),
    }

    # Descriptions
    _rst_descriptions = {
        "Inputs": "dictionary of component indices for named comps",
        "KeepTemplateComponents": "add to template ``component_parameters``",
        "KineticDataFile": "template ``kinetic_data`` file",
        "MovingBodyDefns": "definitions for components in each moving body",
        "MovingBodyInputFile": "template ``moving_body.input`` file",
        "RubberDataFile": "template for ``rubber.data`` file",
        "SpeciesThermoDataFile": "template ``species_thermo_data`` file",
        "TDataFile": "template for ``tdata`` file",
    }

   # --- Component Mapping ---
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

   # --- MovingBodys ---
    def get_ConfigNBody(self) -> int:
        r"""Get a number of moving bodies defined in "Config" section

        :Call:
            >>> n = opts.get_ConfigNBody()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                Number of moving bodies
        :Versions:
            * 2025-05-15 ``@ddalle``: v1.0
        """
        # Get MovingBodyDefns opt
        defns = self.get_opt("MovingBodyDefns", j=None)
        # Return length
        return 0 if defns is None else len(defns)

    def get_ConfigMovingBody(self, k: int) -> list:
        r"""Get a list of component names or indices for a moving body

        :Call:
            >>> comps = opts.get_ConfigMovingBody(k)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *k*: :class:`int`
                Body number (0-based)
        :Outputs:
            *comps*: :class:`list`\ [:class:`str` | :class:`int`]
                List of components
        :Versions:
            * 2025-05-15 ``@ddalle``: v1.0
        """
        # Get option
        defns = self.get_opt("MovingBodyDefns", vdef=[])
        # Ensure list
        defns = defns if isinstance(defns, list) else [defns]
        # Subset to entry *k*
        comps = getel(defns, j=k)
        # Ensure list again
        return comps if isinstance(comps, list) else [comps]


# Add properties
ConfigOpts.add_properties(ConfigOpts._optlist)
