r"""
Basic options for CAPE run matrix options

This module provides the class :class:`RunMatrixOpts`, which reads the
settings in the ``"RunMatrix"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict, BOOL_TYPES, FLOAT_TYPES


# Float + string
SF_TYPES = FLOAT_TYPES + (str,)


# Option for a definition
class RunMatrixDefn(OptionsDict):
    # No attributes
    __slots__ = ()

    # List of options
    _optlist = {
        "Abbreviation",
        "Format",
        "Group",
        "Label",
        "Type",
        "Value",
    }

    # Types
    _opttypes = {
        "Abbreviation": str,
        "Format": str,
        "Group": BOOL_TYPES,
        "Label": BOOL_TYPES,
        "Type": str,
        "Value": str,
    }

    # Permissible values
    _optvals = {
        "Type": ("float", "int", "str"),
    }

    # Defaults
    _rc = {
        "Format": "%s",
        "Group": False,
        "Label": True,
        "Type": "value",
        "Value": "float",
    }


# Definitions with frestream state
class RunMatrixPDefn(RunMatrixDefn):
    # Attributes
    __slots__ = ()

    # List of additional options
    _optlist = {
        "CompID",
        "RefPressure",
        "RefTemperature",
        "TotalPressure",
        "TotalTemperature",
    }

    # Types
    _opttypes = {
        "CompID": str,
        "RefPressure": SF_TYPES,
        "RefTemperature": SF_TYPES,
        "TotalPressure": SF_TYPES,
        "TotalTemperature": SF_TYPES,
    }

    # List depth
    _optlistdepth = {
        "CompID": 1,
    }

    # Defaults
    _rc = {
        "CompID": [],
        "RefPressure": 1.0,
        "RefTemperature": 1.0,
        "TotalTemperature": "T0",
    }


# Class for a collection of definitions
class RunMatrixDefnCollection(OptionsDict):
    # No attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Section map
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": RunMatrixDefn,
    }


# Class for generic mesh settings
class RunMatrixOpts(OptionsDict):
    # No attbitues
    __slots__ = ()

    # List of options
    _optlist = {
        "Definitions",
        "File",
        "Freestream",
        "GroupPrefix",
        "Keys",
        "Prefix",
    }

    # Aliases
    _optmap = {
        "Cols": "Keys",
        "Defns": "Definitions",
        "cols": "Keys",
        "defns": "Definitions",
        "file": "File",
        "gas": "Freestream",
        "keys": "Keys",
        "prefix": "Prefix",
    }

    # Types
    _opttypes = {
        "Definitions": dict,
        "Keys": str,
        "File": str,
        "Freestream": dict,
        "GroupPrefix": str,
        "Prefix": str,
    }

    # List depth
    _optlistdepth = {
        "Keys": 1,
    }

    # Defaults
    _rc = {
        "GroupPrefix": "Grid",
        "Keys": ["mach", "alpha", "beta"],
        "Prefix": "",
    }

    # Sections
    _sec_cls = {
        "Definitions": RunMatrixDefnCollection,
    }

    # Descriptions
    _rst_descriptions = {
        "Definitions": "definitions for each run matrix variable",
        "File": "run matrix data file name",
        "Freestream": "properties of freestream gas model",
        "GroupPrefix": "default prefix for group folders",
        "Keys": "list of run matrix variables",
        "Prefix": "default prefix for case folders",
    }


# Add getters/setters
RunMatrixOpts.add_properties(RunMatrixOpts._optlist, prefix="RunMatrix")
