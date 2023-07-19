r"""
Basic options for CAPE run matrix options

This module provides the class :class:`RunMatrixOpts`, which reads the
settings in the ``"RunMatrix"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict, BOOL_TYPES, FLOAT_TYPES


# Float + string
SF_TYPES = FLOAT_TYPES + (str,)


# Map of alternate values for key "Type"
KEY_TYPEMAP = {
    "ALPHA": "alpha",
    "ALPHA_P": "aoap",
    "ALPHA_T": "aoap",
    "ALPHA_TOTAL": "aoap",
    "AOA": "alpha",
    "AOAP": "aoap",
    "AOS": "beta",
    "Alpha": "alpha",
    "Alpha_p": "aoap",
    "Alpha_t": "aoap",
    "Alpha_total": "aoap",
    "BETA": "beta",
    "Beta": "beta",
    "M": "mach",
    "MACH": "mach",
    "Mach": "mach",
    "P_INF": "p",
    "P_inf": "p",
    "PINF": "p",
    "PHI": "phip",
    "PHIP": "phip",
    "PHI_P": "phip",
    "Pressure": "p",
    "Q_BAR": "q",
    "Q_INF": "q",
    "QBAR": "q",
    "QINF": "q",
    "RE": "rey",
    "REY": "rey",
    "REYNOLDS": "rey",
    "REYNOLDS_NUMBER": "rey",
    "T_INF": "T",
    "T_inf": "T",
    "TINF": "T",
    "Tinf": "T",
    "alpha_p": "aoap",
    "alpha_t": "aoap",
    "alpha_total": "aoap",
    "aoa": "alpha",
    "aos": "alpha",
    "m": "mach",
    "p_inf": "p",
    "phi": "phip",
    "phi_p": "phip",
    "pinf": "p",
    "pressure": "p",
    "re": "rey",
    "reynolds": "rey",
    "reynolds_number": "rey",
    "temp": "T",
    "temperature": "T",
}


# Option for a definition
class KeyDefnOpts(OptionsDict):
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
        "Value": ("float", "int", "str"),
    }

    # Defaults
    _rc = {
        "Format": "%s",
        "Group": False,
        "Label": True,
        "Type": "value",
        "Value": "float",
    }


# Definitions for Mach number
class MachKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "m",
    }


# Definitions for angle of attack
class AlphaKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "a",
    }


# Definitions for total angle of attack
class AOAPKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "a",
    }


# Definitions for sideslip
class BetaKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "b",
    }


# Definitions for missile-axis roll
class PhiPKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "r",
    }


# Definitions for Reynolds number per length
class ReynoldsKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "Re",
        "Label": False,
    }


# Definitions for static temperature
class TemperatureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "T",
        "Label": False,
    }


# Definitions for static pressure
class PressureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "p",
        "Label": False,
    }


# Definitions for dynamic pressure
class DynamicPressureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "q",
        "Label": False,
    }


# Definitions with frestream state
class SurfCPKeyDefnOpts(KeyDefnOpts):
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
class KeyDefnCollectionOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Section map
    _sec_cls_opt = "Type",
    _sec_cls_optmap = {
        "_default_": KeyDefnOpts,
        "T": TemperatureKeyDefnOpts,
        "alpha": AlphaKeyDefnOpts,
        "aoap": AOAPKeyDefnOpts,
        "beta": BetaKeyDefnOpts,
        "mach": MachKeyDefnOpts,
        "p": PressureKeyDefnOpts,
        "phip": PhiPKeyDefnOpts,
        "q": DynamicPressureKeyDefnOpts,
        "rey": ReynoldsKeyDefnOpts,
    }

    # Preprocess
    def preprocess_dict(self, a: dict):
        r"""Preprocess collection of run matrix key definitions

        :Call:
            >>> opts.preprocess_dict(a)
        :Inputs:
            *opts*: :class:`RunMatrixDefnCollection`
                Options interface for "RunMatrix" > "Definitions"
        :Versions:
            * 2023-07-18 ``@ddalle``: v1.0
        """
        # Loop through items
        for k in a:
            # Get value
            v = a[k]
            # Check if a dictionary
            if isinstance(v, dict):
                # Set *Type* to *k* if not present
                vtyp = v.setdefault("Type", k)
                # Standardize type
                typ = KEY_TYPEMAP.get(vtyp, vtyp)
                # Re-set
                v["Type"] = typ


# Class for generic mesh settings
class RunMatrixOpts(OptionsDict):
    # No attbitues
    __slots__ = ()

    # List of options
    _optlist = {
        "Definitions",
        "File",
        "Freestream",
        "GroupMesh",
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
        "GroupMesh": BOOL_TYPES,
        "GroupPrefix": str,
        "Prefix": str,
    }

    # List depth
    _optlistdepth = {
        "Keys": 1,
    }

    # Defaults
    _rc = {
        "GroupMesh": False,
        "GroupPrefix": "Grid",
        "Keys": ["mach", "alpha", "beta"],
        "Prefix": "",
    }

    # Sections
    _sec_cls = {
        "Definitions": KeyDefnCollectionOpts,
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
