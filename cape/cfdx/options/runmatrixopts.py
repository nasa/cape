r"""
Basic options for CAPE run matrix options

This module provides the class :class:`RunMatrixOpts`, which reads the
settings in the ``"RunMatrix"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict, BOOL_TYPES, FLOAT_TYPES, INT_TYPES


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
    "CONFIG": "config",
    "CP": "SurfCP",
    "CT": "SurfCT",
    "Config": "config",
    "DENSITY": "rho",
    "Density": "rho",
    "GAMMA": "gamma",
    "GROUP_LABEL": "GroupLabel",
    "GROUP_PREFIX": "config",
    "GROUP_SUFFIX": "GroupLabel",
    "GROUPPREFIX": "config",
    "Gamma": "gamma",
    "GroupPrefix": "config",
    "GroupSuffix": "GroupLabel",
    "LABEL": "label",
    "Label": "label",
    "M": "mach",
    "MACH": "mach",
    "Mach": "mach",
    "OTHER": "value",
    "Other": "value",
    "P0": "p0",
    "P0_INF": "p0",
    "P0_inf": "p0",
    "P0INF": "p0",
    "P0inf": "p0",
    "P_0": "p0",
    "P_INF": "p",
    "P_TOTAL": "p0",
    "P_inf": "p",
    "PINF": "p",
    "PHI": "phip",
    "PHIP": "phip",
    "PHI_P": "phip",
    "PTOTAL": "p0",
    "Pressure": "p",
    "Q_BAR": "q",
    "Q_INF": "q",
    "QBAR": "q",
    "QINF": "q",
    "RE": "rey",
    "REY": "rey",
    "REYNOLDS": "rey",
    "REYNOLDS_NUMBER": "rey",
    "RHO": "rho",
    "RHO_INF": "rho",
    "RHOINF": "rho",
    "Re": "rey",
    "Rey": "rey",
    "Reynolds": "rey",
    "Reynolds_number": "rey",
    "ReynoldsNumber": "rey",
    "Rho": "rho",
    "SUFFIX": "label",
    "Suffix": "label",
    "SurfCP": "SurfCp",
    "T0_INF": "T0",
    "T0_inf": "T0",
    "T0INF": "T0",
    "T_INF": "T",
    "T_VIBE": "Tv",
    "T_VIBRATION": "Tv",
    "T_WALL": "Tw",
    "T_inf": "T",
    "T_v": "Tv",
    "T_vibe": "Tv",
    "T_vibration": "Tv",
    "T_wall": "Tw",
    "TAG": "tag",
    "TAGS": "tag",
    "TINF": "T",
    "TV": "Tv",
    "TVIBE": "Tv",
    "TVIBRATION": "Tv",
    "TW": "Tw",
    "TWALL": "Tw",
    "Tag": "tag",
    "Tags": "tag",
    "Tinf": "T",
    "TriRotation": "TriTranslate",
    "TriTranslation": "TriTranslate",
    "U": "V",
    "U_INF": "V",
    "U_inf": "V",
    "UID": "user",
    "UINF": "V",
    "USER": "user",
    "USERFILTER": "user",
    "USERNAME": "user",
    "Uinf": "V",
    "V_INF": "V",
    "V_inf": "V",
    "VALUE": "value",
    "VINF": "V",
    "Value": "value",
    "Vinf": "V",
    "alpha_p": "aoap",
    "alpha_t": "aoap",
    "alpha_total": "aoap",
    "aoa": "alpha",
    "aos": "alpha",
    "density": "rho",
    "m": "mach",
    "other": "value",
    "p0_inf": "p0",
    "p0inf": "p0",
    "p_inf": "p",
    "p_total": "p0",
    "phi": "phip",
    "phi_p": "phip",
    "pinf": "p",
    "pressure": "p",
    "qbar": "q",
    "qinf": "q",
    "re": "rey",
    "reynolds": "rey",
    "reynolds_number": "rey",
    "rho_inf": "rho",
    "rhoinf": "rho",
    "rotate": "TriRotate",
    "rotation": "TriRotate",
    "suffix": "label",
    "tags": "tag",
    "temp": "T",
    "temperature": "T",
    "translate": "TriTranslate",
    "translation": "TriTranslate",
    "u": "V",
    "u_inf": "V",
    "uid": "user",
    "uinf": "V",
    "userfilter": "user",
    "username": "user",
    "v": "V",
    "v_inf": "V",
    "vinf": "V",
}


# Option for a definition
class KeyDefnOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # List of options
    _optlist = {
        "Abbreviation",
        "Format",
        "FormatMultiplier",
        "Group",
        "Label",
        "NonnegativeFormat",
        "AbsoluteValueFormat",
        "SkipIfZero",
        "Type",
        "Units",
        "Value",
    }

    # Aliases
    _optmap = {
        "DType": "Value",
        "DataType": "Value",
    }

    # Types
    _opttypes = {
        "Abbreviation": str,
        "Format": str,
        "FormatMultiplier": FLOAT_TYPES,
        "Group": BOOL_TYPES,
        "Label": BOOL_TYPES,
        "NonnegativeFormat": BOOL_TYPES,
        "AbsoluteValueFormat": BOOL_TYPES,
        "SkipIfZero": BOOL_TYPES,
        "Type": str,
        "Units": str,
        "Value": str,
    }

    # Permissible values
    _optvals = {
        "Value": ("float", "int", "str", "bin", "oct", "hex"),
    }

    # Defaults
    _rc = {
        "Format": "%s",
        "FormatMultiplier": 1.0,
        "Group": False,
        "Label": True,
        "NonnegativeFormat": False,
        "AbsoluteValueFormat": False,
        "SkipIfZero": False,
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


# Definitions for freestream vibration temperature
class VibrationTemperatureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "Tv",
        "Label": False,
    }


# Definitions for stagnation (total) temperature
class StagnationTemperatureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "T0",
        "Label": False,
    }


# Definitions for wall temperature
class WallTemperatureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "Tw",
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


# Definitions for stagnation pressure
class StagnationPressureKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "p0",
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


# Definitions for velocity magnitude
class VelocityKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "V",
    }


# Definitions for density
class DensityKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "r",
        "Label": False,
    }


# Definitions for ratio of specific heats
class GammaKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "r",
        "Label": False,
    }


# Defintions for *Label*, etc.
class LabelKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "",
        "Value": "str",
    }


# General key to move components around
class SurfTransformKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "CompID",
        "CompIDSymmetric",
        "Vector",
        "Points",
        "PointsSymmetric",
    )

    # List depth
    _optlistdepth = {
        "CompID": 1,
        "CompIDSymmetric": 1,
        "Vector": 1,
        "Points": 1,
        "PointsSymmetric": 1,
    }


# Config transforation
class SurfConfigTransformKeyDefnOpts(SurfTransformKeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "TransformationIndex",
    )

    # Types
    _opttypes = {
        "TransformationIndex": INT_TYPES,
    }


# Config translation
class ConfigTranslationDefnKeyOpts(SurfConfigTransformKeyDefnOpts):
    # Attributes
    __slots__ = ()


# Config rotation
class ConfigRotationDefnKeyOpts(SurfConfigTransformKeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Extra attributes
    _optlist = (
        "AngleSymmetry",
        "CompIDTranslate",
        "CompIDTranslateSymmetric",
        "FreezeGMPAxis",
        "FreezeGMPCenter",
        "TranslateRefPoint",
        "TranslateScale",
        "VectorSymmetry",
    )

    # Alternates
    _optmap = {
        "AxisSymmetry": "VectorSymmetry",
        "CenterSymmetry": "VectorSymmetry",
    }

    # Types
    _opttypes = {
        "AngleSymmetry": FLOAT_TYPES,
        "FreezeGMPAxis": BOOL_TYPES,
        "FreezeGMPCenter": BOOL_TYPES,
    }

    # List depth
    _optlistdepth = {
        "CompIDTranslate": 1,
        "CompIDTranslateSymmetric": 1,
    }

    # Defaults
    _rc = {
        "AngleSymmetry": -1.0,
        "FreezeGMPAxis": False,
        "FreezeGMPCenter": False,
    }


# Tri translation
class TriTranslationDefnKeyOpts(SurfTransformKeyDefnOpts):
    # Attributes
    __slots__ = ()


# Tri rotation
class TriRotationDefnKeyOpts(SurfTransformKeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "AngleSymmetry",
        "CompIDTranslate",
        "CompIDTranslateSymmetric",
        "TranslateRefPoint",
        "TranslateScale",
        "VectorSymmetry",
    )

    # Alternates
    _optmap = {
        "AxisSymmetry": "VectorSymmetry",
        "CenterSymmetry": "VectorSymmetry",
    }

    # Types
    _opttypes = {
        "AngleSymmetry": FLOAT_TYPES,
    }

    # List depth
    _optlistdepth = {
        "CompIDTranslate": 1,
        "CompIDTranslateSymmetric": 1,
    }

    # Defaults
    _rc = {
        "AngleSymmetry": -1.0,
    },


# Defintions for *tag*, etc.
class TagKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Abbreviation": "tag_",
        "Value": "str",
        "Label": False,
    }


# Definitions for *config* and related group folder names
class ConfigKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Group": True,
        "Value": "str",
        "Abbreviation": "",
    }


# TriFunction definitions
class TriFunctionKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Extra attributes
    _optlist = (
        "Function",
    )

    # Types
    _opttypes = {
        "Function": str,
    }

    # Defaults
    _rc = {
        "Value": "str",
    }


# ConfigFunction definitions
class ConfigFunctionDefnOpts(TriFunctionKeyDefnOpts):
    # Attributes
    __slots__ = ()


# Definitions for *GroupLabel* and related group folder names
class GroupLabelKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Group": True,
        "Value": "str",
        "Abbreviation": "",
    }


# Definitions for placeholder key
class ValueKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Label": False,
    }


# Definitions for "@user" key
class UserKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Label": False,
        "Abbreviation": "user@",
        "Value": "str",
    }


# Definitions for keys setting stagnation pressure on surface
class SurfCPKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # List of additional options
    _optlist = (
        "CompID",
        "PressureCalibration",
        "PressureOffset",
        "RefPressure",
        "RefTemperature",
        "TemperatureCalibration",
        "TemperatureOffset",
        "TotalPressure",
        "TotalTemperature",
    )

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
        "PressureCalibration": 1.0,
        "PressureOffset": 0.0,
        "RefPressure": 1.0,
        "RefTemperature": 1.0,
        "TemperatureCalibration": 1.0,
        "TemperatureOffset": 0.0,
        "TotalTemperature": "T0",
    }


# Definitions for keys setting thrust coefficient
class SurfCTKeyDefnOpts(SurfCPKeyDefnOpts):
    # Attributes
    __slots__ = ()

    # List of additional options
    _optlist = (
        "AreaRatio",
        "ExitArea",
        "ExitMach",
        "RefArea",
        "RefDynamicPressure",
    )

    # Types
    _opttypes = {
        "ExitArea": FLOAT_TYPES,
        "ExitMach": FLOAT_TYPES,
    }

    # Defaults
    _rc = {
        "RefPressure": None,
        "RefTemperature": None,
    }


# Class for modifying a line of an XML file
class XMLInputKeyDefnOpts(KeyDefnOpts):
    # Attributes
    __slots__ = ()

    # List of additional options
    _optlist = (
        "Name",
    )

    # Types
    _opttypes = {
        "Name": "str",
    }


# Class for a collection of definitions
class KeyDefnCollectionOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Section map
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": KeyDefnOpts,
        "ConfigFunction": ConfigFunctionDefnOpts,
        "ConfigRotate": ConfigRotationDefnKeyOpts,
        "ConfigTranslate": ConfigTranslationDefnKeyOpts,
        "GroupLabel": GroupLabelKeyDefnOpts,
        "SurfCp": SurfCPKeyDefnOpts,
        "SurfCT": SurfCTKeyDefnOpts,
        "T": TemperatureKeyDefnOpts,
        "T0": StagnationTemperatureKeyDefnOpts,
        "TriFunction": TriFunctionKeyDefnOpts,
        "TriRotate": TriRotationDefnKeyOpts,
        "TriTranslate": TriTranslationDefnKeyOpts,
        "Tv": VibrationTemperatureKeyDefnOpts,
        "Tw": WallTemperatureKeyDefnOpts,
        "V": VelocityKeyDefnOpts,
        "XMLInput": XMLInputKeyDefnOpts,
        "alpha": AlphaKeyDefnOpts,
        "aoap": AOAPKeyDefnOpts,
        "beta": BetaKeyDefnOpts,
        "config": ConfigKeyDefnOpts,
        "gamma": GammaKeyDefnOpts,
        "label": LabelKeyDefnOpts,
        "mach": MachKeyDefnOpts,
        "p": PressureKeyDefnOpts,
        "p0": StagnationPressureKeyDefnOpts,
        "phip": PhiPKeyDefnOpts,
        "q": DynamicPressureKeyDefnOpts,
        "rey": ReynoldsKeyDefnOpts,
        "tag": TagKeyDefnOpts,
        "user": UserKeyDefnOpts,
        "value": ValueKeyDefnOpts,
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
        "Values",
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
        "Values": dict,
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
        "Freestream": {},
        "Values": {},
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

    # For 1.0 compatibility: shift raw options -> "Values" section
    def preprocess_dict(self, a: dict):
        r"""Preprocess "RunMatrix" options

        In CAPE 1.0, users may specify values in the top level of
        *RunMatrix*. These should be in *RunMatrix* > *Values*, and this
        function will move them there.

        :Call:
            >>> opts.preprocess_dict(a)
        :Inputs:
            *opts*: :class:`RunMatrixOpts`
                Run matrix options instance
            *a*: :class:`dict`
                Unprocessed options before ``opts.set_opts(a)``
        :Versions:
            * 2023-07-20 ``@ddalle``: v1.0
        """
        # Get keys
        cols = a.get("Keys", [])
        # Get expected potential options
        optlist = self.getx_cls_set("_optlist")
        # Loop through *a* to find raw values
        # (Have to create a list b/c *a.keys()* may change during loop)
        for k in list(a.keys()):
            # Check if *k* is a run matrix key
            if (k in cols) and (k not in optlist):
                # Get "Values" section; create if necessary
                vals = a.setdefault("Values", {})
                # Remove it and put it into the "Values" section
                vals.setdefault(k, a.pop(k))


# Add getters/setters
RunMatrixOpts.add_properties(RunMatrixOpts._optlist, prefix="RunMatrix")
