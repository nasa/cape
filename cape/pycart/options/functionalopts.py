"""
:mod:`cape.pycart.options.functionalopts`: Objective Function Options
=====================================================================

This module provides an interface for defining Cart3D's output
functional for output-based mesh adaptation.  The options read from
this file are written to the ``$__Design_Info`` section of
``input.cntl`1``. Each output function is a linear combination of terms
where each term can be a component of a force, a
component of a moment, or a point/line sensor.

The following is a representative example of a complex output function.

    .. code-block:: javascript

        "Functional": {
            // Term 1: normal force coefficient on "wings"
            "CN": {
                "type": "optForce",
                "force": 2,
                "frame": 0,
                "weight": 1.0,
                "compID": "wings",
                "J": 0,
                "N": 1,
                "target": 0.0
            },
            // Term 2: 0.5 times side force on "entire"
            "CY": {
                "type": "optForce",
                "force": 1,
                "frame": 0,
                "weight": 0.5,
                "compID": "entire"
            },
            // Term 3: 0.001 times point sensor called "p1"
            "p1": {
                "type": "optSensor",
                "weight": 0.001
            }
        }

See the :ref:`JSON "Functional" section <json-Functional>` for a
description of all available options.
"""


# Import options-specific utilities
from ...optdict import OptionsDict, FLOAT_TYPES, INT_TYPES


# Coefficient
class FunctionalCoeffOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Options
    _optlist = {
        "J",
        "N",
        "compID",
        "force",
        "frame",
        "index",
        "moment",
        "parent",
        "target",
        "type",
        "weight",
    }

    # Types
    _opttypes = {
        "J": INT_TYPES,
        "N": INT_TYPES,
        "compID": str,
        "force": INT_TYPES,
        "frame": INT_TYPES,
        "index": INT_TYPES,
        "moment": INT_TYPES,
        "parent": str,
        "target": FLOAT_TYPES,
        "type": str,
        "weight": FLOAT_TYPES,
    }

    # Values
    _optvals = {
        "J": (0, 1),
        "type": (
            "optForce",
            "optMoment",
            "optSensor",
        ),
        "force": (0, 1, 2),
        "frame": (0, 1),
        "index": (0, 1, 2),
        "moment": (0, 1, 2),
    }

    # Defaults
    _rc = {
        "J": 0,
        "N": 1,
        "force": 0,
        "frame": 0,
        "index": 0,
        "moment": 0,
        "target": 0.0,
        "type": "optForce",
        "weight": 1.0,
    }

    # Descriptions
    _rst_descriptions = {
        "compID": "name of component from which to calculate force/moment",
        "force": "axis number of force to use (0-based)",
        "frame": "force frame; ``0`` for body axes and ``1`` for stability",
        "index": "index of moment reference point to use (0-based)",
        "moment": "axis number of moment to use (0-based)",
        "parent": "name of parent coefficient",
        "target": "target value; functional is ``weight*(F-target)**N``",
        "type": "output type",
        "weight": "weight multiplier for force's contribution to total",
    }


# Class for collection of coefficient defns
class FunctionalOpts(OptionsDict):
    r"""Dictionary-based options for *Functional* section
    """
    # Attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Sections
    _sec_cls_opt = "type"
    _sec_cls_optmap = {
        "_default_": FunctionalCoeffOpts,
    }

    # Get option for a function
    def get_FunctionalCoeffOpt(self, coeff: str, opt: str, j=None, **kw):
        r"""Get option for a specific functional coefficient

        :Call:
            >>> v = opts.get_FunctionalCoeffOpt(coeff, opt, j=None, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *coeff*: :class:`str`
                Name of coefficient
            *opt*: :class:`str`
                Name of functional option
        :Outputs:
            *v*: :class:`object`
                Value of ``opts[fn][opt]`` or as appropriate
        :Versions:
            * 2023-05-16 ``@ddalle``: v1.0
        """
        return self.get_subopt(coeff, opt, j=j, **kw)

    # Function to return all the optForce dicts found
    def filter_FunctionalCoeffsByType(self, typ: str):
        r"""Return a subset of function coefficients by type

        :Call:
            >>> copts = opts.filter_FunctionalCoeffsByType(typ)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *typ*: :class:`str`
                Requested functional type
        :Outputs:
            *copts*: :class:`dict`\ [:class:`FunctionalCoeffOpts`]
                Subset of functional options with correct type
        :Versions:
            * 2023-05-16 ``@ddalle``: v1.0
        """
        # List of functions
        copts = {}
        # Loop through keys
        for coeff, coeffopts in self.items():
            # Check type
            if self.get_FunctionalCoeffOpt(coeff, "type") == typ:
                copts[coeff] = coeffopts
        # Output
        return copts

