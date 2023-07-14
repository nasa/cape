r"""
:mod:`cape.pyfun.options.refineopts`: FUN3D refine options
=================================================================

Options interface for aspects of running ``Refine`` with Fun3D for
grid adaptation.

"""

# Local imports
from ...cfdx.options.util import ExecOpts
from ...optdict import BOOL_TYPES, INT_TYPES, FLOAT_TYPES


# Class for ref translate cli options
class RefineTranslateOpts(ExecOpts):
    r"""Class for refine translate command line settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "input_grid",
        "output_grid",
    }

    # Types
    _opttypes = {
        "input_grid": str,
        "output_grid": str,
    }

    # Allowed Values
    _optvals = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
        "input_grid": "base grid (ugrid) to convert prior to adaptation",
        "output_grid": "converted grid (mesb) name needed for refine",
    }


# Add properties
RefineTranslateOpts.add_properties(
    RefineTranslateOpts._optlist, prefix="refine_trans_")


# Class for ref distance cli options
class RefineDistanceOpts(ExecOpts):
    r"""Class for refine distance command line settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "grid",
        "mapbc",
        "dist_solb",
    }

    # Types
    _opttypes = {
        "grid": str,
        "mapbc": str,
        "dist_solb": str,
    }

    # Allowed Values
    _optvals = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
        "grid": "base grid (ugrid) to compute cell distances for refine",
        "mapbc": "FUN3D BC file for define walls in grid for refine",
        "dist_solb": "computed distances needed for refine adaptaion",
    }


# Add properties
RefineDistanceOpts.add_properties(
    RefineDistanceOpts._optlist, prefix="refine_dist_")


# Class for ref loop cli options
class RefineLoopOpts(ExecOpts):
    r"""Class for refine loop command line settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "initial_complexity",
        "ramp_complexity",
        "target_complexity"
    }

    # Types
    _opttypes = {
        "initial_complexity": str,
        "ramp_complexity": str,
        "target_complexity": str,
    }

    # Allowed Values
    _optvals = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
        "initial_complexity": "starting projected grid complexity",
        "ramp_complexity": "amount to increase complexity between iterations",
        "target_complexity": "final complexity",
    }


# Add properties
RefineLoopOpts.add_properties(
    RefineLoopOpts._optlist, prefix="refine_loop_")
