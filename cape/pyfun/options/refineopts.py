r"""
:mod:`cape.pyfun.options.refineopts`: FUN3D refine options
=================================================================

Options interface for aspects of running ``Refine`` with Fun3D for
grid adaptation.

"""

# Local imports
from ...cfdx.options.execopts import ExecOpts


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
        * 2023-06-29 ``@jmeeroff``: v1.0
    """
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "options for FUN3D's ``refine`` command"

    # Accepted options
    _optlist = {
        "input_grid",
        "output_grid",
    }

    # Aliases
    _optmap = {
        "i": "input_grid",
        "o": "output_grid",
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

    # General option getter
    def get_RefineTranslateOpt(self, opt: str, j=None, **kw):
        return self.get_opt(opt, j=j, **kw)

    # General option setter
    def set_RefineTranslateOpt(self, opt: str, val, j=None, **kw):
        self.set_opt(opt, val, j=j, **kw)


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
        "input_grid",
        "mapbc",
        "dist_solb",
    }

    # Types
    _opttypes = {
        "input_grid": str,
        "mapbc": str,
        "dist_solb": str,
    }

    # Map
    _optmap = {
        "grid": "input_grid",
    }

    # Allowed Values
    _optvals = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
        "input_grid": "base grid (ugrid) to compute cell distances for refine",
        "mapbc": "FUN3D BC file for define walls in grid for refine",
        "dist_solb": "computed distances needed for refine adaptaion",
    }

   # General option getter
    def get_RefineDistanceOpt(self, opt: str, j=None, **kw):
        return self.get_opt(opt, j=j, **kw)

    # General option setter
    def set_RefineDistanceOpt(self, opt: str, val, j=None, **kw):
        self.set_opt(opt, val, j=j, **kw)


# Class for ref loop cli options
class RefineOpts(ExecOpts):
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
        "complexity",
        "sweeps",
        "initial_complexity",
        "interpolant",
        "ramp_complexity",
        "target_complexity",
        "input",
        "output",
        "mapbc",
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
        "sweeps": 5,
    }

    # Descriptions
    _rst_descriptions = {
        "initial_complexity": "starting projected grid complexity",
        "ramp_complexity": "amount to increase complexity between iterations",
        "target_complexity": "final complexity",
    }

   # General option getter
    def get_RefineOpt(self, opt: str, j=None, **kw):
        return self.get_opt(opt, j=j, **kw)

    # General option setter
    def set_RefineOpt(self, opt: str, val, j=None, **kw):
        self.set_opt(opt, val, j=j, **kw)


# Add properties
RefineOpts.add_properties(
    RefineOpts._optlist, prefix="refine_")
