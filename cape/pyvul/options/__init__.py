r"""
:mod:`cape.pyvul.options`: VULCAN-CFD options interface module
===============================================================

This module provides the options interface for :mod:`cape.pyvul`. Many
settings are inherited from :mod:`cape.cfdx.options`, and there are
some additional options specific to FUN3D for pyfun.

:See also:
    * :mod:`cape.cfdx.options`

"""

# Local imports
from .configopts import ConfigOpts
from .runctlopts import RunControlOpts
from .vulcaninpopts import VulcanInpOpts
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pyvul`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    """
   # === Class attributes ===
    # Additional attributes
    __slots__ = ()

    # Identifiers
    _name = "CAPE inputs for a FUN3D run matrix"

    # Additional options
    _optlist = {
        "VulcanInpFile",
        "Vulcan",
        "MapBC",
    }

    # Aliases
    _optmap = {
        "BCs": "MapBC",
        "mapbc": "MapBC",
        "VulcanInputFile": "VulcanInpFile",
    }

    # Known option types
    _opttypes = {
        "VulcanInpFile": str,
    }

    # Option default list depth

    # Defaults
    _rc = {
        "VulcanInpFile": "vulcan.inp",
    }

    # Descriptions for methods
    _rst_descriptions = {
        "VulcanInpFile": "template VULCAN-CFD input file, usually ``.inp``",
    }

    # New or replaced sections
    _sec_cls = {
        "Config": ConfigOpts,
        "Fun3D": VulcanInpOpts,
        "RunControl": RunControlOpts,
    }

   # === Configuration ===
    # Initialization hook
    def init_post(self):
        # Add extra folders to path
        self.AddPythonPath()


# Add properties
Options.add_properties(
    (
        "VulcanInpFile",
    ))
# Add methods from subsections
Options.promote_sections()
