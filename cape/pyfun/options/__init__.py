r"""
:mod:`cape.pyfun.options`: FUN3D options interface module
==========================================================

This module provides the options interface for :mod:`cape.pyfun`. Many
settings are inherited from :mod:`cape.cfdx.options`, and there are
some additional options specific to FUN3D for pyfun.

:See also:
    * :mod:`cape.cfdx.options`

"""

# Local imports
from . import util
from .runctlopts import RunControlOpts
from .DataBook import DataBook
from .Report import Report
from .fun3dnml import Fun3DNml
from .Mesh import Mesh
from .configopts import ConfigOpts
from .Functional import Functional
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pyfun`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    :Versions:
        * 2014-07-28 ``@ddalle``: Version 1.0
    """
   # ================
   # Class attributes
   # ================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "DualFun3D",
        "Fun3D",
        "Fun3DNamelist",
        "Functional",
        "MovingBodyInput",
    }

    # Aliases
    _optmap = {
        "Namelist": "Fun3DNamelist",
    }

    # Known option types
    _opttypes = {
        "Fun3DNamelist": str,
    }

    # Option default list depth
    _optlistdepth = {
    }

    # Defaults
    _rc = {
        "Fun3DNamelist": "fun3d.nml",
    }

    # Descriptions for methods
    _rst_descriptions = {
        "Fun3DNamelist": "template ``fun3d.nml`` file",
    }

    # New or replaced sections
    _sec_cls = {
        "Config": ConfigOpts,
        "DataBook": DataBook,
        "DualFun3D": Fun3DNml,
        "Fun3D": Fun3DNml,
        "Functional": Functional,
        "Mesh": Mesh,
        "MovingBodyInput": Fun3DNml,
        "Report": Report,
        "RunControl": RunControlOpts,
    }
   # >

   # =============
   # Configuration
   # =============
   # <
    # Initialization hook
    def init_post(self):
        r"""Initialization hook for :class:`Options`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Versions:
            * 2022-10-23 ``@ddalle``: Version 1.0
        """
        # Read the defaults
        defs = util.getPyFunDefaults()
        # Apply the defaults.
        self = util.applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()
   # >


# Add properties
Options.add_properties(
    (
        "Fun3DNamelist",
    ))
# Add methods from subsections
Options.promote_sections()
