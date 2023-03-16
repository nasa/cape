r"""
:mod:`cape.pycart.options`: pyCart options interface module
============================================================

This module provides tools to read, access, modify, and write settings
for :mod:`cape.pycart`. It is a fork of

    :mod:`cape.cfdx.options`

with several Cart3D-specific options and provides the class

    :mod:`cape.pycart.options.Options`

:See Also:
    * :mod:`cape.cfdx.options`
    * :mod:`cape.pycart.options.runctlopts`
"""

# Local imports
from .runctlopts import RunControlOpts
from .meshopts import MeshOpts
from .configopts import ConfigOpts
from .Functional import Functional
from .databookopts import DataBookOpts
from .Report import Report
from .util import get_pycart_defaults, applyDefaults
from ...cfdx import options
from ...optdict import ARRAY_TYPES


# Class definition
class Options(options.Options):
    r"""Options class for :mod:`cape.pycart`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary of raw options
    :Outputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2014-07-28 ``@ddalle``: Version 1.0
        * 2022-11-04 ``@ddalle``: Version 2.0; use :mod:`optdict`
    """
   # =================
   # Class attributes
   # =================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "AeroCsh",
        "Functional",
        "InputCntl",
    }

    # Aliases
    _optmap = {
        "aero.csh": "AeroCsh",
        "input.cntl": "InputCntl",
        "inputCntl": "InputCntl",
    }

    # Types
    _opttypes = {
        "InputCntl": str,
    }

    # Defaults
    _rc = {
        "AeroCsh": "aero.csh",
        "InputCntl": "input.cntl",
        "ZombieFiles": [
            "*.out",
            "history.dat",
            "adapt??/history.dat",
            "adapt??/*.out"
        ],
    }

    # Descriptions
    _rst_descriptions = {
        "AeroCsh": "template ``aero.csh`` file",
        "InputCntl": "template ``input.cntl`` file",
    }

    # Additional or replaced sections
    _sec_cls = {
        "RunControl": RunControlOpts,
        "Config": ConfigOpts,
        "Functional": Functional,
        "Mesh": MeshOpts,
        "DataBook": DataBookOpts,
        "Report": Report,
    }
   # >

   # ======
   # Config
   # ======
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
            * 2022-11-03 ``@ddalle``: Version 1.0
        """
        # Read the defaults
        defs = get_pycart_defaults()
        # Apply the defaults.
        self = applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()
   # >

   # ================
   # Global settings
   # ================
   # <
    # Method to get the number of multigrid levels
    def get_mg(self, j=None, **kw) -> int:
        r"""Return the number of multigrid levels

        :Call:
            >>> mg = opts.get_mg(j=None, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *mg*: :class:`int`
                Maximum of *mg_fc* and *mg_ad*
        :Versions:
            * 2014-08-01 ``@ddalle``: Version 1.0
            * 2022-11-03 ``@ddalle``: Version 1.1; modern type checks
        """
        # Get the two values.
        mg_fc = self.get_mg_fc(j=j, **kw)
        mg_ad = self.get_mg_ad(j=j, **kw)
        # Deal with arrays
        if isinstance(mg_fc, ARRAY_TYPES):
            mg_fc = mg_fc[0]
        if isinstance(mg_ad, ARRAY_TYPES):
            mg_ad = mg_ad[0]
        # Deal with ``None``
        if mg_fc is None:
            mg_fc = 0
        if mg_ad is None:
            mg_ad = 0
        # Return max
        return max(mg_fc, mg_ad)
   # >


# Add global properties
Options.add_properties(("AeroCsh", "InputCntl"))
# Add methods from subsections
Options.promote_sections()
