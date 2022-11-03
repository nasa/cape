r"""
This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyover`.  The class is based off of the built-int :class:`dict` class.

In addition, this module controls default values of each pyOver
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    1. Values directly specified in the input file, :file:`pyOver.json`
    
    2. Values specified in the default control file,
       :file:`$PYOVER/settings/pyOver.default.json`
    
    3. Hard-coded defaults from this module
"""

# Import template module
import cape.cfdx.options

# Import modules for controlling specific parts of Cart3D
from .util import getPyOverDefaults, applyDefaults
from .runctlopts import RunControlOpts
from .overnml import OverNml
from .gridSystem import GridSystemNml
from .Mesh import Mesh
from .DataBook import DataBook
from .Report import Report
from .Config import Config
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pyover`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary of raw options
    :Outputs:
        *opts*: :class:`cape.pyover.options.Options`
            Options interface
    :Versions:
        * 2014-07-28 ``@ddalle``: Version 1.0
        * 2022-11-03 ``@ddalle``: Version 2.0
    """
   # ======================
   # Class Attributes
   # ======================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "OverNamelist",
        "Overflow",
        "Grids",
    }

    # Types
    _opttypes = {
        "OverNamelist": str,
    }

    # Defaults
    _rc = {
        "OverNamelist": "overflow.inp",
    }

    # Descriptions
    _rst_descriptions = {
        "OverNamelist": "name of template ``overflow.inp`` file",
    }

    # Replaced or renewed sections
    _sec_cls = {
        "RunControl": RunControlOpts,
        "Config": Config,
        "Mesh": Mesh,
        "DataBook": DataBook,
        "Report": Report,
        "Overflow": OverNml,
        "Grids": GridSystemNml,
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
        # Read the defaults.
        defs = getPyOverDefaults()
        # Apply the defaults.
        self = applyDefaults(self, defs)
   # >


# Add properties
Options.add_properties(("OverNamelist",))
# Promote new subsections
Options.promote_sections()
