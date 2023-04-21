r"""
:mod:`cape.pyover.options`: Options interface for :mod:`cape,pyover`
====================================================================

This is the OVERFLOW-specific implementation of the CAPE options
package, based on

    :mod:`cape.cfdx.options`
"""

# Local imports
from .util import getPyOverDefaults, applyDefaults
from .runctlopts import RunControlOpts
from .overnml import OverNml
from .gridSystem import GridSystemNml
from .meshopts import MeshOpts
from .databookopts import DataBookOpts
from .reportopts import ReportOpts
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
        "Mesh": MeshOpts,
        "DataBook": DataBookOpts,
        "Report": ReportOpts,
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
