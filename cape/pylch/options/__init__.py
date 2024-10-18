r"""
:mod:`cape.pylch.options`: CAPE options specific to Loci/CHEM
==============================================================

This module provides defintions for and an interface to all options that
are unique to :mod:`cape.pylcnh.

:See also:
    * :mod:`cape.cfdx.options`
"""

# Local imports
from .meshopts import MeshOpts
from .runctlopts import RunControlOpts
from ...cfdx import options


# Class definition
class Options(options.Options):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "VarsFile",
    )

    # Types
    _opttypes = {
        "VarsFile": str,
    }

    # New or replaced sections
    _sec_cls = {
        "Mesh": MeshOpts,
        "RunControl": RunControlOpts,
    }


# Add properties
Options.add_properties(
    (
        "VarsFile",
    ))
# Add methods from subsections
Options.promote_sections()
