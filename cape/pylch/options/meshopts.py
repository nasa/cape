"""
:mod:`cape.pylch.options.meshopts`
======================================

This module provides options for surface and volume meshes thar are
specific to Loci/CHEM. It is very similar to
:mod:`cape.pyfun.options.meshopts`.

    * Provides the name of the ``.mapbc`` file using the option
      *MapBCFile*. This specifies the boundary condition for each
      surface component

    * Specifies the name of a volume mesh, *MeshFile* (which can also be
      grown from the surface using AFLR3)

    * Specifies a surface triangulation either for creating a volume
      mesh with AFLR3 and/or providing a surface for thrust BC
      definitions

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.cfdx.options.aflr3opts`
"""

# Local imports
from ...cfdx.options import meshopts


# Class for FUN3D mesh settings
class MeshOpts(meshopts.MeshOpts):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "MapBCFile",
    }

    # Aliases
    _optmap = {
        "BCFile": "MapBCFile",
        "MapBC": "MapBCFile",
    }

    # Types
    _opttypes = {
        "MapBCFile": str,
    }

    # Descriptions
    _rst_descriptions = {
        "MapBCFile": "name of the boundary condition map file",
    }


# Add properties
_PROPS = (
    "MapBCFile",
)
MeshOpts.add_properties(_PROPS)
