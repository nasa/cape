r"""


This module provides options for surface and  volume meshes in US3D.
This consists of three parts, although the second or third option (but
never both) may be optional depending of the configuration 

    * Provides the name of any extra  US3D  boundary condition files

    * Specifies the name of a volume mesh, *MeshFile* (which can also be grown
      from the surface using AFLR3)

    * Specifies a surface triangulation either for creating a volume mesh with
      AFLR3 and/or providing a surface for thrust BC definitions

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.pyus.options.runctlopts`
"""

# Local imports
from ...cfdx.options import meshopts


# Class for FUN3D mesh settings
class MeshOpts(meshopts.MeshOpts):
    """Dictionary-based interface for FUN3D meshing options"""
        
    pass
# class Mesh

