"""
:mod:`pyUS.options.Mesh`: US3D Meshing Options
==================================================

This module provides options for surface and  volume meshes in US3D.  This
consists of three parts, although the second or third option (but never both)
may be optional depending of the configuration 

    * Provides the name of any extra  US3D  boundary condition files
      
    * Specifies the name of a volume mesh, *MeshFile* (which can also be grown
      from the surface using AFLR3)
      
    * Specifies a surface triangulation either for creating a volume mesh with
      AFLR3 and/or providing a surface for thrust BC definitions

:See Also:
    * :mod:`cape.options.Mesh`
    * :mod:`cape.options.aflr3`
    * :mod:`pyUS.options.runControl`
    * :mod:`pyUS.options.Config`
"""

# Import options-specific utilities
from .util import rc0, odict
# Import Cape template
import cape.options.Mesh


# Class for FUN3D mesh settings
class Mesh(cape.options.Mesh):
    """Dictionary-based interface for FUN3D meshing options"""
        
    pass
# class Mesh

