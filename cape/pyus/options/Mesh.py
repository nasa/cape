"""
:mod:`cape.pyus.options.Mesh`: US3D Meshing Options
====================================================

This module provides options for surface and  volume meshes in US3D.  This
consists of three parts, although the second or third option (but never both)
may be optional depending of the configuration 

    * Provides the name of any extra  US3D  boundary condition files
      
    * Specifies the name of a volume mesh, *MeshFile* (which can also be grown
      from the surface using AFLR3)
      
    * Specifies a surface triangulation either for creating a volume mesh with
      AFLR3 and/or providing a surface for thrust BC definitions

:See Also:
    * :mod:`cape.cfdx.options.Mesh`
    * :mod:`cape.cfdx.options.aflr3`
    * :mod:`cape.pyus.options.runControl`
    * :mod:`cape.pyus.options.Config`
"""

# Import options-specific utilities
from .util import rc0, odict
# Import Cape template
import cape.cfdx.options.Mesh


# Class for FUN3D mesh settings
class Mesh(cape.cfdx.options.Mesh):
    """Dictionary-based interface for FUN3D meshing options"""
        
    pass
# class Mesh

