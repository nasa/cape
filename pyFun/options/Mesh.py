"""Interface for FUN3D meshing"""

# Import options-specific utilities
from util import rc0, odict
# Import Cape template
import cape.options.Mesh


# Class for FUN3D mesh settings
class Mesh(cape.options.Mesh):
    """Dictionary-based interface for FUN3D meshing options"""
        
    # Get the surface BC map
    def get_MapBCFile(self, i=None):
        """Return the name of the boundary condition map file
        
        :Call:
            >>> fname = opts.get_MapBCFile(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int`
                Phase index
        :Outputs:
            *fname*: :class:`str`
                Boundary condition file name
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        return self.get_key('BCFile', i)
    
# class Mesh

