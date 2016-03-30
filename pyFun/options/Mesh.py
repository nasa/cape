"""Interface for FUN3D meshing"""

# Import options-specific utilities
from util import rc0, odict


# Class for FUN3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for FUN3D meshing options"""
    
    # Mesh filenames
    def get_MeshFile(self, i=None):
        """Return the original mesh file names
        
        :Call:
            >>> fname = opts.get_MeshFile(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Optional index
        :Outputs:
            *fname*: :class:`str` | :class:`list` (:class:`str`)
                Mesh file name or list of files
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        return self.get_key('MeshFile', i)

    # Get the triangulation file(s)
    def get_TriFile(self, i=None):
        """Return the surface triangulation file
        
        :Call:
            >>> fname = opts.get_TriFile(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Run index
        :Outputs:
            *fname*: :class:`str` or :class:`list`(:class:`str`)
                Surface triangulation file
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
            * 2016-03-29 ``@ddalle``: Copied from :mod:`pyCart.options`
        """
        return self.get_key('TriFile', i)
        
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

