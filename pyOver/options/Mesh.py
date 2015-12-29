"""Interface for OVERFLOW meshing"""

# Import options-specific utilities
from util import rc0, odict


# Class for FUN3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for OVERFLOW meshing options"""
    
    # Mesh filenames
    def get_MeshFile(self, i=None):
        """Return the original mesh file names
        
        :Call:
            >>> fname = opts.get_MeshFile(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Optional index
        :Outputs:
            *fname*: :class:`str` | :class:`list` (:class:`str`)
                Mesh file name or list of files
        :Versions:
            * 2015-12-29 ``@ddalle``: First version
        """
        return self.get_key('MeshFile', i)

# class Mesh

