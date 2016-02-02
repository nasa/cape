"""Interface for OVERFLOW meshing"""

# Import options-specific utilities
from util import rc0, odict


# Class for FUN3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for OVERFLOW meshing options"""
    
    # Mesh filenames
    def get_MeshFiles(self):
        """Return the original mesh file names
        
        :Call:
            >>> fname = opts.get_MeshFiles(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str` | :class:`list` (:class:`str`)
                Mesh file name or list of files
        :Versions:
            * 2015-12-29 ``@ddalle``: First version
        """
        return self.get_MeshCopyFiles() + self.get_MeshLinkFiles()
        
    # Mesh filenames to copy
    def get_MeshCopyFiles(self):
        """Return the names of mesh files to copy
        
        :Call:
            >>> fmsh = opts.get_MeshCopyFiles()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fmsh*: :class:`list` (:class:`str`)
                List of mesh file names to be copied to each case folder
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get value
        fmsh = self.get_key('CopyFiles')
        # Ensure list
        if fmsh is None:
            # No files
            return []
        elif type(fmsh).__name__ not in ['list', 'ndarray']:
            # Single file
            return [fmsh]
        else:
            # Return the list
            return fmsh
    
    # Mesh filenames to copy
    def get_MeshLinkFiles(self):
        """Return the names of mesh files to link
        
        :Call:
            >>> fmsh = opts.get_MeshLinkFiles()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fmsh*: :class:`list` (:class:`str`)
                List of mesh file names to be copied to each case folder
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get value
        fmsh = self.get_key('LinkFiles')
        # Ensure list
        if fmsh is None:
            # No files
            return []
        elif type(fmsh).__name__ not in ['list', 'ndarray']:
            # Single file
            return [fmsh]
        else:
            # Return the list
            return fmsh
# class Mesh

