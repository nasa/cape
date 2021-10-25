r"""
:mod:`cape.pykes.options.Mesh`: Kestrel meshing options
==========================================================

This module provides options for Kestrel grid systems.

A typical example JSON section is showed below.

    .. code-block:: javascript
    
        "Mesh": {
            "LinkFiles": [
            ],
            "CopyFiles": [
                "inputs/mesh/pykes.avm"
            ]
        }
"""

# Import options-specific utilities
from .util import odict


# Class for FUN3D mesh settings
class Mesh(odict):
    r"""Dictionary-based interface for Kestrel meshing options"""

    # Mesh filenames
    def get_MeshFiles(self, config=None):
        r"""Return the original mesh file names

        :Call:
            >>> meshfiles = opts.get_MeshFiles(i=None)
        :Inputs:
            *opts*: :class:`odict`
                Options interface
        :Outputs:
            *meshfiles*: :class:`list`\ [:class:`str`]
                List of mesh file names to copy/link into case folders
        :Versions:
            * 2021-10-25 ``@ddalle``: Version 1.0
        """
        return self.get_MeshCopyFiles() + self.get_MeshLinkFiles()

    # Mesh filenames to copy
    def get_MeshCopyFiles(self):
        r"""Return the names of mesh files to copy
        
        :Call:
            >>> meshfiles = opts.get_MeshCopyFiles()
        :Inputs:
            *opts*: :class:`odict`
                Options interface
        :Outputs:
            *meshfiles*: :class:`list`\ [:class:`str`]
                List of mesh file names to copy into each case folder
        :Versions:
            * 2021-10-25 ``@ddalle``: Version 1.0
        """
        # Get option
        meshfiles = self.get("CopyFiles")
        # Check if found
        if meshfiles is None:
            return []
        # Check if list
        if isinstance(meshfiles, (list, tuple)):
            # Already list
            return meshfiles
        else:
            # Return singleton
            return [meshfiles]
    
    # Mesh filenames to copy
    def get_MeshLinkFiles(self):
        r"""Return the names of mesh files to link
        
        :Call:
            >>> meshfiles = opts.get_MeshLinkFiles()
        :Inputs:
            *opts*: :class:`odict`
                Options interface
        :Outputs:
            *meshfiles*: :class:`list`\ [:class:`str`]
                List of mesh file names to link into each case folder
        :Versions:
            * 2021-10-25 ``@ddalle``: Version 1.0
        """
        # Get option
        meshfiles = self.get("LinkFiles")
        # Check if found
        if meshfiles is None:
            return []
        # Check if list
        if isinstance(meshfiles, (list, tuple)):
            # Already list
            return meshfiles
        else:
            # Return singleton
            return [meshfiles]

