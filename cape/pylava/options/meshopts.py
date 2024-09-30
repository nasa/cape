r"""
:mod:`cape.pylava.options.meshopts`: LAVACURV mesh options
===================================================================

This module provides options for LAVA Curvilinear grid
systems. Curvilinear grid systems have a complex file structure, but
the pyLava options are relatively simple.  The user then specifies a
home folder for the mesh files. Finally, the user specifies the names
of various mesh (and related) files to copy or link (relative to the
*ConfigDir*) into the case folder.

A typical example JSON section is showed below.

    .. code-block:: javascript

        "Mesh": {
            "ConfigDir": "common",
            "LinkFiles": [
                "grid.in.ihc",
                "grid.b2b",
                "grid.bc.in",
                "INTOUT",
                "usurp/panel_weights.dat",
            ]
        }

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.pyfun.options.overnmlopts`
"""

# Local imports
from ...optdict import OptionsDict


# Class for LAVACURV mesh settings
class MeshOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "ConfigDir",
        "CopyFiles",
        "LinkFiles",
    }

    # Aliases
    _optmap = {
        "File": "CopyFiles",
        "Folder": "ConfigDir",
    }

    # Types
    _opttypes = {
        "ConfigDir": str,
        "CopyFiles": str,
        "LinkFiles": str,
    }

    # Allowed values
    _optvals = {}

    # List depth
    _optlistdepth = {
        "CopyFiles": 1,
        "LinkFiles": 1,
    }

    # Defaults
    _rc = {
        "CopyFiles": [],
        "LinkFiles": [
            "grid.in.ihc",
            "grid.b2b",
            "grid.bc.in",
            "INTOUT",
            "usurp/panel_weights.dat",
        ],
    }

    # Descriptions
    _rst_descriptions = {
        "ConfigDir": "folder from which to copy/link mesh files",
        "CopyFiles": "list of files to copy into case folder",
        "LinkFiles": "list of files to link into case folder",
    }

    # Mesh filenames
    def get_MeshFiles(self, **kw):
        r"""Return full list of mesh file names

        :Call:
            >>> fnames = opts.get_MeshFiles(**kw)
        :Inputs:
            *opts*: :class:`cape.pylava.options.Options`
                Options interface
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of mesh file names
        :Versions:
            * 2024-08-06 ``@sneuhoff``: v1.0
        """
        # Get categories
        copy_files = self.get_opt("CopyFiles", **kw)
        link_files = self.get_opt("LinkFiles", **kw)
        # Combine
        return copy_files + link_files


# Add properties
MeshOpts.add_properties(MeshOpts._optlist, prefix="Mesh")
