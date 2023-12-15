r"""

This module provides options for Kestrel grid systems. This contains an
optional list of files to copy and an optional list of files to link
into each case folder.

A typical example JSON section is showed below.

    .. code-block:: javascript

        "Mesh": {
            "LinkFiles": [
                "grid.avm"
            ],
            "CopyFiles": [
                "aux.avm"
            ]
        }

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
"""

# Local imports
from ...optdict import OptionsDict


# Class for OVERFLOW mesh settings
class MeshOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "CopyFiles",
        "LinkFiles",
    }

    # Types
    _opttypes = {
        "CopyFiles": str,
        "LinkFiles": str,
    }

    # List depth
    _optlistdepth = {
        "CopyFiles": 1,
        "LinkFiles": 1,
    }

    # Defaults
    _rc = {
        "CopyFiles": [],
        "LinkFiles": [],
    }

    # Descriptions
    _rst_descriptions = {
        "CopyFiles": "list of files to copy into case folder",
        "LinkFiles": "list of files to link into case folder",
    }

    # Mesh filenames
    def get_MeshFiles(self, **kw):
        r"""Return full list of mesh file names

        :Call:
            >>> fnames = opts.get_MeshFiles(**kw)
        :Inputs:
            *opts*: :class:`cape.pyover.options.Options`
                Options interface
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of mesh file names
        :Versions:
            * 2015-12-29 ``@ddalle``: v1.0
            * 2023-03-17 ``@ddalle``: v2.0; use :class:`OptionsDict`
        """
        # Get categories
        copy_files = self.get_opt("CopyFiles", **kw)
        link_files = self.get_opt("LinkFiles", **kw)
        # Combine
        return copy_files + link_files


# Add properties
MeshOpts.add_properties(MeshOpts._optlist, prefix="Mesh")

