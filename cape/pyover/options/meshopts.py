r"""
:mod:`cape.pyover.options.meshopts`: OVERFLOW mesh options
===================================================================

This module provides options for OVERFLOW grid systems.  OVERFLOW grid
systems have a complex file structure, but the pyOver options are
relatively simple. First the user specifies either ``dcf``, ``peg5``, or
another overall interpolation type using :func:`MeshOpts.get_MeshType`.
The user then specifies a home folder for the mesh files. Finally, the
user specifies the names of various mesh (and related) files to copy or
link (relative to the *ConfigDir*) into the case folder.

A typical example JSON section is showed below.

    .. code-block:: javascript

        "Mesh": {
            "ConfigDir": "common",
            "Type": "dcf",
            "LinkFiles": [
                "grid.in",
                "xrays.in",
                "fomo/grid.ibi",
                "fomo/panel_weights.dat"
            ],
            "CopyFiles": [
                "fomo/mixsur.fmp"
            ]
        }

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.pyfun.options.overnmlopts`
"""

# Local imports
from ...optdict import OptionsDict


# Class for OVERFLOW mesh settings
class MeshOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "ConfigDir",
        "CopyFiles",
        "LinkFiles",
        "Type",
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
        "Type": str,
    }

    # Allowed values
    _optvals = {
        "Type": ("dcf", "peg5"),
    }

    # List depth
    _optlistdepth = {
        "CopyFiles": 1,
        "LinkFiles": 1,
    }

    # Defaults
    _rc = {
        "CopyFiles": [],
        "LinkFiles": [
            "grid.in",
            "xrays.in",
            "fomo/grid.ibi",
            "fomo/grid.nsf",
            "fomo/grid.ptv",
            "fomo/mixsur.fmp",
        ],
    }

    # Descriptions
    _rst_descriptions = {
        "ConfigDir": "folder from which to copy/link mesh files",
        "CopyFiles": "list of files to copy into case folder",
        "LinkFiles": "list of files to link into case folder",
        "Type": "overall meshing stragety",
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

