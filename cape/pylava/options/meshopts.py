r"""
:mod:`cape.pylava.options.meshopts`: LAVA mesh options
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
from ...cfdx.options import meshopts


# Class for LAVACURV mesh settings
class MeshOpts(meshopts.MeshOpts):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "ConfigDir",
    }

    # Aliases
    _optmap = {
        "File": "CopyFiles",
        "Folder": "ConfigDir",
    }

    # Types
    _opttypes = {
        "ConfigDir": str,
    }

    # Descriptions
    _rst_descriptions = {
        "ConfigDir": "folder from which to copy/link mesh files",
    }


# Add properties
MeshOpts.add_properties(MeshOpts._optlist, prefix="Mesh")
