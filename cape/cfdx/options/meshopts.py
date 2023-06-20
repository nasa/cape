r"""
:mod:`cape.cfdx.options.meshopts`: Basic options for mesh settings
==================================================================

This module provides the class :class:`MeshOpts`, which reads the
settings in the ``"Mesh"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict


# Class for generic mesh settings
class MeshOpts(OptionsDict):
    r"""Dictionary-based interface for *Mesh* section

    :Versions:
        * 2023-03-16 ``@ddalle``: v1.0
    """
    # No attbitues
    __slots__ = ()

    # List of options
    _optlist = {
        "MeshFile",
        "TriFile",
    }

    # Types
    _opttypes = {
        "MeshFile": str,
        "TriFile": str,
    }

    # Descriptions
    _rst_descriptions = {
        "MeshFile": "original mesh file name(s)",
        "TriFile": "original surface triangulation file(s)",
    }


# Add getters/setters
MeshOpts.add_properties(MeshOpts._optlist)
MeshOpts.add_extenders(MeshOpts._optlist)
