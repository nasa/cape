"""
:mod:`cape.pyfun.options.meshopts`
======================================

This module provides options for surface and volume meshes thar are
specific to FUN3D. This consists of three parts, although the second or
third option (but never both) may be optional depending on the
configuration.

    * Provides the name of the FUN3D ``.mapbc`` file using the option
      *MapBCFile*. This specifies the FUN3D boundary condition for each
      surface component

    * Specifies the name of a volume mesh, *MeshFile* (which can also be
      grown from the surface using AFLR3)

    * Specifies a surface triangulation either for creating a volume
      mesh with AFLR3 and/or providing a surface for thrust BC
      definitions

The FUN3D version also provides options for *Faux* and *Freeze* (or point to an
external one) in combination with mesh adaptation.

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.cfdx.options.aflr3opts`
"""

# Local imports
from ...cfdx.options import meshopts
from ...optdict import FLOAT_TYPES, INT_TYPES, OptionsDict


class FauxItemOpts(OptionsDict):
    r"""Options for ``"Faux"`` section of pyfun :class:`MeshOpts`

    This just requires all entries to have :class:`float` types.
    """
    # No additional attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": FLOAT_TYPES,
    }

    # List depth
    _optlistdepth = {
        "_default_": 1,
    }


# Class for Faux input
class FauxOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": FauxItemOpts,
    }


# Class for FUN3D mesh settings
class MeshOpts(meshopts.MeshOpts):
    # No additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Faux",
        "FauxFile",
        "FreezeComponents",
        "FreezeFile",
        "MapBCFile",
    }

    # Aliases
    _optmap = {
        "BCFile": "MapBCFile",
        "MapBC": "MapBCFile",
        "faux": "Faux",
    }

    # Types
    _opttypes = {
        "Faux": FauxOpts,
        "FauxFile": str,
        "Freezecomponents": INT_TYPES,
        "FreezeFile": str,
        "MapBCFile": str,
    }

    # Descriptions
    _rst_descriptions = {
        "Faux": "manual ``faux_input`` settings",
        "FauxFile": "name of ``faux_input`` template file",
        "FreezeFile": "name of file w/ compIDs to freeze during adaptation",
        "MapBCFile": "name of the boundary condition map file",
    }

    # Get faux geometry for a component
    def get_Faux(self, comp=None):
        r"""Get the geometry information for a ``faux_input`` component

        :Call:
            >>> faux = opts.get_Faux(comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: {``None``} | :class:`str`
                Name or number of component to process (all if ``None``)
        :Outputs:
            *faux*: :class:`dict` | :class:`float` | :class:`list`
                ``faux_input`` plane definition(s)
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
            * 2023-03-17 ``@ddalle``: v2.0; use OptionsDict
        """
        # Get full dictionary of faux geometry
        faux = self.get("Faux", {})
        # Check if seeking a single component
        if comp is None:
            # Return full set of instructions
            return faux
        else:
            # Return component instructions
            return faux[comp]


# Add properties
_PROPS = (
    "MapBCFile",
)
MeshOpts.add_properties(_PROPS)

# Add getters
_GETTER_PROPS = (
    "FauxFile",
    "FreezeComponents",
    "FreezeFile",
)
MeshOpts.add_properties(_GETTER_PROPS)
