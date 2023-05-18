r"""
Cart3D-specific volume meshing options

This module provides options for creating volume meshes in Cart3D. This
consists of three parts:

    * Provides the name of a template ``input.c3d`` file; overrides the
      file created by ``autoInputs``

    * Specifies instructions for Cart3D bounding boxes (``"BBox"``) with
      an automated interface to specify boxes by giving the name of a
      component along with a margin, in addition to the capability to
      manually give the coordinates

    * Specifies instructions for Cart3D surface refinement options
      (``"XLev"``) by name of component(s) or component index

These *BBox* and *XLev* instructions edit the file ``preSpec.c3d.cntl``.
The *BBox* instructions are applied via the method
:func:`cape.pycart.cntl.Cntl.PreparePreSpecCntl`. This file is an input
file to ``cubes`` and affects the resolution of the volume created.

:See Also:
    * :mod:`cape.cfdx.options.meshopts`
    * :mod:`cape.pycart.preSpecCntl.PreSpecCntl`
"""


# Local imports
from ...cfdx.options import meshopts
from ...optdict import OptionsDict, BOOL_TYPES, INT_TYPES, FLOAT_TYPES


# Class for BBox instructions
class BBoxOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Recognized options
    _optlist = {
        "compID",
        "n",
        "pad",
        "xm",
        "xp",
        "xpad",
        "ym",
        "yp",
        "ypad",
        "zm",
        "zp",
        "zpad",
    }

    # Aliases
    _optmap = {
        "CompID": "compID",
        "comp": "compID",
        "face": "compID",
        "margin": "pad",
    }

    # Types
    _opttypes = {
        "compID": (str, int),
        "n": INT_TYPES,
        "pad": FLOAT_TYPES,
        "xm": FLOAT_TYPES,
        "xp": FLOAT_TYPES,
        "xpad": FLOAT_TYPES,
        "ym": FLOAT_TYPES,
        "yp": FLOAT_TYPES,
        "ypad": FLOAT_TYPES,
        "zm": FLOAT_TYPES,
        "zp": FLOAT_TYPES,
        "zpad": FLOAT_TYPES,
    }

    # Defaults
    _rc = {
        "n": 7,
    }

    # Descriptions
    _rst_descriptions = {
        "compID": "component ID number or name",
        "n": "number of refinement levels inside BBox",
        "pad": "extra BBox margin for axes, both plus and minus",
        "xm": "BBox margin for minimum *x*-coord",
        "xp": "BBox margin for maximum *x*-coord",
        "xpad": "BBox margin on min and max of *x*-coords",
        "ym": "BBox margin for minimum *y*-coord",
        "yp": "BBox margin for maximum *y*-coord",
        "ypad": "BBox margin on min and max of *y*-coords",
        "zm": "BBox margin for minimum *z*-coord",
        "zp": "BBox margin for maximum *z*-coord",
        "zpad": "BBox margin on min and max of *z*-coords",
    }


# Class for XLev settings
class XLevOpts(OptionsDict):
    # No additional attributes
    __slots__ = ()

    # Options list
    _optlist = {
        "compID",
        "n",
    }

    # Aliases
    _optmap = {
        "CompID": "compID",
        "comp": "compID",
        "face": "compID",
    }

    # Types
    _opttypes = {
        "compID": (str, int),
        "n": INT_TYPES,
    }

    # Defaults
    _rc = {
        "n": 1,
    }

    # Descriptions
    _rst_descriptions = {
        "compID": "component ID number or name",
        "n": "number of refinement levels for surfaces on *compID*",
    }


# Class for Cart3D mesh settings
class MeshOpts(meshopts.MeshOpts):
    # No additional attributes
    __slots__ = ()

    # Additional allowed options
    _optlist = {
        "BBox",
        "XLev",
        "inputC3d",
        "mesh2d",
    }

    # Aliases
    _optmap = {
        "2D": "mesh2d",
        "2d": "mesh2d",
        "Mesh2D": "mesh2d",
        "InputC3D": "inputC3d",
        "InputC3d": "inputC3d",
    }

    # Types
    _opttypes = {
        "BBox": BBoxOpts,
        "XLev": XLevOpts,
        "inputC3d": str,
        "mesh2d": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "inputC3d": "input.c3d",
        "mesh2d": False,
    }

    # Descriptions
    _rst_descriptions = {
        "BBox": r"""list of bounding boxes for volume mesh creation

        This defines bounding boxes using the name of a component taken
        from a Cart3D ``Config.xml`` file. The :class:`cape.tri.Tri`
        class automatically finds the smallest bounding box that
        contains this component, and then the user can specify
        additional margins beyond this box (margins can also be
        negative). Separate margins (or "pads") on theminimum and
        maximum coordinates can be given following the convention
        ``"xm"`` (short for *x-minus*) on the minimum coordinate and
        ``"xp"`` for the maximum coordinate.

            .. code-block:: python

                {
                    "compID": "fin2",
                    "n": 9,
                    "pad": 2.0,
                    "xpad": 3.0,
                    "ym": -1.0,
                    "yp": 3.0
                }""",
        "XLev": r"""list of *XLev* specs for add'l surface refinements

        An *XLev* specification tells ``cubes`` to perform a number of
        additional refinements on any cut cells that intersect a
        triangle from a named (or numbered) component.  This refinement
        can violate the *maxR* command-line input to ``cubes`` and is
        very useful for ensuring that small features of the surface have
        adequate resolution in the initial volume mesh.

        The following example specifies two additional refinements
        (after the initial run-through by ``cubes``) on all triangles in
        the component ``"fins"``.  These instructions are then written
        to ``preSpec.c3d.cntl``.

            .. code-block:: python

                {
                    "compID": "fins",
                    "n": 2
                }""",
        "inputC3d": "file name of pre-generated ``input.c3d``",
        "mesh2d": "option to build 2D mesh",
    }


# Add properties
MeshOpts.add_properties(MeshOpts._optlist)
