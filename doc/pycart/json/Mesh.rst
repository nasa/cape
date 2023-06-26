
.. _pycart-json-mesh:

********************
Mesh Section Options
********************
The options below are the available options in the Mesh Section of the ``pycart.json`` control file

..
    start-Mesh-trifile

*TriFile*: {``None``} | :class:`str`
    original surface triangulation file(s)

..
    end-Mesh-trifile

..
    start-Mesh-xlev

*XLev*: {``None``} | :class:`XLevOpts`
    list of *XLev* specs for add'l surface refinements

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
                }
        

..
    end-Mesh-xlev

..
    start-Mesh-meshfile

*MeshFile*: {``None``} | :class:`str`
    original mesh file name(s)

..
    end-Mesh-meshfile

..
    start-Mesh-inputc3d

*inputC3d*: {``'input.c3d'``} | :class:`str`
    file name of pre-generated ``input.c3d``

..
    end-Mesh-inputc3d

..
    start-Mesh-mesh2d

*mesh2d*: {``False``} | :class:`bool` | :class:`bool_`
    option to build 2D mesh

..
    end-Mesh-mesh2d

..
    start-Mesh-bbox

*BBox*: {``None``} | :class:`BBoxOpts`
    list of bounding boxes for volume mesh creation

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
                }
        

..
    end-Mesh-bbox

