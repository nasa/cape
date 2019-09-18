
------------
Mesh Options
------------

Much like the :file:`pyCart.json` file itself, the meshing options are split
into several sections.  There are several options that are not in any section,
and the remaining options are split into sections according to the Cart3D
binaries to which those settings apply.  The relevant Cart3D meshing binary
programs are `autoInputs` and `cubes`, while the settings for `mgPrep` are
determined based on the number of multigrid levels requested in the "flowCart"
and "adjointCart" sections.

Below is a sample of the list of pyCart meshing options along with their default
values as they could be written in a JSON pyCart settings file

    .. code-block:: javascript
    
        "Mesh": {
            "intersect": false,
            "verify": false,
            
            // File names
            "TriFile": "Components.i.tri",
            "preSpecCntl": "preSpec.c3d.cntl",
            "inputC3d": "input.c3d",
            "mesh2d": false,
            
            // Manual volume meshing customizations
            "BBox": [],
            "XLev": []
        }
        
Preprocessing Steps
===================

Two important optional steps in the Cart3D grid preparation process are
`intersect` and `verify`.  The former of these turns self-intersecting surface
triangulations into water-tight surfaces, and `verify` checks a surface for
various errors.  Open edges, zero-length edges, and intersections are all things
that `verify` checks for.  If either of these options are activated, pyCart
tries to run them as part of the job (i.e. as part of
:func:`pyCart.case.run_flowCart`), but they must be run before `cubes`.

    *intersect*: ``true`` | {``false``}
        Whether or not to run `intersect` before creating volume mesh
        
    *verify*: ``true`` | {``false``}
        Whether or not to run `verify` before creating volume mesh
            
File Names and Basic Settings
=============================

The dictionary of simple stand-alone options is shown below.  In many cases,
these settings can be omitted entirely because the defaults are often adequate. 
An important section is if the user wants to use a different surface
triangulation file name or store it in a different location, e.g. ``"TriFile":
inputs/vehicle.tri"``.  Note that the file created in the folder to hold the
actual Cart3D run, the prepared surface triangulation file will be called
:file:`Components.i.tri` regardless of the value of this option.

    *TriFile*: {``"Components.i.tri"``} | :class:`str` | :class:`list` (*str*)
        This can be either the name of a single triangulation file (specified
        relative to the pyCart root folder, which is the folder from which
        either `pycart` is called or :func:`cape.pycart.cntl.Cntl` is called, or
        a list of such triangulation files.  If a list, the files will be read
        in the order listed, which affects node and face numbering.  If the
        component ID numbers of the multiple tri files do not overlap, they will
        be read unchanged into the combined surface; otherwise an offset will be
        applied to triangulations read in later.
        
    *preSpecCntl*: {``preSpec.c3d.cntl``} | :class:`str`
        Name of mesh prespecification file template to use
        
    *inputC3d*: {``"input.c3d"``} | :class:`str`
        If not using `autoInputs`, name of mesh input file that contains mesh
        dimensions and several other settings.
        
    *mesh2d*: ``true`` | {``false``}
        Whether or not the mesh is two-dimensional.  This setting is not yet
        implemented.
        
Volume Mesh Customization Settings
==================================

Two types of settings can be used in the :file:`preSpec.c3d.cntl` file, which is
an input file for `cubes`, that allow the user to manually increase the mesh
resolution in specific regions.  These are *BBox*\ es and *XLev*\ s.  When not
using pyCart, the *BBox*\ es have to be specified as six coordinates that define
the minimum and maximum coordinates in all three dimensions.  However, pyCart
allows these bounding boxes to be defined as the box that contains a certain
component with optional padding on each side.  The components can be component
numbers, names from :file:`Config.xml`, or lists of either.

Similarly, the *XLev* option is used to specify additional mesh refinements
adjacent to the surface of a component.  This is useful in many instances, but
it is particularly so if the geometry has small regions with powered boundary
conditions such as attitude control motors.

A description of the format of these two options is presented below.  Hopefully
this is a useful reference, but it may be confusing without seeing an example.

    *BBox*: {``[]``} | ``[BB]`` | :class:`list` (:class:`dict`)
        List of individual bounding box objects *BB*
        
        *BB*: :class:`dict`
            Individual bounding box :class:`dict`.  The required fields are
            *compID* and *n*, while the padding parameters are optional.
            
            *n*: {``7``} | :class:`int`
                Minimum number of refinements within the BBox
                
            *compID*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components around which to build box
                
            *pad*: {``false``} | :class:`float`
                Margin to add to box limits on both minimum and maximum sides
                for all three dimensions
                
            *xpad*: {``false``} | :class:`float`
                Extra dimensions to add to both min and max of box x-limits
                
            *ypad*: {``false``} | :class:`float`
                Extra dimensions to add to both min and max of box y-limits
                
            *zpad*: {``false``} | :class:`float`
                Extra dimensions to add to both min and max of box z-limits
                
            *xm*: {``false``} | :class:`float`
                Extra padding for box in only minus-x direction
                
            *xp*: {``false``} | :class:`float`
                Extra padding for box in only plus-x direction
                
            *ym*: {``false``} | :class:`float`
                Extra padding for box in only minus-y direction
                
            *yp*: {``false``} | :class:`float`
                Extra padding for box in only plus-y direction
                
            *zm*: {``false``} | :class:`float`
                Extra padding for box in only minus-z direction
                
            *zp*: {``false``} | :class:`float`
                Extra padding for box in only plus-z direction
                
    *XLev*: {``[]``} | ``[XL]`` | :class:`list` (:class:`dict`)
        List of individual x-level objects *XL*
        
        *XL*: :class:`dict`
            Individual additional surface refinement :class:`dict`.  Both fields
            are required.
            
            *n*: {``7``} | :class:`int`
                Number of additional surface refinements
                
            *compID*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components near which to refine volume mesh
                
As suggested, an example is a much more appropriate way to demonstrate these
useful capabilities, although the above documentation is a thorough reference
once the user is familiarized.

    .. code-block:: javascript
    
        "Mesh": {
            "BBox": [
                {"compID": [1, 2], "n": 8, "ypad": 5, "zpad": 5},
                {"compID": "LeftElevon", "n": 11},
                {"compID": "RightElevon", "n": 11}
            ],
            
            "XLev": [{"compID": ["RightNozzle", "LeftNozzle"], "n": 2}]
            
            ...
        }

This example increases the refinement near the surface of two components
(although, note that ``"RightNozzle"`` and ``"LeftNozzle"`` could actually be
groups of components, and pyCart would work equally well) and adds three manual
refinement boxes.

The first refinement box finds the smallest box that contains all triangles
with component ID of either 1 or 2 and adds some margin in the y- and
z-directions to that box. It tells `cubes` that everything within that box must
have at least 8 refinements, which is a modes number. This is the kind of box
that can be used to slightly increase the resolution away from the body.

The other two bounding boxes specify a higher resolution (at least 11
refinements) in the smallest box containing an elevon. It might be tempting to
combine these two into a single *BBox*, but that would yield a different result
because pyCart would create a *single* box that contains all triangles in
*both* elevons. That would add refinement in the region between the two elevons
that is probably not intended.

The *XLev* is slightly more straightforward to use. The number (2 in this
example) tends to be smaller because this is a number of *additional*
refinements. Furthermore, there is no danger to grouping components into lists.

