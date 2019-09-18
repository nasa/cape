
.. _pycart-ex-thrust:

Demo 8: Powered Rocket and Thrust Vectoring
===========================================

This example explains how to use powered boundary conditions in Cart3D with
pyCart along with a triangulation rotation and intersection as part of a thrust
vector model.  Primarily, this example seeks to introduce how to simulate
rocket engines.  In addition, component rotations and intersections are
demonstrated, and a detailed report including a customized Tecplot layout are
also included.

To demonstrate these capabilities, we begin with the arrow example of previous
examples as shown in :numref:`fig-pycart-ex08-arrow` and add a stand-alone
engine geometry as shown in :numref:`fig-pycart-ex08-engine`.

    .. _fig-pycart-ex08-arrow:
    .. figure:: ../arrow01.png
        :width: 3.0 in
        
        Trimmed arrow surface from previous examples
        
    .. _fig-pycart-ex08-engine:
    .. figure:: engine01.png
        :width: 3.0 in
        
        Stand-alone engine block triangulation

These are two water-tight closed geometries so that they can be used with
``intersect``.  In their unmodified positions, the engine intersects the back
plane of the arrow and produces thrust aligned with the centerline of the
arrow.  In this example, we add a pitch angle to the engine so that it is
rotated by an amount set in the run matrix, and this rotation is performed
prior to the intersection.

This example can be found in ``$PYCART/examples/pycart/08_thrust``, and as
always the :file:`pyCart.json` file in that folder is a good supplement to this
document.  :numref:`fig-pycart-ex08-slice-y0` shows the main product of the
example.

    .. _fig-pycart-ex08-slice-y0:
    .. figure:: slice-y0-mach.png
        :width: 3.5 in
        
        Surface pressure coefficient and :math:`y{=}0` Mach number slice for
        the Mach 1.5 conditions from the example.


Inputs and Run Matrix Description
---------------------------------
Setting up variables to change the thrust of an example is usually contained
within the ``"RunMatrix"`` section of the JSON file.  The following
definitions are used for this example:

    .. code-block:: javascript
    
        "RunMatrix": {
            "Keys": [
                "mach", "alpha", "q", "T",
                "tilt", "CT",
                "config", "Label"
            ],
            "File": "matrix.csv",
            "GroupMesh": false,
            "Definitions": {
                "mach": {
                    "Format": "%0.2f"
                },
                "tilt": {
                    "Type": "rotation",
                    "Value": "float",
                    "Group": false,
                    "Center": [8.0, 0.0, 0.0],
                    "Axis":   [0.0, 1.0, 0.0],
                    "CompID": [
                        "engine_mount",
                        "noz_exterior",
                        "noz_interior",
                        "noz_bc"
                    ],
                    "Abbreviation": "_t",
                    "Format": "%.1f"
                },
                "CT": {
                    "Type": "SurfCT",
                    "Value": "float",
                    "TotalTemperature": 8500.0,
                    "AreaRatio": 4.0,
                    "RefPressure": "freestream",
                    "RefTemperature": "freestream",
                    "RefDynamicPressure": "freestream",
                    "CompID": "noz_bc",
                    "Abbreviation": "T",
                    "Format": "%.1f"
                }
            }
        }
        
In addition to our usual *mach*, *alpha*, *config*, and *Label* parameters that
are part of the standard pyCart setup, we have added a few *RunMatrix>Keys*.
The first two are dynamic pressure (*q*) and freestream static temperature
(*T*).  These are both recognized by pyCart as standard variables, and no
descriptions are needed in the *RunMatrix>Definitions* section.

The next trajectory key is *tilt*, which is defined so that it pitches the
engine block from :numref:`fig-pycart-ex08-engine` by an angle equal to the
value of this variable.  The fact that this is a rotation is set in the *Type*
option within *RunMatrix>Definitions>tilt*.  The center of rotation is set as
``[8.0, 0.0, 0.0]``, which is the center of the back plane of the arrow.  The
value of *Axis* makes this a pitch rotation.  *CompID* is a list of components
that are rotated, which can be either strings or component numbers.  This is a
pretty standard rotation, but users are advised that there are many more
rotation & translation options available.

The last key is *CT*, whose *Type* of ``"SurfCT"`` tells pyCart that the value
of this key is used to set a surface boundary condition with the intent of
setting a nozzle to attain a desired thrust.  The ``"SurfCT"`` and ``"SurfBC"``
types are both targeted at powered boundary conditions, but ``"SurfBC"``
generally targets a desired stagnation pressure while ``"SurfCT"``  targets a
desired thrust.

We should also take this opportunity to discuss the effects of including *q*
and *T*.  Normally, since Cart3D is an inviscid solver, these dimensional
parameters have no effect at all, and the results are truly nondimensional.
However, introducing an engine partially breaks this symmetry to freestream
conditions.  For one thing, a rocket producing an amount of thrust in pounds
will have a different thrust coefficient depending on the freestream dynamic
pressure.  Similarly, a particular temperature at the rocket boundary has
different normalized temperatures for different freestream temperatures.  While
it is possible in pyCart to use a ``"SurfCT"`` key without *q* and *T*, this is
unlikely to be a physically relevant setup.

Going back to the JSON settings for *CT*, we see a *TotalTemperature* of
8500.0, which sets the stagnation temperature at the boundary condition plane
to a constant temperature in degrees Rankine.  If we wanted to set the
*TotalTemperature* relative to the freestream temperature instead of a fixed
dimensional value, we would set *RefTemperature* to ``1.0`` instead of its
``"freestream"`` value.  It is also possible to use the value of another
variable to change the stagnation temperature from case to case by setting the
value of *TotalTemperature* to the name of another trajectory key.  See the
following example for how this could work.

    .. code-block:: javascript
    
        "CT": {
            "Type": "SurfCT",
            "Value": "float",
            "TotalTemperature": "T0",
            ...
        },
        "T0": {
            "Type": "value",
            "Value": "float"
        }
        
We have also set *AreaRatio* here; for Cart3D thrust setup we usually need this
parameter for Cart3D's internal calculation of anticipated thrust.  It is
typically recommended to set the boundary condition on a plane where the Mach
number is 1.0 in Cart3D, but the Mach number on the plane can be set to a
different value using *Mach* within the *Definitions>CT*.  pyCart then uses
this information to calculate the static pressure and density at the boundary
condition plane that should give the corresponding thrust.

While pyCart automatically calculates the surface normal of that plane (since
the velocity has to be set on that plane including its three components), this
simplified thrust calculation is not perfect. In order to get the correct
thrust, there is also a *PressureCalibration* option that can be used to
linearly scale the surface pressure.


.. _pycart-ex08-intersect:

Intersection Process
--------------------
Intersecting closed volumes that each have multiple component IDs marked is a
nontrivial process.  Because ``intersect`` is expecting an input triangulation
in which each component is a water-tight surface with one component, pyCart has
to do some extra preprocessing and postprocessing steps.  To get things to work
properly, we use two separate ``tri`` files and set the following settings in
the JSON.

    .. code-block:: javascript
    
        "Mesh": {
            // Surface triangulation
            "TriFile": ["arrow.tri", "engine.tri"],
            // Extra refinements
            "XLev": [
                {"n": 2, "compID": "noz_bc"},
                {"n": 1, "compID": "noz_interior"}
            ],
            // Extra bounding boxes for adaptation regions
            "BBox": [
                {"n": 8, "compID": "noz_exterior", "xp": 2.5}
            ]
        }
        
The key parameter here is that *Mesh>TriFile* is a list of two files.  As a
result, pyCart assumes that each individual file is a single closed volume.
The *XLev* descriptions specify additional refinements on any cut cells that
intersect specified components, while *BBox* gives rectangular prisms in which
to make a specified number *n* of refinements.

:numref:`fig-pycart-ex08-c-png` shows the original surface triangulation after
rotations but before performing the intersection  operation.  It contains the
same component breakdown as the original input files and is labeled
:file:`Components.c.tri` in the folder.  pyCart also writes the file
:file:`Components.tri` which contains the same nodes and triangles but only has
two components, and a visualization is shown in :numref:`fig-pycart-ex08--png`.

    .. _fig-pycart-ex08-c-png:
    .. figure:: Components_c.png
        :width: 3.2 in
        
        Raw self-intersecting surface with original component IDs,
        :file:`Components.c.tri`
        
    .. _fig-pycart-ex08--png:
    .. figure:: Components.png
        :width: 3.2 in
        
        Self-intersecting surface with one component ID for each closed volume,
        :file:`Components.tri`
        
Then a call is made to Cart3D's ``intersect`` tool such that the input is
:file:`Components.tri`, and the output is :file:`Components.o.tri`, which is
shown in :numref:`fig-pycart-ex08-o-png`.
        
    .. _fig-pycart-ex08-o-png:
    .. figure:: Components_o.png
        :width: 3.2 in
        
        Intersected or trimmed surface with one component ID for each original
        closed volume, :file:`Components.o.tri`
        
In order to get the original components requested by the user, pyCart then
performs an additional step of remapping the component IDs to create
:file:`Components.i.tri`, shown in :numref:`fig-pycart-ex08-i-png`.  Each
triangle has the component ID copied from the closest triangle of
:file:`Components.c.tri`.
        
    .. _fig-pycart-ex08-i-png:
    .. figure:: Components_i.png
        :width: 3.2in
        
        Intersected or trimmed surface with original component IDs mapped,
        :file:`Components.i.tri`

Results and Report Generation
-----------------------------
The run matrix in ``$PYCART/examples/pycart/08_thrust/matrix.csv`` has only one
case, which has a Mach number of 1.5, an angle of attack of 2 degrees.  The
engine is pitched downward 4.5 degrees and a thrust coefficient of 8.5.  A
status while running the case would look something like the following.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory       Status  Iterations  Que CPU Time 
        ---- -------------------------- ------- ----------- --- --------
        0    poweron/m1.50a2.0_t4.5T8.5 RUN     50/700      .      452.9 
        
        RUN=1, 

:numref:`fig-pycart-ex08-slice-y0-mesh` shows a flow visualization of this case
that is generated using the ``"slice-y0-mesh"`` subfigure from
:file:`pyCart.json`.  (The results of the ``"slice-y0"`` subfigure is shown in
:numref:`fig-pycart-ex08-slice-y0`.)  These figures show some of the more
advanced procedures from customizing a Tecplot layout.

        
    .. _fig-pycart-ex08-slice-y0-mesh:
    .. figure:: slice-y0-mach-mesh.png
        :width: 4in
        
        Surface pressure coefficient (:math:`C_p`) and :math:`y{=}0` Mach
        number slice showing volume mesh
        
The process for this example begins with opening the output flow visualization
files created by Cart3D: :file:`Components.i.plt` and :file:`cutPlanes.plt`.
Actually those files are in the ``adapt03/`` folder in this case, but pyCart
automatically creates symbolic links to the most recent ``plt`` files.

Then, after opening those files, the user should create the desired image and
save it as a layout.  A hidden step necessary for this example is that the user
has to customize the color map for the Mach slice.  Since layout files do not
have ``CREATECOLORMAP`` commands for built-in color maps, there is no color map
in the layout file to edit.  It may be possible without this step, but this
documents one known process.  Simply enter the contour details dialouge in
Tecplot and change one of the colors or slide one of the handles in the color
map interface.  This needs to be performed for both color maps since we are
using separate contours on the surface and the slice.

The JSON description for the two flow visualization subfigures is shown below:

    .. code-block:: javascript
    
        "TecBase": {
            "Type": "Tecplot",
            "FigWidth": 1024,
            "Width": 0.48,
            "Caption": "Surface $C_p$ and $y{=}0$ Mach slice",
            "ContourLevels": [
                {
                    "NContour": 1,
                    "MinLevel": -0.4,
                    "MaxLevel": 1.2,
                    "Delta": 0.1
                },
                {
                    "NContour": 2,
                    "MinLevel": 0.0,
                    "MaxLevel": 4.0,
                    "Delta": 0.1
                }
            ],
            "ColorMaps": [
                {
                    "NContour": 1,
                    "ColorMap": {
                        "-0.4": "blue",
                        "0.0": "white",
                        "1.2": "red"
                    }
                },
                {
                    "NContour": 2,
                    "Constraints": ["mach > 1.25"],
                    "ColorMap": {
                        "0.0": "darkpurple",
                        "1.0": ["#b55fbf", "green"],
                        "$mach": "white",
                        "4.0": "darkorange"
                    }
                }
            ]
        },
        // With mesh
        "slice-y0-mach": {
            "Type": "TecBase",
            "Layout": "slice-y0-mach.lay"
        },
        "slice-y0-mach-mesh": {
            "Type": "TecBase",
            "Layout": "slice-y0-mach-mesh.lay"
        }

The two subfigures share most of their options, so they cascade from a common
subfigure called ``"TecBase"``.  Only the name of the layout file is changed.
However, the two layouts are very similar; we could use the following alternate
definition.

    .. code-block:: javascript
    
        "slice-y0-mach-mesh": {
            "Type": "TecBase",
            "Layout": "slice-y0-mach.lay",
            "Keys": {
                "FIELDLAYERS": {
                    "SHOWMESH": "YES"
                }
            }
            
The minimum and maximum values for the two contour maps are set in the
*ContourLevels* section.  Of course, these fixed values could have just been
set within Tecplot, but this allows for min and max values to depend on the
trajectory keys.

To see how this works, see the more complex *ColorMaps* section. Here we set
the surface pressure map so that ``"blue"`` is at the minimum pressure of
``"-0.4"``, white is at *Cp=0*, and the maximum value is red. This simplifies
the process of getting white to lie on *Cp=0* with an asymmetric range of
values.

The color map for the Mach slice is more complicated.  Here we have set
``"darkpurple"`` at Mach 0, a lighter purple of ``"#b55fbf"`` on the lower side
of Mach 1, ``"green"`` on the upper side of Mach 1.  This list of two colors at
Mach 1 leads to a sharp purple/green divide at the sonic line.  Then we set
``"white"`` as the color for ``"$mach"``; the ``$`` tells pyCart to replace
this with the value of the trajectory key *mach* for this color.  Finally, we
use ``"darkorange"`` for top of the color map.

The result is a very informative color map that clearly identifies subsonic
flow, low supersonic flow, the freestream Mach condition, and high supersonic
flow.  Furthermore, this color map setup, by setting ``"$mach": "white"``, it
applies to a range of conditions.  The color map shown above could lead to
problems if the Mach number is lower than about 1.2, so the actual JSON file
contains three different color map specifications.  Which one gets applied is
determined by the *Constraints* key, which is visible in the code snippet show
above.
