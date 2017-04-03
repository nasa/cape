
.. _common-report-tecplot:

Automated Tecplot Figures
--------------------------
One special type of subfigure is the ``"Tecplot"`` subfigure.  This type
applies a Tecplot layout file (``*.lay``), modifies it slightly for each case,
and uses it to export an image.

The process for making a Tecplot subfigure is more complicated than other
subfigures.  Unlike other subfigures it requires quite a bit of setup other
than editing the JSON file.  The process can be outlined as follows.

    1. Go into a case folder with a solution.
    2. Launch Tecplot in this folder using the desired files and create the
       desired image.   Then save the layout file (or several layout files).
    3. Perform minor edits to the text of the layout file.
    4. Copy the file common location in the parent folder.
    5. Edit the JSON file to use the layout.
    
Of these, most steps are pretty straightforward and brief, but task 3 is a
little bit particular.  The process has the complexity it has because it has to
be useful for a range of conditions.  For example, the most up-to-date solution
file may have the iteration number in it, so the file  name in the layout file
would have to change (or something else happen).

Furthermore, there is an issue that's particular to Cart3D in that the cut
planes file (``cutPlanes.plt``) do not have pressure coefficient, and
calculating it requires the freestream Mach number (also not contained in the
file).  So Tecplot layouts in pyCart have the capability of setting the
freestream Mach number to that of the actual case.  As an example, the pressure
coefficient is defined in a typical Cart3D layout file using the following
text.

    .. code-block:: none
    
        $!ALTERDATA
          EQUATION = '{C_p} = ({Pressure} - 1/1.4) / (0.5*1.5*1.5)'
          
In order to make this calculation work for all cases (including those in which
the Mach number is not 1.5), it should be changed to the following.

    .. code-block:: none
    
        $!VarSet |Minf| = 1.5
        $!ALTERDATA
          EQUATION = '{C_p} = ({Pressure} - 1/1.4) / (0.5*|Minf|*|Minf|)'
          
Once this change has been made, pyCart will automatically change the Mach
number for each case.

Another task that is somewhat particular can be handled in either step 2 or 3. 
Before pyCart launches Tecplot, it creates some file links depending on the
solution files in a folder.  Specifically, it will link the most recent version
of a recurring PLT or other solution file to a fixed file name. These depend on
the flow solver, but here is an mostly-complete list.

    +----------------------------------------+----------------------------+
    | Source glob                            | Link Created               |
    +========================================+============================+
    | *Cart3D*                                                            |
    +----------------------------------------+----------------------------+
    | ``cutPlanes.[0-9]*.plt``               | ``cutPlanes.plt``          |
    +----------------------------------------+----------------------------+
    | ``Components.[0-9]*.plt``              | ``Components.i.plt``       |
    +----------------------------------------+----------------------------+
    | *FUN3D*                                                             |
    +----------------------------------------+----------------------------+
    | ``pyfun_tec_boundary_timestep*.plt``   | ``pyfun_tec_boundary.plt`` |
    +----------------------------------------+----------------------------+
    | ``pyfun_${surf}_timestep*.plt``        | ``pyfun_${surf}.plt``      |
    +----------------------------------------+----------------------------+
    | *OVERFLOW*                                                          |
    +----------------------------------------+----------------------------+
    | ``q.{avg,save,restart,[0-9]*}``        | ``q.pyover.p3d``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*[0-9]``                      | ``q.pyover.vol``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*.avg``                       | ``q.pyover.avg``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*.srf``                       | ``q.pyover.srf``           |
    +----------------------------------------+----------------------------+
    | ``x.{save,restart,[0-9]*}``            | ``x.pyover.p3d``           |
    +----------------------------------------+----------------------------+
    | ``x.{[0-9]*.srf,srf}``                 | ``x.pyover.srf``           |
    +----------------------------------------+----------------------------+
    
Therefore, when creating layouts, the user is encouraged to use file names from
the column on the right.


.. _report-tecplot-layout-vars:

Setting Layout Variables
------------------------
The freestream Mach number mentioned above is an example of a "layout variable"
that is set by pyCart while creating a Tecplot layout.  There is another
special layout variable called ``"FieldMap"`` that is useful for adaptive
OVERFLOW solutions.  Adaptive OVERFLOW solutions have an unpredictable number
of grids, which causes a problem for Tecplot.  Furthermore, slight changes to
the grid system changes the number of surface zones, which is also a problem.

However, it may be useful for other variables to be set.  The following example
utilizes the two special cases above (which is unlikely since ``"Mach"`` is
mostly for Cart3D and ``"FieldMap"`` is mostly for OVERFLOW) along with a
dictionary of general layout variables.

    .. code-block:: javascript
    
        "Tec-Y0": {
            "Type": "Tecplot",
            "Layout": "inputs/tec-y0.lay",
            "Mach": "mach",
            "FieldMap": [1, 143, 10000],
            "VarSet": {
                "Tinf": "$T",
                "mu": 1.5e-2
            }
        }

The use of ``"mach"`` tells pyCart that the freestream Mach number can be found
as the value of the trajectory key called ``"mach"``.  The ``"VarSet"``
dictionary can set variables to constants or utilize equations.  The sigil in
``"$T"`` means the value of the trajectory key ``"T"`` for that case.  It is
also possible to use values such as ``"$T - 273.15"``.


.. _report-tecplot-contour-levels:

Altering Contour Level Maps
---------------------------
Another difficult issue for using the same Tecplot layout for a range of cases
is that the appropriate minimum and maximum values for a contour plot often
change based on the conditions.  As a very common example, minimum and maximum
pressure coefficients are higher for transonic cases than high supersonic
cases.  Even more obvious, a contour plot of local Mach number obviously should
have different limits depending on the freestream Mach number.

The following gives examples of both *Cp* and *mach* plots.  This sets the
limits of the *Cp* contour plot are set to
:math:`\pm0.9/\sqrt{\max(1,0.85M_\infty^2-1)}`, and the Mach number plots have
limits of 0 and :math:`1.4M_\infty`.  Any functions from the standard
:mod:`numpy` module must be referenced using syntax such as ``np.sqrt()``.

    .. code-block:: javascript
    
        "TecCp": {
            "Type": "Tecplot",
            "ContourLevels": [
                {
                    "NContour": 1,
                    "MinLevel": "-0.9/np.sqrt(max(1.0, 0.85*$mach**2-1))",
                    "MaxLevel": "0.9/np.sqrt(max(1.0, 0.85*$mach**2-1))",
                    "Delta": 0.025
                },
                {
                    "NContour": 2,
                    "Constraints": ["mach <= 1.4"],
                    "MinLevel": 0.0,
                    "MaxLevel": "max(1.1, 1.4*$mach)",
                    "Delta": 0.025
                },
                {
                    "NContour": 2,
                    "Constraints": ["mach > 1.4"],
                    "MinLevel": 0.0,
                    "MaxLevel": "max(1.1, 1.4*$mach)",
                    "Delta": 0.05
                }
            ]
        }

The assignment of the first instruction to *Cp* while the other two are
assigned to *mach* is based on the ``"NContour"`` parameter.  These are based
on the order in which the contour information occurs in the layout (``.lay``)
file and use 1-based indexing.

The ``"Constraints"`` key allows the contour levels instructions only to be
applied to cases matching those constraints.  The example here uses the same
limits for low-speed and supersonic conditions, but the supersonic cases use a
different value of ``"Delta"`` so that there are not too many contour levels.

Any trajectory key/run matrix variable can be accessed in these equations using
a dollar sign.  The Mach number is the most common variable used here, but
something like the total angle of attack may also affect the limits, too.


.. _report-tecplot-color-map:

Altering Color Maps
-------------------
Partially related to the difficulty of contour levels is the issue of
customizing color maps.  The classic rainbow color maps are not particularly
useful, and even a more typical blue/white/red color map used for *Cp* is
difficult if the range is asymmetric.  That is, if the *Cp* contour limits are
between ``-0.1`` and ``0.8``, then the color map needs to be customized in
order to put white at :math:`C_p{=}0`.  The following example shows how pyCart
could be used to fix this automatically.

    .. code-block:: javascript
    
        "ColorMaps": [
            {
                "NContour": 1,
                "ColorMap": {
                    "-0.1": "blue",
                    "0.0": "white",
                    "0.8": "red"
                }
            }
        ]

In order to make this work, the color map needs to be first edited within
Tecplot.  Any change at all will be ok; it does not need to be altered to match
the target color map.  Making some change to the color map will cause a custom
color map to appear in the layout file; it's possible this is not strictly
needed, but it's a good idea.

Color maps for Mach number are particularly challenging.  The appropriate color
maps for a transonic and high supersonic case are likely to be different.  Here
is a set of three that seems to work well.

    .. code-block:: javascript
    
        "ColorMaps": [
            {
                "NContour": 2,
                "Constraints": ["mach" < 0.8"],
                "ColorMap": {
                    "0.0": "darkpurple",
                    "$mach": "white"
                    "1.0": ["cyan", "palegreen"],
                    "max(1.1,1.4*$mach)": "darkgreen"
                }
            }, {
                "NContour": 2,
                "Constraints": ["mach >= 0.8, "mach <= 1.25"],
                "ColorMap": {
                    "0.0": "darkpurple",
                    "$mach": "white",
                    "max(1.1,1.4*$mach)": "darkorange"
                }
            }, {
                "NContour": 2,
                "Constraints": ["mach > 1.25"],
                "ColorMap": {
                    "0.0": "darkpurple",
                    "1.0": ["#b55fbf", "green"],
                    "$mach": "white",
                    "1.4*$mach": "darkorange"
                }
            }
        ]

This set of Mach number color maps divides the flight envelope into three
regions: a purple-to-cyan plot for subsonic cases, a purple-to-orange plot for
transonic cases, and a purple-green-white-orange map for supersonic cases.

The instructions at Mach 1 are set to have a sharp change in color, which helps
identify the transonic line.  In all three cases, the freestream Mach number is
set to white.  However, for transonic cases, there is no Mach 1 transition line
because it tends to make the contour plots confusing.  These suggested color
maps can certainly be further customized, but hopefully they demonstrate the
various possibilities using the pyCart color maps.


.. _report-tecplot-fieldmap:


Changing the FIELDMAP Parameter
-------------------------------
Tecplot subfigures have an additional parameter called *FieldMap* that are very
useful for situations where the number of zones is changing.  For example,
FUN3D writes each surface component as a separate zone, so changing the
geometry results in a different number of zones.  OVERFLOW results with mesh
adaption furthermore have a zone for each grid, so that each case has a
different number of zones even for the same geometry.

Rather than trying to create new layouts for each necessary case, the
*FieldMap* can be used to alter the contiguous families of zones.  Layout files
have groups of zones that are numbered 1 to *N* and labels them

    .. code-block:: none
    
        $!ACTIVEFIELDMAPS [1-500]
        ...
        $!FIELDMAP  [1-499]
          MESH
          {
            SHOW = NO
            COLOR = BLACK
            LINETHICKNESS = 0.02
          }
          ...
        $!FIELDMAP  [500]
          MESH
          {
            SHOW = NO
            COLOR = BLACK
            LINETHICKNESS = 0.02
          }
          ...
        ...

Then if we have the following *FieldMap* settings for a subfigure in the JSON
file:

    .. code-block:: javascript
    
        "FieldMap": [487, 488]
        
the layout will change the relevant lines of the layout to the following

    .. code-block:: none
    
        $!ACTIVEFIELDMAPS [1-488]
        $!FIELDMAP  [1-487]
        $!FIELDMAP  [488]
        
For OVERFLOW layouts, it is generally advisable to set the last number of
*FieldMap* to something huge.  Since the adapted meshes are at the end of the
mesh, setting the field map maximum to a large number keeps them all with the
same format.


.. _report-tecplot-keys:

Altering Other Layout Parameters
--------------------------------
Using the parameter *Keys*, it is also possible to alter other parameters of
the layout file.  Two common examples of this are turning the mesh on or off
and changing the camera position.

    .. code-block:: javascript
    
        "protb01": {
            "Type": "Tecplot",
            "Layout": "surf-cp.lay",
            "Keys": {
                "THREEDVIEW": {
                    "PSIANGLE": 152,
                    "THETAANGLE": 0,
                    "ALPHAANGLE": 0,
                    "VIEWERPOSITION": {
                        "X": 1950.0,
                        "Y": -2408.0,
                        "Z": -4301.0
                    }
                }
            }
        }
        
In the example above we define a subfigure called ``"protub01"`` that uses the
layout file :file:`surf-cp.lay`.  However, we want to move the camera to view a
something around *x=1950*, so we use the *Keys* parameter above.  This will
attempt to change ``PSIANGLE`` in the the ``THREEDVIEW`` section of the layout
file to ``152``. 

In general the *Keys* section allows the user to change any number of options
within sections of the Tecplot layout using this dictionary setting.  However,
in its present format, there is no handling for repeat sections.  For example,
if the layout file had two ``THREEDVIEW`` sections, pyCart will always just
alter the first one.

