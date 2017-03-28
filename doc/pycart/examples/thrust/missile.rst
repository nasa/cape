
.. _pycart-ex-thrust:

Demo 8: Powered Rocket and Thrust Vectoring
===========================================

This example explains how to use powered boundary conditions in Cart3D with
pyCart along with a triangulation rotation and intersection as part of a thrust
vector model.



Intersection Process
--------------------

    .. _fig-pycart-ex08-c-png:
    .. figure:: Components.c.png
        :width: 3.2 in
        
        Raw self-intersecting surface with original component IDs
        
    .. _fig-pycart-ex08--png:
    .. figure:: Components.png
        :width: 3.2 in
        
        Self-intersecting surface with one component ID for each closed volume
        
    .. _fig-pycart-ex08-o-png:
    .. figure:: Components.o.png
        :width: 3.2 in
        
        Intersected or trimmed surface with one component ID for each original
        closed volume
        
    .. _fig-pycart-ex08-i-png:
    .. figure:: Components.i.png
        :width: 3.2in
        
        Intersected or trimmed surface with original component IDs mapped

Results
-------
:numref:`fig-pycart-ex08-slice-y0` shows a flow visualization of the case
included in the example at Mach 1.5, 2 degrees angle of attack with the engine
tiled downward 4.5 degrees and a thrust coefficient of 8.5.

    .. _fig-pycart-ex08-slice-y0:
    .. figure:: slice-y0-mach.png
        :width: 4in
        
        Surface pressure coefficient (*Cp*) and Mach number slice through *y=0*
        
    .. _fig-pycart-ex07-slice-y0-mesh:
    .. figure:: slice-y0-mach-mesh.png
        :width: 4in
        
        Surface pressure coefficient (*Cp* and *y=0* Mach number slice showing
        volume mesh
        
        