
Demo 3: Moving Fins
===================

The following example demonstrates how to use pyCart and Cart3D to analyze a
configuration with moving components.  The example is similar to
:ref:`pycart-ex-arrow`, except that the fins go inside the body of the bullet.

    .. figure:: fins01.png
        :width: 4in
        
        Example showing fin 4 deflected 15 degrees after intersecting with body

This allows for the fins to be rotated about some hinge axis, and then once the
fins are in the correct position, Cart3D's ``intersect`` tool can be used to
transform this self-intersecting surface into a single water-tight geometry.

Let's run a case to see how this works.

    .. code-block:: none
    
        $ pycart -f fins.json -I 7
        Case Config/Run Directory                   Status  Iterations  Que 
        ---- -------------------------------------- ------- ----------- ---
        0    poweroff_d2+05_d4-15/m1.75a1.0r30.0 ---     /           .   
          Group name: 'poweroff_d2+05.0d4-15.0' (index 4)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.tri'
             Writing triangulation: 'Components.c.tri'
         > autoInputs -r 8 -t Components.tri -maxR 10
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05_d4-15')
             (STDOUT = 'autoInputs.out')
         > intersect -i Components.tri -o Components.o.tri
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05_d4-15')
             (STDOUT = 'intersect.out')
             Writing triangulation: 'Components.i.tri'
         > cubes -pre preSpec.c3d.cntl -maxR 10 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05_d4-15')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05_d4-15')
             (STDOUT = 'mgPrep.out')
        Using template for 'input.cntl' file
             Starting case 'poweroff_d2+05.0d4-15.0/m1.75a1.0r30.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05_d4-15/m1.75a1.0r30.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1,

The output is pretty similar to a case without the fin deflections, but there
are a few differences that are helpful to explain.  The first is the name of the
case, ``poweroff_d2+05_d4-15/m1.75a1.0r30.0``, which is especially different in
that the name of the folder contains the fin deflections.  That is all
controlled in the pyCart input file :file:`fins.json`, and we will discuss it
shortly.  This example is configured with the fin deflections in the folder name
because each set of cases with same fin positions can use the same mesh.

The next difference is that pyCart reports writing two triangulation files,
:file:`Components.tri` and :file:`Components.c.tri`, instead of the usual
:file:`Components.i.tri`.  The reason for this pair of files is that
``intersect`` requires each body to have a single component ID, which destroys
the surface component naming that is defined in our inputs (like splitting off
the nose cap, body, and base of the bullet shape into separate components).  So
:file:`Components.tri` has only five components (the bullet shape and one for
each fin) while :file:`Components.c.tri` has seven components.

Then ``intersect`` is run with the command run above, which generates
:file:`Components.o.tri`.  This file also has only five component IDs, and these
are mapped back into the original component ID numbering by comparing to
:file:`Components.c.tri` to generate the final triangulation
:file:`Components.i.tri` with its seven component IDs.  This may seem like a
minor point, but pyCart has been used for configurations with more than 200
component IDs in three bodies.  Splitting the intersected triangulations back
into organized component IDs would be exceedingly difficult without this tool.


