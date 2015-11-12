
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
:file:`Components.i.tri` with its seven component IDs.

Otherwise, the solution proceeds in the same manner as a non-intersecting case. 
Let's take a closer look at the ``"Mesh"`` and ``"Trajectory"`` sections of the
pyCart input file :file:`fins.json` to explain how this was set up.

    .. code-block:: javascript
    
        "Mesh": {
            // Intersect
            "intersect": true,
            // Surface triangulation
            "TriFile": [
                "inputs/bullet.tri",
                "inputs/fin1.tri",
                "inputs/fin2.tri",
                "inputs/fin3.tri",
                "inputs/fin4.tri"
            ]
        },
        
The ``"Mesh"`` section is relatively simple but contains a little bit more
information than the default section.  The individual water-tight volumes are
split into separate ``tri`` files, which provides pyCart two layers of
information about how to split up the surface.  Each ``tri`` file may contain
multiple component IDs (in this case, only :file:`bullet.tri` contains more than
one component ID), but each file should contain a single closed surface.  Then
pyCart combines all these triangulations before intersecting them.  If 
*intersect* is not set to ``true``, using multiple triangulation files has
little effect.

    .. code-block:: javascript
    
        // Trajectory (i.e. run matrix) description
        "Trajectory": {
            // Global run matrix definitions
            "Keys": ["Mach", "alpha_t", "phi", "d2", "d4"],
            "File": "inputs/matrix.csv",
            "GroupMesh": true,
            "GroupPrefix": "poweroff",
            // Customized key definitions
            "Definitions": {
                // Rotate fin 2
                "d2": {
                    "Group": true,
                    "Type": "rotation",
                    "CompID": "fin2",
                    "Vector": [[7.2,0,0], [7.2,-1,0]],
                    ++"Value": "float",
                    "Format": "%+03i_"
                },
                // Rotate fin 4
                "d4": {
                    "Group": true,
                    "Type": "rotation",
                    "CompID": "fin4",
                    "Vector": [[7.2,0,0], [7.2,1,0]],
                    "Value": "float",
                    "Format": "%+03i"
                }
            }
        }
        
The ``"Trajectory"`` section, which defines the run matrix input variables, is
more interesting, so let's go through the settings one-by-one.  The *Keys* input
sets the list of input variables.  The first three are common variables for many
configurations and as such are automatically recognized by pyCart.  The *File*
parameter simply points to a file that contains the values of each input
variable at which to run the configuration.

Setting *GroupMesh* to true tells pyCart that the run matrix can be split into
groups of cases such that each case in one group can use the same mesh, and
*GroupPrefix* sets the base name of the group folders.

The last parameter, *Definitions*, is the interesting part of this example.
Because *Mach*, *alpha_t*, and *phi* are such common input variables (called
"trajectory keys" in CAPE terminology) that we can rely on the default
definitions.  (Default trajectory key definitions can be altered by editing the
file ``$PYCART/settings/pyCart.default.json``.)  The other two parameters are
fin rotations, which require customization.

The trajectory key *d2* is set up to rotate fin #2.  We set *Group* to ``true``
because cases with the same fin deflections can use the same mesh.  The *Type*
is set to ``"rotation"``, which pyCart recognizes and reduces some of our work
in defining it here.  We set *CompID* to ``"fin2"``, which tells pyCart to
rotate any triangles in the component defined as ``"fin2"`` in the
:file:`Config.xml` file.  Then *Vector* gives a list of two points that define a
vector about which to rotate the points.

Finally, *Format* sets a ``printf`` style format string for how the value is
printed in the folder name.  It's set to integer in this example, which would
create problems for fin deflection angles like ``2.1``.  Anyway, this example
shows how to set up general component rotations very quickly.


