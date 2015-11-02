
Demo 2: Closer Analysis of Simple Arrow Shape
=============================================

The second example is similar to the first pyCart demo except that four fins
have been added and more details of the input files are explained.  The example
is found in ``$PYCART/examples/pycart/arrow`` where ``$PYCART`` is the
installation folder.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
This example is set up with a larger run matrix in order to demonstrate more of
the features of pyCart.

    .. code-block:: none
    
        $ pycart -c
        Case Config/Run Directory    Status  Iterations  Que 
        ---- ----------------------- ------- ----------- ---
        0    poweroff/m1.25a0.0r0.0  ---     /           .   
        1    poweroff/m1.25a1.0r0.0  ---     /           .   
        2    poweroff/m1.25a1.0r15.0 ---     /           .   
        3    poweroff/m1.25a1.0r30.0 ---     /           .   
        4    poweroff/m1.25a1.0r45.0 ---     /           .   
        5    poweroff/m1.5a1.0r0.0   ---     /           .   
        6    poweroff/m1.5a1.0r15.0  ---     /           .   
        7    poweroff/m1.5a1.0r30.0  ---     /           .   
        8    poweroff/m1.5a1.0r45.0  ---     /           .   
        9    poweroff/m1.75a1.0r0.0  ---     /           .   
        10   poweroff/m1.75a1.0r15.0 ---     /           .   
        11   poweroff/m1.75a1.0r30.0 ---     /           .   
        12   poweroff/m1.75a1.0r45.0 ---     /           .   
        13   poweroff/m2.0a1.0r0.0   ---     /           .   
        14   poweroff/m2.0a1.0r15.0  ---     /           .   
        15   poweroff/m2.0a1.0r30.0  ---     /           .   
        16   poweroff/m2.0a1.0r45.0  ---     /           .   
        17   poweroff/m2.5a1.0r0.0   ---     /           .   
        18   poweroff/m2.5a1.0r15.0  ---     /           .   
        19   poweroff/m2.5a1.0r30.0  ---     /           .   
        20   poweroff/m2.5a1.0r45.0  ---     /           .   
        
        ---=21, 
        
Input Files
-----------
Let's look at the files in this folder.

    * ``pyCart.json``: Master settings for running pyCart, input to ``pycart``
    * ``arrow.tri``: Surface triangulation (ASCII)
    * ``arrow.xml``: Names for components of the surface
    * ``matrix.csv``: Run matrix
    
JSON Settings
^^^^^^^^^^^^^
The :file:`pyCart.json` file contains the master settings divided into several
sections, which we will discuss in more detail.  The overall contents of the
file look something like the following, with the ``...`` replaced by more
content.

    .. code-block:: javascript
    
        {
            // Setup for run scripts
            "ShellCmds": [
                "ulimit -S -s 4194304"
            ],
            
            ...
            
            // Trajectory (i.e. run matrix) description
            "Trajectory": {
                "Keys": ["Mach", "alpha_t", "phi"],
                "File": "matrix.csv",
                "GroupMesh": true,
                "GroupPrefix": "poweroff"
            }
        }

The first entry ``"ShellCmds"`` provides a list of commands to run in each
script.  This is primarily important when submitting jobs to a PBS server,
because PBS jobs start without loading any environment settings, etc.  The
contents of this command, ``"ulimit -S -s 4194304"`` increases the stack size.
Many CFD solvers, including, Cart3D recommend removing the limit to this
setting, but some other programs don't like that.

Many users will simply put ``"source ~/.bashrc"`` or something similar here,
which can work but with problems for sharing projects with multiple users.
Often the commands ``"module load pycart"`` and ``"module load cart3d"`` will
appear here in well-organized projects.

    .. code-block:: javascript
    
        "flowCart": {
            // Run sequence
            "InputSeq": [0],
            "IterSeq": [200],
            "it_fc": 200,
            // Options for ``flowCart``
            "mpi_fc": 0,
            "use_aero_csh": 0,
            "cfl": 1.1,
            "mg_fc": 3,
            "y_is_spanwise": true,
            "nProc": 4
        },
        
The ``"flowCart"`` section contains the main settings for running Cart3D.  Many
of the variable names, such as *it_fc*, are copied from Cart3D's template
:file:`aero.csh` scripts or command-line inputs to Cart3D's ``flowCart``.
The three main options (which are required for any pyCart project) are
*InputSeq*, *IterSeq*, and *it_fc*.

    +------------+---------------------------------------------------------+
    | Variable   | Description                                             |
    +============+=========================================================+
    | *it_fc*    | Number of iterations for each call to ``flowCart``,     |
    |            | short for ``iterations_flowCart``; command-line input   |
    |            | is ``flowCart -N $it_fc``                               |
    +------------+---------------------------------------------------------+
    | *InputSeq* | Input sequence, tells pyCart to run input 0; in more    |
    |            | complex projects, this will be a list like ``[0,1,3]``  |
    +------------+---------------------------------------------------------+
    | *IterSeq*  | Iterations for each sequence; this tells pyCart to      |
    |            | continue calling ``flowCart`` until 200 iterations have |
    |            | been run.  If this was ``400``, pyCart would            |
    |            | automatically run ``flowCart`` twice using the first    |
    |            | run's results as inputs to the second                   |
    +------------+---------------------------------------------------------+
    
For a simple case, these parameters seem unnecessarily confusing.  Why not just
tell ``flowCart`` how many iterations to run and be done with it?  For one
thing, *IterSeq* specifies a required number of iterations whereas *it_fc* just
suggests to ``flowCart`` or ``mpix_flowCart`` how many iterations to run.  If
``flowCart`` exits early due to some kind of failure, this convention means that
pyCart will clearly alert us.

Secondly, some applications require more sophisticated approach.  A common
example is a hypersonic case that needs to be run in first-order mode for a few
iterations first.  It might have something like ``"IterSeq": [0, 400]`` and
``"InputSeq": [0, 1]``.  This tells pyCart to run input set ``0`` until it has
run at least ``0`` iterations and then input set ``1`` until it has run at least
``400`` iterations.

The remaining inputs are quite a bit simpler.  For example *mpi_fc* is a boolean
flag to use the MPI ``mpix_flowCart`` command, and *use_aero_csh* is a flag that
tells pyCart to run adaptively using :file:`aero.csh`.  Also, *nProc* sets the
total number of cores or threads to use.

    .. code-block:: javascript
    
        "Mesh": {
            // Surface triangulation
            "TriFile": "arrow.tri",
            // Defines the flow domain automatically
            "autoInputs": {"r": 8},
            // Volume mesh options
            "cubes": {
                "maxR": 10,
                "pre": "preSpec.c3d.cntl",
                "cubes_a": 10,
                "cubes_b": 2,
                "reorder": true
            }
        },
        
The *Mesh* section controls inputs to the Cart3D commands that produce the
volume mesh.  The *TriFile* setting is relatively obvious and points to the name
of the surface triangulation.  The next section allows pyCart to use the Cart3D
binary ``autoInputs`` to create the flow domain and basic volume mesh
parameters with the command ``autoInputs -r 8``, which sets the farfield
boundary at roughly 8 times the size of your surface triangulation.

Running ``autoInputs`` creates files ``input.c3d`` and ``preSpec.c3d.cntl``,
which are given as inputs to the volume generator ``cubes``.

    .. code-block:: javascript
    
        "Config": {
            // Defer to a file for most things.
            "File": "arrow.xml",
            // Declare forces and moments
            "Force": ["cap", "body", "fins", "bullet_no_base", "bullet_total"],
            "RefPoint": {"bullet_no_base": [0.0, 0.0, 0.0]}
            // Reference quantities
            "RefArea": 3.14159,
            "RefLength": 1.0,
        },
        
The *Config* section gives instructions about which components to track, what
moment reference points to use, and similar definitions.  The XML file allows
Cart3D and pyCart to refer to define groups of components and refer to
components by name instead of memorizing their numbers.  The *Force* option
specifies a list of components on which ``flowCart`` should track the force at
each iteration.  This creates files :file:`cap.dat`, :file:`body.dat`,
:file:`fins.dat`, etc.  Then *RefPoint* specifies the list of components for
which to also track the moments, and the moment reference point to use for each
such component.  In this case, the moments will be reported alongside the forces
in :file:`bullet_no_base.dat`.

The *RefArea* and *RefLength* parameters are used here to specify global
reference values, but it is possible to use different reference lengths or areas
for different components in the same run.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["Mach", "alpha_t", "phi"],
            "File": "matrix.csv",
            "GroupMesh": true,
            "GroupPrefix": "poweroff"
        }

The final section (actually, the order is irrelevant, but it's the last section
in this file) describes the run matrix, i.e. trajectory.  The *Keys* parameter
lists the names of variables that will change in the run matrix, i.e. the
independent variables.  In this case, we are using Mach number, total angle of
attack, and velocity roll angle.  There is a set of predefined trajectory keys,
and all three of these examples are in that set, but later examples will show
how to define customized trajectory keys in this section.

The *File* parameter points to a file in which the cases to run are listed, and
*GroupMesh* specifies whether or not each case can use the same mesh.  Setting
it to ``true`` means that ``cubes`` is only run once for the matrix (more
accurately, once for each group, but this example has only one group).  The
*GroupPrefix* gives a name for the folder in which to put all the cases, which
explains why a typical case is named ``poweroff/m1.50a2.00r0.00``, for example.

There are two more sections in the :file:`pyCart.json`, which describe various
products.

Triangulation File: :file:`arrow.tri`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The surface geometry is defined in an ASCII file in a straightforward Cart3D
format.  A summary of the contents is shown below.

    .. code-block:: none
    
        4610  9216
        +4.81527351e-03 +9.80171422e-02 +0.00000000e+00
        +1.92147203e-02 +1.95090326e-01 +0.00000000e+00
        ...
        +6.37716534e+00 +1.58689240e-08 +1.06574801e+00
        385 386 16
        386 387 17
        ...
        2565 4257 2530
        1
        1
        ...
        11

The first line is a summary of the contents of the file.  It states that there
are ``4610`` nodes, i.e. three-dimensional points in space, and ``9216``
triangles.  What follows is 4610 lines with three floating point numbers per
line.  Next is 9216 lines in which each line defines one triangle.  For example,
the first triangle connects node ``385`` to node ``386`` to node ``16``.  After
9216 such lines, there are 9216 more lines with a single integer on each line
that defines the component ID of each triangle.  Thus triangle 1 is part of
component 1, triangle 2 is part of component 1, and the last triangle is part of
component 11.

Component Names: :file:`arrow.xml`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cart3D uses an optional XML file that associates names with each component.  It
uses a standard XML format with component IDs (the numbers at the end of the
:file:`.tri` file discussed above) with a ``Face Label`` value inside a
``<Data>`` tag.  It also allows for the definition of a "container" component
that is the combination of several other components.  This makes it possible to 
track ``fin1`` separately while also tracking all the ``fins`` as a group.  The
contents of the file are shown below.

    .. code-block:: xml
    
        <?xml version="1.0" encoding="ISO-8859-1"?>

        <Configuration Name="bullet sample" Source="bullet.tri">
        
         <!-- Containers -->
          <Component Name="bullet_no_base" Type="container" Parent="bullet_total">
          </Component>
          <Component Name="fins" Type="container" Parent="bullet_no_base">
          </Component>
         
          <Component Name="bullet_total"   Type="container">
          </Component>
         <!-- Containers -->
        
         <!-- body -->
          <Component Name="cap" Type="tri">
           <Data> Face Label=1 </Data>
          </Component>
         
          <Component Name="body" Type="tri">
           <Data> Face Label=2 </Data>
          </Component>
         
          <Component Name="base" Parent="bullet_total" Type="tri">
           <Data> Face Label=3 </Data>
          </Component>
         <!-- body -->
         
         <!-- fins -->
          <Component Name="fin1" Parent="fins" Type="tri">
           <Data> Face Label=11 </Data>
          </Component>
          
          <Component Name="fin2" Parent="fins" Type="tri">
           <Data> Face Label=12 </Data>
          </Component>
          
          <Component Name="fin3" Parent="fins" Type="tri">
           <Data> Face Label=13 </Data>
          </Component>
          
          <Component Name="fin4" Parent="fins" Type="tri">
           <Data> Face Label=14 </Data>
          </Component>
         <!-- fins -->
        
        </Configuration>

Run Matrix File: :file:`matrix.csv`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The conditions at which Cart3D are read from this file, which is a simple list
of conditions.

    .. code-block:: none
    
        # Mach, alpha, phi
        1.25,   0.00,   0.0
        1.25,   1.00,   0.0
        1.25,   1.00,   15.0
        ...
        2.50,   1.00,   45.0

The comment line at the top is not read by pyCart but is placed there for
readability.  Further, the commas are not required; pyCart and other CAPE
modules read trajectory files in a pretty general way.

Run Directives
--------------
Let's run one case, but not the first case.  We can do this by using the
``pycart -I`` command to pick out a specific index or a range of indices.

    .. code-block:: none
    
        $ pycart -I 12
        Case Config/Run Directory    Status  Iterations  Que 
        ---- ----------------------- ------- ----------- ---
        0    poweroff/m1.75a1.0r15.0 ---     /           .   
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 10
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff')
             (STDOUT = 'autoInputs.out')
         > cubes -pre preSpec.c3d.cntl -maxR 10 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff')
             (STDOUT = 'mgPrep.out')
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.75a1.0r15.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff/m1.75a1.0r15.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 

We can check the status of all the cases at Mach 1.75 using the following.

    .. code-block:: none
    
        $ pycart -I 11:15 -c
        Case Config/Run Directory    Status  Iterations  Que 
        ---- ----------------------- ------- ----------- ---
        0    poweroff/m1.75a1.0r0.0  ---     /           .   
        1    poweroff/m1.75a1.0r15.0 DONE    200/200     .   
        2    poweroff/m1.75a1.0r30.0 ---     /           .   
        3    poweroff/m1.75a1.0r45.0 ---     /           .   
        
        ---=3, DONE=1, 

We can use a more direct method to select cases with a certain Mach number using
a constraint.  Let's run the remaining Mach 1.75 cases using that capability.

    .. code-block:: none
    
        $ pycart --cons "Mach==1.75, alpha_t==1.0"
        Case Config/Run Directory    Status  Iterations  Que 
        ---- ----------------------- ------- ----------- ---
        0    poweroff/m1.75a1.0r0.0  ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.75a1.0r0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff/m1.75a1.0r0.0')
             (STDOUT = 'flowCart.out')
        1    poweroff/m1.75a1.0r15.0 DONE    200/200     .   
        2    poweroff/m1.75a1.0r30.0 ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.75a1.0r30.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff/m1.75a1.0r30.0')
             (STDOUT = 'flowCart.out')
        3    poweroff/m1.75a1.0r45.0 ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.75a1.0r45.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/arrow/poweroff/m1.75a1.0r45.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 3 job(s).
        
        ---=3, DONE=1,
        
Run Folders and Output Files
----------------------------
Let's take a look at the files that pyCart created.  First, let's look at the 
files that define the mesh in the ``poweroff/`` folder.

    .. code-block:: none
    
        $ cd poweroff/
        $ ls
        autoInputs.out    input.c3d       m1.75a1.0r45.0  mgPrep.out
        Components.i.tri  m1.75a1.0r0.0   Mesh.c3d.Info   preSpec.c3d.cntl
        Config.xml        m1.75a1.0r15.0  Mesh.mg.c3d   
        cubes.out         m1.75a1.0r30.0  Mesh.R.c3d   

The :file:`.out` files save STDIO printouts from the mesh-generation commands.
The :file:`Mesh.mg.c3d` is the actual mesh file, including multigrid levels
(i.e., coarsened grids).  Our surface triangulation, :file:`arrow.tri` is copied
to :file:`Components.i.tri` in this folder; and the configuration file
:file:`arrow.xml` is copied to :file:`Config.xml`.  The single mesh without
multigrid levels is :file:`Mesh.R.c3d`, and the remaining files are created by
``autoInputs``.

The contents of :file:`input.c3d` set the minimum and maximum *x*, *y*, and *z*
coordinates for the domain on which Cart3D is solved, and is a pretty unique
file.  In this case, it is created automatically by ``autoInputs`` based on the
physical size of the :file:`Components.i.tri` surface.  The other auto-created
file, :file:`preSpec.c3d.cntl` defines regions in which the volume mesh should
have increased resolution.  Calling ``cubes`` also generates regions of
increased resolution based on distance from the surface, but this file can be
used to request more detail.  In addition to some header lines, the contents
look something like the following.

    .. code-block:: none
    
        # BBox: level   Xmin   Xmax      Ymin   Ymax      Zmin    Zmax
        #       (int)  (float) (float) (float) (float)  (float) (float)
        
        
        $__Prespecified_Adaptation_Regions:     # <-Section head (req'd)
        BBox: 6   -0.800   8.800   -4.800   4.800   -4.800   4.800   #  Config BBox
        BBox: 7   -0.299   1.299   -0.800   0.800   -0.799   0.799   #  Comp #0
        BBox: 7    1.700   7.300   -0.800   0.800   -0.800   0.800   #  Comp #1
        BBox: 7    7.201   8.799   -0.800   0.800   -0.799   0.799   #  Comp #2
        BBox: 7    6.479   7.653   -0.401   0.401    1.099   1.900   #  Comp #10
        BBox: 7    6.479   7.653   -1.900  -1.099   -0.401   0.401   #  Comp #11
        BBox: 7    6.479   7.653   -0.401   0.400   -1.900  -1.099   #  Comp #12
        BBox: 7    6.479   7.653    1.099   1.900   -0.400   0.401   #  Comp #13

The third row of *BBox* commands define a region with *x*-coordinates between
1.7 and 7.3, *y*-coordinates between -0.8 and +0.8, and *z*-coordinates between
-0.8 and +0.8.  Within this region, ``cubes`` must make a mesh that has been
refined at least 7 times.  In other words, the mesh size must be at least 128
times smaller than the original mesh.

Now let's look at the files in a run folder.

    .. code-block:: none
    
        $ cd m1.75a1.00r0.00
        $ ls
        body.dat              Components.i.tri     history.dat    moments.dat
        bullet_no_base.dat    Components.i.triq    input.00.cntl  preSpec.c3d.cntl
        bullet_total.dat      conditions.json      input.c3d      run.00.200
        cap.dat               Config.xml           input.cntl     run_cart3d.pbs
        case.json             cutPlanes.00200.plt  loadsCC.dat     
        check.00200           entire.dat           Mesh.c3d.Info  
        checkDT.00200         forces.dat           Mesh.mg.c3d    
        Components.00200.plt  functional.dat       Mesh.R.c3d     

Obviously, there are quite a few files, although many of them are links.  For
example, the files that are listed here and in the parent folder discussed above
are either links or copies.  The :file:`input.c3d` and :file:`preSpec.c3d.cntl`
files are copied because they are small.

Most of the files ending with ``.dat`` are iterative history files.  Some of
these are standard results of running ``flowCart``, and others are specifically
requested.  The most special of these is :file:`history.dat`, which contains the
residual history.  In pyCart, this file is used to determine how many iterations
have been run.


