
.. _pyover-example-bullet:

------------------------
OVERFLOW Bullet Example
------------------------

This pyOver example looks at the process from grid generation to execution and
post-processing for a simple bullet geometry.  This example is located in 

    * ``$CAPE/examples/pyover/01_bullet/``

and the folder is fairly empty before beginning this example. This section
guides the user through generating a grid system using Chimera Grid Tools. The
resulting surface grid system is shown in :numref:`tab-pyover-bullet-01`, and
one view of the volume grid is shown in :numref:`fig-pyover-bullet-01`.

    .. _tab-pyover-bullet-01:
    .. table:: OVERFLOW surface grid for bullet example
    
        +------------------------------+------------------------------+
        |.. image:: bullet-surf-01.png |.. image:: bullet-surf-02.png |
        |    :width: 3.1in             |    :width: 3.1in             |
        |                              |                              |
        |Front and side view           |Aft view                      |
        +------------------------------+------------------------------+
        
    .. _fig-pyover-bullet-01:
    .. figure:: bullet-vol-01.png
        :width: 4.0in
        
        OVERFLOW volume grid slice for bullet example
        
This is a three-grid system that demonstrates a lot of the basic features of
running OVERFLOW and creating a simple grid system.  
        
Grid generation
----------------
While many users are opting to create overset grid systems using a graphical
user interface (for example Pointwise), this example guides users through a
traditional script-based grid generation process.  While there are really no
special contributions of pyOver to this process, it may increase understanding
of later tasks to explain how the grid system was generated.

Some aspects of the grid generation example may rely on recent features of
Chimera Grid Tools.

Starting from the base directory for the example,
``$CAPE/examples/pyover/01_bullet/``, the grid generation takes place in the
``dcf/`` folder.  The initial contents of this folder include a ``geom/``
folder that contains the basic definitions for the geometry and some TCL
scripts for generating the grid.

    * ``dcf/``
    
        - *GlobalDefs.tcl*: script to set overall variables for grid system
        - *inputs.tcl*: various local variables such as grid spacing
        - *config.tcl*: instructions for names of grids
        - ``geom/``: geometry definitions folder
        
            * *bullet.i.tri*: surface geometry triangulation
            * *bullet.stp*: STEP file of relevant curves
            * *bullet.lr8.crv*: little-endian curve of axisymmetric radius
            
        - ``bullet/``: grid building are for **bullet** component
        
            * *BuildBullet.tcl*: main script to generate **bullet** grids
            * *localinputs.tcl*: local settings for **bullet** component
            * *Makefile*: ``make`` instructions for building surface grids

Geometry definitions
^^^^^^^^^^^^^^^^^^^^^
The surface triangulation and curves generated from the natural boundaries of
this triangulation are shown in :numref:`fig-pyover-bullet-02`.

    .. _fig-pyover-bullet-02:
    .. figure:: bullet-geom-01.png
        :width: 3.5in
        
        Surface triangulation and curves for bullet example
        
The curve file was generated using the STEP file in addition to the
:mod:`pc_StepTri2Crv` script:

    .. code-block:: console
    
        $ pc_StepTri2Crv.py bullet -lr8 -o bullet.lr8.crv
        
Grid script setup
^^^^^^^^^^^^^^^^^
The contents of the ``dcf/`` directory are detailed above, but some aspects of
the TCL scripts are explained here.  This example has only a single logical
"component," called **bullet**, but a more general use case for the Chimera
Grid Tools grid script system may have many such components.  For example, if
we added fins to this example, we may create the grids for those fins using
another folder called ``fins/``.

Grid scripts rely on several hard-coded TCL file names, which can be guessed
from the layout of this ``dcf/`` example.  The ``GlobalDefs.tcl`` script sets a
few global variables for the grid script.  None of the variables set in this
file are universal requirements, but those that are set in this TCL script can
become available to all of the other scripts.  The contents of this particular
example of the ``GlobalDefs.tcl`` are shown below.

    .. code-block:: tcl
    
        #!/usr/bin/env tclsh

        global Par
        
        # Source folder stuff
        set ScriptFile [file normalize [info script]]
        set ScriptDir  [file dirname $ScriptFile]
        set RootDir    [file join {*}[lrange [file split $ScriptDir] 0 end]]
        set GeomDir    [file join $RootDir geom]
        
        set Par(ScriptFile) $ScriptFile
        set Par(ScriptDir)  $ScriptDir
        set Par(GeomDir)    $GeomDir
        
        # Global switch for OVERFLOW solver
        set ovfi_inputs "ssor"
        
        # List of parts included
        set IncludeBullet    1
        
        # Grid scaling parameter
        set GlobalScaleFactor 1.0
        
Some of this unusual TCL syntax is just intended to save the absolute path to
various folders, including the one containing the script (*ScriptDir* and
*RootDir* in this example) and the input geometry files (*GeomDir*).  The
*GlobalScaleFactor* can also be used to change the overall resolution of grid
as long as all the other spacing variables are programmed to change with
*GlobalScaleFactor*.

The ``inputs.tcl`` file is much longer but is also a script that basically just
sets variables for use elsewhere.  It defines basic grid resolution settings
with syntax such as

    .. code-block:: none
    
        # ------
        # Wall spacing and stretching ratio
        # ------
        set Par(ds,wall) 0.001
        set Par(sr,wall) 1.2
        set Par(klayer) 3
        # ------
        # Surface stretching ratio
        # ------
        set Par(sr)      1.2
        set Par(sr,slow) 1.1
        # ------
        # Main marching distance
        # ------
        set Par(md)              5.0
        set Par(md,protub)       2.0
        set Par(md,protub,small) 1.0
        
It is a common convention to use *Par* as the TCL variable that stores
parameters for grid spacing.  The ``inputs.tcl`` script also contains default
volume grid options

    .. code-block:: none 
    
        # ------
        # Default hypgen inputs
        # ------
        set Par(smu) 0.5
        set default(zreg)  $Par(md)
        set default(dz0)   $Par(ds,wall)
        set default(dz1)   $Par(ds,glb)
        set default(srmax) $Par(sr)
        set default(ibcja) -10
        set default(ibcjb) -10
        set default(ibcka) -10
        set default(ibckb) -10
        set default(imeth) 2
        
The volume options (``hypgen`` options) can be overridden for individual
surface grids as needed.  In addition the syntax

    .. code-block:: tcl
    
        # ------
        # Volume grids created by other means
        # ------
        set bullet_body(nomakevol) 1

instructs the ``BuildVol`` command not to grow a volume grid for the grid named
``bullet_body`` because that volume grid is already created during the
execution of ``BuildBullet.tcl``.

Within ``inputs.tcl``, there are also instructions for what settings to use in
the template OVERFLOW namelist, ``overflow.inp``:

    .. code-block:: tcl
    
        # ------
        # Inputs for the OVERFLOW flow solver
        # ------
        set Ovr(incore) .T.
        set Ovr(nsteps) 100
        set Ovr(restrt) .F.
        set Ovr(fmg)    .T.
        set Ovr(fmgcyc) "1000, 1000, 0"
        set Ovr(nglvl)  4
        set Ovr(nfomo)  2
        set Ovr(dtphys) 0.0
        set Ovr(nitnwt) 0
        set Ovr(walldist) 2
        
Of course, these can be altered later by :mod:`pyOver` using the
:mod:`pyOver.overNamelist` interface.  Finally, the *mixsurcomp* variable can
be used to group surface families into larger components, which affects the
file ``mixsur.i`` that is built by ``BuildMixsuri``.

The file ``config.tcl`` describes the list of grids to include (for each
component, in examples where that's appropriate).

    .. code-block:: tcl
        
        #!/usr/bin/env tclsh

        source [GetIfile GlobalDefs.tcl]
        source [GetIfile inputs.tcl]
        
        # List of bullet grids
        set grids "bullet/bullet_body
                   bullet/bullet_cap
                   bullet/bullet_base "
        
        # List of xrays
        set xrays "bullet/bullet "
        
        # Convert variable names
        set rootnames "$grids"
        set xraynames "$xrays"

This script is fairly self-explanatory for a simple example such as this, but
in more general cases this file often contains more logic for including or not
including grids based on component on/off switches in ``GlobalDefs.tcl``.  The
variables *rootnames* and *xraynames* are hard-coded and used by the grid
script system.

Surface grid generation
^^^^^^^^^^^^^^^^^^^^^^^^
From the ``dcf/`` folder, run the Chimera Grid Tools command

    .. code-block:: console
    
        $ BuildSurf
        
However, users should take care to match endianness.  The input file is
little-endian, so the one of the following system commands may be necessary.
Note that the ``csh`` versions of these commands would need to use ``setenv``.

    .. code-block:: console
    
        $ export GFORTRAN_CONVERT_UNIT="little_endian"
        $ export F_UFMTENDIAN="little"
        
This command reads the *rootnames* variable and makes a list of all the folders
referenced by any grid, which in our simple example is simply ``bullet/``.
Then the surface grid builder goes into each such folder and just calls

    .. code-block:: console
    
        $ make
        
Therefore the contents of the ``Makefile`` in each component folder have a
direct impact.  The contents for this ``Makefile`` are shown below.  Basically
it instructs ``make`` to run the local script ``BuildBullet.tcl`` if any of
four files are missing or if any of two TCL files are newer than the grid
output files.

    .. code-block:: make
    
        SurfGrids = bullet_cap.srf \
                    bullet_body.srf \
                    bullet_base.srf \
                    bullet.xry
        
        all: $(SurfGrids)
        
        clobber:
            /bin/rm -f \
            bullet_cap.srf \
            bullet_body.srf \
            bullet_base.srf
            
        $(SurfGrids): BuildBullet.tcl localinputs.tcl
            ./BuildBullet.tcl

The other fixed-name file in the ``bullet/`` folder is called
``localinputs.tcl``.  This TCL script is sourced during the generation of
surface grids and of volume grids.  The first part of this script sets spacings
and point counts specific to this component.

    .. code-block:: none
    
        #!/usr/bin/env tclsh

        global Ovr Par 
        
        # Body spacings
        set Par(ds,bullet,cap)  [expr 0.10*$Par(ds,glb)]
        set Par(ds,bullet,crn)  [expr 0.05*$Par(ds,glb)]
        set Par(ds,bullet,body) [expr 0.25*$Par(ds,glb)]
        set Par(ds,bullet,aft)  [expr 0.15*$Par(ds,glb)]
        
        # Number of points around the bullet
        set Par(npcirc,bullet) 73
        
Within the ``BuildBullet.tcl`` script contains many calls to the TCL utilities
of Chimera Grid Tools.  After running this script (via ``BuildSurf`` or a
direct call) the following files are created in the ``bullet/`` folder.

    * **bullet_base.srf**: surface grid ``bullet_base``
    * **bullet_body.srf**: surface grid ``bullet_body``
    * **bullet_cap.srf**: surface grid ``bullet_cap``
    * **bullet_body.vol**: volume grid ``bullet_body``
    * *bullet_base.ovfi*: OVERFLOW inputs for grid ``bullet_base``
    * *bullet_body.ovfi*: OVERFLOW inputs for grid ``bullet_body``
    * *bullet_cap.ovfi*: OVERFLOW inputs for grid ``bullet_body``
    * **bullet.xry**: X-Ray cutter file for bullet's body
    
These files demonstrate that one component may have multiple grids, and thus
the decision on what is a "component" and what is multiple components is
decided by the user for the specific situation.  The grid script system keeps
all grid files separate (although to be clear these are multiple-grid format
with one grid).

Regarding the ``ovfi`` files, they contain namelists specifically for each
grid.  These are assembled into the ``overflow.inp`` namelist for each included
grid (order is important).

Volume grid generation
^^^^^^^^^^^^^^^^^^^^^^^
Creating the volume grids is performed using the following system command, also
run from the ``dcf/`` root folder.

    .. code-block:: console
    
        $ BuildVol
        
This creates a volume grid for the two grids that did not have a previously
generated grid.  The ``bullet_body.vol`` grid is generated by rotating a 2D
grid about the *x*-axis, so this volume does not need to be generated by
``hypgen``.  After running ``BuildVol``, the following additional files are
created.

    * **bullet_base.vol**: volume grid ``bullet_base``
    * **bullet_cap.vol**: volume grid ``bullet_cap``
    * *bullet_base.bvinp*: ``makevol`` inputs
    * *bullet_base.hypi*: ``hypgen``  inputs
    * *bullet_base.mvlog*: ``makevol`` output log
    * *bullet_cap.bvinp*: ``makevol`` inputs
    * *bullet_cap.hypi*: ``hypgen`` stream inputs
    * *bullet_cap.mvlog*: ``makevol`` output log
    
Grid assembly
^^^^^^^^^^^^^^
To create the assembled volume and surface grids, the following (not
necessarily obvious) commands are run.

    .. code-block:: console
    
        $ BuildPlot

This results in the surface grid file ``Composite.srf``, which contains all
three surface grids combined into a single file.

    .. code-block:: console
        
        $ BuildPlot -vol
        
This file creates ``Composite.vol``, which is the primary volume grid that we
need as input to run OVERFLOW.  Copy this file into the ``common/``
subdirectory of the parent folder.  The surface grid file is not required, but
can be convenient to have in a common location.

    .. code-block:: console
    
        $ cp Composite.vol ../common/grid.in
        $ cp Composite.srf ../common/grid.srf

Assembling inputs
^^^^^^^^^^^^^^^^^^
The following two commands create the template OVERFLOW input namelist and
``mixsur`` input file, respectively.

    .. code-block:: console
    
        $ BuildOveri
        $ BuildMixsuri
    
After running the first command, the files ``overflow.inp`` and ``xrays.in``
are created.  Both of these files are also required for running, so they can be
copied into the ``../common/`` folder, too.  However, the ``overflow.inp`` file
is already provided; users can compare them to check that they are identical.

    .. code-block:: console
    
        $ cp xrays.in ../common/
        
The ``BuildMixsuri`` command creates the file ``mixsur.i``.  We will need this
file later, first let's apply the xrays by running OVERFLOW for zero
iterations.  To run OVERFLOW in this manner, we set the namelist parameter
*OMIGLB* > *IRUN* to ``2``.  The normal value is ``0``.  Fortunately, the
``overflow.inp`` file we created already has *IRUN*\ =2.  Now we create a
folder called ``irun2/`` and copy the necessary files into it.  The following
commands can be run from the ``dcf/`` folder.

    .. code-block:: console
    
        $ mkdir -p irun2
        $ cp Composite.vol irun2/grid.in
        $ cp overflow.inp irun2/
        $ cp xrays.in irun2/
        
Now we can enter this folder and run OVERFLOW.

    .. code-block:: console
    
        $ cd irun2
        $ overrunmpi -np 6 overflow
        
Users who do not have a compiled MPI version of OVERFLOW can try .

    .. code-block:: console
    
        $ overrun overflow

This will run OVERFLOW and create quite a few output files. Most of these we
can ignore, but we will need ``x.save`` to run ``mixsur``.  In addition, for
more complex grids, this is the file that we inspect to see interpolation
quality and check the number of orphan points.

To run ``mixsur``, let's go up two folders and set things up to run ``mixsur``
in the ``common/fomo/`` folder.  The term *fomo* is a common portmanteau for
"force and moment" in the OVERFLOW world.

    .. code-block:: console
    
        $ cd ../..
        $ pwd
        .../pyover/01_bullet
        $ cp dcf/irun2/x.save common/fomo
        
The ``mixsur.i`` file is already in the ``fomo/`` folder.  Now we can enter
that folder and run ``mixsur``.

    .. code-block:: console
    
        $ cd common/fomo
        $ mixsur < mixsur.i > mixsur.o
        
This creates a significant number of files, most of which are useful for at
least one OVERFLOW data analysis scenario.  The file ``mixsur.fmp`` is critical
because it provides instructions to OVERFLOW on how to integrate the surface
pressures and viscous loads into component forces & moments.  In addition, the
``grid.i.tri`` file is a unique surface triangulation created from the surface
grid.

    .. _fig-pyover-bullet-03:
    .. figure:: bullet-tri-01.png
        :width: 3.5 in
        
        Surface tri from ``mixsur`` of OVERFLOW bullet surface grid
        
The surface triangulation created by ``mixsur`` is shown in
:numref:`fig-pyover-bullet-03`.  It shows that the surface has been divided
into three families, a cap, fuselage, and base, and that these do not
correspond to the boundaries between grids or something similar.  These
boundaries are set within ``BuildBullet.tcl``.  In regions of overlapping
grids, ``mixsur`` picks a unique triangle (roughly the smallest available,
although this process becomes very complex in the general case) and then
creates "zipper" triangles to join together the triangles that are selected
from dividing the surface grid quads in half.

At this point, we have created all of the grid files that are needed, and we
are ready to start running OVERFLOW using pyOver.


Execution
----------
In addition to the grid input files, ``overflow.inp`` template namelist, and
``mixsur.fmp`` file all described in the previous section, the ``01_bullet/``
folder contains a master settings file ``pyOver.json`` and a run matrix
``inputs/matrix.csv``.

To run one case, we can run the following command.  This will run the second
case in the matrix (index 1 according to Python's 0-based indexing).

    .. code-block:: console
    
        $ pyover -I 1
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 ---     /           .            
          Case name: 'poweroff/m0.8a4.0b0.0' (index 1)
             Starting case 'poweroff/m0.8a4.0b0.0'
         > overrunmpi -np 6 run 01
             (PWD = '/examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.07 hrs (phase 0)
           Wall time used: 0.07 hrs
           Previous phase: 0.07 hrs
         > overrunmpi -np 6 run 02
             (PWD = '/examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.08 hrs (phase 1)
           Wall time used: 0.14 hrs
           Previous phase: 0.08 hrs
         > overrunmpi -np 6 run 03
             (PWD = /examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.05 hrs (phase 2)
        
        Submitted or ran 1 job(s).
        
        ---=1, 


