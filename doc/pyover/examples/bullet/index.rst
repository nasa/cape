
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
        
        
        

Execution
----------

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

