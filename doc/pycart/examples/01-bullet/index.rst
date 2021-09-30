
.. _pycart-ex-bullet:

Demo 1: Basic Usage on a Bullet Shape
=====================================

The first example demonstrates how pyCart and other Cape interfaces generate
parametrically named files, interact with the master JSON file, and set up
Cart3D input files. This example is found in the file

    ``pycart01-bullet.tar.gz``

To get started, download this file and run the following easy commands:

    .. code-block:: console

        $ tar -xzf pycart01-bullet.tar.gz
        $ cd pycart01-bullet
        $ ./copy-files.py
        $ cd work/

This will copy all of the files into a newly created ``work/`` folder. Follow
the instructions below by entering that ``work/`` folder; the purpose is that
you can easily delete the ``work/`` folder and restart the tutorial at any
time.

The geometry used for this shape is a simple capped cylinder with 4890 faces
and three components.  The surface triangulation, :file:`bullet.tri`, is shown
below.

    .. figure:: bullet01.png
        :width: 4in
        
        Simple bullet shape triangulation with three components
        
The files in the folder before starting the demo are shown below.  The required
Cart3D input file :file:`input.cntl` is not present.  In this case, pyCart will
use a template file without any problems.

    * ``pyCart.json``: Master input file for pyCart
    * ``matrix.csv``: List of run conditions for each variable
    * ``bullet.tri``: Surface triangulation
    * ``Config.xml``: Names for surface components
    
A good first test command is the following, which checks the status of each
case in the matrix.

    .. code-block:: bash
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 ---     /           .   
        1    poweroff/m2.0a0.0b0.0 ---     /           .   
        2    poweroff/m2.0a2.0b0.0 ---     /           .   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
        
        ---=4, 
        
Somehow pyCart has determined that there are four configurations to run at a
variety of Mach numbers, angles of attack, and sideslip angles.  Since the
master input file :file:`pyCart.json` has the default name, the script finds it
automatically.  We could have used ``pycart -c -f pyCart.json`` as well.

Now let's set up and run the first case using the command ``pycart -n 1``.
This will create the results folder ``poweroff/``, create a volume mesh in that
folder, create a case folder for the first set of conditions to run, and call
``flowCart`` to analyze the first case.  The output is shown below.

    .. code-block:: none
    
        $ pycart -n 1 
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 ---     /           .   
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 10
         > cubes -pre preSpec.c3d.cntl -maxR 10 -reorder -a 10 -b 2
         > mgPrep -n 3
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.5a0.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
        
        Submitted or ran 1 job(s).
        
        ---=1,
        
Obviously in these examples the value of ``PWD`` will differ from what is shown
in these examples. This command tells pyCart to loop through the cases until if
finds the first case to run. Because no cases had been run prior to executing
this command, the following steps are taken as a result of this command.

    1. Read project settings from :file:`pyCart.json` and conditions from
       :file:`matrix.csv`
        
    2. Create the mesh
    
      A. Create the ``poweroff`` folder
      B. Read the ``bullet.tri`` file and write it to the ``poweroff`` folder
      C. Run ``autoInputs`` to create ``input.c3d`` and ``preSpec.c3d.cntl``
      D. Run ``cubes`` to create volume mesh :file:`Mesh.c3d`
      E. Run ``mgPrep`` to prepare the grid for multigrid
       
    3. Prepare the case
    
      A. Create the ``m1.50a0.0b0.0`` folder
      B. Link the mesh files created in the previous step
      C. Copy the template ``input.cntl`` and set Mach, alpha, and beta
      D. Create a PBS script :file:`run_cart3d.pbs`
    
    4. Run the case by calling ``bash run_cart3d.pbs``

Let's run another case.

    .. code-block:: none
    
        $ pycart -n 1
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   0.0
        1    poweroff/m2.0a0.0b0.0 ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m2.0a0.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
        
        Submitted or ran 1 job(s).
        
        ---=1, DONE=1,

This time, there is a lot less output because the different cases can use the
same mesh.  In the description of the tasks performed for the first case, step
2 can be skipped for subsequent runs.

Now let's check the status again using ``pycart -c``.  The CPU time is listed as
0.0 for both cases because this simple case takes about 0.02 total hours, and
the display is rounded down to the nearest tenth of an hour.

    .. code-block:: none
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   0.0
        1    poweroff/m2.0a0.0b0.0 DONE    200/200     .   0.0
        2    poweroff/m2.0a2.0b0.0 ---     /           .   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
                
        ---=2, DONE=2,
    
That's it.  Now we have two cases run in separate folders, and each looks like a
standard Cart3D run.  Finally, the default call to ``pycart`` is equivalent to
``pycart -f pyCart.json -n 10``.  Running this case in the current folder gives
the following results.

    .. code-block:: none
    
        $ pycart
        Case Config/Run Directory  Status  Iterations  Que CPU Time
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 DONE    200/200     .   0.0
        1    poweroff/m2.0a0.0b0.0 DONE    200/200     .   0.0
        2    poweroff/m2.0a2.0b0.0 ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m2.0a2.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
        Using template for 'input.cntl' file
             Starting case 'poweroff/m2.0a2.0b2.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
        
        Submitted or ran 2 job(s).
        
        ---=2, DONE=2,

This attempts to run 10 cases, but the first two cases are already completed.
Since there are only two cases remaining, the job quits before it can get to 10
cases.
