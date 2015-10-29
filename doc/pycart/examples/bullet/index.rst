
Demo 1: Basic Usage on a Bullet Shape
=====================================

The first example demonstrates how pyCart and other Cape interfaces generate
parametrically named files, interact with the master JSON file, and set up
Cart3D input files.  This example is found in ``$PYCART/examples/pycart/bullet``
where ``$PYCART`` is the installation folder.

The geometry used for this shape is a simple capped cylinder with 4890 faces and
three components.  The surface triangulation, :file:`bullet.tri`, is shown
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
    
To get started, make sure that the ``$PYCART/scriptlib`` folder is part of your
path and ``$PYCART`` is listed in the environment variable *PYTHONPATH*.  It is
a good idea to put these in your startup file or create a module.  For a BASH
environment, the following commands set up the environment for using pyCart
assuming that ``$HOME/pycart`` is where pyCart was installed.

    .. code-block:: bash
    
        export PATH="$PATH:$HOME/pycart/scriptlib"
        export PYTHONPATH="$PYTHONPATH:$HOME/pycart"
        
For a C-shell environment, use the following.

    .. code-block:: csh
    
        setenv PATH "$PATH:$HOME/pycart/scriptlib"
        setenv PYTHONPATH "$PYTHONPATH:$HOME/pycart"
        
Assuming the present working directory is in this demo folder, i.e.
``$PYCART/examples/pycart/bullet``, a good first test command is the following,
which checks the status of each case in the matrix.

    .. code-block:: bash
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que 
        ---- --------------------- ------- ----------- ---
        0    poweroff/m1.5a0.0b0.0 ---     /           .   
        1    poweroff/m2.0a0.0b0.0 ---     /           .   
        2    poweroff/m2.0a2.0b0.0 ---     /           .   
        3    poweroff/m2.0a2.0b2.0 ---     /           .   
        
        ---=4, 
        
Somehow pyCart has determined that there are four configurations to run at a
variety of Mach numbers, angles of attack, and sideslip angles.  Since the
master input file :file:`pyCart.json` has the default name, the script finds it
automatically.  We could have used ``pycart -c -f pyCart.json`` as well.

Now let's set up and run the first case using the command ``pycart -n 1``.  This
will create the results folder ``poweroff/``, create a volume mesh in that
folder, create a case folder for the first set of conditions to run, and call
``flowCart`` to analyze the first case.  The output is shown below.

    .. code-block:: none
    
        $ pycart -n 1 
        Case Config/Run Directory  Status  Iterations  Que 
        ---- --------------------- ------- ----------- ---
        0    poweroff/m1.5a0.0b0.0 ---     /           .   
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 10
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bullet/poweroff')
             (STDOUT = 'autoInputs.out')
         > cubes -pre preSpec.c3d.cntl -maxR 10 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bullet/poweroff')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bullet/poweroff')
             (STDOUT = 'mgPrep.out')
        Using template for 'input.cntl' file
             Starting case 'poweroff/m1.5a0.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bullet/poweroff/m1.5a0.0b0.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1,
        
Specifically