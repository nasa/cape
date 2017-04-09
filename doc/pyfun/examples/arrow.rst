
.. _pyfun-ex-arrow:

Demo 1: Basic Viscous Usage on Arrow with Fins
===============================================

This first example is found in ``$PYCART/examples/pyfun/01_arrow`` where
``$PYCART`` is the installation folder.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
The files in this folder are listed below with a short description.  In this
case, the run matrix is defined within the ``pyFun.json`` file.

    * ``pyFun.json``: Master input control file for pyFun
    * ``fun3d.nml``: Template namelist file
    * ``arrow-far.ugrid``: Volume grid, ASCII AFLR3 format
    * ``arrow-far.mapbc``: Boundary conditions file
    * ``arrow-far.tri``: (Not used) Cart3D surface triangulation
    * ``arrow-far.xml``: (Not used) XML files used to define larger components
    
To get started, make sure that the ``$PYCART/bin`` folder is part of your
path and ``$PYCART`` is listed in the environment variable *PYTHONPATH*.  It is
a good idea to put these in your startup file or create a module.  For a BASH
environment, the following commands set up the environment for using pyCart
assuming that ``$HOME/pycart`` is where pyCart was installed.

    .. code-block:: bash
    
        export PATH="$PATH:$HOME/pycart/bin"
        export PYTHONPATH="$PYTHONPATH:$HOME/pycart"
        
For a C-shell environment, use the following.

    .. code-block:: csh
    
        setenv PATH "$PATH:$HOME/pycart/bin"
        setenv PYTHONPATH "$PYTHONPATH:$HOME/pycart"
        
Assuming the present working directory is in this demo folder, i.e.
``$PYCART/examples/pycart/01_bullet``, a good first test command is the
following, which checks the status of each case in the matrix.

    .. code-block:: bash
    
        $ pyfun -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    arrow/m0.80a0.0b0.0   ---     /           .            
        1    arrow/m0.80a2.0b0.0   ---     /           .            
        2    arrow/m0.84a0.0b0.0   ---     /           .            
        3    arrow/m1.20a0.0b0.0   ---     /           .  

Here pyFun has read the default master file name, ``pyFun.json`` and determined
that this is a setup with four cases.  The run matrix has at least three input
variables, which are Mach number, angle of attack, and sideslip, although the
user can't tell yet whether or not there are more input variables that don't
affect the parametric folder names.

As it turns out, this is a viscous example, so we should expect Reynolds number
per inch and freestream static temperature to be inputs.   It is permitted in
pyFun not to specify these inputs and just keep a constant value inherited from
the ``fun3d.nml`` namelist, but it would be unusual to have varying Mach
numbers and a constant freestream state.  If we look at the final section in
``pyFun.json``, we see that Reynolds number and temperature are indeed
variables.

    .. code-block:: javascript
    
        // Run matrix description
        "Trajectory": {
            // File and variable list
            "File": "",
            "Keys": [
                "mach", "alpha", "beta", "Re", "T", "config", "Label"
            ],
            // Modify one definition
            "Definitions": {
                "mach": {"Format": "%.2f"}
            },
            // Group settings
            "GroupMesh": false,
            // Label universal
            "Label": "",
            "config": "arrow",
            // Local values
            "mach":  [0.8, 0.8, 0.84, 1.2],
            "alpha": [0.0, 2.0, 0.0,  0.0],
            "beta":  [0.0, 0.0, 0.0,  0.0],
            "Re":    [1e3, 1e3, 1e4,  1e4],
            "T":     [478, 478, 478,  478]
        }
        
The second line, *Keys*, lists the five input variables we have discussed above
and two additional input variables used for bookkeeping.  Each of these
variables has a standard name, so pyFun provides a default definition and
interpretation.  However, it is possible to modify any aspect of a variable's
behavior in the *Definitions* section.

Here we have modified the *mach* definition so that pyFun explicitly includes
exactly two digits after the decimal place in the folder name (otherwise we may
have difficulty with the Mach 0.84 case).  There are many more capabilities of
this *Definitions* section.  Some of them are discussed in other examples, and
the complete guide can be found in :ref:`the "Trajectory" section of the JSON
guide <cape-json-Trajectory>`.

In this case, we have decided to specify the values of the variables within the
JSON file.  We can specify a list with one value for each case, as in ``"mach":
[0.8, 0.8, 0.84, 1.2]`` or a constant value that applies to all cases as in
``"config": "arrow"``.

To actually run a case, specifically the first case, run the following command.
It will show some status updates as it runs, but this may take a respectable
amount of time (about 7.5 CPU hours) to demonstrate a semi-realistic case.  It
can be aborted with a ``Ctrl-C`` command if desired.

    .. code-block:: bash
    
        $ pyfun -I 0
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    arrow/m0.80a0.0b0.0   ---     /           .            
          Case name: 'arrow/m0.80a0.0b0.0' (index 0)
             Starting case 'arrow/m0.80a0.0b0.0'.
         > nodet --animation_freq 500
             (PWD = '/home/dalle/usr/pycart/examples/pyfun/arrow/arrow/m0.80a0.0b0.0')
             (STDOUT = 'fun3d.out')
         > nodet --animation_freq 500
             (PWD = '/home/dalle/usr/pycart/examples/pyfun/arrow/arrow/m0.80a0.0b0.0')
             (STDOUT = 'fun3d.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 

While this is running, we can open another window and navigate to the same
folder.  Then we can check the status using another ``pyfun -c`` call.

    .. code-block:: bash
    
        $ pyfun -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    arrow/m0.80a0.0b0.0   RUN     57/1000     .        0.7 
        1    arrow/m0.80a2.0b0.0   ---     /           .            
        2    arrow/m0.84a0.0b0.0   ---     /           .            
        3    arrow/m1.20a0.0b0.0   ---     /           .            
        
        ---=3, RUN=1,

Once the case is complete (not fully necessary for this demo), the status will
change to the following.

    .. code-block:: bash
    
        $ pyfun -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    arrow/m0.80a0.0b0.0   DONE    1000/1000   .        7.5 
        1    arrow/m0.80a2.0b0.0   ---     /           .            
        2    arrow/m0.84a0.0b0.0   ---     /           .            
        3    arrow/m1.20a0.0b0.0   ---     /           .            
        
        ---=3, DONE=1, 

The ``arrow/m0.80a0.0b0.0`` folder itself contains the hallmark files of a
FUN3D run with a project prefix of ``arrow`` (which is set within
``pyFun.json`` or ``fun3d.nml``) with a few additional files used by pyFun to
keep track of status.
