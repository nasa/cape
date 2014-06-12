
.. automodule:: pc_SetupRuns
    :members:
    
    Customization guide
    -------------------
    
    This script uses methods directly from the pyCart modules and a set
    methodology.  It performs the following tasks, and it assumes that the
    appropriate mesh files exist in each grid folder.
    
        #. Loop through the cases,
        #. create the folder for each run case (if necessary),
        #. copy or link the relevant files to each case folder,
        #. edit the input files to values appropriate for the case,
        #. create a script to run ``flowCart`` or ``aero.csh`` in each folder.
        
    Most of this process is fairly globally applicable, but that may not always
    be the case.  Task 4, editing the input files, is the most likely to 
    require customization.  If the case requires changing a setting other than
    the Mach number, angle of attack, or sideslip angle, then that step will
    require additional work.  Keep in mind that settings that apply to all 
    cases in a grid folder should be altered before running this script
    (although that is not strictly necessary; it's just a good idea). 
    
    To do so, the user should look at the source code of this function and
    modify it.  Use the :func:`pyCart.trajectory.Trajectory.GetFullFolderNames`
    function to get the names of each case folder, and loop through them.
    During the loop, it should be relatively straightforward to duplicate the
    steps above with user-written customizations.
    
    
    Methods
    -------
    
    This script can also be imported as a module if the :file:`scriptlib/`
    folder is on the Python path.  If so, the following methods are created.
