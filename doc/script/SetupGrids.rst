
.. automodule:: pc_SetupGrids
    :members:
    
    Customization guide
    -------------------
    
    This script uses methods directly from the pyCart modules, especially
    :func:`pyCart.cart3d.Cart3d.CreateMesh`.  This uses a very specific
    methodology that is good for the creation of many aero models, which
    involves calling :file:`autoInputs` followed by :file:`cubes` followed by
    :file:`mgPrep`.  It creates a mesh in each mesh folder; see
    :func:`pyCart.trajectory.Trajectory.GetGridFolderNames` for more
    information.
    
    However, in many cases a user will want to define grids that cannot conform
    to this simple scheme.  For example if there is a fin that needs to be
    deflected to a different position for each grid or if there are multiple
    bodies with changing relative positions, a custom action needs to take
    place before :file:`autoInputs` can be run.
    
    To do so, the user should look at the source code of this function and
    modify it.  Use the :func:`pyCart.trajectory.Trajectory.GetGridFolderNames`
    function to get the names of each grid folder, and loop through them.
    During the loop, it should be relatively straightforward to
    
        #. perform the custom action (e.g. rotation and translation), and
        #. run the combination of Cart3D scripts desired.
    
    
    Methods
    -------
    
    This script can also be imported as a module if the :file:`scriptlib/`
    folder is on the Python path.  If so, the following methods are created.
