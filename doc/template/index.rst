.. Documentation for the pyCart scripts

================
Script Templates
================
        
The scripts in the script library are intended to provide a basic level of
functionality for rapidly setting up cases with multiple conditions.  If the run
matrix consists only of changes in Mach number, angle of attack, and sideslip,
the basic routine

    .. code-block:: console
        
        $ pc_SetupGrids.py
        $ pc_SetupRuns.py
        
is adequate to prepare each of the run cases.

However, in many cases, this will provide enough customization to prepare more
challenging or detailed run matrices.  To name a few of the possible variables
that could be involved, there could be changes in boundary conditions due to
thrust settings, changes in relative positions of separate bodies, changes in
fin deflection angles, and many more.  It would be difficult to come up with
enough settings in the :file:`pyCart.json` file to support a significant
amount of these situations.  In any case, doing so would probably result in a
file format that is impossibly difficult to read, anyway.

Fortunately, pyCart is set up with the intent that it can provide utility in
cases even when the provided setup scripts cannot be used.  To do so requires
writing a custom script.  Some templates and ideas for how such scripts can be
created and used are included in the script library folder.

Usually, these scripts can be run without modification, but they will not do
much.  However, they provide ideas in examples in portions of code that are
commented out.

.. automodule:: pc_CaseTemplate
    :members:
