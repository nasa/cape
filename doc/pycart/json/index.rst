.. Documentation for the overall pyCart module.

**********************************************
Control Files for pyCart (:file:`pyCart.json`)
**********************************************

This section describes the JSON files that provide the master control to the
pyCart package. The file format, `JSON <http://json.org>`_, stands for
"JavaScript Object Notation," which is a standard file format but relatively
recent. There is one extension for pyCart JSON files, which is that comments are
allowed, using either ``//`` (preferred) or ``#`` as the comment character.
Newer versions of `vi` or `vim` recognize this file format for syntax
highlighting, as do some other text editors, and setting the highlight mode to
javascript is a useful way to convince other programs to provide useful syntax
highlighting.

Creating JSON files is very similar to creating a text file that contains a
single Python :class:`dict`, with a few differences.

    #. Only double quotes are allowed; single quotes are not valid string
       characters.
    #. Each key name must be a string; syntax like ``{1: "a"}`` is not valid.
    #. True and false are denoted ``true``, ``false``, and ``null`` instead of
        the Python standard ``True``, ``False``, and ``None``.
    #. The only available types are :class:`int`, :class:`float`,
       :class:`unicode`, :class:`bool`, :class:`list`, and :class:`dict`.
    #. Other JSON files can be imported using ``JSONFile(othersettings.json)``.

The pyCart control file, which by default is called :file:`pyCart.json` but can
also have other names, is split into several sections.  Most aspects of the
control file have defaults that will go into effect if the user does not
specify that option, but several entries are required.  The user can also
customize these defaults by editing the file
:file:`$PYCART/settings/pyCart.default.json`, where ``$PYCART`` is the path to
the pyCart root directory.

The following is a nearly minimal pyCart control file that is presented to show
the major parts of the file.

    .. code-block:: javascript
    
        {
            // Stand-alone options
            "ShellCmds": [],
            "InputCntl": "input.cntl",
            "AeroCsh": "aero.csh",
            
            // Settings for creating PBS scripts, if applicable
            "PBS": {},
            
            // Primary settings for running Cart3D
            "RunControl": {
                // Overall control of mode and number of iterations
                "InputSeq": [0, 1],
                "IterSeq": [0, 1500],
                "qsub": false,
                // Inputs to flowCart
                "flowCart": {
                    "it_fc": 500,
                    "mg_fc": 3
                },
                // Meshing options
                "autoInputs": {},
                "cubes": {},
                // Adaptation settings
                "adjointCart": {},
                "Adaptation": {
                    "n_adapt_cycles": [2, 5],
                    "ws_it": [200]
                },
            },
            
            // Settings to define the initial mesh
            "Mesh": {},
            
            // Settings to define component names, reference points, etc.
            "Config": {},
            
            // Settings for the output functional for adaptation
            "Functional": {},
            
            // Settings for folder management and archiving
            "Management": {},
            
            // Describe what forces and moments to record in the data book.
            "DataBook": {},
            
            // Describe reports to create to summarize results
            "Report": JSONFile("Report.json"),
            
            // Mandatory definition of run matrix.
            "Trajectory": {
                // This method points to a file that contains run conditions.
                "File": "Trajectory.csv",
                // It is mandatory to define what the input variables are.
                "Keys": ["Mach", "alpha", "beta"],
                "GroupPrefix": "poweroff",
                "GroupMesh": false
            }
        }
        
        
For more information on these options see the following sections, and the
documentation for the module :mod:`pyCart.options` also provides descriptions
of how the options are used.


.. toctree::
    :maxdepth: 3
    
    usage
    uncategorized
    PBS
    runControl
    flowCart
    adjointCart
    Adaptation
    Mesh
    Config
    Functional
    Management
    Plot
    DataBook
    Report
    Trajectory
