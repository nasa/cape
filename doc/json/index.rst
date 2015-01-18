.. Documentation for the overall pyCart module.

**********************************************
Control Files for pyCart (:file:`pyCart.json`)
**********************************************

This section describes the JSON files that provide the master control to the
pyCart package.  The file format, JSON, stands for "Javascript Object
Notation," which is a standard file format but relatively recent.  There is one
extension for pyCart JSON files, which is that comments are allowed, using
either ``//`` (preferred) or ``#`` as the comment character.  Newer versions of
`vi` or `vim`  recognize this file format for syntax highlighting, as do some
other text editors, and setting the highlight mode to javascript is a useful
way to convince other programs to provide useful syntax highlighting.

Creating JSON files is very similar to creating a text file that contains a
single Python :class:`dict`, with a couple of differences.

    #. Only double quotes are allowed; single quotes are not valid string
       characters.
    #. Each key name must be a string; syntax like ``{1: "a"}`` is not valid.
    #. True and false are denoted ``true`` and ``false`` instead of the Python
       standard ``True`` and ``False``.
    #. The only available types are :class:`int`, :class:`float`, :class:`str`,
       :class:`bool`, :class:`list`, and :class:`dict`.  However, nesting is
       entirely allowed.

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
            "GroupMesh": true,
            
            // Settings for creating PBS scripts, if applicable
            "PBS": {},
            
            // Primary settings for running flowCart
            "flowCart": {
                // Some of these variables are specific to pyCart
                "InputSeq": [0, 1],
                "IterSeq": [0, 1500],
                "qsub": false,
                // Most variables use their names from `aero.csh`
                "it_fc": 500,
                "mg_fc": 3
            },
            
            // Options for running adjointCart, if an adaptive run
            "adjointCart": {},
            
            // Settings to control adaptation.
            "Adaptation": {
                // These share names with variables in `aero.csh`
                "n_adapt_cycles": [2, 5],
                "ws_it": [200]
            },
            
            // Settings to define the initial mesh
            "Mesh": {},
            
            // Settings to define component names, reference points, etc.
            "Config": {},
            
            // Settings for the output functional for adaptation
            "Functional": {},
            
            // Settings for folder management and archiving
            "Management": {},
            
            // Definitions of iterative history plots to make
            "Plot": {},
            
            // Describe what forces and moments to record in the data book.
            "DataBook": {},
            
            // Mandatory definition of run matrix.
            "Trajectory": {
                // This method points to a file that contains run conditions.
                "File": "Trajectory.dat",
                // It is mandatory to define what the input variables are.
                "Keys": ["Mach", "alpha", "beta"],
                "GroupPrefix": "poweroff"
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
    flowCart
    adjointCart
    Adaptation
    Mesh
    Config
    Functional
    Management
    Plot
    DataBook
