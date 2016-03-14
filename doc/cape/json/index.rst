
**************************************
Baseline JSON Settings for :mod:`cape`
**************************************

The following JSON settings are available to the :mod:`cape` basis module and
apply to each of the solvers.  Most of these settings will have the same meaning
for :mod:`pyCart`, :mod:`pyFun`, and :mod:`pyOver`, although some of them are
superseded by modified options in those derived modules.

The template name for the CAPE baseline control file is :file:`cape.json`.  It
can be invoked via the command-line function ``cape`` via either of the
following two commands.

    .. code-block:: console
    
        $ cape
        $ cape -f cape.json

A basic template is shown below.

    .. code-block:: javascript
    
        {
            // Universal Python settings
            "PythonPath": [],
            
            // Commands to run at beginning of script
            "ShellCmds": [],
            
            // Maximum number of commands to submit or run
            "nSubmit": 20,
            
            // File settings mask (not applicable to Windows systems)
            "umask": "0027"
            
            // Primary settings for iterations, phases, etc.
            "RunControl": {
                // List of phases to run
                "PhaseSequence": [0, 1],
                // Iteration minimums in each phase
                "PhaseIters": [200, 400],
                // Other local options
                "qsub": False,
                "MPI": False,
                "mpicmd": "mpiexec",
                "Resubmit": false,
                "Continue": true,
                // Special subsections
                "Environ": { },
                "ulimit": { },
                "Archive": { }
            },
            
            // PBS job settings
            "PBS": { },
            
            // Mesh settings
            "Mesh": { },
            
            // Data book settings
            "DataBook": { },
            
            // Automated Report settings
            "Report": { },
            
            // Run matrix settings
            "Trajectory": {
                // List of trajectory keys
                "Keys": ["mach", "alpha", "beta"],
                // File containing conditions
                "File": "matrix.csv"
            }
            
        }
        
An interactive instance of these setting scan be accessed via the following
Python commands.

    .. code-block:: python
    
        import cape
        
        # Import CAPE control instance
        cntl = cape.Cntl()
        # Options interface
        opts = cntl.opts
        
A detailed description of each class of options can be found in the pages that
follow.

.. toctree::
    :maxdepth: 3
    
    uncategorized
    PBS
    runControl
    Archive
    
    
