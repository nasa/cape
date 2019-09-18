.. Documentation for the overall pyOver module.

.. _pyover-json:

**********************************************
Control Files for pyOver (:file:`pyOver.json`)
**********************************************

This section describes the JSON files that provide the master control to the
pyFun package. The file format, `JSON <http://json.org>`_, stands for
"JavaScript Object Notation," which is a standard file format but relatively
recent.  There is one extension for pyCart JSON files, which is that comments are
allowed, using either ``//`` (preferred) or ``#`` as the comment character.
Newer versions of `vi` or `vim` recognize this file format for syntax
highlighting, as do some other text editors, and setting the highlight mode to
javascript is a useful way to convince other programs to provide useful syntax
highlighting.

Creating :ref:`JSON files <json-syntax>` is very similar to creating a text file
that contains a single Python :class:`dict`, with a few differences.

    #. Only double quotes are allowed; single quotes are not valid string
       characters.
    #. Each key name must be a string; syntax like ``{1: "a"}`` is not valid.
    #. True and false are denoted ``true``, ``false``, and ``null`` instead of
        the Python standard ``True``, ``False``, and ``None``.
    #. The only available types are :class:`int`, :class:`float`,
       :class:`unicode`, :class:`bool`, :class:`list`, and :class:`dict`.
    #. Other JSON files can be imported using ``JSONFile(othersettings.json)``.

The pyFun control file, which by default is called :file:`pyFun.json` but can
also have other names, is split into several sections.  Most aspects of the
control file have defaults that will go into effect if the user does not
specify that option, but several entries are required.  The user can also
customize these defaults by editing the file
:file:`$CAPE/settings/pyOver.default.json`, where ``$CAPE`` is the path to
the Cape root directory.  Many of the settings are common to all solvers, and
their description can be found in the :ref:`Cape JSON section <cape-json>`.

The master settings file is loaded in one of two ways: a command-line call to
the script `pyover` or loading an instance of the
:class:`cape.pyover.cntl.Cntl` class. In both cases, the default name of the
file is :file:`pyOver.json`, but it is possible to use other file names. The
following two examples show the status of the run matrix; the first load the
inputs from :file:`pyOver.json`, and the second loads the inputs from
:file:`run/poweron.json`.

    .. code-block:: bash
    
        $ pyover -c
        $ pyover -f run/poweron.json -c
        
Within a Python script, the settings can be loaded with the following code.

    .. code-block:: python
    
        import cape.pyover
        
        # Loads pyOver.json
        c1 = pyOver.Cntl()
        # Loads run/poweron.json
        c2 = pyOver.Cntl('run/poweron.json')

The location from which either of these two methods is called (i.e., the current
working directory) is remembered as the root directory for the run.  Locations
of other files are relative to this directory.

The following is a nearly minimal pyOver control file that is presented to show
the major parts of the file.

    .. code-block:: javascript
    
        {
            // Stand-alone options
            "ShellCmds": [],
            
            // Settings for creating PBS scripts, if applicable
            "PBS": {},
            
            // Namelist inputs
            "Overflow": {},
            
            // Primary settings for running Cart3D
            "RunControl": {
                // Overall control of mode and number of iterations
                "InputSeq": [0, 1],
                "IterSeq": [0, 1500],
                "qsub": false
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
            "RunMatrix": {
                // This method points to a file that contains run conditions.
                "File": "RunMatrix.csv",
                // It is mandatory to define what the input variables are.
                "Keys": ["Mach", "alpha", "beta"],
                "GroupPrefix": "poweroff",
                "GroupMesh": false
            }
        }
        
        
For more information on these options see the following sections, and the
documentation for the module :mod:`cape.pyover.options` also provides descriptions
of how the options are used.


.. toctree::
    :maxdepth: 3
    
    uncategorized
    PBS
    slurm
    runControl
    Archive
    DataBook
    Report
    RunMatrix
    
