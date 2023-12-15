.. _cape-json:

-----------------------
Library of JSON Options
-----------------------


The following JSON settings are available to the :mod:`cape` basis module and
apply to each of the solvers. Most of these settings will have the same
meaning for :mod:`cape.pycart`, :mod:`cape.pyfun`, and :mod:`cape.pyover`,
although some may be superseded by modified options in those derived modules.

The template name for the baseline control file is ``cape.json``. It can
be invoked via the command-line function ``cape`` via either of the following
two commands.

    .. code-block:: console

        $ cape
        $ cape -f cape.json

Similarly, the default file for ``pycart`` is ``pyCart.json``; the default
file for ``pyfun`` is ``pyFun.json``; and the default file for ``pyover``
is ``pyOver.json``.

A basic template for such files is shown below.

    .. code-block:: javascript

        {
            // Universal Python settings
            "PythonPath": [],
            "Modules": [],
            // Special Python functions to run using API
            "InitFunction": [],
            "CaseFunction": [],

            // Commands to run at beginning of run job
            "ShellCmds": [],
            // Commands to run at beginning of batch job
            "BatchShellCmds": [],

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
            // Batch job PBS settings different from *PBS*
            "BatchPBS": { },

            // Surface subset and point settings
            "Config": { },

            // Data book settings
            "DataBook": { },

            // Automated Report settings
            "Report": { },

            // Run matrix settings
            "RunMatrix": {
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
follow. The options listed on this page are top-level options of the main JSON
file that are not part of any section.
    

**Option aliases:**

* *Trajectory* -> *RunMatrix*
* *nSubmit* -> *NSubmit*

**Recognized options:**

*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*ZombieFiles*: {``['*.out']``} | :class:`list`\ [:class:`str`]
    file name flobs to check mod time for zombie status
*umask*: {``None``} | :class:`int` | :class:`str`
    value of option "umask"

**Subsections:**

.. toctree::
    :maxdepth: 1

    BatchPBS
    BatchSlurm
    Config
    DataBook
    Mesh
    PBS
    PostPBS
    PostSlurm
    Report
    RunControl
    RunMatrix
    Slurm
