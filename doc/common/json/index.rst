.. _cape-json:

-------------------
CAPE CFD{X} Options
-------------------

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
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
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
    Slurm
