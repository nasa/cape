-------------------
CAPE CFD{X} Options
-------------------

*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*RunControl*: {``None``} | :class:`object`
    value of option "RunControl"
*Report*: {``None``} | :class:`object`
    value of option "Report"
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*PostSlurm*: {``None``} | :class:`object`
    value of option "PostSlurm"
*Slurm*: {``None``} | :class:`object`
    value of option "Slurm"
*PostPBS*: {``None``} | :class:`object`
    value of option "PostPBS"
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*PBS*: {``None``} | :class:`object`
    value of option "PBS"
*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*Mesh*: {``None``} | :class:`object`
    value of option "Mesh"
*umask*: {``None``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`str`
    value of option "umask"
*BatchPBS*: {``None``} | :class:`object`
    value of option "BatchPBS"
*DataBook*: {``None``} | :class:`object`
    value of option "DataBook"
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
*BatchSlurm*: {``None``} | :class:`object`
    value of option "BatchSlurm"
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*ZombieFiles*: {``['*.out']``} | :class:`list`\ [:class:`str`]
    file name flobs to check mod time for zombie status
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*Config*: {``None``} | :class:`object`
    value of option "Config"


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
