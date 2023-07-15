-------------------
CAPE CFD{X} Options
-------------------

*BatchPBS*: {``None``} | :class:`object`
    value of option "BatchPBS"
*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*BatchSlurm*: {``None``} | :class:`object`
    value of option "BatchSlurm"
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*Config*: {``None``} | :class:`object`
    value of option "Config"
*DataBook*: {``None``} | :class:`object`
    value of option "DataBook"
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
*Mesh*: {``None``} | :class:`object`
    value of option "Mesh"
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*PBS*: {``None``} | :class:`object`
    value of option "PBS"
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*PostPBS*: {``None``} | :class:`object`
    value of option "PostPBS"
*PostSlurm*: {``None``} | :class:`object`
    value of option "PostSlurm"
*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*Report*: {``None``} | :class:`object`
    value of option "Report"
*RunControl*: {``None``} | :class:`object`
    value of option "RunControl"
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*Slurm*: {``None``} | :class:`object`
    value of option "Slurm"
*ZombieFiles*: {``['*.out']``} | :class:`list`\ [:class:`str`]
    file name flobs to check mod time for zombie status
*umask*: {``None``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`str`
    value of option "umask"


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
