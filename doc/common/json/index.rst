-------------------
CAPE CFD{X} Options
-------------------

*BatchPBS*: {``None``} | :class:`object`
    value of option "BatchPBS"
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
*Slurm*: {``None``} | :class:`object`
    value of option "Slurm"
*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*RunControl*: {``None``} | :class:`object`
    value of option "RunControl"
*PostPBS*: {``None``} | :class:`object`
    value of option "PostPBS"
*BatchSlurm*: {``None``} | :class:`object`
    value of option "BatchSlurm"
*Config*: {``None``} | :class:`object`
    value of option "Config"
*Mesh*: {``None``} | :class:`object`
    value of option "Mesh"
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*Report*: {``None``} | :class:`object`
    value of option "Report"
*umask*: {``None``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`str`
    value of option "umask"
*PBS*: {``None``} | :class:`object`
    value of option "PBS"
*DataBook*: {``None``} | :class:`object`
    value of option "DataBook"
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
*PostSlurm*: {``None``} | :class:`object`
    value of option "PostSlurm"
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*ZombieFiles*: {``['*.out']``} | :class:`list`\ [:class:`str`]
    file name flobs to check mod time for zombie status


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
