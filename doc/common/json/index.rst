-------------------
CAPE CFD{X} Options
-------------------

*PostPBS*: {``None``} | :class:`object`
    value of option "PostPBS"
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*BatchSlurm*: {``None``} | :class:`object`
    value of option "BatchSlurm"
*Mesh*: {``None``} | :class:`object`
    value of option "Mesh"
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*RunControl*: {``None``} | :class:`object`
    value of option "RunControl"
*PBS*: {``None``} | :class:`object`
    value of option "PBS"
*umask*: {``None``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`str`
    value of option "umask"
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*Report*: {``None``} | :class:`object`
    value of option "Report"
*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*Slurm*: {``None``} | :class:`object`
    value of option "Slurm"
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*PostSlurm*: {``None``} | :class:`object`
    value of option "PostSlurm"
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
*Config*: {``None``} | :class:`object`
    value of option "Config"
*BatchPBS*: {``None``} | :class:`object`
    value of option "BatchPBS"
*DataBook*: {``None``} | :class:`object`
    value of option "DataBook"
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
