-------------------
CAPE CFD{X} Options
-------------------

*BatchShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    additional shell commands for batch jobs
*PostSlurm*: {``None``} | :class:`object`
    value of option "PostSlurm"
*RunControl*: {``None``} | :class:`object`
    value of option "RunControl"
*PBS_map*: {``None``} | :class:`dict`
    value of option "PBS_map"
*umask*: {``None``} | :class:`int` | :class:`int8` | :class:`int16` | :class:`int32` | :class:`int64` | :class:`uint8` | :class:`uint16` | :class:`uint32` | :class:`uint64` | :class:`str`
    value of option "umask"
*Report*: {``None``} | :class:`object`
    value of option "Report"
*BatchPBS*: {``None``} | :class:`object`
    value of option "BatchPBS"
*PostPBS*: {``None``} | :class:`object`
    value of option "PostPBS"
*PythonPath*: {``None``} | :class:`list`\ [:class:`str`]
    folder(s) to add to Python path for custom modules
*NSubmit*: {``10``} | :class:`object`
    maximum number of jobs to submit at one time
*Mesh*: {``None``} | :class:`object`
    value of option "Mesh"
*Config*: {``None``} | :class:`object`
    value of option "Config"
*ShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "ShellCmds"
*DataBook*: {``None``} | :class:`object`
    value of option "DataBook"
*PythonExec*: {``None``} | :class:`str`
    specific Python executable to use for jobs
*BatchSlurm*: {``None``} | :class:`object`
    value of option "BatchSlurm"
*Slurm*: {``None``} | :class:`object`
    value of option "Slurm"
*CaseFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to execute in case right before starting
*Modules*: {``None``} | :class:`list`\ [:class:`str`]
    list of Python modules to import
*RunMatrix*: {``None``} | :class:`dict`
    value of option "RunMatrix"
*PBS*: {``None``} | :class:`object`
    value of option "PBS"
*InitFunction*: {``None``} | :class:`list`\ [:class:`str`]
    function(s) to run immediately after parsing JSON
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
