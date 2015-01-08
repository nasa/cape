

------------------
PBS Script Options
------------------

The "PBS" section of :file:`pyCart.json` controls settings for PBS scripts when
Cart3D runs are submitted as PBS jobs to a high-performance computing
environment.  Regardless of whether or not the user intends to submit PBS jobs,
pyCart creates scripts called :file:`run_cart3d.pbs` or
:file:`run_cart3d.01.pbs`, :file:`run_cart3d.02.pbs`, etc.  At the top of these
files is a collection of PBS directives, which will be ignored if the script is
launched locally.

To run such a script locally without submitting it, call it with the appropriate
shell.

    .. code-block:: bash
    
        $ bash run_cart3d.pbs
        
If the job is to be submitted, simply submit it.  A special script called
`pqsub` is recommended over the usual `qsub` because it also creates a file
:file:`jobID.dat` that saves the PBS job number

    .. code-block:: bash
    
        $ pqsub run_cart3d.pbs
        

Introduction
============

The PBS options present in the :file:`pyCart.json` control file with default
values are shown below.

    .. code-block:: javascript
    
        "PBS": {
            "j": "oe",
            "r": "n",
            "S": "/bin/bash",
            "select": 1,
            "ncpus": 20,
            "mpiprocs": 20,
            "model": "ivy",
            "W": "",
            "q": "normal",
            "walltime": "2:00:00"
        }
        
These are standard PBS script option names that lead to lines being created at
the top of :file:`run_cart3d.pbs`, unless the option is empty.  For example, if
the PBS options above are in :file:`pyCart.json`, the first few lines of
:file:`run_cart3d.pbs` would be the following.

    .. code-block:: bash
    
        #!/bin/bash
        #PBS -S /bin/bash
        #PBS -N *casename*
        #PBS -r n
        #PBS -j oe
        #PBS -l select=1:ncpus=20:mpiprocs=20:model=ivy
        #PBS -l walltime=2:00:00
        #PBS -q normal

The case label, shown as ``*casename*`` above, is a short label shown as the job
name with `qstat` or similar commands.  The actual value of this label is
determined elsewhere and is related to name of the run directory.
        
The options have names that map directly to the PBS code, and notice that there
is no ``PBS -W`` line because that option is empty.  Because the user can
control the shell that is used with the *S* option, either `bash`, `sh`, `csh`,
or others are compatible with pyCart.

PBS Option Dictionary
=====================

This section provides a list of what the options mean and provides a
description of what values might be available.  The format of the list is the
name of the variable in italics followed by a list of possible values separated
by ``|`` characters.  The default value is surrounded in curly braces ``{}``.

Specific architectures, queue names, and potentially other options will vary
from one high-performance computing environment to another.  The values
suggested here may not be applicable to other systems.

    *j*: {``"oe"``} | ``""`` | :class:`str`
        This is the "join" option.  Setting the vallue to "oe" causes the PBS
        server to write standard output and standard error to a common output
        file.
        
    *S*: {``"/bin/bash"``} | ``"/bin/csh"`` | :class:`str`
        Specify the *full* path to the shell to use.
        
    *r*: {``'"n"``} | ``"r"`` | :class:`str` 
        PBS job rerunable status
        
    *select*: {``1``} | :class:`int`
        Number of *nodes* to request, i.e. the number of independent computing
        machines.
        
    *ncpus*: {``20``} | :class:`int`
        Number of CPUs *per* node
        
    *mpiprocs*: {``20``} | :class:`int`
        Number of MPI processes *per* node
        
    *model*: ``"has"`` | {``"ivy"``} | ``"san"`` | ``"wes"`` | :class:`str`
        Architecture, usually a three-letter code for a vendor-specific model.
        The options listed above are Intel's Haswell, Ivy Bridge, Sandy Bridge,
        and Westmere, which are common values.
        
    *W*: {``""``} | ``"group_list=e0847"`` | :class:`str`
        This option is used to specify the group with in which files should be
        created.  On some systems, this is also used to determine the group to
        which a case is charged.
        
    *walltime*: {``"2:00:00"``} | :class:`str`
        Maximum wall clock time to request
        
    *q*: {``"normal"``} | ``"devel"`` | ``"long"`` | :class:`str`
        The name of the queue to submit the job to.  This can be any queue that
        exists on the high-performance computing system in use.
        
