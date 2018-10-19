
.. _cape-json-slurm:

---------------------
SLURM Script Options
---------------------

The "Slurm" section of :file:`cape.json` controls settings for SLURM scripts when
Cart3D/OVERFLOW/etc. runs are submitted as SLURM jobs to a high-performance
computing environment.  Regardless of whether or not the user intends to submit
SLURM jobs, Cape creates scripts called :file:`run_cart3d.pbs`,
:file:`run_overflow.pbs`, or :file:`run_fun3d.pbs`.  The files are called 
:file:`run_cart3d.01.pbs`, :file:`run_cart3d.02.pbs`, etc. if there are
multiple phases.  At the top of these files is a collection of SLURM directives,
which will be ignored if the script is launched locally.

To run such a script locally without submitting it, call it with the
appropriate shell.

    .. code-block:: bash
    
        $ bash run_cart3d.pbs
        
If the job is to be submitted, simply submit it.  A special script called
`psbatch` is recommended over the usual `sbatch` because it also creates a file
:file:`jobID.dat` that saves the SLURM job number

    .. code-block:: bash
    
        $ psbatch run_cart3d.pbs
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-slurm>`
    * :ref:`FUN3D <pyfun-json-slurm>`
    * :ref:`OVERFLOW <pyover-json-slurm>`
        

Introduction
============

The SLURM options present in the :file:`cape.json` control file with default
values are shown below.

    .. code-block:: javascript
    
        "Slurm": {
            "N": 1,
            "n": 20,
            "mpiprocs": 20,
            "A": "account",
            "p": "normal",
            "gid": "e0847",
            "time": "2:00:00"
        },
        "BatchSlurm": { },
        "PostSlurm": { }
        
These are standard SLURM script option names that lead to lines being created at
the top of :file:`run_cart3d.pbs`, unless the option is empty.  For example, if
the SLURM options above are in :file:`pyCart.json`, the first few lines of
:file:`run_cart3d.pbs` would be the following.

    .. code-block:: bash
    
        #!/bin/bash
        #SBATCH -name *casename*
        #SBATCH -N 20
        #SBATCH -n 20
        #SBATCH -A account
        #SBATCH -gid e0847
        #SBATCH -time=2:00:00
        #SBATCH -p normal
        
The *BatchSlurm* dictionary provides options that override those of *Slurm* for
``pycart --batch`` commands. Options not specified in *BatchSlurm* fall back to
those defined in the first phase of *Slurm*. However, it is common for batch
jobs run on a single node. Here is a typical example in which run jobs utilize
10 Haswell nodes (``"has"``) for 12 hrs, but batch jobs select only one node
for two hours and submit instead to the ``devel`` queue.

    .. code-block:: javascript
    
        "Slurm": {
            "N": 1,
            "n": 20,
            "mpiprocs": 20,
            "A": "account",
            "p": "normal",
            "gid": "e0847",
            "time": "2:00:00"
        },
        "BatchSlurm": {
            "N": 1,
            "p": "devel",
            "time": "2:00:00"
        }

The case label, shown as ``*casename*`` above, is a short label shown as the
job name with ``squeue`` or similar commands. The actual value of this label is
determined elsewhere and is related to name of the run directory.

Each of these options may also be lists, in which case each list entry will
apply to one phase.  A common application for this in Cart3D is when the user
wants to run adaptively for a while (which can only be performed on a single
node) with further iterations using ``mpix_flowCart``.  Then the ``"N"``
option, which sets the number of nodes, becomes

    .. code-block:: javascript
    
        "N": [1, 10]

SLURM Option Dictionary
========================

This section provides a list of what the options mean and provides a
description of what values might be available.  The format of the list is the
name of the variable in italics followed by a list of possible values separated
by ``|`` characters.  The default value is surrounded in curly braces ``{}``.

Specific architectures, queue names, and potentially other options will vary
from one high-performance computing environment to another.  The values
suggested here may not be applicable to other systems.
        
    *N*: {``1``} | :class:`int`
        Number of *nodes* to request, i.e. the number of independent computing
        machines.
        
    *n*: {``20``} | :class:`int`
        Number of CPUs *per* node
        
    *A*: :class:`str`
        Account to charge for resources
        
    *gid*: :class:`str`
        Group ID
        
    *time*: {``"2:00:00"``} | :class:`str` | :class:`list` (:class:`str`)
        Maximum wall clock time to request
        
    *p*: {``"normal"``} | ``"devel"`` | ``"long"`` | :class:`str`
        The name of the queue or "partition" to submit the job to
        
