
.. _pycart-json-slurm:

--------------------
SLURM Script Options
--------------------

The "Slurm" section of :file:`pyCart.json` controls settings for SLURM scripts
when Cart3D runs are submitted as SLURM jobs to a high-performance computing
environment. Regardless of whether or not the user intends to submit SLURM
jobs, pyCart creates scripts called :file:`run_cart3d.pbs` or
:file:`run_cart3d.01.pbs`, :file:`run_cart3d.02.pbs`, etc. At the top of these
files is a collection of SLURM directives, which will be ignored if the script
is launched locally.

To run such a script locally without submitting it, call it with the appropriate
shell.

    .. code-block:: bash
    
        $ bash run_cart3d.pbs
        
If the job is to be submitted, simply submit it.  A special script called
`psbatch` is recommended over the usual `sbatch` because it also creates a file
:file:`jobID.dat` that saves the SLURM job number

    .. code-block:: bash
    
        $ pqsub run_cart3d.pbs

The full description of the SLURM section can be found in the corresponding
:ref:`Cape SLURM section <cape-json-slurm>`.

