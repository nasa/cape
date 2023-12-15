----------------------------------
Options for ``RunControl`` section
----------------------------------

**Option aliases:**

* *PostCmds* -> *PostShellCmds*
* *sbatch* -> *slurm*

**Recognized options:**

*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*MPI*: {``False``} | ``True``
    whether or not to run MPI in phase
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*PostShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    list of commands to run after each cycle
*PreMesh*: {``False``} | ``True``
    whether or not to generate volume mesh before submitting
*Resubmit*: {``False``} | ``True``
    whether or not to submit new job at end of phase *j*
*Verbose*: {``False``} | ``True``
    "RunControl" verbosity flag
*WarmStart*: {``False``} | ``True``
    whether to warm start a case
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*slurm*: {``False``} | ``True``
    wheter or not to submit jobs with Slurm

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunControl-Archive
    RunControl-aflr3
    RunControl-intersect
    RunControl-ulimit
    RunControl-verify
