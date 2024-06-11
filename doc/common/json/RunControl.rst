----------------------------------
Options for ``RunControl`` section
----------------------------------

**Option aliases:**

* *CAPEFile* -> *JSONFile*
* *Continue* -> *ContinuePhase*
* *PostCmds* -> *PostShellCmds*
* *Resubmit* -> *ResubmitNextPhase*
* *nJob* -> *NJob*
* *sbatch* -> *slurm*

**Recognized options:**

*ContinuePhase*: {``True``} | ``False``
    whether restarts of same phase can use same job
*JSONFile*: {``None``} | :class:`str`
    name of JSON file from which settings originated
*MPI*: {``False``} | ``True``
    whether or not to run MPI in phase
*NJob*: {``0``} | :class:`int`
    number of jobs to run concurrently
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*PostShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    list of commands to run after each cycle
*PreMesh*: {``False``} | ``True``
    whether or not to generate volume mesh before submitting
*RestartSamePhase*: {``True``} | ``False``
    whether to restart same phase if needed
*ResubmitNextPhase*: {``False``} | ``True``
    whether new job submits at end of phase
*ResubmitSamePhase*: {``False``} | ``True``
    whether same-phase repeats need new job
*RootDir*: {``None``} | :class:`str`
    (absolute) base folder from which CAPE settings were read
*StartNextPhase*: {``True``} | ``False``
    whether to start next phase or stop CAPE
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
    number of cores/threads to use per case
*qsub*: {``False``} | ``True``
    whether or not to submit jobs with PBS
*slurm*: {``False``} | ``True``
    whether or not to submit jobs with Slurm

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunControl-Archive
    RunControl-aflr3
    RunControl-intersect
    RunControl-ulimit
    RunControl-verify
