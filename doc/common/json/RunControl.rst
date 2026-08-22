--------------
``RunControl``
--------------

**Option aliases:**

* *CAPEFile* ? *JSONFile*
* *ConcurrentPythonFuncs* ? *WorkerPythonFuncs*
* *ConcurrentShellCmds* ? *WorkerShellCmds*
* *Continue* ? *ContinuePhase*
* *GPU* ? *gpu*
* *NSubProc* ? *Maxforks*
* *NSubProcess* ? *MaxForks*
* *PostCmds* ? *PostShellCmds*
* *PostFuncs* ? *PostPythonFuncs*
* *PostPyFuncs* ? *PostPythonFuncs*
* *PreCmds* ? *PreShellCmds*
* *PreFuncs* ? *PrePythonFuncs*
* *PrePyFuncs* ? *PrePythonFuncs*
* *Resubmit* ? *ResubmitNextPhase*
* *WatcherCmds* ? *WorkerShellCmds*
* *WatcherFuncs* ? *WorkerPythonFuncs*
* *WatherTimeout* ? *WorkerTimeout*
* *WorkerCmds* ? *WorkerShellCmds*
* *WorkerFuncs* ? *WorkerPythonFuncs*
* *WorkerMaxWaitTime* ? *WorkerTimeout*
* *WorkerTimeOut* ? *WorkerTimeout*
* *nJob* ? *NJob*
* *sbatch* ? *slurm*

**Recognized options:**

*BatchSize*: {``1000``} | :class:`int`
    number of iters to store in single batch
*ContinuePhase*: {``None``} | ``True`` | ``False``
    whether restarts of same phase can use same job
*JSONFile*: {``None``} | :class:`str`
    name of JSON file from which settings originated
*MPI*: {``False``} | ``True``
    whether or not to run MPI in phase
*MaxForks*: {``4``} | :class:`int`
    maximum forks to use for parallel post-processing
*MaxWorkers*: {``8``} | :class:`int`
    maximum threads to use for parallel post-processing
*NJob*: {``0``} | :class:`int`
    number of jobs to run concurrently
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*PhaseWatcherCmds*: {``None``} | :class:`object`
    value of option "PhaseWatcherCmds"
*PhaseWatcherFuncs*: {``None``} | :class:`object`
    value of option "PhaseWatcherFuncs"
*PostPythonFuncs*: {``None``} | :class:`list`\ [:class:`dict` | :class:`str`]
    Python functions to run after each cycle
*PostShellCmds*: {``None``} | :class:`list`\ [:class:`dict` | :class:`str`]
    list of commands to run after each cycle
*PreMesh*: {``False``} | ``True``
    option to generate mesh before submitting job
*PrePythonFuncs*: {``None``} | :class:`list`\ [:class:`dict` | :class:`str`]
    Python functions to run before each cycle
*PreShellCmds*: {``None``} | :class:`list`\ [:class:`dict` | :class:`str`]
    list of commands to run before each cycle
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
*WarmStartFolder*: {``None``} | :class:`str`
    folder from which to get warm-start file
*WarmStartPhase*: {``None``} | :class:`int`
    phase from which to warm-start a case
*WorkerPythonFuncs*: {``None``} | :class:`list`\ [:class:`dict` | :class:`str`]
    functions to run concurrently during phase
*WorkerShellCmds*: {``None``} | :class:`list`\ [:class:`str`]
    shell commands to run concurrently during phase
*WorkerSleepTime*: {``10.0``} | :class:`int` | :class:`float`
    value of option "WorkerSleepTime"
*WorkerTimeout*: {``600.0``} | :class:`int` | :class:`float`
    max time to wait for workers after CFD phase
*ZombieFiles*: {``None``} | :class:`list`\ [:class:`str`]
    file name flobs to check mod time for zombie status
*ZombieTimeout*: {``45.0``} | :class:`int` | :class:`float`
    minutes to wait before considering a case a zombie
*gpu*: {``None``} | ``True`` | ``False``
    Whether to use GPU-version of code, if separate
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*nProc*: {``None``} | :class:`int` | :class:`float`
    number (or fraction) of cores to use (or omit if negative)
*qsub*: {``False``} | ``True``
    whether or not to submit jobs with PBS
*quickstart*: {``False``} | ``True``
    option to generate mesh before submitting job
*slurm*: {``False``} | ``True``
    whether or not to submit jobs with Slurm

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunControl-Archive
    RunControl-aflr3
    RunControl-comp2tri
    RunControl-intersect
    RunControl-mpi
    RunControl-ulimit
    RunControl-verify
