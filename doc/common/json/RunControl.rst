----------------------------------
Options for ``RunControl`` section
----------------------------------

*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*WarmStart*: ``True`` | {``False``}
    whether to warm start a case
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*verify*: {``None``} | :class:`object`
    value of option "verify"

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunControl-Archive
    RunControl-Environ
    RunControl-aflr3
    RunControl-intersect
    RunControl-ulimit
    RunControl-verify
