--------------------
"RunControl" section
--------------------

*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*verify*: {``None``} | :class:`object`
    value of option "verify"
*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*WarmStart*: ``True`` | {``False``}
    whether to warm start a case
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name


.. toctree::
    :maxdepth: 1

    Archive
    Environ
    aflr3
    intersect
    ulimit
    verify
