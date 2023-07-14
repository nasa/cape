--------------------
"RunControl" section
--------------------

*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*verify*: {``None``} | :class:`object`
    value of option "verify"
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*WarmStart*: ``True`` | {``False``}
    whether to warm start a case
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run


.. toctree::
    :maxdepth: 1

    Archive
    Environ
    aflr3
    intersect
    ulimit
    verify
