--------------------
"RunControl" section
--------------------

*WarmStart*: ``True`` | {``False``}
    whether to warm start a case
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*verify*: {``None``} | :class:`object`
    value of option "verify"
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job


.. toctree::
    :maxdepth: 1

    Archive
    Environ
    aflr3
    intersect
    ulimit
    verify
