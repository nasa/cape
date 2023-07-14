--------------------
"RunControl" section
--------------------

*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*verify*: {``None``} | :class:`object`
    value of option "verify"
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*WarmStart*: ``True`` | {``False``}
    whether to warm start a case
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*


.. toctree::
    :maxdepth: 1

    Archive
    Environ
    aflr3
    intersect
    ulimit
    verify
