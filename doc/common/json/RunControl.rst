--------------
RunControlOpts
--------------

*qsub*: {``True``} | ``False``
    wheter or not to submit jobs with PBS
*slurm*: ``True`` | {``False``}
    wheter or not to submit jobs with Slurm
*verify*: {``None``} | :class:`object`
    value of option "verify"
*intersect*: {``None``} | :class:`object`
    value of option "intersect"
*Archive*: {``None``} | :class:`object`
    value of option "Archive"
*PhaseIters*: {``None``} | :class:`int`
    check-point iterations for phase *j*
*PreMesh*: ``True`` | {``False``}
    whether or not to generate volume mesh before submitting
*ulimit*: {``None``} | :class:`object`
    value of option "ulimit"
*Verbose*: ``True`` | {``False``}
    "RunControl" verbosity flag
*MPI*: ``True`` | {``False``}
    whether or not to run MPI in phase
*aflr3*: {``None``} | :class:`object`
    value of option "aflr3"
*WarmStartFolder*: {``None``} | :class:`object`
    folder from which to get warm-start file
*PhaseSequence*: {``None``} | :class:`int`
    list of phase indices to run
*Environ*: {``None``} | :class:`object`
    value of option "Environ"
*nIter*: {``None``} | :class:`int`
    number of iterations to run in phase *j*
*mpicmd*: ``'mpiexec'`` | ``'mpirun'``
    MPI executable name
*Continue*: {``True``} | ``False``
    whether restarts of same phase can use same job
*Resubmit*: ``True`` | {``False``}
    whether or not to submit new job at end of phase *j*
*nProc*: {``None``} | :class:`int`
    number of cores/threads to run
*WarmStart*: ``True`` | {``False``}
    whether to warm start a case


.. toctree::
    :maxdepth: 1

    Archive
    Environ
    aflr3
    intersect
    ulimit
    verify
