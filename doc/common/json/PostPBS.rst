-----------------
"PostPBS" section
-----------------

*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time
*e*: {``None``} | :class:`object`
    explicit STDERR file name
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*aoe*: {``None``} | :class:`str`
    architecture operating environment
*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*p*: {``None``} | :class:`object`
    PBS priority
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*select*: {``1``} | :class:`int`
    number of nodes
*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*model*: {``None``} | :class:`str`
    model type/architecture

