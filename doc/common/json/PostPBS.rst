-----------------
"PostPBS" section
-----------------

*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*aoe*: {``None``} | :class:`str`
    architecture operating environment
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*model*: {``None``} | :class:`str`
    model type/architecture
*e*: {``None``} | :class:`object`
    explicit STDERR file name
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time
*p*: {``None``} | :class:`object`
    PBS priority
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*select*: {``1``} | :class:`int`
    number of nodes

