------------
BatchPBSOpts
------------

*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*model*: {``None``} | :class:`str`
    model type/architecture
*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*e*: {``None``} | :class:`object`
    explicit STDERR file name
*select*: {``1``} | :class:`int`
    number of nodes
*aoe*: {``None``} | :class:`str`
    architecture operating environment
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*p*: {``None``} | :class:`object`
    PBS priority
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time

