------------------
"BatchPBS" section
------------------

*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time
*e*: {``None``} | :class:`object`
    explicit STDERR file name
*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*select*: {``1``} | :class:`int`
    number of nodes
*model*: {``None``} | :class:`str`
    model type/architecture
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*p*: {``None``} | :class:`object`
    PBS priority
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*aoe*: {``None``} | :class:`str`
    architecture operating environment

