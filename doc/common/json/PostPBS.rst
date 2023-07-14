-----------------
"PostPBS" section
-----------------

*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*aoe*: {``None``} | :class:`str`
    architecture operating environment
*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time
*p*: {``None``} | :class:`object`
    PBS priority
*model*: {``None``} | :class:`str`
    model type/architecture
*select*: {``1``} | :class:`int`
    number of nodes
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*e*: {``None``} | :class:`object`
    explicit STDERR file name

