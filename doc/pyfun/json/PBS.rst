
.. _pyfun-json-pbs:

***************************
Options for ``PBS`` Section
***************************
The options below are the available options in the ``PBS`` Section of the ``pyfun.json`` control file


*r*: {``'n'``} | ``'y'``
    rerun-able setting



*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group



*ncpus*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores (roughly CPUs) per node



*q*: {``'normal'``} | :class:`str`
    PBS queue name



*model*: {``'rom'``} | :class:`str`
    model type/architecture



*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job



*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time



*aoe*: {``None``} | :class:`str`
    architecture operating environment



*p*: {``None``} | :class:`object`
    PBS priority



*A*: {``None``} | :class:`object`
    account name(s) or number(s)



*mpiprocs*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of MPI processes per node



*o*: {``None``} | :class:`object`
    explicit STDOUT file name



*select*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of nodes



*ompthreads*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of OMP threads



*e*: {``None``} | :class:`object`
    explicit STDERR file name



*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting


