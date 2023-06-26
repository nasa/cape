
.. _pyover-json-batchpbs:

************************
BatchPBS Section Options
************************
The options below are the available options in the BatchPBS Section of the ``pyover.json`` control file

..
    start-BatchPBS-q

*q*: {``'normal'``} | :class:`str`
    PBS queue name

..
    end-BatchPBS-q

..
    start-BatchPBS-p

*p*: {``None``} | :class:`object`
    PBS priority

..
    end-BatchPBS-p

..
    start-BatchPBS-select

*select*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of nodes

..
    end-BatchPBS-select

..
    start-BatchPBS-aoe

*aoe*: {``None``} | :class:`str`
    architecture operating environment

..
    end-BatchPBS-aoe

..
    start-BatchPBS-o

*o*: {``None``} | :class:`object`
    explicit STDOUT file name

..
    end-BatchPBS-o

..
    start-BatchPBS-model

*model*: {``'rom'``} | :class:`str`
    model type/architecture

..
    end-BatchPBS-model

..
    start-BatchPBS-walltime

*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time

..
    end-BatchPBS-walltime

..
    start-BatchPBS-a

*A*: {``None``} | :class:`object`
    account name(s) or number(s)

..
    end-BatchPBS-a

..
    start-BatchPBS-ncpus

*ncpus*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores (roughly CPUs) per node

..
    end-BatchPBS-ncpus

..
    start-BatchPBS-ompthreads

*ompthreads*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of OMP threads

..
    end-BatchPBS-ompthreads

..
    start-BatchPBS-j

*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting

..
    end-BatchPBS-j

..
    start-BatchPBS-r

*r*: {``'n'``} | ``'y'``
    rerun-able setting

..
    end-BatchPBS-r

..
    start-BatchPBS-s

*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job

..
    end-BatchPBS-s

..
    start-BatchPBS-e

*e*: {``None``} | :class:`object`
    explicit STDERR file name

..
    end-BatchPBS-e

..
    start-BatchPBS-mpiprocs

*mpiprocs*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of MPI processes per node

..
    end-BatchPBS-mpiprocs

..
    start-BatchPBS-w

*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group

..
    end-BatchPBS-w

