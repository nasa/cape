
.. _pycart-json-postpbs:

***********************
PostPBS Section Options
***********************
The options below are the available options in the PostPBS Section of the ``pycart.json`` control file

..
    start-PostPBS-walltime

*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time

..
    end-PostPBS-walltime

..
    start-PostPBS-q

*q*: {``'normal'``} | :class:`str`
    PBS queue name

..
    end-PostPBS-q

..
    start-PostPBS-r

*r*: {``'n'``} | ``'y'``
    rerun-able setting

..
    end-PostPBS-r

..
    start-PostPBS-select

*select*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of nodes

..
    end-PostPBS-select

..
    start-PostPBS-model

*model*: {``'rom'``} | :class:`str`
    model type/architecture

..
    end-PostPBS-model

..
    start-PostPBS-ncpus

*ncpus*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of cores (roughly CPUs) per node

..
    end-PostPBS-ncpus

..
    start-PostPBS-j

*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting

..
    end-PostPBS-j

..
    start-PostPBS-p

*p*: {``None``} | :class:`object`
    PBS priority

..
    end-PostPBS-p

..
    start-PostPBS-s

*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job

..
    end-PostPBS-s

..
    start-PostPBS-w

*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group

..
    end-PostPBS-w

..
    start-PostPBS-mpiprocs

*mpiprocs*: {``128``} | :class:`int` | :class:`int32` | :class:`int64`
    number of MPI processes per node

..
    end-PostPBS-mpiprocs

..
    start-PostPBS-e

*e*: {``None``} | :class:`object`
    explicit STDERR file name

..
    end-PostPBS-e

..
    start-PostPBS-a

*A*: {``None``} | :class:`object`
    account name(s) or number(s)

..
    end-PostPBS-a

..
    start-PostPBS-aoe

*aoe*: {``None``} | :class:`str`
    architecture operating environment

..
    end-PostPBS-aoe

..
    start-PostPBS-o

*o*: {``None``} | :class:`object`
    explicit STDOUT file name

..
    end-PostPBS-o

..
    start-PostPBS-ompthreads

*ompthreads*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    number of OMP threads

..
    end-PostPBS-ompthreads

