
.. _pyfun-json-batchslurm:

**************************
BatchSlurm Section Options
**************************
The options below are the available options in the BatchSlurm Section of the ``pyfun.json`` control file

..
    start-BatchSlurm-shell

*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name

..
    end-BatchSlurm-shell

..
    start-BatchSlurm-n

*n*: {``40``} | :class:`int` | :class:`int32` | :class:`int64`
    number of CPUs per node

..
    end-BatchSlurm-n

..
    start-BatchSlurm-p

*p*: {``'normal'``} | :class:`str`
    Slurm queue name

..
    end-BatchSlurm-p

..
    start-BatchSlurm-other

*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char

..
    end-BatchSlurm-other

..
    start-BatchSlurm-time

*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time

..
    end-BatchSlurm-time

..
    start-BatchSlurm-n

*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Slurm nodes

..
    end-BatchSlurm-n

..
    start-BatchSlurm-gid

*gid*: {``None``} | :class:`str`
    Slurm job group ID

..
    end-BatchSlurm-gid

..
    start-BatchSlurm-a

*A*: {``None``} | :class:`str`
    Slurm job account name

..
    end-BatchSlurm-a

