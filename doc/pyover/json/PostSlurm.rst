
.. _pyover-json-postslurm:

*************************
PostSlurm Section Options
*************************
The options below are the available options in the PostSlurm Section of the ``pyover.json`` control file

..
    start-PostSlurm-n

*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Slurm nodes

..
    end-PostSlurm-n

..
    start-PostSlurm-p

*p*: {``'normal'``} | :class:`str`
    Slurm queue name

..
    end-PostSlurm-p

..
    start-PostSlurm-other

*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char

..
    end-PostSlurm-other

..
    start-PostSlurm-shell

*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name

..
    end-PostSlurm-shell

..
    start-PostSlurm-gid

*gid*: {``None``} | :class:`str`
    Slurm job group ID

..
    end-PostSlurm-gid

..
    start-PostSlurm-time

*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time

..
    end-PostSlurm-time

..
    start-PostSlurm-n

*n*: {``40``} | :class:`int` | :class:`int32` | :class:`int64`
    number of CPUs per node

..
    end-PostSlurm-n

..
    start-PostSlurm-a

*A*: {``None``} | :class:`str`
    Slurm job account name

..
    end-PostSlurm-a

