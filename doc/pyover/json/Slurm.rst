
.. _pyover-json-slurm:

*********************
Slurm Section Options
*********************
The options below are the available options in the Slurm Section of the ``pyover.json`` control file

..
    start-Slurm-n

*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Slurm nodes

..
    end-Slurm-n

..
    start-Slurm-p

*p*: {``'normal'``} | :class:`str`
    Slurm queue name

..
    end-Slurm-p

..
    start-Slurm-other

*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char

..
    end-Slurm-other

..
    start-Slurm-shell

*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name

..
    end-Slurm-shell

..
    start-Slurm-gid

*gid*: {``None``} | :class:`str`
    Slurm job group ID

..
    end-Slurm-gid

..
    start-Slurm-time

*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time

..
    end-Slurm-time

..
    start-Slurm-n

*n*: {``40``} | :class:`int` | :class:`int32` | :class:`int64`
    number of CPUs per node

..
    end-Slurm-n

..
    start-Slurm-a

*A*: {``None``} | :class:`str`
    Slurm job account name

..
    end-Slurm-a

