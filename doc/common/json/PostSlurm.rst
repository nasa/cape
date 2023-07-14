-------------------
"PostSlurm" section
-------------------

*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time
*A*: {``None``} | :class:`str`
    Slurm job account name
*gid*: {``None``} | :class:`str`
    Slurm job group ID
*N*: {``1``} | :class:`int`
    number of Slurm nodes
*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name
*n*: {``40``} | :class:`int`
    number of CPUs per node
*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char
*p*: {``'normal'``} | :class:`str`
    Slurm queue name

