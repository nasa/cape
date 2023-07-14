--------------------
"BatchSlurm" section
--------------------

*n*: {``40``} | :class:`int`
    number of CPUs per node
*N*: {``1``} | :class:`int`
    number of Slurm nodes
*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name
*A*: {``None``} | :class:`str`
    Slurm job account name
*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time
*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char
*gid*: {``None``} | :class:`str`
    Slurm job group ID
*p*: {``'normal'``} | :class:`str`
    Slurm queue name

