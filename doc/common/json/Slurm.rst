---------------
"Slurm" section
---------------

*gid*: {``None``} | :class:`str`
    Slurm job group ID
*N*: {``1``} | :class:`int`
    number of Slurm nodes
*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char
*n*: {``40``} | :class:`int`
    number of CPUs per node
*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name
*p*: {``'normal'``} | :class:`str`
    Slurm queue name
*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time
*A*: {``None``} | :class:`str`
    Slurm job account name

