-----------------------------
Options for ``Slurm`` section
-----------------------------

**Option aliases:**

* *q* -> *p*
* *t* -> *time*

**Recognized options:**

*A*: {``None``} | :class:`str`
    Slurm job account name
*N*: {``1``} | :class:`int`
    number of Slurm nodes
*gid*: {``None``} | :class:`str`
    Slurm job group ID
*n*: {``40``} | :class:`int`
    number of CPUs per node
*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char
*p*: {``'normal'``} | :class:`str`
    Slurm queue name
*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name
*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time

