---------------------------------
Options for ``PostSlurm`` section
---------------------------------

**Option aliases:**

* *account* -> *A*
* *begin* -> *b*
* *constraint* -> *C*
* *q* -> *p*
* *t* -> *time*

**Recognized options:**

*A*: {``None``} | :class:`str`
    Slurm job account name
*C*: {``None``} | :class:`str`
    constraint(s) on Slurm job
*N*: {``1``} | :class:`int`
    number of Slurm nodes
*b*: {``None``} | :class:`object`
    constraints on when to acllocate job
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

