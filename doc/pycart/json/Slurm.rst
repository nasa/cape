
.. _pycart-json-slurm:

*****************************
Options for ``Slurm`` Section
*****************************
The options below are the available options in the ``Slurm`` Section of the ``pycart.json`` control file


*n*: {``40``} | :class:`int` | :class:`int32` | :class:`int64`
    number of CPUs per node



*shell*: {``'/bin/bash'``} | :class:`str`
    Slurm job shell name



*gid*: {``None``} | :class:`str`
    Slurm job group ID



*time*: {``'8:00:00'``} | :class:`str`
    max Slurm job wall time



*other*: {``None``} | :class:`dict`
    dict of additional Slurm options using ``=`` char



*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    number of Slurm nodes



*p*: {``'normal'``} | :class:`str`
    Slurm queue name



*A*: {``None``} | :class:`str`
    Slurm job account name


