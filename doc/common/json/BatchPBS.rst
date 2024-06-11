--------------------------------
Options for ``BatchPBS`` section
--------------------------------

**Option aliases:**

* *site-needed* -> *site_needed*

**Recognized options:**

*A*: {``None``} | :class:`object`
    account name(s) or number(s)
*S*: {``'/bin/bash'``} | :class:`str`
    shell to execute PBS job
*W*: {``''``} | :class:`str`
    PBS *W* setting, usually for setting group
*aoe*: {``None``} | :class:`str`
    architecture operating environment
*e*: {``None``} | :class:`object`
    explicit STDERR file name
*j*: {``'oe'``} | :class:`str`
    PBS 'join' setting
*model*: {``None``} | :class:`str`
    model type/architecture
*mpiprocs*: {``None``} | :class:`int`
    number of MPI processes per node
*ncpus*: {``None``} | :class:`int`
    number of cores (roughly CPUs) per node
*o*: {``None``} | :class:`object`
    explicit STDOUT file name
*ompthreads*: {``None``} | :class:`int`
    number of OMP threads
*p*: {``None``} | :class:`object`
    PBS priority
*q*: {``'normal'``} | :class:`str`
    PBS queue name
*r*: {``'n'``} | ``'y'``
    rerun-able setting
*select*: {``1``} | :class:`int`
    number of nodes
*site_needed*: {``None``} | :class:`list`\ [:class:`str`]
    list of manually requested hard drives to mount
*walltime*: {``'8:00:00'``} | :class:`str`
    maximum job wall time

