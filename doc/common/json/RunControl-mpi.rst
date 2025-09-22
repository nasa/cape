------------------------------------------------------------
``mpi``: options for MPI executable and command-line options
------------------------------------------------------------

**Option aliases:**

* *nhost* → *perhost*

**Recognized options:**

*args*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "args"
*executable*: {``'mpiexec'``} | :class:`str`
    executable to launch MPI
*flags*: {``None``} | :class:`dict`
    options to ``mpiexec`` using ``-flag val`` format
*hostfile*: {``None``} | :class:`str`
    add hostfile to ``mpiexec`` call
*np*: {``None``} | :class:`int` | :class:`float`
    explicit number (or fraction) of MPI processes
*perhost*: {``None``} | :class:`int`
    value of option "perhost"
*prefix*: {``None``} | :class:`str`
    preliminary executable run before MPI executable
*run*: {``None``} | ``True`` | ``False``
    whether to execute program

