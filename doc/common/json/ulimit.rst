----------------
"ulimit" section
----------------

*c*: {``0``} | :class:`object`
    core file size limit, ``ulimit -c``
*e*: {``0``} | :class:`object`
    max scheduling priority, ``ulimit -e``
*t*: {``'unlimited'``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``
*f*: {``'unlimited'``} | :class:`object`
    max size of files written by shell, ``ulimit -f``
*l*: {``64``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``
*i*: {``127556``} | :class:`object`
    max number of pending signals, ``ulimit -i``
*s*: {``4194304``} | :class:`object`
    stack size limit, ``ulimit -s``
*m*: {``'unlimited'``} | :class:`object`
    max resident set size, ``ulimit -m``
*v*: {``'unlimited'``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``
*r*: {``0``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``
*u*: {``127812``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``
*n*: {``1024``} | :class:`object`
    max number of open files, ``ulimit -n``
*p*: {``8``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``
*x*: {``'unlimited'``} | :class:`object`
    max number of file locks, ``ulimit -x``
*d*: {``'unlimited'``} | :class:`object`
    process data segment limit, ``ulimit -d``
*q*: {``819200``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``

