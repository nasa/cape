------------------------------
Options for ``ulimit`` section
------------------------------

**Option aliases:**

* *core_file_size* -> *c*
* *data_segment* -> *d*
* *file_locks* -> *x*
* *file_size* -> *f*
* *locked_memory* -> *l*
* *max_processes* -> *u*
* *message_queue_size* -> *q*
* *open_files* -> *n*
* *pending_signals* -> *i*
* *pipe_size* -> *p*
* *processes* -> *u*
* *real_time_priority* -> *r*
* *scheduling_priority* -> *e*
* *set_size* -> *m*
* *stack_size* -> *s*
* *time_limit* -> *t*
* *user_processes* -> *u*
* *virtual_memory* -> *v*

**Recognized options:**

*c*: {``0``} | :class:`object`
    core file size limit, ``ulimit -c``
*d*: {``'unlimited'``} | :class:`object`
    process data segment limit, ``ulimit -d``
*e*: {``0``} | :class:`object`
    max scheduling priority, ``ulimit -e``
*f*: {``'unlimited'``} | :class:`object`
    max size of files written by shell, ``ulimit -f``
*i*: {``127556``} | :class:`object`
    max number of pending signals, ``ulimit -i``
*l*: {``64``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``
*m*: {``'unlimited'``} | :class:`object`
    max resident set size, ``ulimit -m``
*n*: {``1024``} | :class:`object`
    max number of open files, ``ulimit -n``
*p*: {``8``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``
*q*: {``819200``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``
*r*: {``0``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``
*s*: {``4194304``} | :class:`object`
    stack size limit, ``ulimit -s``
*t*: {``'unlimited'``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``
*u*: {``127812``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``
*v*: {``'unlimited'``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``
*x*: {``'unlimited'``} | :class:`object`
    max number of file locks, ``ulimit -x``

