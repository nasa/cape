-------------------------------------------
options for ``ulimit`` environment settings
-------------------------------------------

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

*c*: {``None``} | :class:`object`
    core file size limit, ``ulimit -c``
*d*: {``None``} | :class:`object`
    process data segment limit, ``ulimit -d``
*e*: {``None``} | :class:`object`
    max scheduling priority, ``ulimit -e``
*f*: {``None``} | :class:`object`
    max size of files written by shell, ``ulimit -f``
*i*: {``None``} | :class:`object`
    max number of pending signals, ``ulimit -i``
*l*: {``None``} | :class:`object`
    max size that may be locked into memory, ``ulimit -l``
*m*: {``None``} | :class:`object`
    max resident set size, ``ulimit -m``
*n*: {``None``} | :class:`object`
    max number of open files, ``ulimit -n``
*p*: {``None``} | :class:`object`
    pipe size in 512-byte blocks, ``ulimit -p``
*q*: {``None``} | :class:`object`
    max bytes in POSIX message queues, ``ulimit -q``
*r*: {``None``} | :class:`object`
    max real-time scheduling priority, ``ulimit -r``
*s*: {``None``} | :class:`object`
    stack size limit, ``ulimit -s``
*t*: {``None``} | :class:`object`
    max amount of cpu time in s, ``ulimit -t``
*u*: {``None``} | :class:`object`
    max number of procs avail to one user, ``ulimit -u``
*v*: {``None``} | :class:`object`
    max virtual memory avail to shell, ``ulimit -v``
*x*: {``None``} | :class:`object`
    max number of file locks, ``ulimit -x``

