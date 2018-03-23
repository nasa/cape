"""
:mod:`cape.options.ulimit`: System resource options
====================================================

This module provides a class to access options relating to system resources.
Specifically, this pertains to options usually set by ``ulimit`` from the
command line.

The class provided in this module, :class:`cape.options.ulimit.ulimit`, is
loaded in the ``"RunControl"`` section of the JSON file and the
:class:`cape.options.runControl.RunControl` class.

"""

# Ipmort options-specific utilities
from util import rc0, odict, getel


# Resource limits class
class ulimit(odict):
    """Class for resource limits
    
    :Call:
        >>> opts = ulimit(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of system resource options
    :Outputs:
        *opts*: :class:`cape.options.ulimit.ulimit`
            System resource options interface
    :Versions:
        * 2015-11-10 ``@ddalle``: First version
    """
    
    # Get a ulimit setting
    def get_ulimit(self, u, i=0):
        """Get a resource limit (``ulimit``) setting by its command-line flag
        
        :Call:
            >>> l = opts.get_ulimit(u, i=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *u*: :class:`str`
                Name of the ``ulimit`` flag
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *l*: :class:`int`
                Value of the resource limit
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        # Check for setting
        if u not in self:
            # Default flag name
            rcu = 'ulimit_' + u
            # Check for default
            if rcu in rc0:
                # Use the default setting
                return rc0[rcu]
            else:
                # No setting found
                raise KeyError("Found no setting for 'ulimit -%s'" % u)
        # Process the setting
        V = self[u]
        # Select the value for run sequence *i*
        return getel(V, i)
        
    # Set a ulimit setting
    def set_ulimit(self, u, l=None, i=None):
        """Set a resource limit (``ulimit``) setting by its command-line flag
        
        :Call:
            >>> opts.set_ulimit(u, l=None, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *u*: :class:`str`
                Name of the ``ulimit`` flag
            *l*: :class:`int`
                Value of the limit
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        # Get default
        if l is None: udef = rc0["ulimit_%s"%u]
        # Initialize if necessary.
        self.setdefault(u, None)
        # Set the value.
        self[key] = setel(self[u], l, i)
        
    # Stack size
    def get_s(self, i=0):
        """Get the stack size limit, ``ulimit -s``
        
        :Call:
            >>> s = opts.get_s(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *s*: :class:`int`
                Value of the stack size limit (kbytes)
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        return self.get_ulimit('s', i)
        
    # Stack size
    def set_s(self, s, i=0):
        """Get the stack size limit, ``ulimit -s``
        
        :Call:
            >>> opts.set_s(s, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *s*: :class:`int`
                Value of the stack size limit (kbytes)
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        self.set_ulimit('s', s, i)
    
    # Aliases
    get_stack_size = get_s
    set_stack_size = set_s
    
    # Core file size
    def get_c(self, i=0):
        """Get the core file size limit, ``ulimit -c``
        
        :Call:
            >>> c = opts.get_c(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *c*: :class:`int`
                Value of the core file size limit
        :Versions:
            * 2016-03-13 ``@ddalle``: First version (blocks)
        """
        return self.get_ulimit('c', i)
        
    # Core file size
    def set_c(self, c, i=0):
        """Get the core file size limit, ``ulimit -c``
        
        :Call:
            >>> opts.set_c(c, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *c*: :class:`int`
                Value of the core file size limit (blocks)
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        self.set_ulimit('c', c, i)
        
    # Aliases
    get_core_file_size = get_c
    set_core_file_size = set_c
    
    # Data segment
    def get_d(self, i=0):
        """Get the data segment limit
        
        :Call:
            >>> d = opts.get_d(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *d*: :class:`int`
                Data segment limit (kbytes)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('d', i)
        
    # Data segment
    def set_d(self, d, i=0):
        """Set the data segment limit
        
        :Call:
            >>> opts.set_d(d, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *d*: :class:`int`
                Data segment limit (kbytes)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('d', d, i)
        
    # Aliases
    get_data_seg_limit = get_d
    set_data_seg_limit = set_d
    
    # Scheduling priority
    def get_e(self, i=0):
        """Get the scheduling priority
        
        :Call:
            >>> e = opts.get_e(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *e*: :class:`int`
                Scheduling priority
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('e', i)
        
    # Scheduling priority
    def set_e(self, e, i=0):
        """Set the data segment limit
        
        :Call:
            >>> opts.set_e(e, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *e*: :class:`int`
                Scheduling priority 
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('e', e, i)
        
    # Aliases
    get_scheduling_priority = get_e
    set_scheduling_priority = set_e
        
    # File size
    def get_f(self, i=0):
        """Get the file size limit, ``ulimit -f``
        
        :Call:
            >>> f = opts.get_f(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *f*: :class:`int`
                Value of the file size limit (blocks)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('f', i)
        
    # Core file size
    def set_f(self, f, i=0):
        """Get the file size limit, ``ulimit -f``
        
        :Call:
            >>> opts.set_f(f, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *f*: :class:`int`
                Value of the file size limit (blocks)
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        self.set_ulimit('f', f, i)
        
    # Aliases
    get_file_size = get_f
    set_file_size = set_f
    
    # Pending signals
    def get_i(self, i=0):
        """Get the pending signal limit
        
        :Call:
            >>> e = opts.get_i(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *e*: :class:`int`
                Pending signal limit 
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('e', i)
        
    # Pending signals
    def set_i(self, e, i=0):
        """Set the data segment limit
        
        :Call:
            >>> opts.set_i(e, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *e*: :class:`int`
                Pending signal limit
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('i', e, i)
        
    # Aliases
    get_pending_signal_limit = get_i
    set_pending_signal_limit = set_i
        
    # Locked memory
    def get_l(self, i=0):
        """Get the maximum locked memory
        
        :Call:
            >>> l = opts.get_l(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *l*: :class:`int`
                Maximum locked memory 
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('l', i)
        
    # Pending signals
    def set_l(self, l, i=0):
        """Set the maximum locked memory
        
        :Call:
            >>> opts.set_l(l, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *l*: :class:`int`
                Maximum locked memory
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('l', l, i)
        
    # Aliases
    get_max_locked_memory = get_l
    set_max_locked_memory = set_l
    
    # Max memory size
    def get_m(self, i=0):
        """Get the maximum memory size
        
        :Call:
            >>> m = opts.get_m(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *m*: :class:`int`
                Maximum locked memory (kbytes)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('l', i)
        
    # Maximum memory size
    def set_m(self, l, i=0):
        """Set the maximum memory size
        
        :Call:
            >>> opts.set_m(m, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *m*: :class:`int`
                Maximum locked memory (kbytes)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('l', l, i)
        
    # Aliases
    get_max_memory_size = get_m
    set_max_memory_size = set_m
    
    # Open files
    def get_n(self, i=0):
        """Get the maximum number of open files
        
        :Call:
            >>> n = opts.get_n(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *n*: :class:`int`
                Open file number limit 
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('n', i)
        
    # Open files
    def set_n(self, n, i=0):
        """Set the maximum number of open files
        
        :Call:
            >>> opts.set_n(n, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *n*: :class:`int`
                Open file number limit
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('n', n, i)
        
    # Aliases
    get_open_file_limit = get_n
    set_open_file_limit = set_n
    
    # Pipe size
    def get_p(self, i=0):
        """Get the pipe buffer limit
        
        :Call:
            >>> p = opts.get_p(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *p*: :class:`int`
                Pipe buffer size limit (512 bytes)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('p', i)
        
    # Pipe size
    def set_p(self, p, i=0):
        """Set the pipe buffer limit
        
        :Call:
            >>> opts.set_p(p, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *p*: :class:`int`
                Pipe buffer size limit (512 bytes)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('p', p, i)
        
    # Aliases
    get_pipe_size = get_p
    set_pipe_size = set_p
    
    # POSIX message queu
    def get_q(self, i=0):
        """Get the POSIX message queue limit
        
        :Call:
            >>> q = opts.get_q(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *q*: :class:`int`
                POSIX message queue limit (bytes)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('q', i)
        
    # Maximum memory size
    def set_q(self, q, i=0):
        """Set the maximum memory size
        
        :Call:
            >>> opts.set_q(q, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *q*: :class:`int`
                POSIX message queue limit (bytes)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('q', q, i)
        
    # Aliases
    get_messague_queues = get_q
    set_messague_queues = set_q
    
    # Real-time priority
    def get_r(self, i=0):
        """Get the real-time priority
        
        :Call:
            >>> r = opts.get_r(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *r*: :class:`int`
                Real-time priority 
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('r', i)
        
    # Real-time priority
    def set_r(self, r, i=0):
        """Set the maximum memory size
        
        :Call:
            >>> opts.set_r(r, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *r*: :class:`int`
                Real-time priority 
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('r', r, i)
        
    # Aliases
    get_real_time_priority = get_r
    set_real_time_priority = set_r
    
    # CPU time
    def get_t(self, i=0):
        """Get the maximum CPU time
        
        :Call:
            >>> n = opts.get_t(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *t*: :class:`int`
                CPU time limit (seconds)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('t', i)
        
    # Open files
    def set_t(self, t, i=0):
        """Set the maximum CPU time
        
        :Call:
            >>> opts.set_t(t, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *t*: :class:`int`
                CPU time limit (seconds)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('t', t, i)
        
    # Aliases
    get_time_limit = get_t
    set_time_limit = set_t
    
    # User processes
    def get_u(self, i=0):
        """Get the maximum number of user processes
        
        :Call:
            >>> u = opts.get_u(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *u*: :class:`int`
                Maximum number of user processes
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('u', i)
        
    # User processes
    def set_u(self, u, i=0):
        """Set the maximum number of user processes
        
        :Call:
            >>> opts.set_u(u, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *u*: :class:`int`
                Maximum number of user processes
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('u', u, i)
        
    # Aliases
    get_max_processes = get_u
    set_max_processes = set_u
        
    # Virtual memory
    def get_v(self, i=0):
        """Get the virtual memory limit
        
        :Call:
            >>> v = opts.get_v(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *v*: :class:`int`
                Virtual memory limit (kbytes)
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('v', i)
        
    # Virtual memory
    def set_v(self, v, i=0):
        """Set the virtual memory limit
        
        :Call:
            >>> opts.set_r(r, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *v*: :class:`int`
                Virtual memory limit (kbytes)
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('v', v, i)
        
    # Aliases
    get_virtual_memory_limit = get_v
    set_virtual_memory_limit = set_v
    
    # File locks
    def get_x(self, i=0):
        """Get the file lock limit
        
        :Call:
            >>> x = opts.get_x(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *x*: :class:`int`
                File lock limit
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.get_ulimit('v', i)
        
    # Virtual memory
    def set_x(self, x, i=0):
        """Set the file lock limit
        
        :Call:
            >>> opts.set_x(x, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *x*: :class:`int`
                File lock limit
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-03-13 ``@ddalle``: First version
        """
        return self.set_ulimit('x', x, i)
        
    # Aliases
    get_file_locks_limit = get_x
    set_file_locks_limit = set_x
    
    
# class ulimit

