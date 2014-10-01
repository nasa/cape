"""Interface for PBS script options: :mod:`pyCart.options.pbs`"""


# Import options-specific utilities
from util import rc0, odict

# Class for PBS settings
class PBS(odict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
    # Get number of nodes
    def get_PBS_select(self):
        """Return the number of nodes
        
        :Call:
            >>> n = opts.get_PBS_select()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                PBS number of nodes
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_select',0)
        
    # Set number of nodes
    def set_PBS_select(self, n):
        """Set PBS *join* setting
        
        :Call:
            opts.set_PBS_select(n)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *n*: :class:`int`
                PBS number of nodes
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_select', n)
    
    
    # Get number of CPUs per node
    def get_PBS_ncpus(self):
        """Return the number of CPUs per node
        
        :Call:
            >>> n = opts.get_PBS_ncpus()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                PBS number of CPUs per node
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_ncpus',0)
        
    # Set number of CPUs per node
    def set_PBS_ncpus(self, n):
        """Set PBS number of cpus per node
        
        :Call:
            opts.set_PBS_ncpus(n)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *n*: :class:`int`
                PBS number of CPUs per node
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_ncpus', n)
    
    
    # Get number of MPI processes per node
    def get_PBS_mpiprocs(self):
        """Return the number of CPUs per node
        
        :Call:
            >>> n = opts.get_PBS_ncpus()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                PBS number of MPI processes per node
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_mpiprocs',0)
        
    # Set number of MPI processes per node
    def set_PBS_mpiprocs(self, n):
        """Set PBS number of cpus per node
        
        :Call:
            opts.set_PBS_ncpus(n)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *n*: :class:`int`
                PBS number of MPI processes per node
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_mpiprocs', n)
    
    
    # Get PBS model architecture
    def get_PBS_model(self):
        """Return the PBS model/architecture
        
        :Call:
            >>> s = opts.get_PBS_model()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *s*: :class:`str`
                Name of architecture/model to use
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_model',0)
        
    # Set PBS model architecture
    def set_PBS_model(self, s):
        """Set the PBS model/architecture
        
        :Call:
            opts.set_PBS_model(s)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *s*: :class:`str`
                Name of architecture/model to use
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_model', s)
    
    
    # Get PBS walltime limit
    def get_PBS_walltime(self):
        """Return maximum wall time
        
        :Call:
            >>> t = opts.get_PBS_walltime()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_walltime',0)
        
    # Set PBS walltime limit
    def set_PBS_walltime(self, t):
        """Set the maximum wall time
        
        :Call:
            opts.set_PBS_model(s)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_walltime', t)
        
        
    # Get group setting
    def get_PBS_W(self):
        """Return PBS *W* setting, usually for setting groups
        
        :Call:
            >>> W = opts.get_PBS_W()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *W*: :class:`str`
                PBS *W* setting; usually ``'group_list=$grp'``
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_W',0)
        
    # Set group setting
    def set_PBS_W(self, W):
        """Set PBS *W* setting, usually for setting groups
        
        :Call:
            opts.set_PBS_W(W)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *W*: :class:`str`
                PBS *W* setting; usually ``'group_list=$grp'``
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_W', W)
        
        
    # Get queue setting
    def get_PBS_q(self):
        """Return PBS queue
        
        :Call:
            >>> q = opts.get_PBS_q()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *q*: :class:`str`
                Name of PBS queue to submit to
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_q',0)
        
    # Set queue
    def set_PBS_q(self, q):
        """Set PBS *W* setting, usually for setting groups
        
        :Call:
            opts.set_PBS_q(q)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *q*: :class:`str`
                Name of PBS queue to submit to
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_q', q)
        
        
    # Get "join" setting
    def get_PBS_j(self):
        """
        Return the PBS *j* setting, which can be used to *join* STDIO and
        STDERR into a single file.
        
        :Call:
            >>> j = opts.get_PBS_j()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *j*: :class:`str`
                PBS *j* setting; ``'oe'`` to join outputs
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_j',0)
        
    # Set "join" setting
    def set_PBS_j(self, j):
        """Set PBS *join* setting
        
        :Call:
            opts.set_PBS_j(j)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *j*: :class:`str`
                PBS *j* setting; ``'oe'`` to join outputs
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_j', j)
        
    
    # Get "join" setting
    def get_PBS_r(self):
        """Return the PBS *r* setting, which determines if a job can be rerun
        
        :Call:
            >>> r = opts.get_PBS_r()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *r*: :class:`str`
                PBS *r* setting; ``'n'`` for non-rerunable
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_r',0)
        
    # Set "join" setting
    def set_PBS_r(self, r):
        """Set PBS *rerun* setting
        
        :Call:
            opts.set_PBS_j(j)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *r*: :class:`str`
                PBS *r* setting; ``'n'`` for non-rerunable
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_r', r)
        
    
    # Get shell setting
    def get_PBS_S(self):
        """Return the PBS *S* setting, which determines the shell
        
        :Call:
            >>> r = opts.get_PBS_r()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *S*: :class:`str`
                PBS *shell* setting; ``'/bin/bash'`` in most cases
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('PBS_S',0)
        
    # Set shell setting
    def set_PBS_S(self, S):
        """Set the PBS *S* setting, which determines the shell
        
        :Call:
            opts.set_PBS_S(S)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *S*: :class:`str`
                PBS *shell* setting; ``'/bin/bash'`` in most cases
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
        """
        self.set_key('PBS_S', S)
        
