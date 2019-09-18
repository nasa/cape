"""
:mod:`cape.options.slurm`: SLURM script options
================================================

This portion of the options is universal, and so it is only encoded in the
:mod:`cape` module. The :mod:`cape.pycart` module, for example, does not have a
modified version. It contains options for specifying which architecture to use,
how many nodes to request, etc.

"""


# Import options-specific utilities
from .util import rc0, odict

# Class for Slurm settings
class Slurm(odict):
    """Dictionary-based options for Slurm jobs submitted via ``sbatch``
    
    :Call:
        >>> opts = Slurm(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of Slurm options
    :Outputs:
        *opts*: :class:`cape.options.pbs.Slurm`
            Slurm options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: First version
    """
    
    # Get the number of unique Slurm jobs.
    def get_nSlurm(self):
        """Return the maximum number of unique Slurm inputs
        
        For example, if a case is set up to be run in two parts, and the first
        phase needs only one node (*select=1*) while the second phase needs 10
        nodes (*select=10*), then the input file should have 
        ``"select": [1, 10]``, and the output of this function will be ``2``.
        
        :Call:
            >>> n = opts.get_nSlurm()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                Number of unique Slurm scripts
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        # Initialize the number of jobs.
        n = 1
        # Loop the keys.
        for v in self.values():
            # Test if it's a list.
            if type(v).__name__ in ['list', 'ndarray']:
                # Update the length.
                n = max(n, len(v))
        # Output
        return n
        
    
    # Get number of nodes
    def get_Slurm_N(self, i=None):
        """Return the number of nodes
        
        :Call:
            >>> n = opts.get_Slurm_N(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Index to select
        :Outputs:
            *N*: :class:`int`
                Slurm number of nodes
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_N', i)
        
    # Set number of nodes
    def set_Slurm_N(self, N=rc0('Slurm_N'), i=None):
        """Set Slurm number of nodes
        
        :Call:
            >>> opts.set_Slurm_N(N, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *N*: :class:`int`
                Slurm number of nodes
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_N', n, i)
    
    
    # Get number of CPUs per node
    def get_Slurm_n(self, i=None):
        """Return the number of CPUs per node
        
        :Call:
            >>> n = opts.get_Slurm_n(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *n*: :class:`int`
                Slurm number of CPUs per node
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_n', i)
        
    # Set number of CPUs per node
    def set_Slurm_n(self, n=rc0('Slurm_n'), i=None):
        """Set Slurm number of CPUs per node
        
        :Call:
            >>> opts.set_Slurm_n(n, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *n*: :class:`int`
                Slurm number of CPUs per node
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_n', n, i)
    
    # Get Slurm walltime limit
    def get_Slurm_time(self, i=None):
        """Return maximum wall time
        
        :Call:
            >>> t = opts.get_Slurm_time(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_time', i)
        
    # Set Slurm walltime limit
    def set_Slurm_time(self, t=rc0('Slurm_time'), i=None):
        """Set the maximum wall time
        
        :Call:
            >>> opts.set_Slurm_time(t, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_time', t, i)
        
        
    # Get group setting
    def get_Slurm_gid(self, i=None):
        """Return Slurm *gid* setting, usually for setting groups
        
        :Call:
            >>> gid = opts.get_Slurm_gid(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *gid*: :class:`str`
                Slurm *gid* setting
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_gid', i)
        
    # Set group setting
    def set_Slurm_gid(self, gid=rc0('Slurm_gid'), i=None):
        """Set Slurm *W* setting, usually for setting groups
        
        :Call:
            >>> opts.set_Slurm_gid(gid, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *gid*: :class:`str`
                Slurm *gid* setting
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_gid', gid, i)
        
        
    # Get account setting
    def get_Slurm_p(self, i=None):
        """Return Slurm queue
        
        :Call:
            >>> p = opts.get_Slurm_p()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *p*: :class:`str`
                Name of Slurm queue to submit to
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_p', i)
        
    # Set queue
    def set_Slurm_p(self, p=rc0('Slurm_p'), i=None):
        """Set Slurm queue
        
        :Call:
            >>> opts.set_Slurm_p(p, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *p*: :class:`str`
                Name of Slurm queue to submit to
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_p', p, i)
        
        
    # Get account setting
    def get_Slurm_A(self, i=None):
        """Return the Slurm *A* (account) setting
        
        :Call:
            >>> A = opts.get_Slurm_A(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *A*: :class:`str`
                Slurm account setting
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_A', i)
        
    # Set account setting
    def set_Slurm_A(self, A=rc0('Slurm_A'), i=None):
        """Set the Slurm *A* (account) setting
        
        :Call:
            >>> opts.set_Slurm_A(A, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *A*: :class:`str`
                Slurm account setting
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_A', A, i)
        
    
    # Get shell setting
    def get_Slurm_shell(self, i=None):
        """Return the Slurm *S* setting, which determines the shell
        
        :Call:
            >>> r = opts.get_Slurm_shell(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *S*: :class:`str`
                Slurm *shell* setting; ``'/bin/bash'`` in most cases
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        return self.get_key('Slurm_shell', i)
        
    # Set shell setting
    def set_Slurm_shell(self, S=rc0('Slurm_shell'), i=None):
        """Set the Slurm *S* setting, which determines the shell
        
        :Call:
            >>> opts.set_Slurm_shell(S, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *S*: :class:`str`
                Slurm *shell* setting; ``'/bin/bash'`` in most cases
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2018-10-10 ``@ddalle``: First version
        """
        self.set_key('Slurm_shell', S, i)
# class Slurm

