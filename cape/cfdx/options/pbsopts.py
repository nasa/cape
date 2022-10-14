r"""
:mod:`cape.cfdx.options.pbsopts`: PBS script options
=====================================================

This portion of the options is universal, and so it is only encoded in
the :mod:`cape` module. The :mod:`cape.pycart` module, for example, does
not have a modified version. It contains options for specifying which
architecture to use, how many nodes to request, etc.

"""


# Import options-specific utilities
from .util import OptionsDict, ARRAY_TYPES, INT_TYPES


# Class for PBS settings
class PBSOpts(OptionsDict):
    r"""Options class for PBS jobs
    
    :Call:
        >>> opts = PBS(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of PBS options
    :Outputs:
        *opts*: :class:`cape.options.pbs.PBS`
            PBS options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: Version 1.0
        * 2022-10-13 ``@ddalle``: Version 2.0; :class:`OptionsDict`
    """
   # --- Class attributes ---
    # No extra attributes
    __slots__ = tuple()

    # Allowed parameters
    _optlist = {
        "S",
        "W",
        "aoe",
        "e",
        "j",
        "model",
        "mpiprocs",
        "o",
        "ompthreads",
        "q",
        "r",
        "select",
        "walltime"
    }

    # Types
    _opttypes = {
        "S": str,
        "W": str,
        "j": str,
        "r": str,
        "aoe": str,
        "model": str,
        "mpiprocs": INT_TYPES,
        "ncpus": INT_TYPES,
        "ompthreads": INT_TYPES,
        "q": str,
        "select": INT_TYPES,
        "walltime": str,
    }

    # Defaults
    _rc = {
        "S": "/bin/bash",
        "W": "",
        "j": "oe",
        "model": "rom",
        "mpiprocs": 128,
        "ncpus": 128,
        "q": "normal",
        "r": "n",
        "select": 1,
        "walltime": "8:00:00",
    }

    # Descriptions
    _rst_descriptions = {
        "aoe": "architecture operating environment",
        "model": "model type/architecture",
        "mpiprocs": "number of MPI processes per node",
        "ncpus": "number of cores (roughly CPUs) per node",
        "ompthreads": "number of OMP threads",
        "p": "PBS priority",
        "select": "number of nodes",
    }
    
    # Get the number of unique PBS jobs.
    def get_nPBS(self) -> int:
        r"""Return the maximum number of unique PBS inputs
        
        For example, if a case is set up to be run in two parts, and the
        first phase needs only one node (*select=1*) while the second
        phase needs 10 nodes (*select=10*), then the input file should
        have ``"select": [1, 10]``, and the output of this function will
        be ``2``.
        
        :Call:
            >>> n = opts.get_nPBS()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                Number of unique PBS scripts
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        # Initialize the number of jobs.
        n = 1
        # Loop the keys.
        for v in self.values():
            # Test if it's a list
            if isinstance(v, ARRAY_TYPES):
                # Update the length.
                n = max(n, len(v))
        # Output
        return n
    
    
    # Get PBS walltime limit
    def get_PBS_walltime(self, i=None):
        """Return maximum wall time
        
        :Call:
            >>> t = opts.get_PBS_walltime(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_walltime', i)
        
    # Set PBS walltime limit
    def set_PBS_walltime(self, t=rc0('PBS_walltime'), i=None):
        """Set the maximum wall time
        
        :Call:
            >>> opts.set_PBS_walltime(t, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *t*: :class:`str`
                Maximum wall clock time (e.g. ``'8:00:00'``)
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_walltime', t, i)
        
        
    # Get group setting
    def get_PBS_W(self, i=None):
        """Return PBS *W* setting, usually for setting groups
        
        :Call:
            >>> W = opts.get_PBS_W(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *W*: :class:`str`
                PBS *W* setting; usually ``'group_list=$grp'``
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_W', i)
        
    # Set group setting
    def set_PBS_W(self, W=rc0('PBS_W'), i=None):
        """Set PBS *W* setting, usually for setting groups
        
        :Call:
            >>> opts.set_PBS_W(W, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *W*: :class:`str`
                PBS *W* setting; usually ``'group_list=$grp'``
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_W', W, i)
        
        
    # Get queue setting
    def get_PBS_q(self, i=None):
        """Return PBS queue
        
        :Call:
            >>> q = opts.get_PBS_q()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *q*: :class:`str`
                Name of PBS queue to submit to
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_q', i)
        
    # Set queue
    def set_PBS_q(self, q=rc0('PBS_q'), i=None):
        """Set PBS *W* setting, usually for setting groups
        
        :Call:
            >>> opts.set_PBS_q(q, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *q*: :class:`str`
                Name of PBS queue to submit to
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_q', q, i)
        
        
    # Get "join" setting
    def get_PBS_j(self, i=None):
        """
        Return the PBS *j* setting, which can be used to *join* STDIO and
        STDERR into a single file.
        
        :Call:
            >>> j = opts.get_PBS_j(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *j*: :class:`str`
                PBS *j* setting; ``'oe'`` to join outputs
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_j', i)
        
    # Set "join" setting
    def set_PBS_j(self, j=rc0('PBS_j'), i=None):
        """Set PBS *join* setting
        
        :Call:
            >>> opts.set_PBS_j(j, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`str`
                PBS *j* setting; ``'oe'`` to join outputs
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_j', j, i)
        
        
    # Get "stdout" setting
    def get_PBS_o(self, i=None):
        """Return the explicit STDOUT file
        
        :Call:
            >>> o = opts.get_PBS_o(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *o*: :class:`str`
                PBS *o* setting; explicit STDOUT file
        :Versions:
            * 2017-01-11 ``@ddalle``: First version
        """
        return self.get_key('PBS_o', i)
        
    # Set "stdout" setting
    def set_PBS_o(self, o=rc0('PBS_o'), i=None):
        """Set PBS file for STDOUT
        
        :Call:
            >>> opts.set_PBS_o(o, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *o*: :class:`str`
                PBS *o* setting; explicit STDOUT file
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2017-01-11 ``@ddalle``: First version
        """
        self.set_key('PBS_o', o, i)
        
        
    # Get "stderr" setting
    def get_PBS_e(self, i=None):
        """Return the explicit STDERR file
        
        :Call:
            >>> e = opts.get_PBS_o(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *e*: :class:`str`
                PBS *e* setting; explicit STDERR file
        :Versions:
            * 2017-01-11 ``@ddalle``: First version
        """
        return self.get_key('PBS_e', i)
        
    # Set "stderr" setting
    def set_PBS_e(self, e=rc0('PBS_e'), i=None):
        """Set PBS file for STDERR
        
        :Call:
            >>> opts.set_PBS_e(e, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *e*: :class:`str`
                PBS *o* setting; explicit STDERR file
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2017-01-11 ``@ddalle``: First version
        """
        self.set_key('PBS_e', e, i)
        
    
    # Get "rerun" setting
    def get_PBS_r(self, i=None):
        """Return the PBS *r* setting, which determines if a job can be rerun
        
        :Call:
            >>> r = opts.get_PBS_r(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *r*: :class:`str`
                PBS *r* setting; ``'n'`` for non-rerunable
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_r', i)
        
    # Set "join" setting
    def set_PBS_r(self, r=rc0('PBS_r'), i=None):
        """Set PBS *rerun* setting
        
        :Call:
            >>> opts.set_PBS_r(r, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *r*: :class:`str`
                PBS *r* setting; ``'n'`` for non-rerunable
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_r', r, i)
        
    
    # Get shell setting
    def get_PBS_S(self, i=None):
        """Return the PBS *S* setting, which determines the shell
        
        :Call:
            >>> r = opts.get_PBS_S(i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Index to select
        :Outputs:
            *S*: :class:`str`
                PBS *shell* setting; ``'/bin/bash'`` in most cases
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        return self.get_key('PBS_S', i)
        
    # Set shell setting
    def set_PBS_S(self, S=rc0('PBS_S'), i=None):
        """Set the PBS *S* setting, which determines the shell
        
        :Call:
            >>> opts.set_PBS_S(S, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *S*: :class:`str`
                PBS *shell* setting; ``'/bin/bash'`` in most cases
            *i*: :class:`int` or ``None``
                Index to select
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
            * 2014-12-01 ``@ddalle``: Added index
        """
        self.set_key('PBS_S', S, i)

