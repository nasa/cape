"""
Interface to FUN3D run control options
======================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Ipmort options-specific utilities
from util import rc0, odict

# Class for namelist settings
class RunControl(odict):
    """Dictionary-based interface for FUN3D run control"""
    
    # Run input sequence
    def get_InputSeq(self, i=None):
        """Return the input sequence for `flowCart`
        
        :Call:
            >>> InputSeq = opts.get_InputSeq(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *InputSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        return self.get_key('InputSeq', i)
        
    # Set run input sequence.
    def set_InputSeq(self, InputSeq=rc0('InputSeq'), i=None):
        """Set the input sequence for `flowCart`
        
        :Call:
            >>> opts.get_InputSeq(InputSeq, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *InputSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        self.set_key('InputSeq', InputSeq, i)
        
    
    # Get minimum cumulative iteration count
    def get_IterSeq(self, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *IterSeq[i]*.
        
        :Call:
            >>> IterSeq = opts.get_IterSeq(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *IterSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        return self.get_key('IterSeq', i)
        
    # Set minimum cumulative iteration count
    def set_IterSeq(self, IterSeq, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *IterSeq[i]*.
        
        :Call:
            >>> opts.get_IterSeq(IterSeq, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *IterSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        self.set_key('IterSeq', IterSeq, i)
        
    
    # Number of iterations
    def get_nSeq(self):
        """Return the number of input sets in the sequence
        
        :Call:
            >>> nSeq = opts.get_nSeq()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *nSeq*: :class:`int`
                Number of input sets in the sequence
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        # Get the input sequence.
        InputSeq = self.get_InputSeq()
        # Check if it's a list.
        if type(InputSeq).__name__ == "list":
            # Use the length.
            return len(InputSeq)
        else:
            # Something is messed up.
            return 1
            
    # Minimum required number of iterations
    def get_LastIter(self):
        """Return the minimum number of iterations for case to be done
        
        :Call:
            >>> nIter = opts.get_LastIter()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *nIter*: :class:`int`
                Number of required iterations for case
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        return self.get_IterSeq(self.get_nSeq())
        
    # Get MPI status
    def get_MPI(self, i):
        """Return whether or not to use MPI version
        
        :Call:
            >>> MPI = opts.get_mpi(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int`
                Run sequence index
        :Outputs:
            *MPI*: :class:`bool`
                Whether or not to use MPI
        :Versions:
            * 2015-10-17 ``@ddalle``: First version
        """
        return self.get_key('MPI', i)
        
    # Set MPI status
    def set_MPI(self, MPI=rc0('MPI'), i=None):
        """Set whether or not to use MPI version
        
        :Call:
            >>> q = opts.get_mpi(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int`
                Run sequence index
        :Outputs:
            *MPI*: :class:`bool`
                Whether or not to use MPI
        :Versions:
            * 2015-10-17 ``@ddalle``: First version
        """
        self.set_key('MPI', MPI, i)
        
    # Get the number of threads to use
    def get_nProc(self, i=None):
        """Return the number of threads to use
        
        :Call:
            >>> nProc = opts.get_nProc(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nProc*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
        :Versions:
            * 2014-08-02 ``@ddalle``: First version
            * 2014-10-02 ``@ddalle``: Switched to "nProc"
        """
        return self.get_key('nProc', i)
    
    # Set number of threads to use
    def set_nProc(self, nProc=rc0('nProc'), i=None):
        """Set the number of threads to use
        
        :Call:
            >>> opts.set_nProc(nProc, i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *nProc*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-02 ``@ddalle``: First version
            * 2014-10-02 ``@ddalle``: Switched to "nProc"
        """
        self.set_key('nProc', nProc, i)
        
    # Get the command name for "mpirun" or "mpiexec"
    def get_mpicmd(self, i=None):
        """Return either ``'mpirun'`` or ``'mpiexec``
        
        :Call:
            >>> mpicmd = opts.get_mpicmd(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mpicmd*: :class:`str`
                System command to call MPI
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        return self.get_key('mpicmd', i)
    
    # Set the command name for "mpirun" or "mpiexec"
    def set_mpicmd(self, mpicmd=rc0('mpicmd'), i=None):
        """Set minimum CFL number for `flowCart`
        
        :Call:
            >>> opts.set_mpicmd(mpicmd, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mpicmd*: :class:`str`
                System command to call MPI
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        self.set_key('mpicmd', mpicmd, i)
    
    # Get the submittable-job status
    def get_qsub(self, i=None):
        """Determine whether or not to submit jobs
        
        :Call:
            >>> qsub = opts.get_qsub(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *qsub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to submit case to PBS
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        return self.get_key('qsub', i)
    
    # Set the submittable-job status
    def set_qsub(self, qsub=rc0('qsub'), i=None):
        """Set jobs as submittable or nonsubmittable
        
        :Call:
            >>> opts.set_qsub(qsub, i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *qsub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to submit case to PBS
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        self.set_key('qsub', qsub, i)
        
    
    # Get the resubmittable-job status
    def get_resub(self, i=None):
        """Determine whether or not a job should restart or resubmit itself
        
        :Call:
            >>> resub = opts.get_resub(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *resub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to resubmit/restart a case
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        return self.get_key('resub', i)
    
    # Set the resubmittable-job status
    def set_resub(self, resub=rc0('resub'), i=None):
        """Set jobs as resubmittable or nonresubmittable
        
        :Call:
            >>> opts.set_resub(resub, i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *resub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to resubmit/restart a case
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        self.set_key('resub', resub, i)
    
