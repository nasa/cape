"""
Interface to FUN3D run control options
======================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Ipmort options-specific utilities
from util import rc0, odict, getel

# Environment class
class Environ(odict):
    """Class for environment variables"""
    
    # Get an environment variable by name
    def get_Environ(self, key, i=0):
        """Get an environment variable setting by name of environment variable
        
        :Call:
            >>> val = opts.get_Environ(key, i=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *key*: :class:`str`
                Name of the environment variable
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *val*: :class:`str`
                Value to set the environment variable to
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        # Check for the key.
        if key not in self:
            raise KeyError("Environment variable '%s' is not set in JSON file"
                % key)
        # Get the setting or list of settings
        V = self[key]
        # Select the value for run sequence *i*
        return str(getel(V, i))
        
    # Set an environment variable by name
    def set_Environ(self, key, val, i=None):
        """Set an environment variable setting by name of environment variable
        
        :Call:
            >>> val = opts.get_Environ(key, i=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *key*: :class:`str`
                Name of the environment variable
            *val*: :class:`str`
                Value to set the environment variable to
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        # Initialize the key if necessary.
        self.setdefault(key, "")
        # Set the value by run sequence.
        self[key] = setel(self[key], str(val), i)
# class Environ

# Resource limits class
class ulimit(odict):
    """Class for resource limits"""
    
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
                Value of the stack size limit
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
                Value of the stack size limit
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        self.set_ulimit('s', s, i)
        
    # Stack size
    def get_stack_size(self, i=0):
        """Get the stack size limit, ``ulimit -s``
        
        :Call:
            >>> s = opts.get_stack_size(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *s*: :class:`int`
                Value of the stack size limit
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        return self.get_s(i)
        
    # Stack size
    def set_stack_size(self, s, i=0):
        """Get the stack size limit, ``ulimit -s``
        
        :Call:
            >>> opts.set_stack_size(s, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *s*: :class:`int`
                Value of the stack size limit
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        self.set_s(i)
        
        
# class ulimit

# Class for flowCart settings
class Management(odict):
    """Dictionary-based interfaced for options specific to folder management"""
    
    
    # Archive base folder
    def get_ArchiveFolder(self):
        """Return the path to the archive
        
        :Call:
            >>> fdir = opts.get_ArchiveFolder()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fdir*: :class:`str`
                Path to root directory of archive
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveFolder')
        
    # Set archive base folder
    def set_ArchiveFolder(self, fdir=rc0('ArchiveFolder')):
        """Set the path to the archive
        
        :Call:
            >>> opts.set_ArchiveFolder(fdir)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fdir*: :class:`str`
                Path to root directory of archive
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveFolder', fdir)
        
    # Archive format
    def get_ArchiveFormat(self):
        """Return the format for folder archives
        
        :Call:
            >>> fmt = opts.get_ArchiveFormat()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveFormat')
        
    # Set archive format
    def set_ArchiveFormat(self, fmt=rc0('ArchiveFormat')):
        """Set the format for folder archives
        
        :Call:
            >>> opts.set_ArchiveFormat(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveFormat', fmt)
        
    # Get archive type
    def get_ArchiveType(self):
        """Return the archive type; determines what is deleted before archiving
        
        Note that all types except ``"full"`` will delete files from the working
        folder before archiving.  However, archiving actions, including the
        preliminary deletions, only take place if the case is marked *PASS*.
        
            * ``"full"``: Archives all contents of run directory
            * ``"best"``: Deletes all previous adaptation folders
            * ``"viz"``: Deletes all :file:`Mesh*.c3d` files and check files
            * ``"hist"``: Deletes all mesh, tri, and TecPlot files
        
        :Call:
            >>> atype = opts.get_ArchiveType()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {"full"} | "viz" | "best" | "hist"
                Name of archive type
        :Versions:
            * 2015-02-16 ``@ddalle``: First version
        """
        return self.get_key('ArchiveType')
        
    # Set archive type
    def set_ArchiveType(self, atype=rc0('ArchiveType')):
        """Set the archive type; determines what is deleted before archiving
        
        :Call:
            >>> opts.set_ArchiveType(atype)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *atype*: {"full"} | "viz" | "best" | "hist"
                Name of archive type
        :Versions:
            * 2015-02-16 ``@ddalle``: First version
        """
        self.set_key('ArchiveType', atype)
        
    # Get archiving action
    def get_ArchiveAction(self):
        """Return the action to take after finishing a case
        
        :Call:
            >>> fcmd = opts.get_ArchiveAction()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {""} | "skeleton" | "rm" | "archive"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveAction')
        
    # Set archiving action
    def set_ArchiveAction(self, fcmd=rc0('ArchiveAction')):
        """Set the action to take after finishing a case
        
        :Call:
            >>> opts.set_ArchiveAction(fcmd)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fcmd*: {""} | "skeleton" | "rm" | "archive"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveAction', fcmd)
        
    # Get archiving command
    def get_RemoteCopy(self):
        """Return the command used for remote copying
        
        :Call:
            >>> fcmd = opts.get_RemoteCopy()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {"scp"} | "rsync" | "shiftc --wait"
                Command to use for archiving
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('RemoteCopy')
        
    # Set archiving command
    def set_RemoteCopy(self, fcmd='scp'):
        """Set the command used for remote copying
        
        :Call:
            >>> opts.set_RemoteCopy(fcmd)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fcmd*: {"scp"} | "rsync" | "shiftc"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('RemoteCopy', fcmd)
        
    # Get the archive format for PBS output files
    def get_TarPBS(self):
        """Return the archive format for adapt folders
        
        :Call:
            >>> fmt = opts.get_TarPBS()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarPBS')
        
    # Set the archive format for visualization files
    def set_TarPBS(self, fmt=rc0('TarPBS')):
        """Set the archive format for adapt folders
        
        :Call:
            >>> opts.set_TarPBS(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarPBS', fmt)
# class Management


# Class for iteration & mode control settings and command-line inputs
class RunControl(odict):
    """Dictionary-based interface for generic code run control"""
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._Environ()
        self._ulimit()
        self._Management()
    
    # ===========
    # Environment
    # ===========
   # <
    
    # Environment variable interface
    def _Environ(self):
        """Initialize environment variables if necessary"""
        if 'Environ' not in self:
            # Empty/default
            self['Environ'] = Environ()
        elif type(self['Environ']).__name__ == 'dict':
            # Convert to special class
            self['Environ'] = Environ(**self['Environ'])
    
    # Get environment variable
    def get_Environ(self, key, i=0):
        self._Environ()
        return self['Environ'].get_Environ(key, i)
        
    # Set environment variable
    def set_Environ(self, key, val, i=None):
        self._Environ()
        self['Environ'].set_Environ(key, val, i)
        
    # Copy documentation
    for k in ['Environ']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Environ,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Environ,'set_'+k).__doc__
   # >
   
    # ===============
    # Resource Limits
    # ===============
   # <
   
    # Environment variable interface
    def _ulimit(self):
        """Initialize environment variables if necessary"""
        if 'ulimit' not in self:
            # Empty/default
            self['ulimit'] = ulimit()
        elif type(self['ulimit']).__name__ == 'dict':
            # Convert to special class
            self['ulimit'] = ulimit(**self['ulimit'])
    
    # Get resource limit variable
    def get_ulimit(self, u, i=0):
        self._ulimit()
        return self['ulimit'].get_ulimit(u, i)
        
    # Set resource limit variable
    def set_ulimit(self, u, l, i=None):
        self._ulimit()
        self['ulimit'].set_Environ(u, l, i)
        
    # Stack size
    def get_ulimit_s(self, i=0):
        self._ulimit()
        return self['ulimit'].get_s(i)
        
    # Stack size
    def set_ulimit_s(self, s=rc0('ulimit_s'), i=0):
        self._ulimit()
        self['ulimit'].set_s(s, i)
        
    # Stack size
    def get_stack_size(self, i=0):
        self._ulimit()
        return self['ulimit'].get_stack_size(i)
    
    # Stack size
    def set_stack_size(self, s=rc0('ulimit_s'), i=0):
        self._ulimit()
        self['ulimit'].set_stack_size(s, i)
        
    # Copy documentation
    for k in ['ulimit', 'stack_size']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(ulimit,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(ulimit,'set_'+k).__doc__
   # >
   
    # ==========
    # Management
    # ==========
   # <
    
    # Initialization method for folder management optoins
    def _Management(self):
        """Initialize folder management options if necessary"""
        # Check status.
        if 'Management' not in self:
            # Missing entirely.
            self['Management'] = Management()
        elif type(self['Management']).__name__ == 'dict':
            # Convert to special class
            self['Management'] = Management(**self['Management'])
      
    # Get the archive folder
    def get_ArchiveFolder(self):
        self._Management()
        return self['Management'].get_ArchiveFolder()
        
    # Set the archive folder
    def set_ArchiveFolder(self, fdir=rc0('ArchiveFolder')):
        self._Management()
        self['Management'].set_ArchiveFolder(fdir)
        
    # Get the archive format
    def get_ArchiveFormat(self):
        self._Management()
        return self['Management'].get_ArchiveFormat()
        
    # Set the archive format
    def set_ArchiveFormat(self, fmt=rc0('ArchiveFormat')):
        self._Management()
        self['Management'].set_ArchiveFormat(fmt)
        
    # Get the archive type
    def get_ArchiveType(self):
        self._Management()
        return self['Management'].get_ArchiveType()
        
    # Set the archive type
    def set_ArchiveType(self, atype=rc0('ArchiveType')):
        self._Management()
        self['Management'].set_ArchiveType(atype)
        
    # Get the archive action
    def get_ArchiveAction(self):
        self._Management()
        return self['Management'].get_ArchiveAction()
        
    # Set the archive action
    def set_ArchiveAction(self, fcmd=rc0('ArchiveAction')):
        self._Management()
        self['Management'].set_ArchiveAction(fcmd)
        
    # Get the remote copy command
    def get_RemoteCopy(self):
        self._Management()
        return self['Management'].get_RemoteCopy()
        
    # Set the remote copy command
    def set_RemoteCopy(self, fcmd=rc0('RemoteCopy')):
        self._Management()
        self['Management'].set_RemoteCopy(fcmd)
        
    # Copy over the documentation.
    for k in ['ArchiveFolder', 'ArchiveFormat', 'ArchiveAction', 'ArchiveType',
            'RemoteCopy']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Management,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Management,'set_'+k).__doc__      
   # >
    
    # =============== 
    # Local Functions
    # ===============
   # <
    # Number of iterations
    def get_nIter(self, i=None):
        """Return the number of iterations for run sequence *i*
        
        :Call:
            >>> nIter = opts.get_nIter(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *nIter*: :class:`int` or :class:`list` (:class:`int`)
                Number of iterations to run
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        return self.get_key('nIter', i)
        
    # Set number of iterations
    def set_nIter(self, nIter=rc0('nIter'), i=None):
        """Set the number of iterations for run sequence *i*
        
        :Call:
            >>> nIter = opts.get_nIter(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *nIter*: :class:`int` or :class:`list` (:class:`int`)
                Number of iterations to run
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        self.set_key('nIter', nIter, i)
    

    # Run input sequence
    def get_PhaseSequence(self, i=None):
        """Return the input sequence for `flowCart`
        
        :Call:
            >>> PhaseSeq = opts.get_PhaseSequence(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *PhaseSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
            * 2015-11-27 ``@ddalle``: InputSeq -> PhaseSeq
        """
        return self.get_key('PhaseSequence', i)
        
    # Set run input sequence.
    def set_PhaseSequence(self, PhaseSeq=rc0('PhaseSequence'), i=None):
        """Set the input sequence for `flowCart`
        
        :Call:
            >>> opts.get_PhaseSequence(PhaseSeq, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *PhaseSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        self.set_key('PhaseSequence', PhaseSeq, i)
        
    
    # Get minimum cumulative iteration count
    def get_PhaseIters(self, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *PhaseIters[i]*.
        
        :Call:
            >>> PhaseIters = opts.get_PhaseIters(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *PhaseIters*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        return self.get_key('PhaseIters', i)
        
    # Set minimum cumulative iteration count
    def set_PhaseIters(self, PhaseIters, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *PhaseIters[i]*.
        
        :Call:
            >>> opts.get_PhaseIters(PhaseIters, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *PhaseIters*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        self.set_key('PhaseIters', PhaseIters, i)
        
    
    # Number of phases
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
        PhaseSeq = self.get_PhaseSequence()
        # Check if it's a list.
        if type(PhaseSeq).__name__ == "list":
            # Use the length.
            return len(PhaseSeq)
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
        return self.get_PhaseIters(self.get_nSeq())
        
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
                Phase number
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
                Phase number
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
                Phase number
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
                Phase number
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
                Phase number
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
                Phase number
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        self.set_key('qsub', qsub, i)
        
    
    # Get the resubmittable-job status
    def get_Resubmit(self, i=None):
        """Determine whether or not a job should restart or resubmit itself
        
        :Call:
            >>> resub = opts.get_Resubmit(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *resub*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to resubmit/restart a case
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        return self.get_key('Resubmit', i)
    
    # Set the resubmittable-job status
    def set_Resubmit(self, resub=rc0('Resubmit'), i=None):
        """Set jobs as resubmittable or nonresubmittable
        
        :Call:
            >>> opts.set_Resubmit(resub, i)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *resub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to resubmit/restart a case
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2014-10-05 ``@ddalle``: First version
        """
        self.set_key('Resubmit', resub, i)
        
    # Get the continuance status
    def get_Continue(self, i=None):
        """Determine if restarts of the same run input should be resubmitted
        
        :Call:
            >> cont = opts.get_Continue(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *cont*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to continue restarts of same input sequence
                without resubmitting
        :Versions:
            * 2015-11-08 ``@ddalle``: First version
        """
        return self.get_key('Continue', i)
        
    # Set the continuance status
    def set_Continue(self, cont=rc0('Continue'), i=None):
        """Set the resubmit status for restarts of the same input sequence
        
        :Call:
            >> opts.set_Continue(, cont, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *cont*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to continue restarts of same input sequence
                without resubmitting
        :Versions:
            * 2015-11-08 ``@ddalle``: First version
        """
        self.set_key('Continue', cont, i)
   # >
   
# class RunControl
