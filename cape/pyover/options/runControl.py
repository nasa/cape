"""
:mod:`cape.pyover.options.runControl.RunControl`: Run control options
======================================================================

Options interface for aspects of running a case of OVERFLOW.  The settings are
read from the ``"RunControl"`` of a JSON file, and the contents of this section
are written to :file:`case.json` within each run folder.

The methods of :class:`cape.options.runControl.RunControl` are also present.
These control options such as whether to submit as a PBS job, whether or not to
use MPI, etc.

This contains options that determine how long the solver is ran (primarily 
via the ``"PhaseSequence"`` and ``"PhaseIters"`` options), what basic mode it
is run in (such as a submitted or local job or serial or MPI job), and options
for command-line options to the OVERFLOW binaries.  There is also an
``"Archive"`` section that can be used for copying files and cleaning up after
one or more cases have been completed.

This module primarily provides a class :class:`pyOver.options.RunControl`. Many
of the options that are common to all solvers are inherited from
:class:`cape.options.runControl.RunControl`. This class also has an interface
for environment variables and ``ulimit`` parameters.

In particular, all of the commands available to the classes listed below are
also available to :class:`pyOver.options.runControl.RunControl`.

:Classes:
    * :class:`pyOver.options.runControl.RunControl`
    * :class:`pyOver.options.runControl.overrun`
    * :class:`pyOver.options.Archive.Archive`
    * :class:`cape.options.runControl.Environ`
    * :class:`cape.options.ulimit.ulimit`
:See Also:
    * :mod:`cape.options.runControl`
    * :mod:`cape.options.ulimit`
    * :mod:`cape.pyover.options.Archive`
"""

# Import options-specific utilities
from .util import rc0, getel, odict

# Import template module
import cape.options.runControl
# Submodules
from .Archive import Archive

# Class for OVERFLOW command-line interface
class overrun(odict):
    """Class for ``overrun`` command-line options"""
    
    # Extra inputs
    def get_overrun_aux(self, i=None):
        """Get auxiliary command-line inputs to ``overrun``
        
        :Call:
            >>> aux = opts.get_overrun_aux(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *aux*: :class:`str`
                String to be given with ``-aux`` flag
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_key('aux', i, rck='overrun_aux')

    # Get dictionary of options
    def get_overrun_kw(self, i=None):
        """Get other inputs to ``overrun``

        :Call:
            >>> kw = opts.get_overrun_kw(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of additional arguments to ``overrun``
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Intiialize output
        kw = {}
        # Loop through output
        for k in self:
            # Check for named keys
            if k in ['aux', 'args', 'cmd']: continue
            # Otherwise, append the key, but select the phase
            kw[k] = self.get_key(k, i)
        # Output
        return kw
    
    # Function to get OVERFLOW command
    def get_overrun_cmd(self, i=None):
        """Get the name of the OVERFLOW binary to run
        
        :Call:
            >>> fcmd = opts.get_overrun_cmd(i)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fcmd*: :class:`str`
                Name of the command-line function to use
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_key('cmd', i, rck='overrun_cmd')

    # Function to get number of threads
    def get_overrun_nthreads(self, i=None):
        """Get number of OpenMP threads for ``overrunmpi``

        :Call:
            >>> nt = opts.get_overrun_nthreads(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *nt*: {``None``} | :class:`int` > 0
                Number of OpenMP threads
        :Versions:
            * 2017-04-27 ``@ddalle``: First version
        """
        return self.get_key('nthreads', i)

        
    # Function to get extra OVERFLOW arguments
    def get_overrun_args(self, i=None):
        """Get extra arguments to *overrun_cmd*
        
        :Call:
            >>> fargs = opts.get_overrun_args(i)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fargs*: :class:`str`
                Extra command-line arguments/flags
        :Versions:
            * 2016-02-02 ``@ddalle``: First version
        """
        return self.get_key('args', i, rck='overrun_args')
# class overrun

# Class for Report settings
class RunControl(cape.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = RunControl(**kw)
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._Environ()
        self._ulimit()
        self._Archive()
        self._overrun()
    
   # ============ 
   # Initializers
   # ============
   # <
   
    # Initialization and confirmation for nodet options
    def _overrun(self):
        """Initialize `overrun` options if necessary"""
        if 'overrun' not in self:
            # Empty/default
            self['overrun'] = overrun()
        elif type(self['overrun']).__name__ == 'dict':
            # Convert to special class
            self['overrun'] = overrun(**self['overrun'])
            
   # >
    
   # ===============
   # OVERRUN options
   # ===============
   # <
    
    # Aux options
    def get_overrun_aux(self, i=None):
        self._overrun()
        return self["overrun"].get_overrun_aux(i)
        
    # Name of ``overrun`` binary
    def get_overrun_cmd(self, i=None):
        self._overrun()
        return self["overrun"].get_overrun_cmd(i)

    # Get additional ``overrun`` arguments
    def get_overrun_kw(self, i=None):
        self._overrun()
        return self["overrun"].get_overrun_kw(i)

    # Number of OpenMP threads
    def get_overrun_nthreads(self, i=None):
        self._overrun()
        return self["overrun"].get_overrun_nthreads(i)
        
    # Extra ``overrun`` arguments
    def get_overrun_args(self, i=None):
        self._overrun()
        return self["overrun"].get_overrun_args(i)
        
    # Copy documentation
    for k in ['aux', 'kw', 'cmd', 'nthreads', 'args']:
        eval('get_overrun_'+k).__doc__ = getattr(
            overrun,'get_overrun_'+k).__doc__
    
   # >
   
   # ============== 
   # Local settings
   # ==============
   # <
   
    # Function to get prefix
    def get_Prefix(self, i=None):
        """Get the project rootname, or file prefix
        
        :Call:
            >>> fpre = opts.get_Prefix()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fpre*: :class:`str`
                Name of OVERFLOW prefix
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.get_key("Prefix", i, "project_rootname")
        
   # >
    
   # =================
   # Folder management
   # =================
   # <
    # Initialization method for folder management options
    def _Archive(self):
        """Initialize folder management options if necessary"""
        # Check status
        if 'Archive' not in self:
            # Missing entirely.
            self['Archive'] = Archive()
        elif type(self['Archive']).__name__ == 'dict':
            # Convert to special class
            self['Archive'] = Archive(**self['Archive'])
    
   # >
# class RunControl


