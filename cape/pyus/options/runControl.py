"""
:mod:`cape.pyus.options.runControl.RunControl`: Run control options
=====================================================================

Options interface for aspects of running a case of US3D.  The settings are
read from the ``"RunControl"`` of a JSON file, and the contents of this section
are written to :file:`case.json` within each run folder.

The methods of :class:`cape.cfdx.options.runControl.RunControl` are also present.
These control options such as whether to submit as a PBS job, whether or not to
use MPI, etc.

This contains options that determine how long the solver is ran (primarily 
via the ``"PhaseSequence"`` and ``"PhaseIters"`` options), what basic mode it
is run in (such as a submitted or local job or serial or MPI job), and options
for command-line options to the FUN3D binaries.  There is also an
``"Archive"`` section that can be used for copying files and cleaning up after
one or more cases have been completed.

This module primarily provides a class :class:`pyFun.options.RunControl`. Many
of the options that are common to all solvers are inherited from
:class:`cape.cfdx.options.runControl.RunControl`. This class also has an interface
for environment variables and ``ulimit`` parameters.

In particular, all of the commands available to the classes listed below are
also available to :class:`pyFun.options.runControl.RunControl`.

:Classes:
    * :class:`pyUS.options.runControl.RunControl`
    * :class:`pyUS.options.runControl.nodet`
    * :class:`pyUS.options.Archive.Archive`
    * :class:`cape.cfdx.options.aflr3.aflr3`
    * :class:`cape.cfdx.options.runControl.Environ`
    * :class:`cape.cfdx.options.ulimit.ulimit`
:See Also:
    * :mod:`cape.cfdx.options.runControl`
    * :mod:`cape.cfdx.options.ulimit`
    * :mod:`cape.cfdx.options.intersect`
    * :mod:`cape.pyus.options.Archive`
"""

# Import options-specific utilities
from .util import rc0, getel, odict

# Import template module
import cape.cfdx.options.runControl

# Submodules
from .Archive import Archive

# Class for inputs to the US3D executable



# Class for Report settings
class RunControl(cape.cfdx.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = RunControl(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._Environ()
        self._ulimit()
        self._aflr3()
        self._intersect()
        self._verify()
        self._Archive()
        
   # ============ 
   # Initializers
   # ============
   # <
   
    # Initialization and confirmation for options to US3D executable
            
   # >
   
   # ============== 
   # Local settings
   # ==============
   # <
   
        
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


