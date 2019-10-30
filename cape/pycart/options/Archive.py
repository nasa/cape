"""
:mod:`cape.pycart.options.Archive`: pyCart Archiving Options
============================================================

Options interface for archiving one or more Cart3D solutions that was managed
by pyCart.  Archiving generally means two tasks:

    #. Copy requested files to a backup location (such as a tape drive or other
       external location), possibly in groups that have been put into tar balls
       (usually ``.tar`` so that archives can be modified later).
       
    #. Clean up run folder after archiving by deleting files unnecessary for
       post-processing
       
For the most part, cases can only be archived after it has the status ``PASS``.
See :func:`cape.cntl.Cntl.DisplayStatus` and the 
:ref:`run matrix file format <matrix-syntax>` for more information, but
generally this requires the case running at least the requested number of
iterations and marked with a ``p`` in the run matrix file.

However, there are a few options that delete files as solutions complete
phases.  These options are intended for automatically deleting previous
check-point files as more become available.  For example, it is usually
acceptable to keep only two solution files around, so deleting the third newest
solution file is allowable.

Archiving actions can be issued from the command line via the commands such as
the :ref:`following <cli-archive>`.  :ref:`Subsetting options <cli-subset>`
are all available to this command.

    .. code-block:: console
    
        $ pycart --archive [OPTIONS]
        
        

:See Also:
    * :mod:`cape.cfdx.options.Archive`
    * :mod:`cape.pycart.options.runControl`
"""

# Import options-specific utilities
from .util import rc0
# Base module
import cape.cfdx.options.Archive

# Globs and tarballs to tar most of the time
VizGlob = [
    {'Components.i.stats': 'Components.i.[0-9]*.stats'},
    {'Components.i'      : 'Components.i.[0-9]*'},
    {'cutPlanes'         : 'cutPlanes.[0-9]*'},
    {'pointSensors'      : 'pointSensors.[0-9]*'},
    {'lineSensors'       : 'lineSensors.[0-9]*'}
]
# Tar adapt
AdaptDict = [{"adapt??": 1}]
# Check files
CheckDict = [
    {"check.?????": 1},
    {"check.??????.td": 2}
]
# Visualization files; keep only recent
VizDict = [
    {"Components.i.*.plt": 1},
    {"Components.i.*.dat": 1},
    {"cutPlanes.*.plt"   : 1},
    {"cutPlanes.*.dat"   : 1}
]
# Run files
RunDict = [
    {"run_cart3d.??.pbs": 1},
    {"run.[0-9]*.*"     : 1},
    {"input.??.cntl"    : 1},
    {"aero.??.csh"      : 1}
]
# One-off files
RunFiles = [
    'input.c3d', 'Config.xml', 'jobID.dat',
    'results.dat', 'user_time.dat', 'forces.dat', 'moments.dat',
    'functional.dat', 'loadsCC.dat', 'loadsTRI.dat'
]

# Turn dictionary into Archive options
def auto_Archive(opts):
    """Automatically convert dict to :mod:`cape.pycart.options.Archive.Archive`
    
    :Call:
        >>> opts = auto_Archive(opts)
    :Inputs:
        *opts*: :class:`dict`
            Dict of either global, "RunControl" or "Archive" options
    :Outputs:
        *opts*: :class:`pyCart.options.Archive.Archive`
            Instance of archiving options
    :Versions:
        * 2016-02-29 ``@ddalle``: First version
    """
    # Get type
    t = type(opts).__name__
    # Check type
    if t == "Archive":
        # Good; quit
        return opts
    elif t == "RunControl":
        # Get the sub-object
        return opts["Archive"]
    elif t == "Options":
        # Get the sub-sub-object
        aopts = opts["RunControl"]["Archive"]
        # Set the umask
        aopts.set_umask(opts.get_umask())
        # Output
        return aopts
    elif t in ["dict", "odict"]:
        # Downselect if given parent class
        opts = opts.get("RunControl", opts)
        opts = opts.get("Archive",    opts)
        # Convert to class
        return Archive(**opts)
    else:
        # Invalid type
        raise TypeError("Unformatted input must be type 'dict', not '%s'" % t)
# def auto_Archive

# Class for case management
class Archive(cape.cfdx.options.Archive.Archive):
    """
    Dictionary-based interfaced for options specific to folder management
    
    :Call:
        >>> opts = Archive(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
        * 2016-03-01 ``@ddalle``: Upgraded custom settings
    """
    # Initialization method
    def __init__(self, **kw):
        """Initialization method
        
        :Versions:
            * 2016-03-01 ``@ddalle``: First version
        """
        # Copy from dict
        for k in kw:
            self[k] = kw[k]
        # Apply the template
        self.apply_ArchiveTemplate()
    
    # Apply template
    def apply_ArchiveTemplate(self):
        """Apply named template to set default files to delete/archive
        
        :Call:
            >>> opts.apply_ArchiveTemplate()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get the template
        tmp = self.get_ArchiveTemplate().lower()
        # Extension
        ext = self.get_ArchiveExtension()
        # Check it
        if tmp in ["full"]:
            # Archive almost everything and then clean up slightly
            # Files/folders to delete prior to archiving
            self.add_ArchivePreDeleteFiles([])
            self.add_ArchivePreDeleteDirs(["ADAPT", "EMBED"])
            # Pre-archiving
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            # Files to delete before saving
            self.add_ArchivePreUpdateFiles([])
            # Files/folders to delete after archiving
            self.add_ArchivePostDeleteFiles(
                ['adapt??.'+ext, 'checkDT*'])
            self.add_ArchivePostDeleteDirs([])
            # Post-archiving
            self.add_ArchivePostTarGroups([])
            self.add_ArchivePostTarDirs([])
            # Files to keep only *n*
            self.add_ArchivePostUpdateFiles(CheckDict)
        elif tmp in ["restart"]:
            # Keep everything needed to restart
            # Pre-archiving
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            # Files/folders to delete after archiving
            self.add_ArchivePostDeleteFiles(
                ['adapt??.'+ext, 'checkDT*', 'Components.i.triq'])
            self.add_ArchivePostDeleteDirs(["ADAPT", "EMBED"])
            # Files to keep only *n*
            self.add_ArchivePostUpdateFiles(CheckDict)
            self.add_ArchivePostUpdateFiles(VizDict)
        elif tmp in ["viz"]:
            # Keep visualization stuff only
            # Pre-archiving
            self.add_ArchivePreDeleteFiles(["Mesh*", "checkDT*"])
            self.add_ArchivePreDeleteDirs(["ADAPT", "EMBED"])
            # Pre-archiving
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            self.add_ArchivePreUpdateFiles(CheckDict)
            # Files/folders to delete after archiving
            self.add_ArchivePostDeleteFiles([
                'adapt??.'+ext, 'checkDT*',
                'Components.i.tri', '*.lay', '*.out'
            ])
            self.add_ArchivePostDeleteFiles(RunFiles)
            # Files to keep only *n*
            self.add_ArchivePostUpdateFiles(RunDict)
        elif tmp in ["hist"]:
            # Keep history only
            # Pre-archiving
            self.add_ArchivePreDeleteFiles(["Mesh*"])
            self.add_ArchivePreDeleteDirs(["ADAPT", "EMBED"])
            # Pre-archiving
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            self.add_ArchivePreUpdateFiles(CheckDict)
            # Files/folders to delete after archiving
            self.add_ArchivePostDeleteFiles([
                'adapt??.'+ext, 'checkDT*', 'check*', 
                'Components.*.tri', '*.lay', '*.out'
            ])
            self.add_ArchivePostDeleteFiles(RunFiles)
            # Files to keep only *n*
            self.add_ArchivePostUpdateFiles(VizDict)
            self.add_ArchivePostUpdateFiles(RunDict)
        elif tmp in ["skeleton"]:
            # Keep only basic info
            # Pre-archiving
            self.add_ArchivePreDeleteFiles(["Mesh*", "checkDT*"])
            self.add_ArchivePreDeleteDirs(["ADAPT", "EMBED"])
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            self.add_ArchivePreUpdateFiles(CheckDict)
            # Files/folders to delete after archiving
            self.add_ArchivePostDeleteFiles([
                'adapt??.'+ext, 'checkDT*', 'check*', '*.pbs', 
                'Components*triq', 'Components*tri', '*.lay', '*.out'
            ])
            self.add_ArchivePostDeleteFiles(RunFiles)
        elif tmp in ["none"]:
            # Do nothing at all
            pass
        else:
            # Basic templates
            self.add_ArchivePreDeleteDirs(["ADAPT", "EMBED"])
            self.add_ArchivePreTarGroups(VizGlob)
            self.add_ArchivePreTarDirs(AdaptDict)
            self.add_ArchivePostDeleteFiles(['checkDT*', '*.lay', '*.out'])
            self.add_ArchivePostUpdateFiles(CheckDict)
        
        
    # Get number of check points to keep around
    def get_nCheckPoint(self, i=None):
        """Return the number of check point files to keep
        
        :Call:
            >>> nchk = opts.get_nCheckPoint(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('nCheckPoint', i)
        
    # Set the number of check point files to keep around
    def set_nCheckPoint(self, nchk=rc0('nCheckPoint'), i=None):
        """Set the number of check point files to keep
        
        :Call:
            >>> opts.set_nCheckPoint(nchk)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('nCheckPoint', nchk, i)
        
    # Get the archive format for visualization files
    def get_TarViz(self):
        """Return the archive format for visualization files
        
        :Call:
            >>> fmt = opts.get_TarViz()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarViz')
        
    # Set the archive format for visualization files
    def set_TarViz(self, fmt=rc0('TarViz')):
        """Set the archive format for visualization files
        
        :Call:
            >>> opts.set_TarViz(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarViz', fmt)
        
    # Get the archive format for visualization files
    def get_TarAdapt(self):
        """Return the archive format for adapt folders
        
        :Call:
            >>> fmt = opts.get_TarAdapt()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarAdapt')
        
    # Set the archive format for visualization files
    def set_TarAdapt(self, fmt=rc0('TarAdapt')):
        """Set the archive format for adapt folders
        
        :Call:
            >>> opts.set_TarAdapt(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarAdapt', fmt)
# class Archive

