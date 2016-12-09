"""Case archiving options for Cart3D solutions"""

# Import options-specific utilities
from util import rc0
# Base module
import cape.options.Archive

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
# Plot3D files
Plot3DDict = [
    {"brkset.[0-9]*": 1},
    {"q.[0-9]*":      1},
    {"x.[0-9]*":      1}
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
    """Automatically convert dict to :mod:`pyCart.options.Archive.Archive`
    
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
class Archive(cape.options.Archive.Archive):
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
        # Files/folders to delete prior to archiving
        self.add_ArchivePreDeleteFiles(Plot3DDict)
        self.add_ArchivePreDeleteFiles("*.bomb")
        self.add_ArchivePreDeleteFiles("core.*")
        # Pre-archiving
        self.add_ArchivePreTarGroups(["run.[0-9][0-9]*"])
        self.add_ArchivePreTarGroups(["*.out"])
        self.add_ArchivePreTarDirs(["fomo", "lineload", "aero"])
        # Files to delete before saving
        self.add_ArchivePreUpdateFiles([])
        # Files/folders to delete after archiving
        self.add_ArchivePostDeleteFiles([])
        self.add_ArchivePostDeleteDirs([])
        # Post-archiving
        self.add_ArchivePostTarGroups([])
        self.add_ArchivePostTarDirs([])
# class Archive

