"""Case archiving options for Cart3D solutions"""

# Import options-specific utilities
from util import rc0
# Base module
import cape.options.Archive

# Tecplot files
PltDict = [
    {"pyfun_tec": ["*.plt", "*_tec_*.dat", "*.szplt"]}
]

# Flow files
CopyFiles = [
    {"*_volume.tec": 1},
    {"*.flow": 1},
    {"*.ugrid": 1},
    {"*.cgns": 1}
]

# Base files
RunDict = [
    {"pyfun": [
        "case.json",
        "conditions.json",
        "run.[0-9].*",
        "run_fun3d.*pbs"
    ]},
    {"fun3d": [
        "fun3d.*",
        "*.freeze",
        "*.mapbc",
        "faux_input",
        "rubber.data"
    ]},
    {"fm": [
        "*_fm_*.dat",
        "*hist.dat",
        "*.hist.??.dat"
    ]}
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
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get the template
        tmp = self.get_ArchiveTemplate().lower()
        # Extension
        ext = self.get_ArchiveExtension()
        # Files/folders to delete prior to archiving
        self.add_ArchivePreDeleteFiles("*.bomb")
        self.add_ArchivePreDeleteFiles("core.*")
        self.add_ArchivePreDeleteFiles("nan_locations*")
        # Pre-archiving
        self.add_ArchivePreTarGroups([])
        self.add_ArchivePreTarDirs([])
        # Files to delete before saving
        self.add_ArchivePreUpdateFiles([])
        # Post-archiving
        for dopts in RunDict:
            self.add_ArchivePostTarGroups(dopts)
        for dopts in PltDict:
            self.add_ArchivePostTarGroups(dopts)
        # Folders to TAR
        self.add_ArchivePostTarDirs(["fomo", "lineload", "aero"])
        # Individual archive files
        for dopts in CopyFiles:
            self.add_ArchiveArchiveFiles(dopts)
        # Files/folders to delete after archiving
        self.add_ArchivePostDeleteFiles([])
        self.add_ArchivePostDeleteDirs([])
# class Archive

