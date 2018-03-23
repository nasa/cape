"""
:mod:`pyFun.options.Archive`: FUN3D archiving options
======================================================

This module provides FUN3D-specific modifications to the base archiving options
module in :mod:`cape.options.Archive`.  Default options for which files to
delete or tar are specific to each solver, and thus a few modifications are
necessary for each solver in order to define good default values for archiving.

The following default values are copied from the source code of this module.

Default behavior for FUN3D case archiving is to copy several of the large files
(such as mesh, solution, etc.) and create several tar bombs.  The default tar
bombs that are created within an archive folder are specified in two separate
commands.  For each dictionary, the name of the key is the name of the tar bomb
and the list on the right-hand side is a list of file globs that go into the
tar.  These are set or modified in the *ArchivePostTarGroups* setting of the
FUN3D archiving section.

    .. code-block:: python
    
        # Tecplot files
        PltDict = [
            {"pyfun_tec": ["*.plt", "*_tec_*.dat", "*.szplt"]}
        ]
        
        # Base files
        RunDict = [
            {"pyfun": [
                "case.json",
                "conditions.json",
                "run.[0-9]*.[0-9]*",
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
                "*hist.??.dat"
            ]}
        ]
    
Grid, solution, and post-processing files that are directly copied to the
archive are set using the following code.  This affects the *ArchiveFiles*
setting.

    .. code-block:: python
        
        # Flow files
        CopyFiles = [
            {"*_volume.tec": 1},
            {"*.grid_info": 1},
            {"*.flow": 1},
            {"*.ugrid": 1},
            {"*.cgns": 1}
        ]

Further files to be deleted upon use of the ``--skeleton`` command are defined
using the following code.  This is the *SkeletonFiles* setting.  Note that
*SkeletonFiles* are defined in reverse order; the user specifies the files to
**keep**, not delete.

    .. code-block:: python
    
        # Files to keep
        SkeletonFiles = [
            "case.json",
            "conditions.json",
            "archive.log",
            "run.[0-9]*.[0-9]*",
            "*hist.dat",
            "*hist.[0-9]*.dat",
            "fun3d.out",
            "fun3d.[0-9]*.nml",
            {"*_tec_boundary_timestep*.plt": 1},
            {"*_tec_boundary_timestep*.triq": 1},
        ]

"""

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
    {"*.grid_info": 1},
    {"*.flow": 1},
    {"*.ugrid": 1},
    {"*.cgns": 1}
]

# Base files
RunDict = [
    {"pyfun": [
        "case.json",
        "conditions.json",
        "run.[0-9]*.[0-9]*",
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
        "*hist.??.dat"
    ]}
]

# Files to keep
SkeletonFiles = [
    "case.json",
    "conditions.json",
    "archive.log",
    "run.[0-9]*.[0-9]*",
    "*hist.dat",
    "*hist.[0-9]*.dat",
    "fun3d.out",
    "fun3d.[0-9]*.nml",
    {"*_tec_boundary_timestep*.plt": 1},
    {"*_tec_boundary_timestep*.triq": 1},
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
        # Folders to *keep* during ``--skeleton``
        self.add_ArchiveSkeletonFiles(SkeletonFiles)
# class Archive

