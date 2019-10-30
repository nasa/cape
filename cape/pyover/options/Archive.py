"""
:mod:`cape.pyover.options.Archive`: OVERFLOW archiving options
===============================================================

This module provides OVERFLOW-specific modifications to the base archiving
options module in :mod:`cape.cfdx.options.Archive`. Default options for which files
to delete or tar are specific to each solver, and thus a few modifications are
necessary for each solver in order to define good default values for archiving.

The following default values are copied from the source code of this module.

Default behavior for OVERFLOW case archiving is to copy several of the large
files (such as mesh, solution, etc.) and create several tar bombs. The default
tar bombs that are created within an archive folder are specified in two
separate commands. For each dictionary, the name of the key is the name of the
tar bomb and the list on the right-hand side is a list of file globs that go
into the tar. These are set or modified in the *ArchivePostTarGroups* setting
of the FUN3D archiving section.

    .. code-block:: python
        
        # Plot3D files
        Plot3DDict = [
            {"brkset.[0-9]*": 1},
            {"q.[0-9]*":      1},
            {"x.[0-9]*":      1},
        ]
        # Run output files
        RunDict = [
            {"run":     "run.[0-9frtls][0-9oeup]*"},
            {"out":     "*.out"},
            {"SurfBC":  "SurfBC*.dat"},
            {"pyover": [
                "pyover*",
                "case.json",
                "conditions.json"
            ]},
            {"meshinfo": [
                "Config.xml",
                "grdwghts.save",
                "mixsur.save",
                "walldist.save"
            ]}
        ]
    
Grid, solution, and post-processing files that are directly copied to the
archive are set using the following code.  This affects the *ArchiveFiles*
setting.  The bewildering file glob for ``q``, ``x``, and ``brkset`` files are
trying to capture both ``x.restart`` and ``x.save`` with one glob.

    .. code-block:: python
        
        # Flow files
        CopyFiles = [
            "INTOUT",
            "XINTOUT",
            "q.avg",
            "q.srf",
            "x.srf",
            {"q.[sr0-9][ae0-9][vs0-9][et0-9]*": 1},
            {"x.[sr0-9][ae0-9][vs0-9][et0-9]*": 1},
            {"brkset.[sr0-9][ae0-9][vs0-9][et0-9]*": 1}
        ]

Further files to be deleted upon use of the ``--skeleton`` command are defined
using the following code. This is the *SkeletonFiles* and *TailFiles* settings.
Note that *SkeletonFiles* are defined in reverse order; the user specifies the
files to **keep**, not delete. 

    .. code-block:: python
        
        # Skeleton
        SkeletonFiles = [
            "case.json",
            "conditions.json",
            "archive.log",
            "run.[0-9]*.inp",
            "run.[0-9]*.[0-9]*",
            "lineload/grid.i.triq",
        ]
        # Tail files
        TailFiles = [
            {"run.resid": [1, "run.tail.resid"]},
        ]

The *TailFiles* settings causes pyOver to run the command

    .. code-block:: console
    
        $ tail -n 1 run.resid > run.tail.resid

"""

# Import options-specific utilities
from .util import rc0
# Base module
import cape.cfdx.options.Archive

# Files to archive
CopyFiles = [
    "INTOUT",
    "XINTOUT",
    "q.avg",
    "q.srf",
    "x.srf",
    {"q.[sr0-9][ae0-9][vs0-9][et0-9]*": 1},
    {"x.[sr0-9][ae0-9][vs0-9][et0-9]*": 1},
    {"brkset.[sr0-9][ae0-9][vs0-9][et0-9]*": 1}
]
# Plot3D files
Plot3DDict = [
    {"brkset.[0-9]*": 1},
    {"q.[0-9]*":      1},
    {"x.[0-9]*":      1},
]
# Run output files
RunDict = [
    {"run":     "run.[0-9frtls][0-9oeup]*"},
    {"out":     "*.out"},
    {"SurfBC":  "SurfBC*.dat"},
    {"pyover": [
        "pyover*",
        "case.json",
        "conditions.json"
    ]},
    {"meshinfo": [
        "Config.xml",
        "grdwghts.save",
        "mixsur.save",
        "walldist.save"
    ]}
]

# Skeleton
SkeletonFiles = [
    "case.json",
    "conditions.json",
    "archive.log",
    "run.[0-9]*.inp",
    "run.[0-9]*.[0-9]*",
    "lineload/grid.i.triq",
]
# Tail files
TailFiles = [
    {"run.resid": [1, "run.tail.resid"]},
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
        # Files/folders to delete prior to archiving
        self.add_ArchivePreDeleteFiles(Plot3DDict)
        self.add_ArchivePreDeleteFiles("*.bomb")
        self.add_ArchivePreDeleteFiles("core.*")
        # Pre-archiving
        self.add_ArchivePreTarGroups([])
        self.add_ArchivePreTarDirs([])
        # Files to delete before saving
        self.add_ArchivePreUpdateFiles([])
        # Post-archiving
        for dopts in RunDict:
            self.add_ArchivePostTarGroups(dopts)
        # Folders to archive later
        self.add_ArchivePostTarDirs(["fomo", "lineload", "aero"])
        # Individual archive files
        for dopts in CopyFiles:
            self.add_ArchiveArchiveFiles(dopts)
        # Files/folders to delete after archiving
        self.add_ArchivePostDeleteFiles([])
        self.add_ArchivePostDeleteDirs([])
        # Folders to *keep* during ``--skeleton``
        self.add_ArchiveSkeletonFiles(SkeletonFiles)
        self.add_ArchiveSkeletonTailFiles(TailFiles)
# class Archive

