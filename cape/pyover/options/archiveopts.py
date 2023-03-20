r"""
:mod:`cape.pyover.options.archiveopts`: OVERFLOW archiving options
===================================================================

This module provides OVERFLOW-specific modifications to the base
archiving options module.

Default behavior for OVERFLOW case archiving, which is defined in this
module, is to copy several of the large files (such as mesh, solution,
etc.) and create several tar bombs. The default tar balls that are
created within an archive folder are specified in two separate commands.
For each dictionary, the name of the key is the name of the tar file,
and the list on the right-hand side is a list of file globs that go into
the tar file. These are set or modified in the *ArchivePostTarGroups*
setting.

    .. code-block:: python

        # Plot3D files
        Plot3DDict = [
            {"brkset.[0-9]*": 1},
            {"q.[0-9]*":      1},
            {"x.[0-9]*":      1},
        ]
        # Run output files
        RunDict = [
            {"run": "run.[0-9frtls][0-9oeup]*"},
            {"out": "*.out"},
            {"SurfBC": "SurfBC*.dat"},
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

Grid, solution, and post-processing files that are directly copied to
the archive (without being added to a tar file) are set using the
following code. This affects the *ArchiveFiles* setting.  The
bewildering file glob for ``q``, ``x``, and ``brkset`` files are
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

Further files to be deleted upon use of the ``--skeleton`` command are
defined using the following code. This is the *SkeletonFiles* and
*TailFiles* settings. Note that *SkeletonFiles* are defined in reverse
order; the user specifies the files to **keep**, not delete.

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

:See also:
    * :mod:`cape.cfdx.options.archiveopts`
    * :class:`cape.cfdx.options.archiveopts.ArchiveOpts`
    * :mod:`cape.manage`
"""

# Local imports
from ...cfdx.options import archiveopts


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
    {"run": "run.[0-9frtls][0-9oeup]*"},
    {"out": "*.out"},
    {"SurfBC": "SurfBC*.dat"},
    {
        "pyover": [
            "pyover*",
            "case.json",
            "conditions.json"
        ]
    },
    {
        "meshinfo": [
            "Config.xml",
            "grdwghts.save",
            "mixsur.save",
            "walldist.save"
        ]
    }
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


# Class for case management
class ArchiveOpts(archiveopts.ArchiveOpts):
    r"""Archiving options interface

    :Call:
        >>> opts = ArchiveOpts(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: v1.0
        * 2016-03-01 ``@ddalle``: v1.1; custom settings
        * 2022-10-21 ``@ddalle``: v2.0; use :mod:`cape.optdict`
    """
    # Initialization method
    def init_post(self):
        """Initialization hook for OVERFLOW archiving options

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`ArchiveOpts`
                Archiving options interface
        :Versions:
            * 2022-10-21 ``@ddalle``: v1.0
        """
        # Apply the template
        self.apply_ArchiveTemplate()

    # Apply template
    def apply_ArchiveTemplate(self):
        r"""Apply OVERFLOW-specific archiving defaults

        :Call:
            >>> opts.apply_ArchiveTemplate()
        :Inputs:
            *opts*: :class:`ArchiveOpts`
                Options interface
        :Versions:
            * 2016-02-29 ``@ddalle``: v1.0
        """
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


# Turn dictionary into Archive options
def auto_Archive(opts):
    r"""Automatically convert dict to pyover :class:`ArchiveOpts`

    :Call:
        >>> opts = auto_Archive(opts)
    :Inputs:
        *opts*: :class:`dict`
            Dict of either global, "RunControl" or "Archive" options
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Instance of archiving options
    :Versions:
        * 2016-02-29 ``@ddalle``: v1.0
        * 2022-10-21 ``@ddalle``: v2.0; generic call
    """
    return archiveopts.auto_Archive(opts, ArchiveOpts)
