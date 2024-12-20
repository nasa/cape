r"""
:mod:`cape.pyfun.options.archiveopts`
======================================

This module provides FUN3D-specific modifications to the base archiving
options module in :mod:`cape.cfdx.options.Archive`. Default options for
which files to delete or tar are specific to each solver, and thus a few
modifications are necessary for each solver in order to define good
default values for archiving.

The following default values are copied from the source code of this
module.

Default behavior for FUN3D case archiving is to copy several of the
large files (such as mesh, solution, etc.) and create several tar bombs.
The default tar bombs that are created within an archive folder are
specified in two separate commands. For each dictionary, the name of the
key is the name of the tar bomb and the list on the right-hand side is a
list of file globs that go into the tar. These are set or modified in
the *ArchivePostTarGroups* setting of the FUN3D archiving section.

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

Grid, solution, and post-processing files that are directly copied to
the archive are set using the following code.  This affects the
*ArchiveFiles* setting.

    .. code-block:: python

        # Flow files
        CopyFiles = [
            {"*_volume.tec": 1},
            {"*.grid_info": 1},
            {"*.flow": 1},
            {"*.ugrid": 1},
            {"*.cgns": 1}
        ]

Further files to be deleted upon use of the ``--skeleton`` command are
defined using the following code.  This is the *SkeletonFiles* setting.
Note that *SkeletonFiles* are defined in reverse order; the user
specifies the files to **keep**, not delete.

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

# Base module
from ...cfdx.options import archiveopts


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


# Class for case management
class ArchiveOpts(archiveopts.ArchiveOpts):
    r"""Archiving options for :mod:`cape.pyfun`

    :Call:
        >>> opts = ArchiveOpts(**kw)
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Options interface
    :Versions:
        * 2015-09-28 ``@ddalle``: v1.0
        * 2022-10-21 ``@ddalle``: v2.0; use :mod:`cape.optdict`
    """
    pass


# Turn dictionary into Archive options
def auto_Archive(opts):
    r"""Ensure instance of :class:`ArchiveOpts`

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
        * 2022-10-21 ``@ddalle``: v2.0; solver-agnostic
    """
    return archiveopts.auto_Archive(opts, cls=ArchiveOpts)
