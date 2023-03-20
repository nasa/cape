r"""
:mod:`cape.pykes.options.archiveopts`: Kestrel archiving options
==================================================================

This module provides Kestrel-specific modifications to the base
archiving options module in :mod:`archiveopts`.  Default
options for which files to delete or tar are specific to each solver,
and thus a few modifications are necessary for each solver in order to
define good default values for archiving.

The following default values are copied from the source code of this
module.

Default behavior for Kestrel case archiving is to copy several of the
large files (such as mesh, solution, etc.) and create several tar balls.
The default tar balls that are created within an archive folder are
specified in two separate commands.  For each dictionary, the name of
the key is the name of the tar ball and the list on the right-hand side
is a list of file globs that go into the tar. These are set or modified
in the *ArchivePostTarGroups* setting of the archiving section.

    .. code-block:: python

        # Tecplot files
        PltDict = [
        ]

        # Base files
        RunDict = [
        ]

Grid, solution, and post-processing files that are directly copied to the
archive are set using the following code.  This affects the *ArchiveFiles*
setting.

    .. code-block:: python

        # Flow files
        CopyFiles = [
        ]

Further files to be deleted upon use of the ``--skeleton`` command are defined
using the following code.  This is the *SkeletonFiles* setting.  Note that
*SkeletonFiles* are defined in reverse order; the user specifies the files to
**keep**, not delete.

    .. code-block:: python

        # Files to keep
        SkeletonFiles = [
        ]

:See also:
    * :mod:`cape.cfdx.options.archiveopts`
    * :class:`cape.cfdx.options.archiveopts.ArchiveOpts`
    * :mod:`cape.manage`
"""

# Local immports
from ...cfdx.options import archiveopts


# Tecplot files
PltDict = [
]

# Flow files
CopyFiles = [
]

# Base files
RunDict = [
]

# Files to keep
SkeletonFiles = [
]


# Class for case management
class ArchiveOpts(archiveopts.ArchiveOpts):
    """Archiving options for :mod:`cape.pykes`

    :Call:
        >>> opts = Archive(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: v1.0
        * 2022-10-21 ``@ddalle``: v2.0; use :mod:`cape.optdict`
    """
    # Initialization method
    def init_post(self):
        """Initialization hook for Kestrel archiving options

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
        r"""Apply named template to set default files to delete/archive

        :Call:
            >>> opts.apply_ArchiveTemplate()
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Versions:
            * 2023-03-20 ``@ddalle``: v0.0; no templates yet
        """
        pass


# Turn dictionary into Archive options
def auto_Archive(opts):
    r"""Ensure :class:`ArchiveOpts` instance

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
        * 2022-10-21 ``@ddalle``: v2.0; solver-agnostic template
    """
    return archiveopts.auto_Archive(opts, cls=ArchiveOpts)
