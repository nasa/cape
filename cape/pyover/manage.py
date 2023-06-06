r"""
:mod:`cape.pyover.manage`: Manage pyover case folders
=======================================================

This module is a derivative of the main solution folder management
module :mod:`cape.manage` for OVERFLOW. It provides OVERFLOW-specific
versions of the three top-level functions, which each correspond to a
primary command-line option.

    =======================   ==================
    Function                  Command-line
    =======================   ==================
    :func:`CleanFolder`       ``--clean``
    :func:`ArchiveFolder`     ``--archive``
    :func:`SkeletonFolder`    ``--skeleton``
    =======================   ==================

The ``--unarchive`` command does not require any specific customizations
for OVERFLOW, and the generic version of
:func:`cape.manage.UnarchiveFolder` is just called directly.

:See also:
    * :mod:`cape.manage`
    * :mod:`cape.pyover.options.archiveopts`
    * :mod:`cape.cfdx.options.archiveopts`
"""

# Local imports
from .options import archiveopts
from .. import manage


# Subdirectories
fsub = []


# Clear folder
def CleanFolder(opts, phantom=False):
    r"""Delete files before archiving and regardless of status

    :Call:
        >>> pyOver.manage.CleanFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2017-03-10 ``@ddalle``: Version 1.0
        * 2017-12-15 ``@ddalle``: Version 1.1; *phantom* kwarg
    """
    # Restrict options to correct class
    opts = archiveopts.auto_Archive(opts)
    # Call the :mod:`cape` version
    manage.CleanFolder(opts, fsub=fsub, phantom=phantom)


# Archive folder
def ArchiveFolder(opts, phantom=False):
    r"""Archive a folder to a backup location and clean up
    
    :Call:
        >>> pyOver.manage.ArchiveFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2016-12-09 ``@ddalle``: Version 1.0
        * 2017-12-15 ``@ddalle``: Version 1.1; *phantom* kwarg
    """
    # Restrict options to correct class
    opts = archiveopts.auto_Archive(opts)
    # Call the :mod:`cape` version
    manage.ArchiveFolder(opts, fsub=fsub, phantom=phantom)


# Replace folder contents with skeleton
def SkeletonFolder(opts, phantom=False):
    r"""Archive a folder and delete all but most essential files
    
    :Call:
        >>> pyOver.manage.SkeletonFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2017-12-14 ``@ddalle``: Version 1.0
    """
    # Restrict options to correct class
    opts = archiveopts.auto_Archive(opts)
    # Call the :mod:`cape` version
    manage.SkeletonFolder(opts, fsub=fsub, phantom=phantom)
