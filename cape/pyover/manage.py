"""
:mod:`cape.pyover.manage`: Manage pyOver case folders
=======================================================

This module is a derivative of the main solution folder management module
:mod:`cape.manage`. It provides OVERFLOW-specific versions of the three
top-level functions, which each correspond to a primary command-line option.
    
    =======================   ==================
    Function                  Command-line
    =======================   ==================
    :func:`CleanFolder`       ``--clean``
    :func:`ArchiveFolder`     ``--archive``
    :func:`SkeletonFolder`    ``--skeleton``
    =======================   ==================
    
The OVERFLOW-specific versions of these commands use the function
:func:`pyOver.options.Archive.auto_Archive`, which apply the default settings
appropriate to pyOver, and then call the generic version of the function with
the same name from :mod:`cape.manage`.  In addition, this module sets

    .. code-block:: python
        
        # Subdirectories
        fsub = []
        
so the functions do not look inside any subfolders while archiving.

The ``--unarchive`` command does not require any specific customizations for
OVERFLOW, and the generic version of :func:`cape.manage.UnarchiveFolder` is
just called directly.

:See also:
    * :mod:`cape.manage`
    * :mod:`cape.pyover.options.Archive`
    * :mod:`cape.options.Archive`
"""

# Options module
from .options import Archive
# Basis module
import cape.manage

# Subdirectories
fsub = []

# Clear folder
def CleanFolder(opts, phantom=False):
    """Delete files before archiving and regardless of status
    
    :Call:
        >>> pyOver.manage.CleanFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2017-03-10 ``@ddalle``: First version
        * 2017-12-15 ``@ddalle``: Added *phantom* option
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.CleanFolder(opts, fsub=fsub, phantom=phantom)

# Archive folder
def ArchiveFolder(opts, phantom=False):
    """Archive a folder to a backup location and clean up nonessential files
    
    :Call:
        >>> pyOver.manage.ArchiveFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2016-12-09 ``@ddalle``: First version
        * 2017-12-15 ``@ddalle``: Added *phantom* option
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.ArchiveFolder(opts, fsub=fsub, phantom=phantom)

# Replace folder contents with skeleton
def SkeletonFolder(opts, phantom=False):
    """Archive a folder to a backup location and clean up nonessential files
    
    :Call:
        >>> pyOver.manage.SkeletonFolder(opts, phantom=False)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *phantom*: ``True`` | {``False``}
            Write actions to ``archive.log``; only delete if ``False``
    :Versions:
        * 2017-12-14 ``@ddalle``: First version
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.SkeletonFolder(opts, fsub=fsub, phantom=phantom)
# def SkeletonFolder

