"""
Manage Run Directory Folders: :mod:`pyOver.manage`
==================================================

This module contains functions to manage files and folders by archiving them to
a different backup location and cleaning up by deleting or grouping files.
Files can be deleted before or after copying to the archive, and groups of
files matching a list of file globs can be grouped into tar balls.  There is
also an option to replace each folder with a tar ball.

The tar balls that are created during archiving can also be deleted after being
copied to the archive.

:Versions:
    * 2016-12-09 ``@ddalle``: First version
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

