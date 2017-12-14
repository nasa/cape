"""
Manage Run Directory Folders: :mod:`pyFun.manage`
==================================================

This module contains functions to manage files and folders by archiving them to
a different backup location and cleaning up by deleting or grouping files.
Files can be deleted before or after copying to the archive, and groups of
files matching a list of file globs can be grouped into tar balls.  There is
also an option to replace each folder with a tar ball.

The tar balls that are created during archiving can also be deleted after being
copied to the archive.

:Versions:
    * 2016-12-27 ``@ddalle``: First version
"""

# Options module
from .options import Archive
# Basis module
import cape.manage

# Subdirectories
fsub = ["Flow"]

# Clear folder
def CleanFolder(opts):
    """Delete files before archiving and regardless of status
    
    :Call:
        >>> pyFun.manage.CleanFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
    :Versions:
        * 2017-03-10 ``@ddalle``: First version
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.CleanFolder(opts, fsub=fsub)

# Archive folder
def ArchiveFolder(opts):
    """Archive a folder to a backup location and clean up nonessential files
    
    :Call:
        >>> pyFun.manage.ArchiveFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
    :Versions:
        * 2016-12-09 ``@ddalle``: First version
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.ArchiveFolder(opts, fsub=fsub)

# Replace folder contents with skeleton
def SkeletonFolder(opts):
    """Archive a folder to a backup location and clean up nonessential files
    
    :Call:
        >>> pyFun.manage.SkeletonFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
    :Versions:
        * 2017-12-14 ``@ddalle``: First version
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.SkeletonFolder(opts, fsub=fsub)
# def SkeletonFolder

