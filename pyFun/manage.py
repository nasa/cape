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

# Archive folder
def ArchiveFolder(opts):
    """Archive a folder to a backup location and clean up nonessential files
    
    :Call:
        >>> pyOver.manage.ArchiveFolder(opts, fsub=[])
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface including management/archive interface
        *fsub*: :class:`list` (:class:`str`)
            List of globs of subdirectories that are adaptive run folders
    :Versions:
        * 2016-12-09 ``@ddalle``: First version
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Call the :mod:`cape` version
    cape.manage.ArchiveFolder(opts, fsub=fsub)
    
# def ArchiveFolder
