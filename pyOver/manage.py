"""
Manage Run Directory Folders: :mod:`pyCart.manage`
==================================================

This module contains functions to manage files and folders, especially file
count, for run directories.  The most important usage is to compress
``adaptXX/`` folders except for the most recent folder.

However, it can also be used to backup a run directory, expand tar balls, or
delete extra material.

:Versions:
    * 2016-12-09 ``@ddalle``: First version
"""

# Options module
from .options import Archive
# Basis module
import cape.manage

# Subdirectories
fsub = []

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
