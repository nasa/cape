#!/usr/bin/env python
"""
Manage Run Directory Folders: :mod:`cape.manage`
================================================

This module provides methods to manage and archive files for run folders.  It
provides extensive tools for archiving results to other locations either as a
tar ball or zip archive of the entire folder or a folder of several smaller zip
files.

It also provides tools for deleting files either before or after creating the
archive.  In addition, there is an easy interface for keeping only the most
recent *n* files of a certain glob.  For example, if there are solution files
``flow.01``, ``flow.02``, ``flow.03``, and ``flow.04``, setting the
*PostUpdateFiles* parameter to ``{"flow.??": 2}`` will delete only ``flow.01``
and ``flow.02``.

:Versions:
    * 2016-03-01 ``@ddalle``: First version
"""

# File management modules
import os, shutil, glob
# Command-line interface
import subprocess as sp
# Options module
from .options import Archive

# Archive group
def process_ArchiveGroup(grp):
    """Process an archive group, which has a precise format
    
    It must be a dictionary with one entry
    
    :Call:
        >>> ftar, fpat = cape.manage.process_ArchiveGroup(grp)
        >>> ftar, fpat = cape.manage.process_Archivegroup({ftar: fpat})
    :Inputs:
        *grp*: :class:`dict` len=1
            Single dictionary with name and pattern or list of patterns
    :Outputs:
        *ftar*: :class:`str`
            Name of archive to create without extension
        *fpat*: :class:`str` | :class:`list` (:class:`str`)
            File name pattern or list of file name patterns
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Check type
    if (type(grp) != dict) or (len(grp) != 1):
        # Wront length
        raise ValueError(
            ("Received improper archive group: '%s'\n" % grp) +
            "Archive group must be a dict with one entry"
            )
    # Get group
    fgrp = grp.keys()[0]
    # Get value
    fpat = grp[fgrp]
    # Output
    return fgrp, fpat
    
# File name count
def process_ArchiveFile(f, n=1):
    """Process a file glob description
    
    This can be either a file name, file name pattern, or dictionary of file
    name pattern and number of files to keep
    
    :Call:
        >>> fname, nkeep = process_ArchiveFile(fname, n=1)
        >>> fname, nkeep = process_ArchiveFile(fdict, n=1)
        >>> fname, nkeep = process_ArchiveFile({fname: nkeep}, n=1)
    :Inputs:
        *fname*: :class:`str`
            File name or file name pattern to delete/group/etc.
        *nkeep*: :class:`int`
            Number of matching files to keep
        *n*: :class:`int`
            Default number of files to keep
    :Outputs:
        *fname*: :class:`str`
            File name or file name pattern to delete/group/etc.
        *nkeep*: :class:`int`
            Number of matching files to keep
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Input type
    tf = type(f).__name__
    # Check type
    if tf == "dict":
        # Check length
        if len(f) != 1:
            raise ValueError(
                "Received improper archive file description '%s'" % f)
        # Get name and number
        fname = f.keys()[0]
        nkeep = f[fname]
        # Check output name
        if not type(nkeep).__name__.startswith('int'):
            raise TypeError("Number of files to keep must be an integer")
        # Output
        return fname, nkeep
    elif tf in ['str', 'unicode']:
        # String file name; use default number of files to keep
        return f, n
    else:
        # Unprocessed type
        raise TypeError("File descriptor '%s' must be int or dict" % f)


# Get list of folders in which to search
def GetSearchDirs(fsub=None):
    """Get list of current folder and folders matching pattern
    
    :Call:
        >>> fdirs = GetSearchDirs()
        >>> fdirs = GetSearchDirs(fsub)
        >>> fdirs = GetSearchDirs(fsubs)
    :Inputs:
        *fsub*: :class:`str`
            Folder name of folder name pattern
        *fsubs*: :class:`list` (:class:`str`)
            List of folder names or folder name patterns
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Initialize
    fdirs = []
    # Check for null output
    if fsub is None:
        return ['.']
    # Ensure input list
    if type(fsub).__name__ not in ['list', 'ndarray']:
        fsub = [fsub]
    # Loop through names/patterns
    for fi in fsub:
        # Ensure string
        if type(fi).__name__ not in ['str', 'unicode']: continue
        # Get the matching glob
        fglob = glob.glob(fi)
        # Loop through matches
        for fdir in fglob:
            # Make sure it's a directory
            if not os.path.isdir(fdir): continue
            # Make sure it's not in the output already
            if fdir in fdirs: continue
            # Append
            fdirs.append(fdir)
    # Sort
    fdirs.sort()
    # Append working directory (and make sure it's last)
    fdirs.append('.')
    # Output
    return fdirs
    
# Get file/link matches
def GetFileMatches(fname, fsub=None):
    """Get list of all files or links matching a pattern
    
    :Call:
        >>> fglob = GetFileMatches(fname, fsub=None)
        >>> fglob = GetFileMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`str`
            File name or file name pattern
        *fsub*: :class:`str`
            Folder name of folder name pattern
        *fsubs*: :class:`list` (:class:`str`)
            List of folder names or folder name patterns
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get folders
    fdirs = GetSearchDirs(fsub)
    # Initialize output
    fglob = []
    # Loop through folders
    for fdir in fdirs:
        # Construct pattern for this folder
        if fdir == "." or not fdir:
            # Just use the pattern as is
            fn = fname
        else:
            # Join the folder name
            fn = os.path.join(fdir, fname)
        # Get the glob
        fglobn = glob.glob(fn)
        # Sort it
        fglobn.sort()
        # Loop through the matches
        for fgn in fglobn:
            # Ensure file
            if os.path.isdir(fgn):
                # Directory; continue
                continue
            elif not (os.path.isfile(fgn) or os.path.islink(fgn)):
                # Not a file (how did this happen?)
                continue
            elif fgn in fglob:
                # Already matched a previous pattern
                continue
            # Append
            fglob.append(fgn)
    # Output
    return fglob
    
# Get file/link matches for a list of patterns
def GetFileMatchesList(flist, fsub=None):
    """Get list of all files or links matching any of a list of patterns
    
    :Call:
        >>> fglob = GetFileMatchesList(flist, fsub=None)
        >>> fglob = GetFileMatchesList(flist, fsubs)
    :Inputs:
        *flist*: :class:`list` (:class:`str`)
            List of file names or file name patterns
        *fsub*: :class:`str`
            Folder name of folder name pattern
        *fsubs*: :class:`list` (:class:`str`)
            List of folder names or folder name patterns
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Ensure list
    if type(flist).__name__ not in ['list', 'ndarray']:
        flist = [flist]
    # Initialize
    fglob = []
    # Loop through patterns
    for fname in flist:
        # Get matches for this pattern
        fgn = GetFileMatches(fname)
        # Append contents to output glob
        for fn in fgn:
            # Check if already present
            if fn in fglob:
                continue
            # Append to list
            fglob.append(fn)
    # Output
    return fglob
                
# Get folder matches
def GetDirMatches(fname, fsub=None):
    """Get list of all folders matching a pattern
    
    :Call:
        >>> fglob = GetDirMatches(fname, fsub=None)
        >>> fglob = GetDirMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`str`
            Folder name or folder name pattern
        *fsub*: :class:`str`
            Folder name of folder name pattern
        *fsubs*: :class:`list` (:class:`str`)
            List of folder names or folder name patterns
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of folders matching input pattern
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get folders
    fdirs = GetSearchDirs(fsub)
    # Initialize output
    fglob = []
    # Loop through folders
    for fdir in fdirs:
        # Construct pattern for this folder
        if fdir == "." or not fdir:
            # Just use the pattern as is
            fn = fname
        else:
            # Join the folder name
            fn = os.path.join(fdir, fname)
        # Get the glob
        fglobn = glob.glob(fn)
        # Loop through the matches
        for fgn in fglobn:
            # Ensure file
            if not os.path.isdir(fgn):
                # Not directory; continue
                continue
            elif fgn in fglob:
                # Already matched a previous pattern
                continue
            # Append
            fglob.append(fgn)
    # Sort
    fglob.sort()
    # Output
    return fglob
    

# Function to delete files according to a glob
def DeleteFiles_SubDir(fdel, n=1, fsub=None):
    """Delete files that match a glob
    
    The function also searches in any folder matching the directory glob or list
    of directory globs *fsub*.
    
    :Call:
        >>> cape.manage.DeleteFiles_SubDir(fdel, n=1, fsub=None)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *n*: :class:`int`
            Number of files to keep
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder, list of folders, or glob of folders to also search
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get list of files to delete
    fmatch = GetFileMatches(fdel, fsub)
    # Remove from list any files to keep around
    if n == 0:
        # Delete all files
        fglob = fmatch
    elif len(fmatch) <= n:
        # Nothing to delete
        return
    else:
        # Strip last *n* files
        fglob = fmatch[:-n]
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not (os.path.isfile(fn) or os.path.islink(fn)):
            continue
        # Delete it
        os.remove(fn)
        
# Function to delete folders according to a glob
def DeleteDirs_SubDir(fdel, n=1, fsub=None):
    """Delete folders that match a glob
    
    The function also searches in any folder matching the directory glob or list
    of directory globs *fsub*.
    
    :Call:
        >>> cape.manage.DeleteDirs_SubDir(fdel, n=1, fsub=None)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder, list of folders, or glob of folders to also search
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get list of files to delete
    fmatch = GetDirMatches(fdel, fsub)
    # Remove from list any files to keep around
    if n == 0:
        # Delete all files
        fglob = fmatch
    elif len(fmatch) <= n:
        # Nothing to delete
        return
    else:
        # Strip last *n* files
        fglob = fmatch[:-n]
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not os.path.isdir(fn):
            continue
        # Delete it
        shutil.rmtree(fn)
        
# Archive groups
def TarFileGroup(cmd, ftar, fname):
    """Archive a group of files and delete the files
    
    Only the present folder will searched for file name matches
    
    :Call:
        >>> TarFileGroup(cmd, ftar, fname)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`str`
            File name pattern or list thereof to combine into archive
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Check input
    if type(cmd).__name__ != 'list':
        raise TypeError("Input command must be a list of one or two strings")
    # Get list of files
    fglob = GetFileMatchesList(fname)
    # Exit if not matches
    if len(fglob) < 2: return
    # Create command
    cmdc = cmd + [ftar] + fglob
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete matches
    for fn in fglob:
        if os.path.isfile(fn) or os.path.islink(fn):
            os.remove(fn)
        
# Tar all links
def TarLinks(cmd, ext):
    """Tar all links existing in the current folder
    
    :Call:
        >>> TarLinks()
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get all files
    flist = os.listdir('.')
    # Initialize
    flink = []
    # Loop through files
    for fn in flist:
        # Check if it's a link
        if os.path.islink(fn): flink.append(fn)
    # Exit if no links
    if len(flink) < 2: return
    # Create command
    cmdc = cmd + ['links.'+ext] + flink
    # Run command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete links
    for fn in flink:
        if os.path.islink(fn):
            os.remove(fn)
    
# Tar a folder
def TarDir(cmd, ftar, fdir):
    """Archive a folder and delete the folder
    
    :Call:
        >>> TarDir(cmd, ftar, fdir)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`str`
            File name pattern or list thereof to combine into archive
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Check if the folder exists
    if not os.path.isdir(fdir): return
    # Create command
    cmdc = cmd + [ftar, fdir]
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete the folder
    if os.path.isdir(fdir):
        shutil.rmtree(fdir)
        
# Untar a folder
def Untar(cmd, ftar):
    """Unarchive a tar ball and then delete it
    
    :Call:
        >>> cape.manage.Untar(cmd, ftar)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`str`
            File name pattern or list thereof to combine into archive
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Create command
    cmdc = cmd + [ftar]
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete the folder
    if os.path.isfile(ftar):
        os.remove(ftar)
    



