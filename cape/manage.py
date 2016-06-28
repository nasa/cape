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

The module provides methods to perform deletions, conditional deletions, and
grouping files into tar or zip archives at multiple stages.  For example,
:func:`PreDeleteFiles` deletes files after a case has been completed and before
generating the archive of the case.  On the other hand, :func:`PostDeleteFiles`
deletes files after creating the archive; the difference is whether or not the
file in question is included in the archive before it is deleted.

Functions such as :func:`ProgressDeleteFiles` deletes files between phases, and
as such it is much more dangerous.  Other methods will only delete or archive
once the case has been marked as PASS by the user.
"""

# File management modules
import os, shutil, glob
# Command-line interface
import subprocess as sp
# Numerics
import numpy as np
# Options module
from .options import Archive

# File tests
def isfile(fname):
    """Handle to test if file **or** link
    
    :Call:
        >>> q = cape.manage.isfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file
    :Outputs:
        *q*: :class:`bool`
            Whether or not file both exists and is either a file or link
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    return os.path.isfile(fname) or os.path.islink(fname)
    
# File is broken link
def isbrokenlink(fname):
    """Handle to test if a file is a broken link
    
    :Call:
        >>> q = cape.manage.isbrokenlink(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of link
    :Outputs:
        *q*: :class:`bool`
            Whether or not it is a link that is broken
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    return os.path.islink(fname) and not os.path.isfile(fname)
    
# Sort files by time
def sortfiles(fglob):
    """Sort a glob of files based on the time of their last edit
    
    :Call:
        >>> fsort = sortfiles(fglob)
    :Inputs:
        *fglob*: :class:`list` (:class:`str`)
            List of file names
    :Outputs:
        *fsort*: :class:`list` (:class:`str`)
            Above listed by :func:`os.path.getmtime`
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Time function
    ft = lambda f: os.path.getmtime(f) if os.path.exists(f) else 0.0
    # Initialize times
    t = np.array([ft(f) for f in fglob])
    # Get the order
    i = np.argsort(t)
    # Return the files in order
    return [fglob[j] for j in i]

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
        *fpat*: :class:`str` | :class:`dict` | :class:`list` (:class:`str`)
            File name pattern or list of file name patterns
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Check type
    if (type(grp) != dict) or (len(grp) != 1):
        # Wront length
        raise ValueError(
            ("Received improper tar group: '%s'\n" % grp) +
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
        raise TypeError("File descriptor '%s' must be str or dict" % f)


# Get list of folders in which to search
def GetSearchDirs(fsub=None, fsort=None):
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
        *fsort*: :class:`function`
            Non-default sorting function
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
    # Sort it
    if fsort is None:
        # Default sorting function
        fdirs = sortfiles(fdirs)
    else:
        # Sort using input
        fdirs = fsort(fdirs)
    # Append working directory (and make sure it's last)
    fdirs.append('.')
    # Output
    return fdirs
    
# Get list of matches, generic
def GetMatches(fname, fsub=None, fkeep=None, ftest=None, n=0, fsort=None):
    """Get matches based on arbitrary rules
    
    :Call:
        >>> fglob = GetMatches(fname, fsub=None, fkeep=None, ftest=None, n=0)
    :Inputs:
        *fname*: :class:`str`
            File name or file name pattern
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Subfolder name of folder name patterns in which to search
        *fkeep*: :class:`list` (:class:`str`)
            List of file names matching a negative file glob
        *ftest*: :class:`func`
            Function to test file type, e.g. :func:`os.path.isdir`
        *n*: :class:`int`
            Default number of files to ignore from the end of the glob
        *fsort*: :class:`function`
            Non-default sorting function
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Process the input
    fname, nkeep = process_ArchiveFile(fname, n=n)
    # Get folders
    fdirs = GetSearchDirs(fsub, fsort=fsort)
    # Initialize result
    fglob = []
    # Loop through folders
    for fdir in fdirs:
        # Construct relative file name glob for this folder
        if fdir == "." or not fdir:
            # Name as is
            fn = fname
        else:
            # Prepend the folder name to the file name glob
            fn = os.path.join(fdir, fname)
        # Apply the glob
        fglobn = glob.glob(fn)
        # Sort it
        if fsort is None:
            # Default sorting function
            fglobn = sortfiles(fglobn)
        else:
            # Sort using input
            fglobn = fsort(fglobn)
        # Check how many files to keep
        if len(fglobn) <= nkeep:
            # Nothing to delete (yet)
            continue
        # Strip last *nkeep* matches
        if nkeep > 0:
            fglobn = fglobn[:-nkeep]
        # Loop through the matches
        for fgn in fglobn:
            # Test if file, folder, etc.
            if ftest is not None and not ftest(fgn):
                # Failed test, e.g. a directory and not a file
                continue
            elif fgn in fglob:
                # Already matched a previous pattern
                continue
            elif fkeep is not None and fgn in fkeep:
                # Match superseded by a negative file glob
                continue
            # Append to the list of matches
            fglob.append(fgn)
    # Output
    return fglob
    
# Get list of matches to a list of globs, generic
def GetMatchesList(flist, fsub=None, ftest=None, n=0):
    """Get matches from a list of file glob descriptors
    
    :Call:
        >>> fglob = GetMatchesList(flist, fsub=None, ftest=None, n=0)
    :Inputs:
        *flist*: :class:`list` (:class:`str` | :class:`dict`)
            List of file names or file name patterns
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Subfolder name of folder name patterns in which to search
        *ftest*: :class:`func`
            Function to test file type, e.g. :func:`os.path.isdir`
        *n*: :class:`int`
            Default number of files to ignore from the end of the glob
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Ensure list
    if type(flist).__name__ not in ['list', 'ndarray']:
        flist = [flist]
    # Initialize negative matches
    fkeep = []
    # Loop through patterns for negative globs
    for fname in flist:
        # Check for negative
        if len(fname) < 2:
            # Not enough characters
            continue
        elif not fname.startswith('!'):
            # Not a negative glob
            continue
        # Get matches for the negative of this pattern
        fgn = GetMatches(fname[1:], fsub=fsub, ftest=ftest, n=0)
        # Append to keep glob
        for fn in fgn:
            # Check if already present
            if fn in fkeep: continue
            # Append to list
            fkeep.append(fn)
    # Initialize
    fglob = []
    # Loop through patterns
    for fname in flist:
        # Get matches for this pattern
        fgn = GetMatches(fname, fsub=fsub, fkeep=fkeep, ftest=ftest, n=n)
        # Append contents to output glob
        for fn in fgn:
            # Check if already present
            if fn in fglob: continue
            # Append to list
            fglob.append(fn)
    # Output
    return fglob
    
# Get file/link matches
def GetFileMatches(fname, fsub=None, n=0):
    """Get list of all files or links matching a list of patterns
    
    :Call:
        >>> fglob = GetFileMatches(fname, fsub=None, n=0)
        >>> fglob = GetFileMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list` (:class:`dict` | :class:`str`)
            List of file name patterns
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname, fsub=fsub, ftest=isfile, n=n)
    # Output
    return fglob
    
# Get link matches
def GetLinkMatches(fname, fsub=None, n=0):
    """Get list of all links matching a list of patterns
    
    :Call:
        >>> fglob = GetLinkMatches(fname, fsub=None, n=0)
        >>> fglob = GetLinkMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list` (:class:`dict` | :class:`str`)
            List of file name patterns
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname, fsub=fsub, ftest=os.path.islink, n=n)
    # Output
    return fglob
                
# Get folder matches
def GetDirMatches(fname, fsub=None, n=0):
    """Get list of all folders matching a list of patterns
    
    :Call:
        >>> fglob = GetDirMatches(fname, fsub=None, n=0)
        >>> fglob = GetDirMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list` (:class:`dict` | :class:`str`)
            List of file name patterns
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
    :Outputs:
        *fglob*: :class:`list` (:class:`str`)
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname, fsub=fsub, ftest=os.path.isdir, n=n)
    # Output
    return fglob

# Function to delete files according to full descriptor
def DeleteFiles(fdel, fsub=None, n=1):
    """Delete files that match a list of glob
    
    The function also searches in any folder matching the directory glob or list
    of directory globs *fsub*.
    
    :Call:
        >>> cape.manage.DeleteFiles_SubDir(fdel, n=1, fsub=None)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder, list of folders, or glob of folders to also search
        *n*: :class:`int`
            Number of files to keep
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get list of matches
    fglob = GetFileMatches(fdel, fsub=fsub, n=n)
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not isfile(fn): continue
        # Delete it.
        print("       Label 0100: rm %s" % fn)
        os.remove(fn)
        
# Function to delete files according to full descriptor
def ArchiveFiles(opts, fname, fsub=None, n=0):
    """Delete files that match a list of glob
    
    The function also searches in any folder matching the directory glob or list
    of directory globs *fsub*.
    
    :Call:
        >>> cape.manage.ArchiveFiles(opts, fname, fsub=None, n=1)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *fname*: :class:`str`
            File name or glob of files to delete
        *fsub*: :class:`str` | :class:`list` (:class:`str`)
            Folder, list of folders, or glob of folders to also search
        *n*: :class:`int`
            Number of files to keep
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get list of matches
    fglob = GetFileMatches(fname, fsub=fsub, n=n)
    
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # If no action, do not backup
    if not flfe: return
    # If not full, do not continue
    if ftyp.lower() == "full": return
    
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the case folder
    frun = os.path.join(fgrp, fdir)
    
    # Setup archive
    CreateArchiveFolder(opts)
    CreateArchiveCaseFolder(opts)
    
    # Loop through matches
    for fsrc in fglob:
        # Destination file
        fto = os.path.join(flfe, frun, fsrc)
        # Status update
        print("  %s --> %s" % (fsrc, fto))
        # Check copy type
        if ':' in flfe:
            # Remote copy the file
            ierr = sp.call([fscp, fsrc, fto])
            if ierr: raise SystemError("Remote copy failed.")
        else:
            # Local copy
            shutil.copy(fsrc, fto)
            
# Archive an entire case as a single tar ball
def ArchiveCaseWhole(opts):
    """Archive an entire run folder
    
    This function must be run from the case folder to be archived
    
    :Call:
        >>> ArchiveCaseWhole(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # Get the archive format, extension, and command
    fmt  = opts.get_ArchiveFormat()
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # If no action, do not backup
    if not flfe: return
    # If not full, do not continue
    if ftyp.lower() != "full": return
    
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    
    # Setup archive
    CreateArchiveFolder(opts)
    CreateArchiveGroupFolder(opts)
    
    # Check if it's remote
    if ':' in flfe:
        # Name of tar file
        ftar = '%s.%s' % (fdir, ext)
        # Split host name
        fhost, fldir = flfe.split(':')
        # Full local destination name (on remote host)
        fltar = os.path.join(fldir, ftar)
        # Full global destination name
        frtar = os.path.join(flfe, ftar)
        # Check if the archive exists
        if sp.call(['ssh', fhost, 'test', '-f', fltar]) == 0:
            print("  Archive exists: %s" % frtar)
            return
        # Status update
        print("  %s --> %s" % (fdir, ftar))
        # Tar the folder locally.
        ierr = sp.call(cmdu, [ftar, fdir])
        if ierr: raise SystemError("Archiving failed.")
        # Status update
        print("  %s --> %s" % (ftar, frtar))
        # Remote copy
        ierr = sp.call([fscp, ftar, frtar])
        if ierr: raise SystemError("Remote copy failed.")
    else:
        # Name of destination
        ftar = os.path.join(flfe, fgrp, '%s.%s'%(fdir, ext))
        # Check if the archive exists
        if os.path.isfile(ftar):
            print("  Archive exists: %s" % ftar)
            return
        # Status update
        print("  %s --> %s" % (fdir, ftar))
        # Tar the folder.
        ierr = sp.call(cmdu + [ftar, fdir])
        if ierr: raise SystemError("Archiving failed.")
        
    # Return to folder
    os.chdir(fdir)

    
# Function to delete folders according to full descriptor
def DeleteDirs(fdel, fsub=None, n=1):
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
        *n*: :class:`int`
            Number of folders to keep at end of glob
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Get list of matches
    fglob = GetDirMatches(fdel, fsub=fsub, n=n)
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not os.path.isdir(fn): continue
        # Delete it
        print("      Label 0101: rm -r %s" % fn)
        shutil.rmtree(fn)
        
# Archive groups
def TarGroup(cmd, ftar, fname, n=0, clean=False):
    """Archive a group of files and delete the files
    
    Only the present folder will searched for file name matches
    
    :Call:
        >>> TarGroup(cmd, ftar, fname, clean=False)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`list` (:class:`dict` | :class:`str`)
            File name pattern or list thereof to combine into archive
        *clean*: :class:`bool`
            Whether or not to clean up after archiving
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
        * 2016-03-14 ``@ddalle``: Generic version
    """
    # Check input
    if type(cmd).__name__ != 'list':
        raise TypeError("Input command must be a list of one or two strings")
    # Get list of files
    fglob = GetFileMatches(fname, n=n)
    # Get extension
    ext = ftar.split('.')[-1]
    # Make sure not to tar any tar balls
    fglob = [f for f in fglob if not f.endswith(ext)]
    # Exit if not matches
    if len(fglob) < 2: return
    # Create command
    cmdc = cmd + [ftar] + fglob
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Check clean-up flag
    if not clean: return
    # Delete matches
    for fn in fglob:
        if isfile(fn):
            os.remove(fn)
        
# Tar all links
def TarLinks(cmd, ext, clean=True):
    """Tar all links existing in the current folder
    
    :Call:
        >>> TarLinks(cmd, ext, clean=True)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *clean*: :class:`bool`
            Whether or not to clean up after archiving
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
    # Form file name for tar ball
    ftar = 'links.' + ext
    # Remove *ftar* from this list to avoid recursions
    fglob = [f for f in fglob if f != ftar]
    # Create command
    cmdc = cmd + [ftar] + flink
    # Run command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete links
    for fn in flink:
        if os.path.islink(fn):
            os.remove(fn)
    
# Tar a folder
def TarDir(cmd, ftar, fdir, clean=True):
    """Archive a folder and delete the folder
    
    :Call:
        >>> TarDir(cmd, ftar, fdir)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fdir*: :class:`str`
            Name of folder
        *clean*: :class:`bool`
            Whether or not to delete folder afterwards
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
    # Check for clean flag
    if not clean: return
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
    
# ----------------------------

# ----------------------------
# PRE-ARCHIVE ACTION FUNCTIONS
# ----------------------------
# Function to pre-delete files
def PreDeleteFiles(opts, fsub=None, aa=None):
    """Delete files that match a list of file name patterns
    
    :Call:
        >>> PreDeleteFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0)
    
# Function to pre-delete dirs
def PreDeleteDirs(opts, fsub=None, aa=None):
    """Delete folders that match a list of file name patterns before archiving
    
    :Call:
        >>> PreDeleteDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0)

# Function to pre-update files
def PreUpdateFiles(opts, fsub=None, aa=None):
    """Delete files that match a list, keeping the most recent file by default
    
    :Call:
        >>> PreUpdateFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1)
    
# Function to pre-tar files
def PreTarGroups(opts, fsub=None, aa=None):
    """Tar file/folder groups
    
    The files are deleted after the tar ball is created 
    
    :Call:
        >>> PreTarGroups(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchivePreTarGroups()
    # Exit if necessary
    if fgrps is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for grp in fgrps:
        # Process the group dictionary
        fgrp, fname = process_ArchiveGroup(grp)
        # Archive file name
        ftar = '%s.%s' % (fgrp, ext)
        # Archive
        TarGroup(cmdu, ftar, fname, n=0, clean=True)

# Function to pre-tar dirs
def PreTarDirs(opts, fsub=None, aa=None):
    """Tar folders before archiving
    
    The folders are deleted after the tar ball is created
    
    :Call:
        >>> PreTarDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fopt = opts.get_ArchivePreTarDirs()
    # Exit if necessary
    if fopt is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Get list of matching directories
    fdirs = GetDirMatches(fopt, fsub=fsub, n=1)
    # Loop through directories
    for fdir in fdirs:
        # Archive file name
        ftar = '%s.%s' % (fdir, ext)
        # Command t
        TarDir(cmdu, ftar, fdir, clean=True)
    
# ----------------------------

# -----------------------------
# POST-ARCHIVE ACTION FUNCTIONS
# -----------------------------
# Function to post-delete files
def PostDeleteFiles(opts, fsub=None, aa=None):
    """Delete files that match a list of file name patterns
    
    :Call:
        >>> PostDeleteFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0)
    
# Function to post-delete dirs
def PostDeleteDirs(opts, fsub=None, aa=None):
    """Delete folders that match a list of file name patterns before archiving
    
    :Call:
        >>> PostDeleteDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0)

# Function to post-update files
def PostUpdateFiles(opts, fsub=None, aa=None):
    """Delete files that match a list, keeping the most recent file by default
    
    :Call:
        >>> PreUpdateFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1)
    
# Function to post-tar files
def PostTarGroups(opts, fsub=None, aa=None, frun=None):
    """Tar file/folder groups
    
    The files are not deleted after the tar ball is created 
    
    :Call:
        >>> PostTarGroups(opts, fsub=None, aa=None, frun=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
        *frun*: :class:`str`
            Case folder name
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchivePostTarGroups()
    # Exit if necessary
    if fgrps is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for grp in fgrps:
        # Process the group dictionary
        fgrp, fname = process_ArchiveGroup(grp)
        # Archive file name
        if (ftype.lower() != "full") and (':' not in flfe):
            # Local tar command; create Tar in place rather than copying it
            ftar = os.path.join(flfe, frun, '%s.%s' * (fgrp,ext))
        else:
            # Otherwise, create the tar ball in this folder
            ftar = '%s.%s' % (fgrp, ext)
        # Archive
        TarGroup(cmdu, ftar, fname, n=0, clean=False)

# Function to post-tar dirs
def PostTarDirs(opts, fsub=None, aa=None, frun=None):
    """Tar folders after archiving
    
    The folders are not deleted after the tar ball is created
    
    :Call:
        >>> PostTarDirs(opts, fsub=None, aa=None, frun=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
        *frun*: :class:`str`
            Name of case folder
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get remote copy destination
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get options
    fopt = opts.get_ArchivePostTarDirs()
    # Exit if necessary
    if fopt is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Get list of matching directories
    fdirs = GetDirMatches(fopt, fsub=fsub, n=1)
    # Loop through directories
    for fdir in fdirs:
        # Archive file name
        if (ftype.lower() != "full") and (':' not in flfe):
            # Local tar command; create Tar in place rather than copying it
            ftar = os.path.join(flfe, frun, '%s.%s' * (fdir,ext))
        else:
            # Otherwise, create the tar ball in this folder
            ftar = '%s.%s' % (fdir, ext)
        # Perform grouping/compression
        TarDir(cmdu, ftar, fdir, clean=False)
    
# ----------------------------

# -------------------------
# PROGRESS ACTION FUNCTIONS
# -------------------------
# Function for in-progress file deletion
def ProgressDeleteFiles(opts, fsub=None, aa=None):
    """Delete files that match a list of file name patterns
    
    :Call:
        >>> ProgressDeleteFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0)
    
# Function for in-progress file archiving
def ProgressArchiveFiles(opts, fsub=None, aa=None):
    """Delete files that match a list of file name patterns
    
    :Call:
        >>> ProgressDeleteFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fglob = opts.get_ArchiveProgressArchiveFiles()
    # Exit if necessary
    if fglob is None: return
    # Copy
    ArchiveFiles(opts, fglob, fsub=fsub, n=0)
    
# Function for in-progress folder deletion
def ProgressDeleteDirs(opts, fsub=None, aa=None):
    """Delete folders that match a list of file name patterns before archiving
    
    :Call:
        >>> ProgressDeleteDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0)

# Function for in-progress file updates
def ProgressUpdateFiles(opts, fsub=None, aa=None):
    """Delete files that match a list, keeping the most recent file by default
    
    :Call:
        >>> ProgressUpdateFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1)
    
# Function to tar groups in progress
def ProgressTarGroups(opts, fsub=None, aa=None):
    """Tar file/folder groups after each run
    
    The files are deleted after the tar ball is created 
    
    :Call:
        >>> ProgressTarGroups(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchiveProgressTarGroups()
    # Exit if necessary
    if fgrps is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # These have to be updatable
    if ext in ['tbz2', 'tgz']:
        # Convert to tar
        cmdu = ['tar', '-uf']
        ext  = 'tar'
    # Loop through groups
    for grp in fgrps:
        # Process the group dictionary
        fgrp, fname = process_ArchiveGroup(grp)
        # Archive file name
        ftar = '%s.%s' % (fgrp, ext)
        # Archive
        TarGroup(cmdu, ftar, fname, n=0, clean=True)

# Function for in-progress folder compression
def ProgressTarDirs(opts, fsub=None, aa=None):
    """Tar folders after each run
    
    The folders are deleted after the tar ball is created
    
    :Call:
        >>> ProgressTarDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list` (:class:`str`) | :class:`str`
            List of sub-directory globs in which to search
        *aa*: :class:`function`
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Convert options
    if type(aa).__name__ == "function":
        opts = aa(opts)
    # Get options
    fopt = opts.get_ArchiveProgressTarDirs()
    # Exit if necessary
    if fopt is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Get list of matching directories
    fdirs = GetDirMatches(fopt, fsub=fsub, n=1)
    # Loop through directories
    for fdir in fdirs:
        # Archive file name
        ftar = '%s.%s' % (fdir, ext)
        # Command t
        TarDir(cmdu, ftar, fdir, clean=False)

# -------------------------

# -------------
# ARCHIVE SETUP
# -------------
# Function to ensure the destination folder exists
def CreateArchiveFolder(opts):
    """Create the folder that will contain the archive, if necessary
    
    :Call:
        >>> CreateArchiveFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # If no action, do not backup
    if not flfe: return
    
    # Ensure folder exists.
    if ':' in flfe:
        # Split off host name
        fhost, fldir = flfe.split(':')
        # Check remotely.
        if sp.call(['ssh', fhost, 'test', '-d', fldir]) != 0:
            # Create it.
            sp.call(['ssh', fhost, 'mkdir', fldir])
    else:
        # Test locally.
        if not os.path.isdir(flfe):
            # Create it.
            opts.mkdir(flfe)

# Create archive group folders
def CreateArchiveCaseFolder(opts):
    """Create the group and run folders in the archive, as appropriate
    
    This function must be run from within the folder that is to be archived
    
    :Call:
        >>> CreateArchiveCaseFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # If no action, do not backup
    if not flfe: return
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the case folder
    frun = os.path.join(fgrp, fdir)
    # Return to the folder
    
    # Ensure group folder exists.
    if ':' in flfe:
        # Split off host name
        fhost, fldir = flfe.split(':')
        # Remote group and case addresses
        flgrp = os.path.join(fldir, fgrp)
        flrun = os.path.join(fldir, frun)
        # Check group folder remotely
        if sp.call(['ssh', fhost, 'test', '-d', flgrp]) != 0:
            # Create it.
            sp.call(['ssh', fhost, 'mkdir', flgrp])
        # Check run folder remotely
        if (ftyp!="full") and sp.call(['ssh',fhost,'test','-d',flrun]) != 0:
            # Create it.
            sp.call(['ssh', fhost, 'mkdir', flrun])
    else:
        # Group and case addresses
        flgrp = os.path.join(flfe, fgrp)
        flrun = os.path.join(flfe, frun)
        # Test locally.
        if not os.path.isdir(flgrp):
            # Create it.
            opts.mkdir(flgrp)
        # Test for run folder
        if (ftyp!="full") and not os.path.isdir(flrun):
            # Create it.
            opts.mkdir(flrun)

# Create archive group folders
def CreateArchiveGroupFolder(opts):
    """Create the group folder in the archive, as appropriate
    
    This function must be run from within the folder that is to be archived
    
    :Call:
        >>> CreateArchiveGroupFolder(opts)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: First version
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # If no action, do not backup
    if not flfe: return
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Return to current folder
    os.chdir(fdir)
    
    # Ensure group folder exists.
    if ':' in flfe:
        # Split off host name
        fhost, fldir = flfe.split(':')
        # Remote group and case addresses
        flgrp = os.path.join(fldir, fgrp)
        # Check group folder remotely
        if sp.call(['ssh', fhost, 'test', '-d', flgrp]) != 0:
            # Create it.
            sp.call(['ssh', fhost, 'mkdir', flgrp])
    else:
        # Remote group and case addresses
        flgrp = os.path.join(flfe, fgrp)
        # Test locally.
        if not os.path.isdir(flgrp):
            # Create it.
            opts.mkdir(flgrp)
    
# -------------

