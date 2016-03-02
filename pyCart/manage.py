"""
Manage Run Directory Folders: :mod:`pyCart.manage`
==================================================

This module contains functions to manage files and folders, especially file
count, for run directories.  The most important usage is to compress
``adaptXX/`` folders except for the most recent folder.

However, it can also be used to backup a run directory, expand tar balls, or
delete extra material.

:Versions:
    * 2014-11-12 ``@ddalle``: Starter version
"""

# File management modules
import os, shutil, glob
# Command-line interface
import subprocess as sp
# Options module
from .options import Archive
# Basis module
import cape.manage

# Subdirectories
fsub = ['adapt??']

# Delete files matching a glob
def DeleteFiles(fdel, n=0):
    """Delete files that match a file name pattern
    
    The function also searches within the `adapt??` folders
    
    :Call:
        >>> pyCart.manage.DeleteFiles(fdel, n=0)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *n*: :class:`int`
            Default number of files to keep
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Process file name and keep number
    fname, nkeep = cape.manage.process_ArchiveFile(fdel, n=n)
    # Call function from CAPE
    cape.manage.DeleteFiles_SubDir(fname, n=nkeep, fsub=fsub)
    
# Delete folders according to a glob
def DeleteDirs(fdel, n=0):
    """Delete folders that match a pattern
    
    :Call:
        >>> pyCart.manage.DeleteDirs(fdel, n=0)
    :Inputs:
        *fdel*: :class:`str`
            Folder name or patter of folder names to delete
        *n*: :class:`int`
            Number of files to keep
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Process file name and keep number
    fname, nkeep = cape.manage.process_ArchiveFile(fdel, n=n)
    # Call function from CAPE
    cape.manage.DeleteDirs_SubDir(fname, n=nkeep)
# def DeleteDirs

# Function to delete files
def PreDeleteFiles(opts=None):
    """Delete files according to *PreDeleteFiles* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PreDeleteFiles(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePreDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteFiles(fpat, n=0)

# Function to delete files afterwards
def PostDeleteFiles(opts=None):
    """Delete files according to *PostDeleteFiles* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PostDeleteFiles(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePostDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteFiles(fpat, n=0)

# Function to delete older files before archiving
def PreUpdateFiles(opts=None):
    """Delete files according to *PreUpdateFiles* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PreUpdateFiles(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePreUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteFiles(fpat, n=1)

# Function to delete older files after archiving
def PostUpdateFiles(opts=None):
    """Delete files according to *PostUpdateFiles* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PreUpdateFiles(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePostUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteFiles(fpat, n=1)

# Function to delete folders
def PreDeleteDirs(opts=None):
    """Delete folders according to *PreDeleteDirs* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PreDeleteDirs(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePreDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteDirs(fpat, n=0)

# Function to delete folders
def PostDeleteDirs(opts=None):
    """Delete folders according to *PreDeleteDirs* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PostDeleteDirs(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get files
    fdel = opts.get_ArchivePostDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Delete
    for fpat in fdel:
        DeleteDirs(fpat, n=0)

# Function to group before archiving
def PreArchiveGroups(opts=None):
    """Archive file groups from *PreArchiveGroups* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PreArchiveGroups(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`pyCart.options.Archive.Archive`
            Archive options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get option
    fgrps = opts.get_ArchivePreArchiveGroups()
    # Exit if necessary
    if fgrps is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for grp in fgrps:
        # Process the dict
        fgrp, fname = cape.manage.process_ArchiveGroup(grp)
        # File name
        ftar = '%s.%s' % (fgrp, ext)
        # Archive
        cape.manage.TarFileGroup(cmdu, ftar, fname)
        
# Function to group after archiving
def PostArchiveGroups(opts=None):
    """Archive file groups from *PostArchiveGroups* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PostArchiveGroups(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`pyCart.options.Archive.Archive`
            Archive options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get option
    fgrps = opts.get_ArchivePostArchiveGroups()
    # Exit if necessary
    if fgrps is None: return
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for grp in fgrps:
        # Process the dict
        fgrp, fname = cape.manage.process_ArchiveGroup(grp)
        # File name
        ftar = '%s.%s' % (fgrp, ext)
        # Archive
        cape.manage.TarFileGroup(cmdu, ftar, fname)
        
# Function to tar folders before archiving
def PreArchiveDirs(opts=None):
    """Archive folders from *PreArchiveDirs* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PostArchiveDirs(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`pyCart.options.Archive.Archive`
            Archive options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get option
    fdirs = opts.get_ArchivePreArchiveDirs()
    # Exit if necessary
    if fdirs is None: return
    # get forma,t command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for fpat in fdirs:
        # Number of files not to tar
        fname, nkeep = cape.manage.process_ArchiveFile(fpat, n=1)
        # Get matches, e.g "adapt??" --> ["adapt00", "adapt01", ...]
        ldir = cape.manage.GetDirMatches(fname)
        # Trim last *n* directories, which are kept
        if nkeep == 0:
            # Delete all files
            Ldir = ldir
        elif len(ldir) <= nkeep:
            # Keep all files
            continue
        else:
            # Strip last *n* files from list of deletions
            Ldir = ldir[:-nkeep]
        # Loop through directories
        for fdir in Ldir:
            # File name
            ftar = '%s.%s' % (fdir, ext)
            # Archive
            cape.mamage.TarDir(cmdu, ftar, fdir)

# Function to tar folders after archiving
def PostArchiveDirs(opts=None):
    """Archive folders from *PostArchiveDirs* setting in *Archive* section
    
    :Call:
        >>> pyCart.manage.PostArchiveDirs(opts=None)
    :Inputs:
        *opts*: ``None`` | :class:`pyCart.options.Archive.Archive`
            Archive options interface
    :Versions:
        * 2016-03-01 ``@ddalle``: First version
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Get option
    fdirs = opts.get_ArchivePostArchiveDirs()
    # Exit if necessary
    if fdirs is None: return
    # get forma,t command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Loop through groups
    for fpat in fdirs:
        # Number of files not to tar
        fname, nkeep = cape.manage.process_ArchiveFile(fpat, n=1)
        # Get matches, e.g "adapt??" --> ["adapt00", "adapt01", ...]
        ldir = cape.manage.GetDirMatches(fname)
        # Trim last *n* directories, which are kept
        if nkeep == 0:
            # Delete all files
            Ldir = ldir
        elif len(ldir) <= nkeep:
            # Keep all files
            continue
        else:
            # Strip last *n* files from list of deletions
            Ldir = ldir[:-nkeep]
        # Loop through directories
        for fdir in Ldir:
            # File name
            ftar = '%s.%s' % (fdir, ext)
            # Archive
            cape.manage.TarDir(cmdu, ftar, fdir)
# def PostArchiveDirs


# Function to compress extra folders
def TarAdapt(opts):
    """Replace ``adaptXX`` folders with a tarball except for most recent
    
    For example, if there are 3 adaptations in the current folder, the
    following substitutions will be made.
    
        * ``adapt00/`` --> :file:`adapt00.tar`
        * ``adapt01/`` --> :file:`adapt01.tar`
        * ``adapt02/`` --> :file:`adapt02.tar`
        * ``adapt03/``
        
    The most recent adaptation is left untouched, and the old folders are
    deleted.
    
    :Call:
        >>> pyCart.manage.TarAdapt(opts)
    :Inputs:
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *opts.get_TarAdapt()*: {``"restart"``} | ``"full"`` | ``"none"``
            Archiving action to take
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
        * 2015-12-02 ``@ddalle``: Options
    """
    # Check for a BEST/ foldoer.
    if not os.path.islink('BEST'):
        # No adaptation cycles to compress.
        return
    # Check format
    fmt = opts.get_ArchiveFormat()
    # Action to take
    topt = opts.get_TarAdapt()
    # Process option
    if topt is None:
        # Lazy no-archive
        return
    elif type(topt).__name__ not in ['str', 'unicode']:
        # Not a string
        raise TypeError(
            '"TarAdapt" option should be in ["none", "full", "restart"')
    elif topt.lower() == "none":
        # Do nothing
        return
    elif topt.lower() not in ["full", "restart"]:
        # Unrecognized option
        raise TypeError(
            '"TarAdapt" option should be in ["none", "full", "restart"')
    # Process command header.
    if fmt in ['gzip', 'tgz']:
        # GZip format
        # Tar command
        cmdu = ['tar', '-czf']
        # Untar command
        cmdx = ['tar', '-xzf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdu = ['tar', '-cJf']
        cmdx = ['tar', '-xJf']
        ext = '.tbz'
    else:
        # Just use tar
        cmdu = ['tar', '-uf']
        cmdx = ['tar', '-xf']
        ext = '.tar'
    # Get the most recent cycle.
    fbest = os.path.split(os.path.realpath('BEST'))[-1]
    # Initial list of adaptXX/ folders.
    fdirs = glob.glob('adapt[0-9][0-9]')
    # Quit if none were found.
    if len(fdirs) == 0: return
    # Max adaptation
    imax = max([int(fdir[5:]) for fdir in fdirs])
    # Loop through adaptXX/ folders.
    for fdir in fdirs:
        # Get the adaptation number.
        i = int(fdir[5:])
        # Check if the folder is in use.
        quse = (fdir == fbest) or (i >= imax)
        # Make sure nothing happened to the folder in the meantime.
        if not os.path.isdir(fdir): continue
        # Don't process the folder if in use.
        if quse: continue
        # Status update
        print("%s --> %s" % (fdir, fdir+'.tar'))
        # Check cleanup option
        if topt.lower() in ["restart", "hist", "skeleton"]:
            # Remove check files before tarring.
            for f in glob.glob(os.path.join(fdir, 'check*')):
                os.remove(f)
            # Remove mesh files before tarring.
            for f in glob.glob(os.path.join(fdir, 'Mesh*.c3d')):
                os.remove(f)
        # Tar the folder.
        ierr = sp.call(cmdu + [fdir+ext, fdir])
        if ierr: continue
        # Remove the folder.
        shutil.rmtree(fdir)
# def TarAdapt
        
# Function to undo the above
def ExpandAdapt(opts):
    """Expand :file:`adaptXX.tar`` files
    
    :Call:
        >>> pyCart.manage.ExpandAdapt(fmt="tar")
    :Inputs:
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
    """
    # Get format
    fmt = opts.get_ArchiveFormat()
    # Process command header.
    if fmt in ['gzip', 'tgz']:
        # GZip format
        # Untar command
        cmdx = ['tar', '-xzf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdx = ['tar', '-xJf']
        ext = '.tbz'
    else:
        # Just use tar
        cmdx = ['tar', '-xf']
        ext = '.tar'
    # Loop through adapt?? tarballs.
    for ftar in glob.glob('adapt[0-9][0-9]'+ext):
        # Create the expected folder name.
        fdir = ftar.split('.')[0]
        # Check for that folder.
        if os.path.exists(fdir): continue
        # Status update
        print("%s --> %s" % (ftar, fdir))
        # Untar
        ierr = sp.call(cmdx + [ftar])
        # Check for errors.
        if ierr: continue
        # Remove the tarball.
        os.remove(ftar)
# def ExpandAdapt
        
# Function to tar up visualization checkpoints
def TarViz(opts):
    """Add all visualization surface and cut plane TecPlot files to tar balls
    
    This reduces file count by tarring :file:`Components.i.*.plt` and
    :file:`cutPlanes.*.plt`.
    
    :Call:
        >>> pyCart.manage.TarViz(opts)
    :Inputs:
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *opts.get_TarViz()*: ``"restart"`` | {``"full"``} | ``"none"``
            Archiving action to take
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-12-18 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
    """
    # Check format
    fmt = opts.get_ArchiveFormat()
    # Action to take
    topt = opts.get_TarAdapt()
    # Process option
    if topt is None:
        # Lazy no-archive
        return
    elif type(topt).__name__ not in ['str', 'unicode']:
        # Not a string
        raise TypeError(
            '"TarViz" option should be in ["none", "full", "restart"')
    elif topt.lower() == "none":
        # Do nothing
        return
    elif topt.lower() not in ["full", "restart"]:
        # Unrecognized option
        raise TypeError(
            '"TarViz" option should be in ["none", "full", "restart"')
    # Process command header.
    if fmt in ['gzip', 'tgz']:
        # GZip format
        # Tar command
        cmdu = ['tar', '-czf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdu = ['tar', '-cJf']
        ext = '.tbz'
    else:
        # Just use tar
        cmdu = ['tar', '-uf']
        ext = '.tar'
    # Globs and tarballs
    VizGlob = [
        'Components.i.[0-9]*.stats',
        'Components.i.[0-9]*',
        'cutPlanes.[0-9]*',
        'pointSensors.[0-9]*',
        'lineSensors.[0-9]*']
    VizTar = [
        'Components.i.stats'+ext,
        'Components.i'+ext,
        'cutPlanes'+ext,
        'pointSensors'+ext,
        'lineSensors'+ext]
    # Loop through the globs.
    for (fglob, ftar) in zip(VizGlob, VizTar):
        # Get the matches
        fnames = glob.glob(fglob+'.dat') + glob.glob(fglob+'.plt')
        fnames.sort()
        # Check for a match to the glob.
        if len(fnames) == 0: continue
        # Status update
        print("  '%s.{dat,plt}' --> '%s'" % (fglob, ftar))
        # Create archive
        ierr = sp.call(cmdu + [ftar] + fnames)
        if ierr: continue
        # Delete files
        ierr = sp.call(['rm'] + fnames[:-1])
# def TarViz

# Clear old check files.
def ClearCheck(n=1):
    """Clear old :file:`check.?????` and :file:`check.??????.td`
    
    :Call:
        >>> pyCart.manage.ClearCheck(n=1)
    :Inputs:
        *n*: :class:`int`
            Keep the last *n* check points.
    :Versions:
        * 2014-12-31 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added *n* setting
    """
    # Exit if *n* is not positive.
    if n <= 0: return
    # Get the check.* files... in order.
    fglob = glob.glob('check.?????')
    fglob.sort()
    # Check for a match.
    if len(fglob) > n:
        # Loop through the glob until the last *n* files.
        for f in fglob[:-n]:
            # Remove it.
            os.remove(f)
            # Build the checkDT.????? file name
            fDT = 'checkDT.' + f.split('.')[1]
            # Delete that file if it exists.
            if os.path.isfile(fDT): os.remove(fDT)
    # Get the check.* files.
    fglob = glob.glob('check.??????.td')
    fglob.sort()
    # Check for a match.
    if len(fglob) > n:
        # Loop through the glob until the last *n* files.
        for f in fglob[:-n]:
            # Remove it.
            os.remove(f)
# def ClearCheck

# Archive folder.
def ArchiveFolder(opts):
    """
    Archive a folder to a backup location and clean up the folder if requested
    
    :Call:
        >>> pyCart.manage.ArchiveFolder(opts)
    :Inputs:
        *opts*: :class:`pyCart.options.Options`
            Options interface of pyCart management interface
    :Versions:
        * 2015-01-11 ``@ddalle``: First version
    """
    # Get clean up action command.
    farch = opts.get_ArchiveAction()
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # If no action, do not backup
    if not farch or not flfe: return
    
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
            
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    
    # Ensure group folder exists.
    if ':' in flfe:
        # Remote group address.
        flgrp = os.path.join(fldir, fgrp)
        # Check remotely.
        if sp.call(['ssh', fhost, 'test', '-d', flgrp]) != 0:
            # Create it.
            sp.call(['ssh', fhost, 'mkdir', flgrp])
    else:
        # Test locally.
        if not os.path.isdir(os.path.join(flfe, fgrp)):
            # Create it.
            os.mkdir(os.path.join(flfe, fgrp), 0750)
    
    # Get the archive format, extension, and command
    fmt  = opts.get_ArchiveFormat()
    cmdu = opts.get_ArchiveCmd()
    ext  = "." + opts.get_ArchiveExtension()
    # Form the destination file name.
    ftar = os.path.join(flfe, fgrp, fdir+ext)
    # Check if it exists.
    if CheckArchive(ftar):
        print("  Archive exists: %s" % ftar)
        return
    # Go to the folder.
    os.chdir(fdir)
    # Delete files that need to go first.
    PreDeleteFiles(opts)
    PreUpdateFiles(opts)
    PreDeleteDirs(opts)
    # Pre-processing groupings if necessary
    PreArchiveGroups(opts)
    PreArchiveDirs(opts)
    # Go back up to group folder.
    os.chdir('..')
    # Check if it's a remote copy.
    if ':' in ftar:
        # Remote copy; get the command
        fcmd = opts.get_RemoteCopy()
        # Status update.
        print("  %s  -->  %s" % (fdir, fdir+ext))
        # Tar the folder locally.
        ierr = sp.call(cmdu + [fdir+ext, fdir])
        if ierr: raise SystemError("Archiving failed.")
        # Status update.
        print("  %s  -->  %s" % (fdir+ext, ftar))
        # Remote copy the archive.
        ierr = sp.call([fcmd, fdir+ext, ftar])
        if ierr: raise SystemError("Remote copy failed.")
        # Clean up.
        os.remove(fdir+ext)
    else:
        # Local destination
        print("  %s  -->  %s" % (fdir, ftar))
        # Tar directly to archive.
        ierr = sp.call(cmdu + [ftar, fdir])
        if ierr: raise SystemError("Archiving failed.")
    # Go back into the folder
    os.chdir(fdir)
    # Check for further clean up.
    PostDeleteFiles(opts)
    PostUpdateFiles(opts)
    PostDeleteDirs(opts)
    # Post-archiving groups
    PostArchiveGroups(opts)
    PostArchiveDirs(opts)
# def ArchiveFolder
        
# Clean up a folder but don't delete it.
def SkeletonFolder():
    """Clean up a folder of everything but the essentials
    
    :Call:
        >>> pyCart.manage.SkeletonFolder()
    :Versions:
        * 2015-01-11 ``@ddalle``: First version
    """
    # Get list of adapt?? folders.
    fglob = glob.glob('adapt??')
    fglob.sort()
    # Delete adapt folders except for the last one.
    for f in fglob[:-1]: shutil.rmtree(f)
    # Clear checkpoint files.
    fglob = glob.glob('check*')
    for f in fglob: os.remove(f)
    # Remove *.tar files
    for f in glob.glob('*.tar'): os.remove(f)
    # Remove *.tgz files
    for f in glob.glob('*.tgz'): os.remove(f)
    # Remove *.tbz files
    for f in glob.glob('*.tbz'): os.remove(f)
    # Remove Mesh.* files
    for f in glob.glob('Mesh.*'): os.remove(f)
# def SkeletonFolder
    
# Check if an archive already exists
def CheckArchive(ftar):
    """Check if an archive already exists
    
    :Call:
        >>> q = pyCart.manage.CheckArchive(ftar)
    :Inputs:
        *ftar*: :class:`str`
            Full path to archive folder
    :Outputs:
        *q*: :class:`bool`
            Returns ``True`` if *ftar* exists
    :Versions:
        * 2015-01-11 ``@ddalle``: First version
    """
    # Check for a remote folder.
    if ':' in ftar:
        # Remote
        fssh, fdir = ftar.split(':')
        # Will return an error code if file does not exist.
        ierr = sp.call(['ssh', fssh, 'test', '-f', fdir])
        # Process it.
        return ierr == 0
    else:
        # Local
        return os.path.isfile(ftar)
# def CheckArchive

