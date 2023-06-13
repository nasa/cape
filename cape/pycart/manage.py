"""
:mod:`cape.pycart.manage`: Manage pyCart case folders
======================================================

This module is a derivative of the main solution folder management module
:mod:`cape.manage`. It provides Cart3D-specific versions of the three
top-level functions, which each correspond to a primary command-line option.

    =======================   ==================
    Function                  Command-line
    =======================   ==================
    :func:`CleanFolder`       ``--clean``
    :func:`ArchiveFolder`     ``--archive``
    :func:`SkeletonFolder`    ``--skeleton``
    =======================   ==================

The Cart3D-specific versions of these commands use the function
:func:`cape.pycart.options.archiveopts.auto_Archive`, which apply the default settings
appropriate to cape.pycart. Because this module was the first folder management
version created, it does not rely heavily on :mod:`cape.manage`. This module
sets

    .. code-block:: python

        # Subdirectories
        fsub = ['adapt??']

which tells the management functions to also look inside the adaptation
solutions while archiving if they are present.

The ``--unarchive`` command does not require any specific customizations for
Cart3D, and the generic version of :func:`manage.UnarchiveFolder` is
just called directly.

:See also:
    * :mod:`cape.manage`
    * :mod:`cape.pycart.options.Archive`
    * :mod:`cape.options.Archive`
"""

# Standard library
import glob
import os
import shutil
import subprocess as sp

# Local imports
from .options import archiveopts
from .. import manage


# Subdirectories
fsub = ['adapt??']
# Auto archive function
aa = archiveopts.auto_Archive


# Perform in-progress file management after each run
def ManageFilesProgress(opts=None):
    r"""Delete or group files and folders at end of each run
    
    :Call:
        >>> cape.pycart.manage.ManageFilesProgress(opts=None)
    :Inputs:
        *opts*: :class:`Options`
            Options interface for archiving
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    opts = archiveopts.auto_Archive(opts)
    # Perform actions
    manage.ProgressDeleteFiles(opts, fsub=fsub)
    manage.ProgressUpdateFiles(opts, fsub=fsub)
    manage.ProgressDeleteDirs(opts)
    manage.ProgressTarGroups(opts)
    manage.ProgressTarDirs(opts)

    
# Perform pre-archive management
def ManageFilesPre(opts=None):
    r"""Delete or group files and folders before creating archive
    
    :Call:
        >>> ManageFilesPre(opts=None)
    :Inputs:
        *opts*: :class:`Options`
            Options interface for archiving
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    opts = archiveopts.auto_Archive(opts)
    # Perform actions
    manage.PreDeleteFiles(opts, fsub=fsub)
    manage.PreUpdateFiles(opts, fsub=fsub)
    manage.PreDeleteDirs(opts)
    manage.PreTarGroups(opts)
    manage.PreTarDirs(opts)
    

# Perform post-archive management
def ManageFilesPost(opts=None):
    """Delete or group files and folders after creating archive
    
    :Call:
        >>> cape.pycart.manage.ManageFilesPost(opts=None)
    :Inputs:
        *opts*: :class:`cape.pycart.options.Options` | :class:`dict`
            Options interface for archiving
    :Versions:
        *opts*: :class:`Options`
            Options interface for archiving
    """
    # Convert options
    opts = archiveopts.auto_Archive(opts)
    # Perform actions
    manage.PostDeleteFiles(opts, fsub=fsub)
    manage.PostUpdateFiles(opts, fsub=fsub)
    manage.PostDeleteDirs(opts)
    manage.PostTarGroups(opts)
    manage.PostTarDirs(opts)


# Function to compress extra folders
def TarAdapt(opts):
    r"""Replace ``adaptXX`` folders with a tarball except most recent
    
    For example, if there are 3 adaptations in the current folder, the
    following substitutions will be made.
    
        * ``adapt00/`` --> :file:`adapt00.tar`
        * ``adapt01/`` --> :file:`adapt01.tar`
        * ``adapt02/`` --> :file:`adapt02.tar`
        * ``adapt03/``
        
    The most recent adaptation is left untouched, and the old folders
    are deleted.
    
    :Call:
        >>> TarAdapt(opts)
    :Inputs:
        *opts*: :class:`cape.pycart.options.Options`
            Options interface
        *opts.get_TarAdapt()*: {``"restart"``} | ``"full"`` | ``"none"``
            Archiving action to take
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: Version 1.0
        * 2015-01-10 ``@ddalle``: Version 1l1; format as an option
        * 2015-12-02 ``@ddalle``: Version 1.2; Options
    """
    # Check for a BEST/ foldoer.
    if not os.path.islink('BEST'):
        # No adaptation cycles to compress.
        return
    # Check format
    fmt = opts.get_ArchiveFormat()
    # Action to take
    topt = opts.get_TarAdapt()
    # Don't tar if turned off
    if not topt:
        return
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
        cmdu = ['tar', '-uf']
        # Untar command
        cmdx = ['tar', '-xf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdu = ['tar', '-uf']
        cmdx = ['tar', '-xf']
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
        if not os.path.isdir(fdir):
            continue
        # Don't process the folder if in use.
        if quse:
            fadapt_best = fdir
            continue
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
    # Do not process further without adapt00.tar
    if not os.path.isfile('adapt00.tar'):
        return
    # Using adapt00/
    fmesh0 = os.path.join("adapt00", "Mesh.c3d.Info")
    # Special file used for statistics
    if not os.path.isfile(fmesh0):
        # Folder we actually want to keep
        fuse = 'adapt%02i' % imax
        # Time to use for adapt00
        t = os.path.getmtime(fuse) - 10.0
        # Revive the old files
        sp.call(['tar', '-xf', 'adapt00.tar', fmesh0])
        # Set the time to something old
        os.utime('adapt00', (t, t))
        os.utime('adapt00.tar', (t, t))


# Function to undo the above
def ExpandAdapt(opts):
    r"""Expand :file:`adaptXX.tar`` files
    
    :Call:
        >>> ExpandAdapt(fmt="tar")
    :Inputs:
        *opts*: :class:`cape.pycart.options.Options`
            Options interface
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: Version 1.0
        * 2015-01-10 ``@ddalle``: Version 1.1; format as an option
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

        
# Function to tar up visualization checkpoints
def TarViz(opts):
    r"""Move visualization surface and cut plane files to tar balls
    
    This reduces file count by tarring :file:`Components.i.*.plt` and
    :file:`cutPlanes.*.plt`.
    
    :Call:
        >>> TarViz(opts)
    :Inputs:
        *opts*: :class:`cape.pycart.options.Options`
            Options interface
        *opts.get_TarViz()*: ``"restart"`` | {``"full"``} | ``"none"``
            Archiving action to take
        *opts.get_ArchiveFormat()*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-12-18 ``@ddalle``: Version 1.0
        * 2015-01-10 ``@ddalle``: Version 1.1; add format option
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


# Clear old check files.
def ClearCheck(n=1):
    r"""Clear old :file:`check.?????` and :file:`check.??????.td`
    
    :Call:
        >>> ClearCheck(n=1)
    :Inputs:
        *n*: :class:`int`
            Keep the last *n* check points.
    :Versions:
        * 2014-12-31 ``@ddalle``: Version 1.0
        * 2015-01-10 ``@ddalle``: Version 1.1; Added *n* setting
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


# Clear check files created during start/stop running
def ClearCheck_iStart(nkeep=1, istart=0):
    r"""Clear check files that were created since iteration *istart*
    
    :Call:
        >>> ClearCheck_iStart(nkeep=1, istart=0)
    :Inputs:
        *nkeep*: :class:`int`
            Number of most recent check files to keep
        *istart*: :class:`int`
            Do not delete check files prior to iteration *istart*
    :Versions:
        * 2016-03-04 ``@ddalle``: Version 1.0
    """
    # Exit if non-positive
    if nkeep is None or nkeep < 0: return
    # Get the check.* files
    fglob = glob.glob('check.?????')
    fglob.sort()
    # check for a match
    if len(fglob) > nkeep:
        # Loop through the glob except for the last *nkeep* files
        for fc in fglob[:-nkeep]:
            # Check iStart
            try:
                # Iteration number
                i = int(fc.split('.')[1])
            except Exception:
                # Not a valid iteration
                continue
            # Check if it is a recent enough iteration
            if i <= istart: continue
            # Delete the file.
            os.remove(fc)
            # Build the checkDT.????? file nae
            fDT = 'checkDT.%05i' % i
            # Remove it, too.
            if os.path.isfile(fDT): os.remove(fDT)
    # Check the check.*.td files
    fglob = glob.glob('check.???????.td')
    fglob.sort()
    # Check for a match
    if len(fglob) > nkeep:
        # Loop through glob until the last *n* files
        for fc in fglob[:-nkeep]:
            # Check iStart
            try:
                # Iteration number
                i = int(fc.split('.')[1])
            except Exception:
                # Not a valid iteration
                continue
            # Check if it is a recent enough iteration
            if i <= istart: continue
            # Delete the file.
            os.remove(fc)


# Archive folder.
def ArchiveFolder(opts):
    r"""Archive a folder to a backup location and clean up if requested
    
    :Call:
        >>> ArchiveFolder(opts)
    :Inputs:
        *opts*: :class:`cape.pycart.options.Options`
            Options interface of pyCart management interface
    :Versions:
        * 2015-01-11 ``@ddalle``: Version 1.0
    """
    # Restrict options to correct class
    opts = archiveopts.auto_Archive(opts)
    # Get archive type
    ftyp = opts.get_ArchiveType()
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # If no action, do not backup
    if not ftyp or not flfe:
        return
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the case folder
    frun = os.path.join(fgrp, fdir)
    # Reenter case folder
    os.chdir(fdir)
    # Ensure folder exists.
    manage.CreateArchiveFolder(opts)
    manage.CreateArchiveCaseFolder(opts)
    # Get the archive format, extension, and command
    ext  = opts.get_ArchiveExtension()
    # Write date
    manage.write_log_date()
    # Pre-archiving file management
    ManageFilesPre(opts)
    # Perform remote copy
    if ftyp.lower() == "full":
        # Archive entire folder
        manage.ArchiveCaseWhole(opts)
        # Post-archiving file management
        ManageFilesPost(opts)
    else:
        # Partial archive; create folder containing several files
        # Form destination folder name
        # Archive end-of-run files
        manage.ProgressArchiveFiles(opts, fsub=fsub)
        # Create tar balls as appropriate before archiving
        manage.PostTarGroups(opts, frun=frun)
        manage.PostTarDirs(opts, frun=frun)
        # Archive existing tar balls
        manage.ArchiveFiles(opts, ['*.%s'%ext, '*.tar', '*.zip'])
        # After archiving, perform clean-up
        manage.PostDeleteFiles(opts, fsub=fsub)
        manage.PostUpdateFiles(opts, fsub=fsub)
        manage.PostDeleteDirs(opts)

        
# Clean up a folder but don't delete it.
def SkeletonFolder():
    r"""Clean up a folder of everything but the essentials
    
    :Call:
        >>> cape.pycart.manage.SkeletonFolder()
    :Versions:
        * 2015-01-11 ``@ddalle``: Version 1.0
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

    
# Check if an archive already exists
def CheckArchive(ftar):
    r"""Check if an archive already exists
    
    :Call:
        >>> q = cape.pycart.manage.CheckArchive(ftar)
    :Inputs:
        *ftar*: :class:`str`
            Full path to archive folder
    :Outputs:
        *q*: :class:`bool`
            Returns ``True`` if *ftar* exists
    :Versions:
        * 2015-01-11 ``@ddalle``: Version 1.0
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

