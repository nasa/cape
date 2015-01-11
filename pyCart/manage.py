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

# Function to compress extra folders
def TarAdapt(fmt="tar"):
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
        >>> pyCart.manage.TarAdapt(fmt="tar")
    :Inputs:
        *fmt*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
    """
    # Check for a BEST/ foldoer.
    if not os.path.islink('BEST'):
        # No adaptation cycles to compress.
        return
    # Process command header.
    if fmt in ['gzip', 'tgz']:
        # GZip format
        # Tar command
        cmdu = ['tar', '-uzf']
        # Untar command
        cmdx = ['tar', '-xzf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdu = ['tar', '-uJf']
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
    # Check for subsequent iterations. (can tar last folder then)
    qtar = os.path.isfile('history.dat') and os.path.islink('Restart.file')
    # Loop through adaptXX/ folders.
    for fdir in fdirs:
        # Get the adaptation number.
        i = int(fdir[5:])
        # Check if the folder is in use.
        quse = (fdir == fbest) or (i >= imax)
        # Make sure nothing happened to the folder in the meantime.
        if not os.path.isdir(fdir):
            # Not a folder; what?
            continue
        # Don't process the folder if in use.
        if quse and (not qtar):
            # We'll get it later.
            continue
        # Status update
        print("%s --> %s" % (fdir, fdir+'.tar'))
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
        # Check for minimal last adaptation.
        if quse:
            # Untar the history file.
            ierr = sp.call(cmdx + [fdir+ext, fdir+'/history.dat'])
            if ierr: continue
        
        
# Function to undo the above
def ExpandAdapt(fmt="tar"):
    """Expand :file:`adaptXX.tar`` files
    
    :Call:
        >>> pyCart.manage.ExpandAdapt(fmt="tar")
    :Inputs:
        *fmt*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
    """
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
def TarViz(fmt="tar"):
    """Add all visualization surface and cut plane TecPlot files to tar balls
    
    This reduces file count by tarring :file:`Components.i.*.plt` and
    :file:`cutPlanes.*.plt`.
    
    :Call:
        >>> pyCart.manage.TarViz(fmt="tar")
    :Inputs:
        *fmt*: {``"tar"``} | ``"gzip"`` | ``"bz2"``
            Format, can be 'tar', 'gzip', or 'bzip'
    :Versions:
        * 2014-12-18 ``@ddalle``: First version
        * 2015-01-10 ``@ddalle``: Added format as an option
    """
    # Process command header.
    if fmt in ['gzip', 'tgz']:
        # GZip format
        # Tar command
        cmdu = ['tar', '-uzf']
        # File extension
        ext = '.tgz'
    elif fmt in ['bz2', 'bz', 'bzip', 'bzip2', 'tbz']:
        # BZip2 format
        cmdu = ['tar', '-uJf']
        ext = '.tbz'
    else:
        # Just use tar
        cmdu = ['tar', '-uf']
        ext = '.tar'
    # Globs and tarballs
    VizGlob = [
        'Components.i.[0-9]*.stats',
        'Components.i.[0-9]*',
        'cutPlanes.[0-9]*']
    VizTar = [
        'Components.i.stats'+ext,
        'Components.i'+ext,
        'cutPlanes'+ext]
    # Loop through the globs.
    for (fglob, ftar) in zip(VizGlob, VizTar):
        # Get the matches
        fnames = glob.glob(fglob+'.dat') + glob.glob(fglob+'.plt')
        # Check for a match to the glob.
        if len(fnames) == 0: continue
        # Status update
        print("  '%s.{dat,plt}' --> '%s'" % (fglob, ftar))
        # Tar surface TecPlot files
        ierr = sp.call(cmdu + [ftar] + fnames)
        if ierr: continue
        # Delete files
        ierr = sp.call(['rm'] + fnames)
        
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
    fglob = glob.glob('check.?????').sort()
    # Check for a match.
    if len(fglob) > n:
        # Loop through the glob until the last *n* files.
        for f in fglob[:-n]:
            # Remove it.
            os.remove(f)
    # Get the check.* files.
    fglob = glob.glob('check.??????.td').sort()
    # Check for a match.
    if len(fglob) > n:
        # Loop through the glob until the last *n* files.
        for f in fglob[:-n]:
            # Remove it.
            os.remove(f)
    
