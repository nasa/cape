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
def TarAdapt():
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
        >>> pyCart.manage.TarAdapt()
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
    """
    # Check for a BEST/ foldoer.
    if not os.path.islink('BEST'):
        # No adaptation cycles to compress.
        return
    # Get the most recent cycle.
    fbest = os.path.split(os.path.realpath('BEST'))[-1]
    # Initial list of adaptXX/ folders.
    fdirs = glob.glob('adapt[0-9][0-9]')
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
        ierr = sp.call(['tar', '-cf', fdir+'.tar', fdir])
        if ierr: continue
        # Untar the history file.
        ierr = sp.call(['tar', '-xf', fdir+'.tar', fdir+'/history.dat'])
        if ierr: continue
        # Remove the folder.
        shutil.rmtree(fdir)
        
        
# Function to undo the above
def ExpandAdapt():
    """Expand :file:`adaptXX.tar`` files
    
    :Call:
        >>> pyCart.manage.ExpandAdapt()
    :Versions:
        * 2014-11-12 ``@ddalle``: First version
    """
    # Loop through adapt?? tarballs.
    for ftar in glob.glob('adapt[0-9][0-9].tar'):
        # Create the expected folder name.
        fdir = ftar.split('.')[0]
        # Check for that folder.
        if os.path.exists(fdir): continue
        # Status update
        print("%s --> %s" % (fdir+'.tar', fdir))
        # Untar
        ierr = sp.call(['tar', '-xf', ftar])
        # Check for errors.
        if ierr: continue
        # Remove the tarball.
        os.remove(ftar)
        
        
# Function to tar up visualization checkpoints
def TarViz():
    """Add all visualization surface and cut plane TecPlot files to tar balls
    
    This reduces file count by tarring :file:`Components.i.*.plt` and
    :file:`cutPlanes.*.plt`.
    
    :Call:
        >>> pyCart.manage.TarViz()
    :Versions:
        * 2014-12-18 ``@ddalle``: First version
    """
    # Globs and tarballs
    VizGlob = [
        'Components.i.[0-9]*.stats',
        'Components.i.[0-9]*',
        'cutPlanes.[0-9]*']
    VizTar = [
        'Components.i.stats.tar',
        'Components.i.tar',
        'cutPlanes.tar']
    # Loop through the globs.
    for (fglob, ftar) in zip(VizGlob, VizTar):
        # Get the matches
        fnames = glob.glob(fglob+'.dat') + glob.glob(fglob+'.plt')
        # Check for a match to the glob.
        if len(fnames) == 0: continue
        # Status update
        print("  '%s.{dat,plt}' --> '%s'" % (fglob, ftar))
        # Tar surface TecPlot files
        ierr = sp.call(['tar', '-uf', ftar] + fnames)
        if ierr: continue
        # Delete files
        ierr = sp.call(['rm'] + fnames)
        
# Clear old check files.
def ClearCheck():
    """Clear old :file:`check.?????` and :file:`check.??????.td`
    
    :Call:
        >>> pyCart.manage.ClearCheck()
    :Versions:
        * 2014-12-31 ``@ddalle``: First version
    """
    # Get the check.* files.
    fglob = glob.glob('check.?????')
    # Check for a match.
    if len(fglob) > 0:
        # Get the max index.
        imax = max([int(f.split('.')[1]) for f in fglob])
        # Loop through glob.
        for f in fglob:
            # Check the index.
            if int(f.split()[1]) < imax:
                # Remove it.
                os.remove(f)
    # Get the check.* files.
    fglob = glob.glob('check.??????.td')
    # Check for a match.
    if len(fglob) > 0:
        # Get the max index.
        imax = max([int(f.split('.')[1]) for f in fglob])
        # Loop through glob.
        for f in fglob:
            # Check the index.
            if int(f.split()[1]) < imax:
                # Remove it.
                os.remove(f)
    
