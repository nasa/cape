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
    # Loop through adaptXX/ folders.
    for fdir in glob.glob('adapt[0-9][0-9]'):
        # Check for things to skip.
        if fdir == fbest:
            # Target of BEST/; leave alone
            continue
        elif not os.path.isdir(fdir):
            # Not a folder; what?
            continue
        elif (fdir == 'adapt00') and (not os.path.isdir('adapt01')):
            # No BEST folder yet
            continue
        # Tar the folder.
        ierr = sp.call(['tar', '-cf', fdir+'.tar', fdir])
        # Check for errors.
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
        # Untar
        ierr = sp.call(['tar', '-xf', ftar])
        # Check for errors.
        if ierr: continue
        # Remove the tarball.
        os.remove(ftar)
    
