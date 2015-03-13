"""
Module for Semiautomatic Folder Archiving: :mod:`pyCart.tar`
"""

# System interface
import os, shutil
import subprocess as sp

# Simple function to untar a folder
def untar(ftar):
    """Untar an archive
    
    :Call:
        >>> ierr = tar.untar(ftar)
    :Inputs:
        *ftar*: :class:`str`
            Name of the archive to untar
    :Outputs:
        *ierr*: :class:`int`
            Exit code from the `tar` command
    :Versions:
        * 2015-03-07 ``@ddalle``: First version
    """
    # Untar the folder.
    ierr = sp.call(['tar', '-xf', ftar])
    # Output
    return ierr
    
# Function to tar a folder
def tar(ftar, *a):
    """Untar an archive
    
    :Call:
        >>> ierr = tar.tar(ftar, fdir, *a)
    :Inputs:
        *ftar*: :class:`str`
            Name of the archive
        *fdir*: :class:`str`
            Name of the folder to archive, for example
        *a*: :class:`list` (:class:`str`)
            Additional arguments are also passed to the tar command
    :Outputs:
        *ierr*: :class:`int`
            Exit code from the `tar` command
    :Versions:
        * 2015-03-07 ``@ddalle``: First version
    """
    # Untar the folder.
    ierr = sp.call(['tar', '-uf', ftar] + list(a))
    # Output
    return ierr

# Function to leave a folder and archive it.
def chdir_up():
    """Leave a folder, archive it, and delete the folder
    
    :Call:
        >>> tar.chdir_up()
    :Versions:
        * 2015-03-07 ``@ddalle``: First version
    """
    # Get the current folder.
    fpwd = os.path.split(os.getcwd())[-1]
    # Go up.
    os.chdir('..')
    # Name of the file.
    ftar = fpwd + '.tar'
    # Check for the file.
    if os.path.isfile(ftar):
        # Check the date.
        if os.path.getmtime(ftar) < os.path.getmtime(fpwd): return
    # Update or create the archive.
    ierr = tar(ftar, fpwd)
    # Check for failure.
    if ierr: return
    # Delete the folder.
    shutil.rmtree(fpwd)
    
# Function to go into a folder that might be archived.
def chdir_in(fdir):
    """Go into a folder that may be archived
    
    :Call:
        >>> tar.chdir_in(fdir)
    :Inputs:
        *fdir*: :class:`str`
            Name of folder
    :Versions:
        * 2015-03-07 ``@ddalle``: First version
    """
    # Name of the archive.
    ftar = fdir + '.tar'
    # Check for the archive.
    if os.path.isfile(ftar):
        # Check for the folder.
        if os.path.isdir(fdir):
            # Check the dates.
            if os.path.getmtime(fdir) < os.path.getmtime(ftar):
                # Folder needs updating.
                untar(ftar)
        else:
            # No folder.
            untar(ftar)
    # Go into the folder.
    os.chdir(fdir)
    
