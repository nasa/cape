#!/usr/bin/env python
"""
Create a release tar ball for Cape

:Versions:
    * 2015-10-29 ``@ddalle``: Started
"""

# System interface
import os, shutil, fnmatch
import subprocess as sp
# The module itself
import cape, cape.argread

# Permissions to use for new folders
dmask = 0o755

# Make function
def MakeRelease(ver):
    # Go to home directory.
    fpwd = os.getcwd()
    os.chdir(cape.CapeFolder)
    os.chdir('..')
    # Folder name
    fdir = 'cape_%s' % ver
    # Make the release folder.
    if os.path.isdir(fdir):
        raise SystemError("Folder '%s' already exists" % fdir)
    os.mkdir(fdir, 0o755)
    
    # Files to ignore for modules
    modignore = ['*.py[oc]']
    # Copy the modules.
    CopyTree('cape',   fdir, modignore, depth=2)
    CopyTree('pyCart', fdir, modignore, depth=2)
    CopyTree('pyFun',  fdir, modignore, depth=2)
    
    # Other settings
    CopyTree('bin',         fdir, modignore, depth=2)
    CopyTree('settings',    fdir, modignore, depth=2)
    CopyTree('modulefiles', fdir, modignore, depth=2)
    
    # Installation files
    shutil.copy('Makefile',   fdir)
    shutil.copy('config.cfg', fdir)
    
    # Files to ignore for documentation
    docignore = ['']
    # Documentation folder to copy
    fdoc = os.path.join('doc', '_build', 'html')
    # Copy the documentation
    CopyTree(fdoc, os.path.join(fdir, 'doc'), docignore, depth=16)
    
    # Files to ignore for examples
    exignore = [
        'powerof*', 'poweron', 'invisc', 'batch-pbs', 'report-*',
        "*.ugrid", "arrow"
    ]
    # Create destination folders for examples
    os.mkdir(os.path.join(fdir, 'examples'), dmask)
    os.mkdir(os.path.join(fdir, 'examples', 'pycart'), dmask)
    os.mkdir(os.path.join(fdir, 'examples', 'pyfun'),  dmask)
    os.mkdir(os.path.join(fdir, 'examples', 'pyover'), dmask)
    # pyCart Examples folder
    fexi = os.path.join('examples', 'pycart')
    fexo = os.path.join(fdir, 'examples', 'pycart')
    # Copy each example individually.
    CopyTree(os.path.join(fexi, '01_bullet'),         fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '02_arrow'),          fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '03_fins'),           fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '04_bJet'),           fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '05_adapt_bJet'),     fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '06_lineload_arrow'), fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '07_data_arrow'),     fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '08_thrust'),         fexo, exignore, depth=2)
    # pyFun Examples folder
    fexi = os.path.join('examples', 'pyfun')
    fexo = os.path.join(fdir, 'examples', 'pyfun')
    # Copy each example individually.
    CopyTree(os.path.join(fexi, '01_arrow'),          fexo, exignore, depth=2)
    CopyTree(os.path.join(fexi, '02_bullet'),         fexo, exignore, depth=2)
    # pyOver Examples folder
    fexi = os.path.join('examples', 'pyover')
    fexo = os.path.join(fdir, 'examples', 'pyover')
    # Copy each example individually.
    CopyTree(os.path.join(fexi, '01_bullet'),         fexo, exignore, depth=3)
    
    # Destination tar
    ftar = os.path.join('release', fdir+'.tar.gz')
    # Check for the tar ball
    if os.path.isfile(ftar):
        raise SystemError("Release '%s' exists; aborting" % ftar)
    # Zip the folder
    sp.call(['tar', '-czf', ftar, fdir])
    # Remove the folder
    shutil.rmtree(fdir)
    
    # Return to original location
    os.chdir(fpwd)
# def MakeRelease
    
# Copy a folder using a glob
def CopyTree(fsrc, fdst, ignore=[], depth=2):
    """Copy a folder while ignoring a list of globs
    
    :Call:
        >>> CopyTree(fsrc, fdst, ignore=[], depth=2)
    :Inputs:
        *fsrc*: :class:`str`
            Name of folder to copy
        *fdst*: :class:`str`
            Destination folder to copy files to
        *ignore*: :class:`list`
            List of globs to ignore
        *depth*: ``None`` or :class:`int`
            Maximum recurse depth
    :Versions:
        * 2015-10-29 ``@ddalle``: First version
    """
    # Check if source exists.
    if not os.path.isdir(fsrc):
        raise SystemError("Source folder '%s' does not exist" % fsrc)
    
    # Create folder if necessary
    if not os.path.isdir(fdst):
        # Create the folder.
        os.mkdir(fdst, dmask)
    else:
        # Get names of folders (last part of absolute path)
        fsrci = os.path.split(os.path.abspath(fsrc))[1]
        fdsti = os.path.split(os.path.abspath(fdst))[1]
        # Check for non-matching names
        if fdsti != fsrci: fdst = os.path.join(fdst, fsrci)
        # Create folder if necessary.
        if not os.path.isdir(fdst): os.mkdir(fdst, dmask)
        
    # Loop through files
    for fname in os.listdir(fsrc):
        # Check the ignore globs
        qi = False
        for fi in ignore:
            if fnmatch.fnmatch(fname, fi):
                # Match
                qi = True
                break
        # Check for ignore
        if qi: continue
        # Relative path.
        frel = os.path.join(fsrc, fname)
        # Check for a directory.
        if os.path.isdir(frel):
            # Check recursion depth.
            if depth is not None:
                # Decrease the allowable depth.
                di = depth - 1
                # Depth
                if di <= 0: continue
            else:
                # Infinite depth allowed.
                di = None
            # Build destination folder
            fdsti = os.path.join(fdst, fname)
            # Recurse into that tree.
            CopyTree(frel, fdsti, ignore=ignore, depth=di)
        else:
            # Copy the file
            shutil.copy(frel, fdst)
# def CopyTree


# Check if called as script
if __name__ == "__main__":
    # Read arguments
    a, kw = cape.argread.readflags(os.sys.argv)
    # Check for version name
    if len(a) > 0:
        # Version specified
        ver = a[0]
    else:
        # Use the current CAPE version
        ver = cape.version
    # Make the release
    MakeRelease(ver)

