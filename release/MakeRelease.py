#!/usr/bin/env python
"""
Create a release tar ball for Cape

:Versions:
    * 2015-10-29 ``@ddalle``: Started
    * 2019-09-16 ``@ddalle``: Update for version 0.9
"""

# Standard library modules
import os
import sys
import shutil
import fnmatch
import subprocess as sp

# The module itself
import cape
import cape.argread

# Permissions to use for new folders
dmask = 0o755


# Make function
def MakeRelease(ver):
    # Remember current folder name
    fpwd = os.getcwd()
    # Go to parent directory of entire git repo
    os.chdir(cape.CapeFolder)
    os.chdir('..')

    # Create zip for the main content
    ZipModule(ver)
    # Create zip of HTML documentation
    ZipHTMLDoc(ver)
    # Zip each example
    ZipExamples(ver)


# Create the ``.zip`` for the main content
def ZipModule(ver):
    # Release name
    fdir = 'cape-%s' % ver
    # Check if the folder already exists
    if os.path.isdir(fdir):
        raise SystemError(
            "Container folder for release '%s' already exists" % fdir)
    # Make the release folder
    os.mkdir(fdir, 0o755)
    
    # Enter documentation folder
    os.chdir("doc")
    # Compile PDF documentation
    ierr = os.system("make latexpdf")
    # Check for error
    if ierr:
        raise SystemError("Error during `make latexpdf` execution")
    # Return
    os.chdir("..")

    # Files to ignore for modules
    modignore = [
        "*.py[oc]",
        "__pycache__",
        "*.so",
        "pyUS*",
        "pyus*",
        "us3d"
    ]
    # Copy the modules.
    CopyTree('cape',   fdir, modignore, depth=2)
    CopyTree('pyCart', fdir, modignore, depth=2)
    CopyTree('pyFun',  fdir, modignore, depth=2)
    CopyTree('pyOver', fdir, modignore, depth=2)
    
    # Other settings and tools
    CopyTree('bin',         fdir, modignore, depth=2)
    CopyTree('settings',    fdir, modignore, depth=2)
    CopyTree('modulefiles', fdir, modignore, depth=2)
    CopyTree('templates',   fdir, modignore, depth=2)
    
    # Installation files
    shutil.copy('Makefile',   fdir)
    shutil.copy('config.cfg', fdir)
    # Manual
    shutil.copy(os.path.join("doc", "_build", "latex", "cape.pdf"),
        os.path.join(fdir, "CAPE-manual-%s.pdf" % ver))
    
    # Name of the zip file
    fzip = os.path.join('release', fdir+'.zip')
    # Delete zip file if it exists
    if os.path.isfile(fzip):
        os.remove(fzip)
    # Zip the folder
    sp.call(['zip', fzip, '-r', fdir])
    # Delete folder
    shutil.rmtree(fdir)
    
    
# Create the ``.zip`` for the HTML documentation
def ZipHTMLDoc(ver):
    # Release name
    fdir = 'cape-%s' % ver
    # Check if the folder already exists
    if os.path.isdir(fdir):
        raise SystemError(
            "Container folder for release '%s' already exists" % fdir)
    # Make the release folder
    os.mkdir(fdir, 0o755)
    
    # Enter documentation folder
    os.chdir("doc")
    # Compile PDF documentation
    ierr = os.system("make html")
    # Check for error
    if ierr:
        raise SystemError("Error during `make latexpdf` execution")
    # Return
    os.chdir("..")
    
    
    # Documentation folder to copy
    fdoc = os.path.join("doc", "_build", "html")
    # Copy the documentation
    CopyTree(fdoc, os.path.join(fdir, "doc"), [], depth=16)
    
    # Destination tar
    fzip = os.path.join('release', fdir+'-HTML-doc.zip')
    # Delete zip file if it exists
    if os.path.isfile(fzip):
        os.remove(fzip)
    # Zip the folder
    sp.call(['zip', fzip, '-r', fdir])
    # Remove the folder
    shutil.rmtree(fdir)


# Tar examples
def ZipExamples(ver):
    # Release name
    fdir = 'cape-%s' % ver
    # Check if the folder already exists
    if os.path.isdir(fdir):
        raise SystemError(
            "Container folder for release '%s' already exists" % fdir)
    # Make the release folder
    os.mkdir(fdir, 0o755)
    # Create destination folders for examples
    os.mkdir(os.path.join(fdir, 'examples'), 0o755)
    os.mkdir(os.path.join(fdir, 'examples', 'pycart'), 0o755)
    os.mkdir(os.path.join(fdir, 'examples', 'pyfun'),  0o755)
    os.mkdir(os.path.join(fdir, 'examples', 'pyover'), 0o755)
    
    # List of examples
    fex_list = [
        os.path.join("pycart", "01_bullet"),
        os.path.join("pycart", "02_arrow"),
        os.path.join("pycart", "03_fins"),
        os.path.join("pycart", "04_bJet"),
        os.path.join("pycart", "05_adapt_bJet"),
        os.path.join("pycart", "06_lineload_arrow"),
        os.path.join("pycart", "07_data_arrow"),
        os.path.join("pycart", "08_thrust"),
        os.path.join("pyfun",  "01_arrow"),
        os.path.join("pyfun",  "02_bullet"),
        os.path.join("pyover", "01_bullet"),
    ]
    
    # Files to ignore for examples
    exignore = [
        "powerof*",
        "poweron",
        "invisc",
        "batch-pbs",
        "report-*",
        "*.ugrid",
        "arrow"
    ]
    
    # Loop through examples
    for fex in fex_list:
        # Full path to example source files and copy destination
        fexi = os.path.join("examples", fex)
        fexo = os.path.join(fdir, fexi)
        # Make folder
        os.mkdir(fexo, 0o755)
        
        # Copy files
        CopyTree(fexi, fexo, exignore, depth=2)
        
        # Example name modified for single file name
        fexz = fdir + "-" + fex.replace(os.sep, "-") + ".zip"
        # Destination tar
        fzip = os.path.join('release', fexz) 
        # Delete zip file if it exists
        if os.path.isfile(fzip):
            os.remove(fzip)
        # Zip the folder
        sp.call(['zip', fzip, '-r', fexo])
        # Remove the folder
        shutil.rmtree(fexo)
    
    # Clear release directory
    shutil.rmtree(fdir)


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
        if fdsti != fsrci:
            fdst = os.path.join(fdst, fsrci)
        # Create folder if necessary
        if not os.path.isdir(fdst):
            os.mkdir(fdst, dmask)
        
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



# Check if called as script
if __name__ == "__main__":
    # Read arguments
    a, kw = cape.argread.readflags(sys.argv)
    # Check for version name
    if len(a) > 0:
        # Version specified
        ver = a[0]
    else:
        # Use the current CAPE version
        ver = cape.version
    # Make the release
    MakeRelease(ver)

