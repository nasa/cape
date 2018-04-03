#!/usr/bin/env python
"""
Attempt to assimilate owners on all objects
============================================

This script searches all files and checks for any that don't have the same
owner as the user running the script. In order to correct this, it will move
each such file to a temporary file, copy it (thereby creating a new file as the
current user), and then deleting the temporary file.

:Call:
    
    .. code-block:: console
    
        $ assim_user.py [OPTIONS]

:Options:
    -h, --help
        Display this help message and exit
        
:Versions:
    * 2018-04-03 ``@ddalle``: First version
"""

# System modules
import os
import glob
import shutil

# Primary function
def assimilate_user(v=False):
    """Attempt to map all files to the current user
    
    :Call:
        >>> assimilate_user(v=False)
    :Inputs:
        *v*: ``True`` | {``False``}
            Verbosity option
    :Versions:
        * 2018-04-03 ``@ddalle``: First version
    """
    # Get the executor's UID
    uid = os.getuid()
    # Get UID of this folder
    if uid != os.stat('.').st_uid:
        raise ValueError("UIDs of process and top-level folder do not match.")
    
    # Save current folder
    fpwd = os.getcwd()
    # Initialize depth
    d = 0
    # Assimilate user of this folder
    assimilate_uid_dir(uid, d, 6, v=v)
    # Confirm return to original location
    os.chdir(fpwd)
    
# Perform UID operations on one folder
def assimilate_uid_dir(uid, d=0, maxdepth=10, v=False):
    """Perform operations to ensure current user owns all files in folder
    
    :Call:
        >>> assimilate_uid_dir(uid, d=0, maxdepth=10, v=False)
    :Inputs:
        *uid*: :class:`int`
            User ID desired for all files' owner
        *d*: {``0``} | :class:`int`
            Current depth
        *maxdepth*: {``10``} | :class:`int`
            Maximum depth
        *v*: ``True`` | {``False``}
            Verbosity option
    :Versions:
        * 2018-04-03 ``@ddalle``: First version
    """
    # Get list of files in this folder
    fglob = os.listdir('.')
    # Save current path
    fpwd = os.getcwd()
    # String header (a number of spaces, 2*d)
    hdr = " " * (2*d)
    # Status update
    if v:
        print(hdr + os.path.split(os.getcwd())[1] + os.sep)
    # Add to header
    hdr += "  "
    # Loop through them
    for fi in fglob:
        # Get user
        ui = os.stat(fi).st_uid
        # Check for matching UID
        if ui == uid:
            # Check if it's a folder
            if os.path.isdir(fi):
                # Enter the folder
                os.chdir(fi)
                # Recurse
                assimilate_uid_dir(uid, d+1, maxdepth, v=v)
                # Return
                os.chdir('..')
            # Move on to next file/folder
            continue
        # Check if it's a folder, link, or regular file
        if os.path.isdir(fi):
            # Status update
            print(hdr + "%s/ (%s -> %s)" % (fi, ui, uid))
            # Temporary folder name
            ft = get_temp_filename(d=True)
            # Move the folder
            try:
                os.rename(fi, ft)
            except Exception:
                if v:
                    print(hdr + ("FAILED: %s --> %i" % (fi, ft)))
                continue
            # Recreate the folder
            try:
                os.mkdir(fi)
            except Exception:
                if v:
                    print(hdr + ("FAILED: mkdir %s" % fi))
                continue
            # Loop through the files in that folder
            for fj in os.listdir(ft):
                # Move it to the new folder
                try:
                    os.rename(os.path.join(ft, fj), os.path.join(fi, fj))
                except Exception:
                    if v:
                        print(hdr +
                            ("FAILED: %s/%s --> %s/%s" % (ft,fj,fi,fj)))
                    continue
            # Remove temporary folder
            try:
                os.rmdir(ft)
            except Exception:
                if v:
                    print(hdr + ("FAILED: rmdir %s" % ft))
            # Check max depth
            if d+1 >= maxdepth:
                continue
            # Enter the folder
            os.chdir(fi)
            # Recurse
            assimilate_uid_dir(uid, d+1, maxdepth, v=v)
            # Return to this folder
            os.chdir('..')
        else:
            # Status update
            print(hdr + "%s (%s -> %s)" % (fi, ui, uid))
            # Temporary file name
            ft = get_temp_filename()
            # Move the file
            try:
                os.rename(fi, ft)
            except Exception:
                if v:
                    print(hdr + ("FAILED: %s --> %s" % (fi,ft)))
                continue
            # Copy the file (creating it with current *uid*)
            try:
                shutil.copy(ft, fi)
            except Exception:
                if v:
                    print(hdr + ("FAILED: cp %s %s" % (ft, fi)))
            # Remove the copied file
            try:
                os.remove(ft)
            except Exception:
                if v:
                    print(hdr + ("FAILED: rm %s" % ft))
    # Return to original folder (safety)
    os.chdir(fpwd)

# Get the name of a temporary file that doesn't exist
def get_temp_filename(d=False):
    """Create the name of a file that does not exist in the current folder
    
    :Call:
        >>> ft = get_temp_filename(d=False)
    :Inputs:
        *d*: ``True`` | {``False``}
            Whether or not to create a "folder" name, otherwise a file
    :Outputs:
        *ft*: {``"tmpfile"``} | ``"tmpdir"`` | :class:`str`
            Available name of a temporary file or folder
    :Versions:
        * 2018-04-03 ``@ddalle``: First version
    """
    # Initialize candidate
    if d:
        # Initial folder candidate
        ft = "tmpdir"
    else:
        # Initial file candidate
        ft = "tmpfile"
    # Loop until not found
    while os.path.exists(ft):
        # Add a character
        ft += "_"
    # Output
    return ft
    
# Check if run as script
if __name__ == "__main__":
    # Get inputs
    argv = os.sys.argv
    # Check for help option
    if ("-h" in argv) or ("--help" in argv):
        print(__doc__)
        sys.exit()
        
    # Process options
    if ('-v' in argv) or ("--verbose" in argv):
        # Verbose option
        v = True
    else:
        # Quiet
        v = False
    # Run main function
    assimilate_user(v=v)

