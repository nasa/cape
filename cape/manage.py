r"""
This module provides methods to manage and archive files for run
folders. It provides extensive tools for archiving results to other
locations either as a tar ball, tar bomb, or zip archive of the entire
folder or a folder of several smaller zip files.

It also provides tools for deleting files either before or after
creating the archive.  In addition, there is an easy interface for
keeping only the most recent *n* files of a certain glob.  For example,
if there are solution files ``flow.01``, ``flow.02``, ``flow.03``, and
``flow.04``, setting the *PostUpdateFiles* parameter to

    .. code-block:: python

        {"flow.??": 2}

will delete only ``flow.01`` and ``flow.02``.

The module provides methods to perform deletions, conditional deletions,
and grouping files into tar or zip archives at multiple stages.  For
example, :func:`PreDeleteFiles` deletes files after a case has been
completed and before generating the archive of the case.  On the other
hand, :func:`PostDeleteFiles` deletes files after creating the archive;
the difference is whether or not the file in question is included in the
archive before it is deleted.

A case cannot be archived until it has been granted the status of
``PASS``, by meeting the requested number of iterations and phases and
by getting marked with a ``p`` in the run matrix file.  However, less
aggressive pruning via the ``--clean`` command can always be performed,
even if the case is currently running.

The archiving process can be "reversed" (although this does not delete
the archived copy) using :func:`UnarchiveFolder`, which copies files
from the archive to the working folder.  This can be useful if you
determine that a necessary file for post-processing was cleaned out or
if the case needs to be extended (run for more iterations).

Functions such as :func:`ProgressDeleteFiles` deletes files between
phases, and as such it is much more dangerous.  Other methods will only
delete or archive once the case has been marked as PASS by the user.

An even more aggressive action can be taken using the ``--skeleton``
command-line option.  After creating or updating the archive, this
deletes even more files from the working copy than ``--archive`` using
the :func:`SkeletonFolder` function.  A common use of this dichotomy is
to set up ``--archive`` so that all post-processing can still be done,
and once the post-processing ``--skeleton`` deletes everything but a
minimal set of information about the case.  The ``--skeleton`` also has
an extra capability to replace the working copy with the last few lines
of that file, which might contain information on the most recent
iteration run, for example.  In order to avoid catastrophe, the
``--skeleton`` command effectively calls ``--archive`` as part of its
normal procedure.

This module contains many supporting methods to perform specific actions
such as comparing modification dates on the working and archived copies.
The main functions that are called from the command line are:

    * :func:`ManageFilesProgress`: delete files progressively, even if
      the case is still running (can be used for example to keep only the
      two most recent check point files); does not archive
    * :func:`ManageFilesPre`: delete or tar files/folders immediately
      prior to creating archive; reduces size or archive but not performed
      until case has ``PASS`` status
    * :func:`ManageFilesPost`: deletes or tar files/folders after creating
      archive

Each of the four phases also has several contributing functions that
perform a specific task at a specific phase, which are outlined below.

    * ``"Progress"``: delete files safely at any time
        - :func:`ProgressDeleteFiles`
        - :func:`ProgressUpdateFiles`
        - :func:`ProgressDeleteDirs`
        - :func:`ProgressTarGroups`
        - :func:`ProgressTarDirs`

    * ``"Pre"``: delete files from complete case before archiving
        - :func:`PreDeleteFiles`
        - :func:`PreUpdateFiles`
        - :func:`PreDeleteDirs`
        - :func:`PreTarGroups`
        - :func:`PreTarDirs`

    * ``"Post"``: delete files after creating/updating archive
        - :func:`PostDeleteFiles`
        - :func:`PostUpdateFiles`
        - :func:`PostDeleteDirs`
        - :func:`PostTarGroups`
        - :func:`PostTarDirs`

    * ``"Skeleton"``: delete files after post-processing
        - :func:`SkeletonTailFiles`
        - :func:`SkeletonDeleteFiles`

The difference between ``DeleteFiles`` and ``UpdateFiles`` is merely on
what action is taken by default.  The user can manually specify how many
most recent files matching a certain glob to keep using a :class:`dict`
such as

    .. code-block:: python

        {"flow[0-9][0-9]": 2}

but if a glob is given without a
:class:`dict`, such as simply

    .. code-block:: python

        "flow[0-9][0-9]"

by default ``DeleteFiles()`` will delete all of them while
``UpdateFiles()`` will keep the most recent.

Finally, the main command-line folder management calls each interface
directly with one function from this module.

    * ``--clean``: :func:`CleanFolder`
    * ``--archive``: :func:`ArchiveFolder`
    * ``--skeleton``: :func:`SkeletonFolder`
    * ``--unarchive``: :func:`UnarchiveFolder`

"""

# Standard library modules
import os
import shutil
import glob
import sys

# Standard library, renamed
import subprocess as sp

# Standard library, partial
from datetime import datetime

# Third-party modules
import numpy as np

# Local modules, partial imports
from .cfdx.options import Archive
from .cfdx.bin     import check_output, tail


# Type helpers
if sys.version_info.major > 2:
    unicode = str


# Write date to archive
def write_log_date(fname='archive.log'):
    r"""Write the date to the archive log

    :Call:
        >>> cape.manage.write_log_date(fname='archive.log')
    :Inputs:
        *fname*: {``"archive.log"``} | :class:`str`
            Name of acrhive log file
    :Versions:
        * 2016-07-08 ``@ddalle``: Version 1.0
    """
    # Open the file
    f = open(fname, 'a')
    # Write the date
    f.write("# --- %s ---\n" %
        datetime.now().strftime('%Y-%m-%d %H: %M:%S %Z'))
    # Close the file
    f.close()


# Write an action to log
def write_log(txt, fname='archive.log'):
    r"""Write the date to the archive log

    :Call:
        >>> cape.manage.write_log(txt, fname='archive.log')
    :Inputs:
        *fname*: {``"archive.log"``} | :class:`str`
            Name of acrhive log file
    :Versions:
        * 2016-07-08 ``@ddalle``: Version 1.0
    """
    # Open the file
    f = open(fname, 'a')
    # Write the line
    f.write('%s\n' % txt.rstrip())
    # Close the file
    f.close()


# File tests
def isfile(fname):
    r"""Handle to test if file **or** link

    :Call:
        >>> q = cape.manage.isfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file
    :Outputs:
        *q*: :class:`bool`
            Whether or not file both exists and is either a file or link
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    return os.path.isfile(fname) or os.path.islink(fname)


# Get modification time of a file
def getmtime(fname):
    r"""Get the modification time of a file, using ``ssh`` if necessary

    For local files, this function uses :func:`os.path.getmtime`.  If
    *fname* contains a ``:``, this function issues an ``ssh`` command
    via :func:`subprocess.Popen`.

    :Call:
        >>> t = getmtime(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file
    :Outputs:
        *t*: :class:`float` | ``None``
            Modification time; ``None`` if file does not exist
    :Versions:
        * 2017-03-17 ``@ddalle``: Version 1.0
    """
    # Check if the path is remote
    if ':' in fname:
        # Remote
        try:
            # Get the remote name
            srv = fname.split(':')[0]
            # Get the file name
            floc = fname.lstrip(srv).lstrip(':')
            # Form the command
            cmd = ['ssh', srv, 'date', '+%s', '-r', floc]
            # Call it
            txt = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]
            # Convert to integer
            return float(txt.strip())
        except Exception:
            # Interpret all errors as nonexistant file
            return None
    else:
        # Local
        if os.path.isfile(fname):
            # Local file
            return os.path.getmtime(fname)
        else:
            # No file
            return None


# Get latest modification time of a glob
def getmtime_glob(fglob):
    r"""Get the modification time of a glob, using ``ssh`` if necessary

    :Call:
        >>> t = getmtime_glob(fglob)
    :Inputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List if names of files
    :Outputs:
        *t*: :class:`float` | ``None``
            Modification time of most recently modified file
    :Versions:
        * 2017-03-13 ``@ddalle``: Version 1.0
    """
    # Initialize output
    t = 0
    # Loop through files
    for fname in fglob:
        # Get modification time of the file
        ti = getmtime(fname)
        # Check if ``None``
        if ti:
            t = max(t, ti)
    # Filter ``0``
    if t == 0:
        t = None
    # Output
    return t


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
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    return os.path.islink(fname) and not os.path.isfile(fname)


# Sort files by time
def sortfiles(fglob):
    r"""Sort a glob of files based on the time of their last edit

    :Call:
        >>> fsort = sortfiles(fglob)
    :Inputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of file names
    :Outputs:
        *fsort*: :class:`list`\ [:class:`str`]
            Above listed by :func:`os.path.getmtime`
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
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
    r"""Process an archive group, which has a precise format

    It must be a dictionary with one entry

    :Call:
        >>> ftar, fpat = cape.manage.process_ArchiveGroup(grp)
        >>> ftar, fpat = cape.manage.process_Archivegroup({ftar: fpat})
    :Inputs:
        *grp*: :class:`dict`, *len*: 1
            Single dictionary with name and pattern or list of patterns
    :Outputs:
        *ftar*: :class:`str`
            Name of archive to create without extension
        *fpat*: :class:`str` | :class:`dict` | :class:`list`
            File name pattern or list of file name patterns
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
    """
    # Check type
    if not isinstance(grp, dict) or (len(grp) != 1):
        # Wront length
        raise ValueError(
            ("Received improper tar group: '%s'\n" % grp) +
            "Archive group must be a dict with one entry"
            )
    # Get group
    fgrp = list(grp.keys())[0]
    # Get value
    fpat = grp[fgrp]
    # Output
    return fgrp, fpat


# File name count
def process_ArchiveFile(f, n=1):
    r"""Process a file glob description

    This can be either a file name, file name pattern, or dictionary of
    file name pattern and number of files to keep

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
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2022-02-02 ``@ddalle``: Version 1.1; python 3 fixes
    """
    # Check type
    if isinstance(f, dict):
        # Check length
        if len(f) != 1:
            raise ValueError(
                "Expected file descriptor '%s' to have len=1, got %i"
                % (f, len(f)))
        # Loop through elements
        for fname, nkeep in f.items():
            # Check output name
            if not type(nkeep).__name__.startswith('int'):
                raise TypeError("Number of files to keep must be an integer")
            # Output
            return fname, nkeep
    elif isinstance(f, (str, unicode)):
        # String file name; use default number of files to keep
        return f, n
    else:
        # Unprocessed type
        raise TypeError("File descriptor '%s' must be str or dict" % f)


# Get list of folders in which to search
def GetSearchDirs(fsub=None, fsort=None):
    r"""Get list of current folder and folders matching pattern

    :Call:
        >>> fdirs = GetSearchDirs()
        >>> fdirs = GetSearchDirs(fsub)
        >>> fdirs = GetSearchDirs(fsubs)
    :Inputs:
        *fsub*: :class:`str`
            Folder name of folder name pattern
        *fsubs*: :class:`list`\ [:class:`str`]
            List of folder names or folder name patterns
        *fsort*: **callable**
            Non-default sorting function
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
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
def GetMatches(fname,
    fsub=None, fkeep=None, ftest=None, n=0, fsort=None, qdel=False):
    r"""Get matches based on arbitrary rules

    :Call:
        >>> fglob = GetMatches(fname, **kw)
    :Inputs:
        *fname*: :class:`str`
            File name or file name pattern
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Subfolder name of folder name patterns in which to search
        *fkeep*: :class:`list`\ [:class:`str`]
            List of file names matching a negative file glob
        *ftest*: :class:`func`
            Function to test file type, e.g. :func:`os.path.isdir`
        *n*: :class:`int`
            Default number of files to ignore from the end of the glob or
            number of files to ignore from beginning of glob if negative
        *fsort*: **callable**
            Non-default sorting function
        *qdel*: ``True`` | {``False``}
            Interpret *n* as how many files to **keep** if ``True``; as how
            many files to **copy** if ``False``
    :Outputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of files matching input pattern
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *qdel* kwarg
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
        if qdel and (len(fglobn) <= nkeep):
            # Nothing to delete (yet)
            continue
        # Strip last *nkeep* matches
        if qdel and (nkeep > 0):
            # Keep the last *nkeep* files
            fglobn = fglobn[:-nkeep]
        elif qdel and (nkeep < 0):
            # Keep the first *nkeep* files (unusual)
            fglobn = fglobn[-nkeep:]
        elif nkeep > 0:
            # Copy the last *nkeep* files
            fglobn = fglobn[-nkeep:]
        elif nkeep < 0:
            # Copy the first *nkeep* files (unusual)
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
def GetMatchesList(flist, fsub=None, ftest=None, n=0, qdel=False):
    r"""Get matches from a list of file glob descriptors

    :Call:
        >>> fglob = GetMatchesList(flist, **kw)
    :Inputs:
        *flist*: :class:`list`\ [:class:`str` | :class:`dict`]
            List of file names or file name patterns
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Subfolder name of folder name patterns in which to search
        *ftest*: :class:`func`
            Function to test file type, e.g. :func:`os.path.isdir`
        *n*: :class:`int`
            Default number of files to ignore from the end of the glob or
            number of files to ignore from beginning of glob if negative
        *qdel*: ``True`` | {``False``}
            Interpret *n* as how many files to **keep** if ``True``; as how
            many files to **copy** if ``False``
    :Outputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of files matching input pattern
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Added *qdel* option
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
        fgn = GetMatches(fname[1:], fsub=fsub, ftest=ftest, n=0, qdel=qdel)
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
        fgn = GetMatches(fname,
            fsub=fsub, fkeep=fkeep, ftest=ftest, n=n, qdel=qdel)
        # Append contents to output glob
        for fn in fgn:
            # Check if already present
            if fn in fglob: continue
            # Append to list
            fglob.append(fn)
    # Output
    return fglob


# Get file/link matches
def GetFileMatches(fname, fsub=None, n=0, qdel=False):
    r"""Get list of all files or links matching a list of patterns

    :Call:
        >>> fglob = GetFileMatches(fname, fsub=None, n=0, qdel=False)
        >>> fglob = GetFileMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list`\ [:class:`dict` | :class:`str`]
            List of file name patterns
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
        *qdel*: ``True`` | {``False``}
            Interpret *n* as how many files to **keep** if ``True``; as
            how many files to **copy** if ``False``
    :Outputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *qdel* kwarg
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname, fsub=fsub, ftest=isfile, n=n, qdel=qdel)
    # Output
    return fglob


# Get link matches
def GetLinkMatches(fname, fsub=None, n=0, qdel=False):
    r"""Get list of all links matching a list of patterns

    :Call:
        >>> fglob = GetLinkMatches(fname, fsub=None, n=0, qdel=False)
        >>> fglob = GetLinkMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list`\ [:class:`dict` | :class:`str`]
            List of file name patterns
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
        *qdel*: ``True`` | {``False``}
            Interpret *n* as how many files to **keep** if ``True``; as how
            many files to **copy** if ``False``
    :Outputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *qdel*
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname,
        fsub=fsub, ftest=os.path.islink, n=n, qdel=qdel)
    # Output
    return fglob


# Expand any links in a glob
def ExpandLinks(fglob):
    r"""Expand any links in a full glob if linked to relative file

    If the link points to an absolute path, the entry is not replaced
    with the target of the link.

    :Call:
        >>> flst = ExpandLinks(fglob)
    :Inputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of file names, possibly including links
    :Outputs:
        *flst*: :class:`list`\ [:class:`str`]
            List of file names plus expanded versions of any links
    :Versions:
        * 2017-12-13 ``@ddalle``: Version 1.0
    """
    # Initialize output
    flst = list(fglob)
    # Loop through files
    for i in len(fglob):
        # Get the file name
        f = fglob[i]
        # Check for link (nothing to do otherwise)
        if not os.path.islink(f): continue
        # Expand the link
        fname = os.readlink(f)
        # Check if it's absolute
        if os.path.isabs(fname): continue
        # Check if it's relative to something outside this folder
        if os.path.normpath(fname).startswith('..'): continue
        # Otherwise, replace the entry
        flst[i] = fname
    # Output
    return flst


# Get folder matches
def GetDirMatches(fname, fsub=None, n=0, qdel=False):
    r"""Get list of all folders matching a list of patterns

    :Call:
        >>> fglob = GetDirMatches(fname, fsub=None, n=0, qdel=False)
        >>> fglob = GetDirMatches(fname, fsubs)
    :Inputs:
        *fname*: :class:`list`\ [:class:`dict` | :class:`str`]
            List of file name patterns
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder name of folder name pattern
        *n*: :class:`int`
            Default number of files to ignore at end of glob
        *qdel*: ``True`` | {``False``}
            Interpret *n* as how many files to **keep** if ``True``; as how
            many files to **copy** if ``False``
    :Outputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of files matching input pattern
    :See also:
        * :func:`cape.manage.process_ArchiveFile`
        * :func:`cape.manage.GetMatchesList`
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *qdel* kwarg
    """
    # Get matches from :func:`GetMatches`
    fglob = GetMatchesList(fname,
        fsub=fsub, ftest=os.path.isdir, n=n, qdel=qdel)
    # Output
    return fglob


# List of folders implied by list of files
def GetImpliedFolders(fglob, fdirs=[]):
    r"""Check a list of files to get implicit matching folders

    For example, ``["case.json", "adapt00/input.nml"]`` implies the
    folder ``adapt00/``

    :Call:
        >>> fsubs = GetImpliedFolders(fglob, fdirs=[])
    :Inputs:
        *fglob*: :class:`list`\ [:class:`str`]
            List of file names including any folder names
        *fdirs*: {``[]``} | :class:`list`\ [:class:`str`]
            List of folders to append to
    :Outputs:
        *fsubs*: :class:`list`\ [:class:`str`]
            Unique list of folders including entries from *fdirs*
    :Versions:
        * 2017-12-13 ``@ddalle``: Version 1.0
    """
    # Initialize output
    fsubs = list(fdirs)
    # Loop through files
    for fname in fglob:
        # Use :func:`os.path.split` to check if a folder is in the name
        fsplit = os.path.split(fname)
        # Check for a split
        if len(fsplit) > 1:
            # Get the folder and file name
            fdir, fn = fsplit
            # Check if it's already in the list (no repeats)
            if fdir not in fsubs:
                fsubs.append(fdir)
    # Output
    return fsubs


# Function to delete files according to full descriptor
def DeleteFiles(fdel, fsub=None, n=1, phantom=False):
    r"""Delete files that match a list of globs

    The function also searches in any folder matching the directory glob
    or list of directory globs *fsub*.

    :Call:
        >>> cape.manage.DeleteFiles(fdel, fsub=None, n=1, phantom=False)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder, list of folders, or glob of folders to also search
        *n*: :class:`int`
            Number of files to keep
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Get list of matches
    fglob = GetFileMatches(fdel, fsub=fsub, n=n, qdel=True)
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not isfile(fn):
            continue
        # Write to log
        write_log('  rm %s' % fn)
        # Check if not actually deleting
        if phantom: continue
        # Delete it.
        os.remove(fn)


# Function to delete all files *except* specified list
def DeleteFilesExcept(fskel, dskel=[], fsub=None, n=0, phantom=False):
    r"""Delete all files except those that match a list of globs

    :Call:
        >>> cape.manage.DeleteFilesExcept(fskel, **kw)
    :Inputs:
        *fskel*: :class:`list`\ [:class:`str` | :class:`dict`]
            List of file names or globs of files to delete
        *dskel*: {``[]``} | :class:`list`
            List of folder names of globs of folder names to delete
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder, list of folders, or glob of folders to also search
        *n*: {``0``} | :class:`int`
            Number of files to keep if not set by dictionary options for
            each file; if ``0``, keep all by default
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2017-12-13 ``@ddalle``: Version 1.0, fork :func:`DeleteFiles`
    """
    # Get list of matches
    fglob = GetFileMatches(fskel, fsub=fsub, n=n, qdel=False)
    # Get list of directory matches
    dglob = GetDirMatches(fskel, fsub=fsub, n=n, qdel=False)
    # Get list of search dirs
    fdirs = GetSearchDirs(fsub)
    # Check for any additional dirs implied by *fglob*
    fimp = GetImpliedFolders(fglob)
    # Loop through search dirs
    for fdir in fdirs:
        # List the files
        fls = os.listdir(fdir)
        # Loop through implied dirs
        for fi in fimp:
            # Check if folder exists
            if not os.path.isdir(fi):
                continue
            # Otherwise, get the files in that folder
            flsi = os.listdir(fi)
            # Append to list
            fls += [os.path.join(fi, fj) for fj in flsi]
        # Loop through the files
        for f in fls:
            # Normalize the path, remove "//", "./", etc.
            fn = os.path.normpath(f)
            # Check if it's a directory
            if os.path.isdir(fn):
                # Check for three reasons to keep it
                if fn in dglob:
                    # Directly specified
                    continue
                elif fn in fimp:
                    # Implied by a file within this folder
                    continue
                elif fn in fdirs:
                    # It's a search directory
                    continue
                # Otherwise, delete the directory
                write_log("rm -r %s" % fn)
                # Check for simulation option
                if phantom:
                    continue
                # Delete it
                shutil.rmtree(fn, ignore_errors=True)
                continue
            # Check if it's in the glob
            if fn in fglob: continue
            # Otherwise, delete it
            write_log("rm %s" % fn)
            # Check if not actually deleting files
            if phantom:
                continue
            # Delete it
            os.remove(fn)


# Function to delete all files *except* specified list
def TailFiles(ftail, fsub=None, n=1, phantom=False):
    r"""Tail a list of files

    An example input is ``[{"run.resid": [2, "run.tail.resid"]}]``;
    this tells the function to *tail* the last ``2`` files from
    ``run.resid`` and put them in a file called ``run.tail.resid``.

    :Call:
        >>> cape.manage.TailFiles(ftail, fsub=None, n=1, phantom=False)
    :Inputs:
        *ftail*: :class:`list`\ [:class:`dict`]
            List of dictionaries of files to tail
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder, list of folders, or glob of folders to also search
        *n*: :class:`int`
            Number of files to keep
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Loop through instructions
    for ft in ftail:
        # Get source
        try:
            # Get name of source file
            fsrc = list(ft.keys())[0]
            # Get number and output file
            nf, fout = ft[fsrc]
            # Ensure correct types
            nf = int(nf)
            fout = str(fout)
        except Exception:
            # Print a warning and move on
            print("  Invalid tail instruction: '%s'" % ft)
            continue
        # Find the file matches
        fglob = GetFileMatches(fsrc, fsub=fsub, n=1)
        # Loop through matches
        for fn in fglob:
            # Triple-check for existence
            if not isfile(fn):
                continue
            # Process folder if possible
            fdir, fi = os.path.split(fn)
            # Full output file
            fo = os.path.join(fdir, fout)
            # Write to log
            write_log('  tail -n %i %s > %s' % (nf, fn, fo))
            # Write deletion log if appropriate
            if fsrc != fout:
                write_log('  rm %s' % fn)
            # Check if not actually doing anything
            if phantom:
                continue
            # Tail the text
            txt = tail(fn, n=nf)
            # Open the output file
            with open(fo, 'w') as f:
                f.write(txt)
            # Delete input file
            if os.path.isfile(fn): os.remove(fn)


# ----------------------------------------------------------------------------
# PHASE ACTIONS
# ----------------------------------------------------------------------------
# Perform in-progress file management after each run
def ManageFilesProgress(opts=None, fsub=None, phantom=False):
    r"""Delete or group files and folders at end of each run

    :Call:
        >>> cape.manage.ManageFilesProgress(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options` | :class:`dict`
            Options interface for archiving
        *fsub*: :class:`list`\ [:class:`str`]
            List of globs of subdirectories that are adaptive run folders
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Perform actions
    ProgressDeleteFiles(opts, fsub=fsub, phantom=phantom)
    ProgressUpdateFiles(opts, fsub=fsub, phantom=phantom)
    ProgressDeleteDirs(opts, phantom=phantom)
    ProgressTarGroups(opts)
    ProgressTarDirs(opts)


# Perform pre-archive management
def ManageFilesPre(opts=None, fsub=None, phantom=False):
    r"""Delete or group files and folders before creating archive

    :Call:
        >>> cape.manage.ManageFilesPre(opts=None, fsub=None)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options` | :class:`dict`
            Options interface for archiving
        *fsub*: :class:`list`\ [:class:`str`]
            Number of files to keep
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Perform actions
    PreDeleteFiles(opts, fsub=fsub, phantom=phantom)
    PreUpdateFiles(opts, fsub=fsub, phantom=phantom)
    PreDeleteDirs(opts, phantom=phantom)
    PreTarGroups(opts)
    PreTarDirs(opts)


# Perform post-archive management
def ManageFilesPost(opts=None, fsub=None, phantom=False):
    r"""Delete or group files and folders after creating archive

    :Call:
        >>> cape.manage.ManageFilesPost(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options` | :class:`dict`
            Options interface for archiving
        *fsub*: :class:`list`\ [:class:`str`]
            Globs of subdirectories that are adaptive run folders
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
        * 2017-03-06 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Convert options
    opts = Archive.auto_Archive(opts)
    # Perform actions
    PostDeleteFiles(opts, fsub=fsub, phantom=phantom)
    PostUpdateFiles(opts, fsub=fsub, phantom=phantom)
    PostDeleteDirs(opts, phantom=phantom)
    PostTarGroups(opts)
    PostTarDirs(opts)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
# Clear folder
def CleanFolder(opts, fsub=[], phantom=False):
    r"""Delete files before archiving and regardless of status

    :Call:
        >>> cape.manage.CleanFolder(opts, fsub=[], phantom=False)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface including management/archive interface
        *fsub*: :class:`list`\ [:class:`str`]
            Globs of subdirectories that are adaptive run folders
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2017-03-10 ``@ddalle``: Version 1.0
        * 2017-12-15 ``@ddalle``: Added *phantom* option
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Perform deletions
    ManageFilesProgress(opts, phantom=phantom)


# Archive folder
def ArchiveFolder(opts, fsub=[], phantom=False):
    r"""Archive a folder and clean up nonessential files

    :Call:
        >>> cape.manage.ArchiveFolder(opts, fsub=[], phantom=False)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface including management/archive interface
        *fsub*: :class:`list`\ [:class:`str`]
            Globs of subdirectories that are adaptive run folders
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-12-09 ``@ddalle``: Version 1.0
        * 2017-12-15 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Restrict options to correct class
    opts = Archive.auto_Archive(opts)
    # Get archive type
    ftyp = opts.get_ArchiveType()
    # Get the archive root directory
    flfe = opts.get_ArchiveFolder()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # If no action, do nothing
    if not ftyp or not flfe:
        return

    # Get the current folder
    fdir = os.path.split(os.getcwd())[-1]
    # Go up one folder to the group directory
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the combined case folder name
    frun = os.path.join(fgrp, fdir)
    # Reenter case folder
    os.chdir(fdir)

    # Ensure folder exists
    CreateArchiveFolder(opts)
    CreateArchiveCaseFolder(opts)

    # Get the archive format, extension, and command
    fmt  = opts.get_ArchiveFormat()
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Write the data
    write_log_date()
    # Pre-archiving file management
    ManageFilesPre(opts, fsub=fsub)
    # Check for single tar ball or finer-grain archiving
    if ftyp.lower() == "full":
        # Archive entire folder
        ArchiveCaseWhole(opts)
        # Post-archiving file management
        MangeFilesPost(opts, fsub=fsub)
    else:
        # Partial archive; create folder containing several files
        # Form destination folder name
        ftar = os.path.join(flfe, fgrp, fdir)
        # Archive end-of-run files
        # ProgressArchiveFiles(opts, fsub=fsub)
        # Create tar balls before archiving
        PostTarGroups(opts, frun=frun)
        PostTarDirs(opts, frun=frun)
        # Archive any tar balls that exist
        ArchiveFiles(opts, fsub=fsub)
        # After archiving, perform clean-up
        PostDeleteFiles(opts, fsub=fsub)
        PostUpdateFiles(opts, fsub=fsub)
        PostDeleteDirs(opts)


# Unarchive folder
def UnarchiveFolder(opts):
    r"""Unarchive a case archived as one or more compressed files

    :Call:
        >>> cape.manage.UnarchiveFolder(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2017-03-13 ``@ddalle``: Version 1.0
    """
    # Restrict options to correct (sub)class
    opts = Archive.auto_Archive(opts)
    # Get archive type
    ftyp = opts.get_ArchiveType()
    # Get the archive root directory
    flfe = opts.get_ArchiveFolder()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # If not action, do nothing
    if not ftyp or not flfe:
        return

    # Get the current fodler
    fdir = os.path.split(os.getcwd())[-1]
    # Go up one folder to the group directory
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the combined case fodler name
    frun = os.path.join(fgrp, fdir)
    # Reenter case folder
    os.chdir(fdir)

    # Get the archive format, extension, and command
    fmt  = opts.get_ArchiveFormat()
    cmdu = opts.get_UnarchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Check for a single tar ball
    if ftyp.lower() == "full":
        # Unarchive single tar ball
        UnarchiveCaseWhole(opts)
        return
    elif ':' in flfe:
        # Remote
        fremote = True
        # Split host name
        fhost, fldir = flfe.split(':')
        # Full remote source name
        frdir = os.path.join(fldir, frun)
        fdir = os.path.join(flfe, frun)
        # Check if the remote archive exists
        if sp.call(['ssh', fhost, 'test', '-d', frdir]) == 1:
            # Archive does not exist
            print("  No remote archive '%s'" % fdir)
            return
        # Use SSH to get files in directory
        cmd = ['ssh', fhost, 'ls', frdir]
        # Get list of remote files
        fglob = check_output(cmd).strip().split('\n')
        # Loop through files
        for fname in fglob:
            # Remote file name
            fsrc = os.path.join(fdir, fname)
            # Check if file is a tar ball
            if fname.endswith(ext):
                # Status update
                print("  ARCHIVE/%s --> %s" % (fname, fname))
                # Copy the file (too much work to check dates)
                ierr = sp.call([fscp, fsrc, fname])
                if ierr:
                    raise SystemError("Remote copy failed.")
                # Status update
                print("  %s %s" % (' '.join(cmdu), fname))
                # Untar
                ierr = sp.call([cmdu, fname])
                if ierr:
                    raise SystemError("Untar command failed.")
            else:
                # Single file
                # Check dates
                if os.path.isfile(fname) and getmtime(fname) > getmtime(fsrc):
                    # Up-to-date
                    continue
                # Status update
                print("  ARCHIVE/%s --> %s" % (fname, fname))
                # Copy the file
                ierr = sp.call([fscp, fsrc, fname])
                if ierr:
                    raise SystemError("Remote copy failed.")
    else:
        # Local
        fremote = False
        # Name of source archive
        fdir = os.path.join(flfe, frun)
        # Check if file exists
        if not os.path.isdir(fdir):
            # Archive does not exist
            print("  No archive '%s'" % fdir)
            return
        # Get list of archive files
        fglob = os.listdir(fdir)
        # Loop through files
        for fname in fglob:
            # Remote file name
            fsrc = os.path.join(fdir, fname)
            # Check if file is a tar ball
            if fname.endswith(ext):
                # Status pdate
                print("  %s ARCHIVE/%s" % (' '.join(cmdu), fname))
                # Untar without copying
                ierr = sp.call(cmdu + [fsrc])
                if ierr: raise SystemError("Untar command failed.")
            else:
                # Single file
                if os.path.isfile(fname) and getmtime(fname) > getmtime(fsrc):
                    # Up-to-date
                    continue
                # Status update
                print("  ARCHIVE/%s --> %s" % (fname, fname))
                # Copy the file
                shutil.copy(fsrc, fname)


# Clean out folder afterward
def SkeletonFolder(opts, fsub=[], phantom=False):
    r"""Perform post-archiving clean-out actions; create a "skeleton"

    :Call:
        >>> cape.manage.SkeletonFolder(opts, fsub=[], phantom=False)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface including management/archive
        *fsub*: :class:`list`\ [:class:`str`]
            Globs of subdirectories that are adaptive run folders
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2017-12-13 ``@ddalle``: Version 1.0
        * 2017-12-15 ``@ddalle``: Version 1.1, add *phantom* option
    """
    # Run the archive command to ensure the archive is up-to-date
    ArchiveFolder(opts, fsub=[])
    # Run the skeleton commands
    SkeletonTailFiles(opts, fsub=fsub, phantom=phantom)
    SkeletonDeleteFiles(opts, fsub=fsub, phantom=phantom)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# SECOND-LEVEL FUNCTIONS
# ----------------------------------------------------------------------------
# Function to copy files to archive for one glob
def ArchiveFiles(opts, fsub=None, phantom=False):
    r"""Delete files that match a list of glob

    The function also searches in any folder matching the directory glob
    or list of directory globs *fsub*.

    :Call:
        >>> manage.ArchiveFiles(opts, fsub=None, phantom=False)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder, list of folders, or glob of folders to also search
        *phantom*: ``True`` | {``False``}
            Only copy files if ``True``
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2016-12-09 ``@ddalle``: Version 1.1, use *ArchiveFiles* option
        * 2017-03-06 ``@ddalle``: Version 1.2, add *phantom* option
    """
    # Archive all tar balls
    opts.add_ArchiveArchiveFiles(["*.tar", "*.gz", "*.zip", "*.bz"])
    # Archive list
    farch = opts.get_ArchiveArchiveFiles()
    # Get list of matches
    fglob = GetFileMatches(farch, fsub=fsub, n=0)

    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # If no action, do not backup
    if not flfe: return
    # If not full, do not continue
    if ftyp.lower() == "full":
        return

    # Write flag
    write_log('<ArchiveFiles>')

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

    # Loop through matches
    for fsrc in fglob:
        # Make sure file still exists
        if not os.path.isfile(fsrc):
            continue
        # Destination file
        fto = os.path.join(flfe, frun, fsrc)
        # Get mod time on target file if it exists
        tto = getmtime(fto)
        # Check mod time compared to local file
        if (tto) and (tto >= os.path.getmtime(fsrc)):
            continue
        # Status update
        print("  %s --> ARCHIVE/%s" % (fsrc, fsrc))
        # Check archive option
        if phantom:
            continue
        # Check copy type
        if ':' in flfe:
            # Status update
            write_log("  %s %s %s" % (fscp, fsrc, fto))
            if phantom: continue
            # Remote copy the file
            ierr = sp.call([fscp, fsrc, fto])
            if ierr:
                raise SystemError("Remote copy failed.")
        else:
            # Status update
            write_log("  cp %s %s" % (fsrc, fto))
            if phantom:
                continue
            # Local copy
            shutil.copy(fsrc, fto)


# Archive an entire case as a single tar ball
def ArchiveCaseWhole(opts):
    r"""Archive an entire run folder

    This function must be run from the case folder to be archived

    :Call:
        >>> ArchiveCaseWhole(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
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
    if not flfe:
        return
    # If not full, do not continue
    if ftyp.lower() != "full":
        return

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
        fltar = os.path.join(fldir, fgrp, ftar)
        # Full global destination name
        frtar = os.path.join(flfe, fgrp, ftar)
        # Check if the archive exists
        if sp.call(['ssh', fhost, 'test', '-f', fltar]) == 0:
            print("  Archive exists: %s" % frtar)
            return
        # Status update
        print("  %s --> %s" % (fdir, ftar))
        # Tar the folder locally.
        ierr = sp.call(cmdu + [ftar, fdir])
        if ierr:
            raise SystemError("Archiving failed.")
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
        if ierr:
            raise SystemError("Archiving failed.")

    # Return to folder
    os.chdir(fdir)


# Restore an archive
def UnarchiveCaseWhole(opts):
    r"""Unarchive a tar ball that stores results for an entire folder

    :Call:
        >>> UnarchiveCaseWhole(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2017-03-13 ``@ddalle``: Version 1.0
    """
    # Get the archive root directory
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get the remote copy command
    fscp = opts.get_RemoteCopy()
    # Get the archive format, extension, and command
    fmt  = opts.get_ArchiveFormat()
    cmdu = opts.get_UnarchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # If no action, do not unarchive
    if not flfe:
        return
    # If not a full archive, do not continue
    if ftyp.lower() != "full":
        return

    # Get the current folder
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]

    # Check if the archive is remote
    if ':' in flfe:
        # Name of tar file
        ftar = '%s.%s' % (fdir, ext)
        # Split host name
        fhost, fldir = flfe.split(':')
        # Full remote source name
        frtar = os.path.join(flfe, fgrp, ftar)
        # Check if the archive exists
        if getmtime(frtar) is None:
            print("  No archive %s" % frtar)
            return
        # Status update
        print("  %s --> %s" % (frtar, ftar))
        # Copy the archive locally
        ierr = sp.call([fscp, frtar, ftar])
        if ierr: raise SystemError("Remote copy failed.")
        # Status update
        print("  %s --> %s" % (ftar, fdir))
        # Unarchive
        ierr = sp.call(cmdu + [ftar])
        if ierr:
            raise SystemError("Unarchiving failed.")
    else:
        # Name of archive
        ftar = os.path.join(flfe, fgrp, '%s.%s'%(fdir, ext))
        # Check if archive exists
        if getmtime(ftar) is None:
            print("  No archive %s" % ftar)
            return
        # Status update
        print("  %s --> %s" % (ftar, fdir))
        # Untar the folder
        ierr = sp.call(cmdu + [ftar])
        if ierr:
            raise SystemError("Unarchiving failed.")

    # Return to folder
    os.chdir(fdir)


# Function to delete folders according to full descriptor
def DeleteDirs(fdel, fsub=None, n=1, phantom=False):
    r"""Delete folders that match a glob

    The function also searches in any folder matching the directory glob
    or list of directory globs *fsub*.

    :Call:
        >>> DeleteDirs_SubDir(fdel, n=1, fsub=None, phantom=False)
    :Inputs:
        *fdel*: :class:`str`
            File name or glob of files to delete
        *fsub*: :class:`str` | :class:`list`\ [:class:`str`]
            Folder, list of folders, or glob of folders to also search
        *n*: :class:`int`
            Number of folders to keep at end of glob
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
    """
    # Get list of matches
    fglob = GetDirMatches(fdel, fsub=fsub, n=n, qdel=True)
    # Loop through matches
    for fn in fglob:
        # Triple-check for existence
        if not os.path.isdir(fn): continue
        # Delete it
        write_log('  rm -r %s' % fn)
        # Check for phantom option
        if phantom: continue
        # Delete the folder
        shutil.rmtree(fn)


# Archive groups
def TarGroup(cmd, ftar, fname, n=0, clean=False):
    r"""Archive a group of files and delete the files

    Only the present folder will searched for file name matches

    :Call:
        >>> TarGroup(cmd, ftar, fname, clean=False)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`list`\ [:class:`dict` | :class:`str`]
            File name pattern or list thereof to combine into archive
        *clean*: :class:`bool`
            Whether or not to clean up after archiving
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
        * 2016-03-14 ``@ddalle``: Version 2.0; generalized
    """
    # Check input
    if not isinstance(cmd, list):
        raise TypeError("Input command must be a list of one or two strings")
    # Get list of files
    fglob = GetFileMatches(fname, n=n)
    # Get extension
    ext = ftar.split('.')[-1]
    # Make sure not to tar any tar balls
    fglob = [f for f in fglob if not f.endswith(ext)]
    # Exit if not matches
    if len(fglob) < 2: return
    # Get modification times
    tsrc = getmtime_glob(fglob)
    tto = getmtime(ftar)
    # Check if archive needs to be created/updated
    if (tsrc is None):
        # No files to copy
        return
    elif (tto is not None) and (tto >= tsrc):
        # Archive is up-to-date
        return
    # Create command
    cmdc = cmd + [ftar] + fglob
    # Write to the log
    write_log('  ' + ' '.join(cmdc))
    # Status update
    print("  tar -cf ARCHIVE/%s" % os.path.split(ftar)[-1])
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Check clean-up flag
    if not clean: return
    # Delete matches
    for fn in fglob:
        if isfile(fn):
            write_log('  rm %s' % fn)
            os.remove(fn)


# Tar all links
def TarLinks(cmd, ext, clean=True):
    r"""Tar all links existing in the current folder

    :Call:
        >>> TarLinks(cmd, ext, clean=True)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *clean*: :class:`bool`
            Whether or not to clean up after archiving
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
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
    # Write command to log
    write_log('  ' + ' '.join(cmdc))
    # Run command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete links
    for fn in flink:
        if os.path.islink(fn):
            write_log('  rm %s' % fn)
            os.remove(fn)


# Tar a folder
def TarDir(cmd, ftar, fdir, clean=True):
    r"""Archive a folder and delete the folder

    :Call:
        >>> TarDir(cmd, ftar, fdir)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fdir*: :class:`str`
            Name of folder
        *clean*: :class:`bool`
            Whether or not to delete folder afterwards
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
    """
    # Check if the folder exists
    if not os.path.isdir(fdir): return
    # List of files to in dir
    fnames = os.listdir(fdir)
    # Check for anything to tar
    if len(fnames) < 1: return
    # Get the modification times
    tto  = getmtime(ftar)
    # Get the time from each file in the line load
    tsrc = [getmtime(os.path.join(fdir,f)) for f in fnames]
    # Check options for already-existing archives
    if tto:
        # Check if archive is already up to date
        if tto >= max(tsrc): return
        # Ensure we use "update" tar option
        if cmd == ["tar", "-cf"]:
            # Change the command to "-uf" for an update
            cmd = ["tar", "-uf"]
    # Create command
    cmdc = cmd + [ftar, fdir]
    # Status update
    cmd = "  %s ARCHIVE/%s %s" % (' '.join(cmd), os.path.split(ftar)[1], fdir)
    print(cmd)
    write_log(cmd)
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Check for clean flag
    if not clean: return
    # Delete the folder
    if os.path.isdir(fdir):
        write_log('  rm -r %s' % fdir)
        shutil.rmtree(fdir)


# Untar a folder
def Untar(cmd, ftar):
    r"""Unarchive a tar ball and then delete it

    :Call:
        >>> cape.manage.Untar(cmd, ftar)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of archiving/compression commands
        *ftar*: :class:`str`
            Name of file to create
        *fname*: :class:`str`
            File name pattern or list thereof to combine into archive
    :Versions:
        * 2016-03-01 ``@ddalle``: Version 1.0
    """
    # Create command
    cmdc = cmd + [ftar]
    # Log
    write_log('  ' + ' '.join(cmdc))
    # Run the command
    ierr = sp.call(cmdc)
    # Exit if unsuccessful
    if ierr: return
    # Delete the folder
    if os.path.isfile(ftar):
        write_log('  rm %s' % ftar)
        os.remove(ftar)
# ----------------------------


# ----------------------------
# PRE-ARCHIVE ACTION FUNCTIONS
# ----------------------------
# Function to pre-delete files
def PreDeleteFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete files that match a list of file name patterns

    :Call:
        >>> PreDeleteFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Write flag
    write_log('<PreDeleteFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0, phantom=phantom)


# Function to pre-delete dirs
def PreDeleteDirs(opts, fsub=None, aa=None, phantom=False):
    r"""Delete folders that match a list of patterns before archiving

    :Call:
        >>> PreDeleteDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreDeleteDirs()
    # Exit if necessary
    if fdel is None:
        return
    # Write flag
    write_log('<PreDeleteDirs>')
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0, phantom=phantom)


# Function to pre-update files
def PreUpdateFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete files that match list, keeping the most recent by default

    :Call:
        >>> PreUpdateFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePreUpdateFiles()
    # Exit if necessary
    if fdel is None:
        return
    # Write flag
    write_log('<PreUpdateFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1, phantom=phantom)


# Function to pre-tar files
def PreTarGroups(opts, fsub=None, aa=None):
    r"""Tar file/folder groups

    The files are deleted after the tar ball is created

    :Call:
        >>> PreTarGroups(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchivePreTarGroups()
    # Exit if necessary
    if fgrps is None: return
    # Write flag
    write_log('<PreTarGroups>')
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
    r"""Tar folders before archiving

    The folders are deleted after the tar ball is created

    :Call:
        >>> PreTarDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fopt = opts.get_ArchivePreTarDirs()
    # Exit if necessary
    if fopt is None: return
    # Write flag
    write_log('<PreTarDirs>')
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
def PostDeleteFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete appropriate files after completing archive

    :Call:
        >>> PostDeleteFiles(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostDeleteFiles()
    # Exit if necessary
    if fdel is None: return
    # Write flag
    write_log('<PostDeleteFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0, phantom=phantom)


# Function to post-delete dirs
def PostDeleteDirs(opts, fsub=None, aa=None, phantom=False):
    r"""Delete appropriate folders after completing archive

    :Call:
        >>> PostDeleteDirs(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostDeleteDirs()
    # Exit if necessary
    if fdel is None: return
    # Write flag
    write_log('<PostDeleteDirs>')
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0, phantom=phantom)


# Function to post-update files
def PostUpdateFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete files after archiving, by default keeping most recent

    :Call:
        >>> PreUpdateFiles(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchivePostUpdateFiles()
    # Exit if necessary
    if fdel is None: return
    # Write flag
    write_log('<PostUpdateFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1, phantom=phantom)


# Function to post-tar files
def PostTarGroups(opts, fsub=None, aa=None, frun=None):
    r"""Tar file/folder groups

    The files are not deleted after the tar ball is created

    :Call:
        >>> PostTarGroups(opts, fsub=None, aa=None, frun=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *frun*: :class:`str`
            Case folder name
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchivePostTarGroups()
    # Exit if necessary
    if fgrps is None:
        return
    # Write flag
    write_log('<PostTarGroups>')
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Get remote copy destination
    flfe = opts.get_ArchiveFolder()
    # Loop through groups
    for grp in fgrps:
        # Process the group dictionary
        fgrp, fname = process_ArchiveGroup(grp)
        # Archive file name
        if (':' not in flfe):
            # Local tar command; create Tar in place rather than copying it
            ftar = os.path.join(flfe, frun, '%s.%s' % (fgrp,ext))
        else:
            # Otherwise, create the tar ball in this folder
            ftar = '%s.%s' % (fgrp, ext)
        # Archive
        TarGroup(cmdu, ftar, fname, n=0, clean=False)


# Function to post-tar dirs
def PostTarDirs(opts, fsub=None, aa=None, frun=None):
    r"""Tar folders after archiving

    The folders are not deleted after the tar ball is created

    :Call:
        >>> PostTarDirs(opts, fsub=None, aa=None, frun=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *frun*: :class:`str`
            Name of case folder
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get remote copy destination
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # Get options
    fopt = opts.get_ArchivePostTarDirs()
    # Exit if necessary
    if fopt is None:
        return
    # Write flag
    write_log('<PostTarGroups>')
    # Get format, command, and extension
    cmdu = opts.get_ArchiveCmd()
    ext  = opts.get_ArchiveExtension()
    # Get list of matching directories
    fdirs = GetDirMatches(fopt, fsub=fsub, n=1)
    # Loop through directories
    for fdir in fdirs:
        # Archive file name
        if (':' not in flfe):
            # Local tar command; create Tar in place rather than copying it
            ftar = os.path.join(flfe, frun, '%s.%s' % (fdir,ext))
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
def ProgressDeleteFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete appropriate files from active (in progress) case folder

    :Call:
        >>> ProgressDeleteFiles(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressDeleteFiles()
    # Exit if necessary
    if fdel is None:
        return
    # Write flag
    write_log('<ProgressDeleteFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=0, phantom=phantom)


# Function for in-progress file archiving
def ProgressArchiveFiles(opts, fsub=None, aa=None, phantom=False):
    """Archive files of active (in progress) case folder

    :Call:
        >>> ProgressArchiveFiles(opts, fsub=None, **kw)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fglob = opts.get_ArchiveProgressArchiveFiles()
    # Exit if necessary
    if fglob is None:
        return
    # Write flag
    write_log('<ProgressArchiveFiles>')
    # Copy
    ArchiveFiles(opts, fsub=fsub, phantom=phantom)


# Function for in-progress folder deletion
def ProgressDeleteDirs(opts, fsub=None, aa=None, phantom=False):
    """Delete appropriate folders of active (in progress) case

    :Call:
        >>> ProgressDeleteDirs(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressDeleteDirs()
    # Exit if necessary
    if fdel is None:
        return
    # Write flag
    write_log('<ProgressDeleteDirs>')
    # Delete
    DeleteDirs(fdel, fsub=fsub, n=0, phantom=phantom)


# Function for in-progress file updates
def ProgressUpdateFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete files in active folder, by default keeping most recent

    :Call:
        >>> ProgressUpdateFiles(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fdel = opts.get_ArchiveProgressUpdateFiles()
    # Exit if necessary
    if fdel is None:
        return
    # Write flag
    write_log('<ProgressUpdateFiles>')
    # Delete
    DeleteFiles(fdel, fsub=fsub, n=1, phantom=phantom)


# Function to tar groups in progress
def ProgressTarGroups(opts, fsub=None, aa=None):
    r"""Tar file/folder groups after each run

    The files are deleted after the tar ball is created

    :Call:
        >>> ProgressTarGroups(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fgrps = opts.get_ArchiveProgressTarGroups()
    # Exit if necessary
    if fgrps is None:
        return
    # Write flag
    write_log('<ProgressTarGroups>')
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
    r"""Tar folders of active (in progress) case

    The folders are deleted after the tar ball is created

    :Call:
        >>> ProgressTarDirs(opts, fsub=None, aa=None)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options
    fopt = opts.get_ArchiveProgressTarDirs()
    # Exit if necessary
    if fopt is None: return
    # Write flag
    write_log('<ProgressTarDirs>')
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


# -------------------------
# SKELETON ACTION FUNCTIONS
# -------------------------
# Function for deleting skeleton files
def SkeletonDeleteFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Delete all files except those matching file name patterns

    :Call:
        >>> SkeletonDeleteFiles(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2017-12-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options for files to keep
    fskel = opts.get_ArchiveSkeletonFiles()
    # Get options for folders to keep
    dskel = opts.get_ArchiveSkeletonDirs()
    # Get options for files to tail
    ftail = opts.get_ArchiveSkeletonTailFiles()
    # Exit if necessary
    if fskel is None:
        return
    # Append tail files
    if ftail is not None:
        # Loop through tail files
        for ftl in ftail:
            # Use **try** to check if it's a valid instruction
            try:
                # Get the name of the source file
                fsrc = list(ftl.keys())[0]
                # Unpack output file
                nt, fout = ftl[fsrc]
            except Exception:
                continue
            # Add pre-tail and post-tail files to *keep* list
            fskel += [fsrc, fout]
    # Write flag
    write_log('<SkeletonDeleteFiles>')
    # Delete the files
    DeleteFilesExcept(fskel, dskel, fsub=fsub, n=0, phantom=phantom)


# Function for tailing files during skeleton action
def SkeletonTailFiles(opts, fsub=None, aa=None, phantom=False):
    r"""Replace files with their last few lines, possibly in a new file

    :Call:
        >>> SkeletonTailFiles(opts, fsub=None, aa=None, phantom=False)
    :Inputs:
        *opts*: ``None`` | :class:`Archive` | :class:`dict`
            Options dictionary or options interface
        *fsub*: :class:`list`\ [:class:`str`] | :class:`str`
            List of sub-directory globs in which to search
        *aa*: **callable**
            Conversion function applied to *opts*
        *phantom*: ``True`` | {``False``}
            Only delete files if ``False``
    :Versions:
        * 2017-12-14 ``@ddalle``: Version 1.0
    """
    # Convert options
    if callable(aa):
        opts = aa(opts)
    # Get options for files to tail
    ftail = opts.get_ArchiveSkeletonTailFiles()
    # Exit if necessary
    if ftail is None:
        return
    # Write flag
    write_log('<SkeletonTailFiles>')
    # Delete the files
    TailFiles(ftail, fsub=fsub, n=1, phantom=phantom)
# -------------------------


# -------------
# ARCHIVE SETUP
# -------------
# Function to ensure the destination folder exists
def CreateArchiveFolder(opts):
    r"""Create the folder that will contain the archive, if necessary

    :Call:
        >>> CreateArchiveFolder(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # If no action, do not backup
    if not flfe:
        return
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
    r"""Create the group and run folders in the archive, as appropriate

    This function must be run within the folder that is to be archived.

    :Call:
        >>> CreateArchiveCaseFolder(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # If no action, do not backup
    if not flfe:
        return
    # Get the current folder.
    fdir = os.path.split(os.getcwd())[-1]
    # Go up a folder.
    os.chdir('..')
    # Get the group folder
    fgrp = os.path.split(os.getcwd())[-1]
    # Get the case folder
    frun = os.path.join(fgrp, fdir)
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
    # Return to the folder
    os.chdir(fdir)


# Create archive group folders
def CreateArchiveGroupFolder(opts):
    r"""Create the group folder in the archive, as appropriate

    This function must be run from within the folder getting archived.

    :Call:
        >>> CreateArchiveGroupFolder(opts)
    :Inputs:
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
    :Versions:
        * 2016-03-14 ``@ddalle``: Version 1.0
    """
    # Get the archive root directory.
    flfe = opts.get_ArchiveFolder()
    # Archive type
    ftyp = opts.get_ArchiveType()
    # If no action, do not backup
    if not flfe:
        return
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

