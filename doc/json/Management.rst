
-------------------------------
Folder Management and Archiving
-------------------------------

Once a run matrix or at least some of the cases in it have been run, it is
usually advantageous to back up the results to some other location and/or delete
some of the files from the active disk space.  To assist in this task, pyCart
provides a "Management" section, which can be invoked using ``pycart
--archive``.  The default options are shown below.

    .. code-block:: javascript
    
        "Management": {
            "ArchiveFolder": "",
            "ArchiveFormat": "tar",
            "ArchiveAction": "skeleton",
            "RemoteCopy": "scp",
            "nCheckPoint": 2,
            "TarViz": "tar",
            "TarAdapt": "tar",
            "TarPBS": "tar"
        }
        
The full dictionary of "Management" options and their possible values is shown
below.

    *ArchiveFolder*: {``""``} | ``"/path/to/archive"`` | \
    ``"host:/path/to/archive"`` | :class:`str`
     
        The base location where archives are created.  If this is empty, no
        archives are created; if it contains a colon, a copy is made locally and
        then sent to the remote host; if it contains no colon, the archive is
        made in place.  Suppose that the name of a case for example is
        ``"poweroff/m0.88"``; the backup copy will be made at *ArchiveFolder*\
        ``/poweroff/m0.88.tar``.
        
    *ArchiveFormat*: {``"tar"``} | ``"gz"`` | ``"bz2"``
        Format to use for archives of cases
        
    *ArchiveAction*: ``""`` | {``"skeleton"``} | ``"delete"``
        Action to take after creating archive; do nothing, remove nonessential
        files, or delete entire run directory
    
    *RemoteCopy*: {``"scp"``} | ``"rsync"`` | :class:`str`
        Command to use when copying archives to remote directories.  For cases
        where special options are preferred, e.g. ``rsync -auz`` or ``shiftc
        --wait``, just use these as the value of this option.  This flexibility
        is somewhat of a security threat because accidental or malicious code
        here may be executed.  For example, putting ``rm -rf`` would yield
        unpleasant results.  However, the archiving task uses
        :func:`subprocess.call`, so advanced code-insertion attacks using pipes
        or other shell features will not work
        
    *nCheckPoint*: {``2``} | :class:`int`
        Number of previous :file:`check.%05i` or :file:`check.%06i.td` volume
        solution files to keep around.
        
    *TarAdapt*: ``""`` | {``"tar"``} | ``"gz"`` | ``"bz2"``
        Format to use for grouping early adaptation folders (no grouping or
        compressing if ``""``)
        
    *TarPBS*: ``""`` | {``"tar"``} | ``"gz"`` | ``"bz2"``
        Format to use for grouping PBS output files
        
    *TarViz*: ``""`` | {``"tar"``} | ``"gz"`` | ``"bz2"``
        Format to use for grouping visualization files (no grouping or
        compressing if ``""``)
