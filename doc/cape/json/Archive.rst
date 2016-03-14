
.. _cape-json-Archive:

----------------
Archive Settings
----------------

The ``"Archive"`` section of a Cape JSON file's ``"RunControl"`` section
contains options for archiving and cleaning up a case that has been completed.
The syntax for these options is shown below.


    .. code-block:: javascript
    
        "Archive": {
            // Descriptors for the archive type
            "ArchiveFolder": "",
            "ArchiveFormat": "tar",
            "ArchiveType": "full",
            "ArchiveTemplate": "viz",
            "RemoteCopy": "scp",
            // Actions to take after each phase
            "ProgressArchiveFiles": [],
            // Actions to take before archiving
            "PreDeleteFiles": [],
            "PreDeleteDirs": [],
            "PreUpdateFiles": [],
            "PreArchiveFiles": [],
            "PreArchiveGroups": [],
            "PreArchiveDirs": [],
            // Cleanup actions to take after archiving
            "PostDeleteFiles": [],
            "PostDeleteDirs": [],
            "PostUpdateFiles": [],
            "PostArchiveGroups": [],
            "PostArchiveDirs": []
        }
        
The primary purpose of this section, which controls actions taken in the
:mod:`cape.manage` module, is to create archives (or backup copies, depending on
the user's point of view) of the run folders.  However, it can also be used to
manage unneeded files upon completion of runs, which often becomes an important
feature.  On some file systems, file count is also an important limitation, and
the methods can also help manage that aspect without deleting files via creation
of tar balls.


Basic Archive Definitions
=========================

The archive/backup is defined by a small number of options.  The first defines
where the archive will be made, set by *ArchiveFolder*.  If the path given in
*ArchiveFolder* does not exist, no archive will be created (although the options
starting with "Pre" may still take effect).  If the *ArchiveFolder* contains a
``:`` character, it is considered a remote path, and the archive is copied using
the value of *RemoteCopy*.

Archives are made in the format corresponding to *ArchiveFormat*, which should
usually be ``"tar"`` or ``"zip"`` because those formats can be updated.  Many
backup systems have hardware compression, and so ``"zip"`` is mostly recommended
for Windows systems, and ``"gzip"`` or ``"bzip2"`` are recommended only when not
using dedicated backup hardware.  The compressed formats (other than ``"zip"``)
are ignored for the pre-archiving groups.

The *ArchiveType* command specifies whether a solution is archived as a single
tar ball of the entire folder or as a folder with several tar balls inside of
it.


