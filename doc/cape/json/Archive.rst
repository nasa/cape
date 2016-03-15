
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
            // Actions to take after each phase (caution!)
            "ProgressDeleteFiles": [],
            "ProgressDeleteDirs": [],
            "ProgressUpdateFiles": [],
            "ProgressArchiveFiles": [],
            "ProgressArchiveGroups": [],
            "ProgressArchiveDirs": [],
            // Files to add to archive after each run
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

.. _cape-json-Archive-basic:

Basic Archive Definitions
=========================

The archive/backup is defined by a small number of options. The first defines
where the archive will be made, set by *ArchiveFolder*. If the path given in
*ArchiveFolder* is empty (e.g. ``""``), no archive will be created (although the
options starting with "Pre" may still take effect). If the *ArchiveFolder*
contains a ``:`` character, it is considered a remote path, and the archive is
copied using the value of *RemoteCopy*.

Archives are made in the format corresponding to *ArchiveFormat*, which should
usually be ``"tar"`` or ``"zip"`` because those formats can be updated.  Many
backup systems have hardware compression, and so ``"zip"`` is mostly recommended
for Windows systems, and ``"gzip"`` or ``"bzip2"`` are recommended only when not
using dedicated backup hardware.  The compressed formats (other than ``"zip"``)
are ignored for the pre-archiving groups.

The *ArchiveType* command specifies whether a solution is archived as a single
tar ball of the entire folder or as a folder with several tar balls inside of
it.  Setting *ArchiveType* to ``"full"`` means that the entire folder containing
the CFD results for one case are archived as a single file.  Otherwise, the
archive is grouped into several components.

Finally, the *ArchiveTemplate* sets default values for all the ``Pre*`` and
``Post*`` archive settings.  This does not apply to the basic :mod:`cape`
module, but it plays an important role in archiving for each of the CFD solvers.

The dictionary of these options, their defaults, and their possible values is
below.

    *ArchiveFolder*: {``""``} | :class:`str`
        Location of the archive folder, remote if this contains ``:``
        
    *ArchiveFormat*: {``"tar"``} | ``"zip"`` | ``"bzip2"`` | ``"gzip"``
        Format for archive files
        
    *ArchiveType*: {``"full"``} | ``"partial"``
        If ``"full"``, then entire solution archived as a single file
        
    *ArchiveTemplate*: {``"full"``} | :class:`str`
        Applies default values for which files to delete or archive
        
    *RemoteCopy*: {``"scp"``} | ``"rsync"``
        Command to copy the files to archive remotely, if necessary

.. _cape-json-Archive-file:

File Glob Inputs
================

Each of the remaining options is used to describe a set of files.  This is
similar to a file glob like ``*.png`` for all the files ending with ``png`` in
the current directory, but the inputs are slightly more complex to handle
typical CFD situations.  First, each input can be either a single file glob or a
list of file globs.

Secondly, each file descriptor can have a parameter that tells Cape to ignore
the most recent *n* files.  This is a relatively common situation for CFD
solvers, which may write large solution files.  A user will want to have these
solution files to be created with some frequency to lessen the consequences of
failures during any run, but having a lot of them is undesirable.  Therefore the
file descriptors in this options section allows the user to tell Cape to delete
most files matching a glob but keep the most recent *n*.  To avoid sorting
issues with files named like ``q.2`` and ``q.10``, sorting is based on the last
time of modification of the file.

The following example shows JSON syntax of the three basic ways that file lists
can be defined.

    .. code-block:: javascript
    
        "PreDeleteFiles": "*.plt",
        "PreUpdateFiles": {"q.*": 2},
        "PostDeleteFiles": [
            "something??.dat",
            "!something03.dat",
            {"else??.dat": 1}
        ]

The specific purpose of each of these options is discussed in subsequent
sections; the intent of this paragraph is to describe what comes after the ``:``
characters.  In the first line, this specifies that all files matching the glob
``*.plt`` should be deleted before creating the archive.  The second line tells
*PreUpdateFiles* to take action on all ``q.*`` files but ignore the two most
recent files that match.  Finally, the *PostDeleteFiles* option contains a list
of three such globs.  Each of these globs applies, but any file glob starting
with ``"!"`` specifies a negative file glob.  In this example,
``something00.dat`` would get deleted, but ``something03.dat`` would not.

This type of object is called :class:`glob` in the following descriptions as an
abbreviation for the three possible classes that it can actually be.

The *TarGroup* options require a slightly different input, which is a file name
for the archive to be created in addition to a list of globs of files to be
placed in the archive.  The following example gives the primary idea.

    .. code-block:: javascript
    
        "PreTarGroups": {"forces": ["forces.dat", "total.dat", "wing.dat"]}
        
This example will group the three files listed into a single file, which will
be called ``forces.tar``, ``forces.tgz``, ``forces.tbz2``, or ``forces.zip``
depending on the value of *ArchiveFormat*.

.. _cape-json-Archive-pre:

Pre-Archive File Inputs
=======================

This section describes options for Cape to delete or group files immediately
before creating the archive.  These actions will only take place after a case
has completed the requested number of iterations and has been marked ``PASS``,
so there is some amount of safety.  However, this may permanently remove the
ability to restart a case since the deletions are applied prior to creating the
archive.

The full dictionary is given below.

    *PreDeleteFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 0 most recent

    *PreUpdateFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 1 most recent
        
    *PreDeleteDirs*: {``[]``} | :class:`glob`
        List of folders to delete, by default keeping 0 most recent
        
    *PreTarGroups*: {``[]``} | :class:`dict` (:class:`glob`)
        List of groups of files to group into tar balls
        
    *PreTarDirs*: {``[]``} | :class:`dict` (:class:`glob`)
        List of folders to replace with tar balls
  
.. _cape-json-Archive-progress:
        
In-Progress File Archiving Inputs
=================================

This section describes options for Cape to delete or group at the end of each
run.  Deleting files before a case has been marked ``PASS`` is dangerous, but
the user may want to use this capability with the appropriate level of caution. 

There is an additional option that does not have the potential to delete files
but instead archives files as they are created.  This option has no effect if
*ArchiveType* is ``"full"``.  It may be useful when multiple solution files are
created, and the user wants to keep a record of these but not keep them on the
same file system where the computations continue to be performed.

    *ProgressArchiveFiles*: {``[]``} | :class:`glob`
        List of files to copy to archive at any time
        
    *ProgressDeleteFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 0 most recent

    *ProgressUpdateFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 1 most recent
        
    *ProgressDeleteDirs*: {``[]``} | :class:`glob`
        List of folders to delete, by default keeping 0 most recent
        
    *ProgressTarGroups*: {``[]``} | :class:`dict` (:class:`glob`)
        List of groups of files to group into tar balls
        
    *ProgressTarDirs*: {``[]``} | :class:`dict` (:class:`glob`)
        List of folders to replace with tar balls
        
.. _cape-json-Archive-post:

Post-Archive File Inputs
========================

This last section describes options for Cap to delete or group files after
creating the archive.  A special aspect of these options is that if the
*ArchiveType* is not ``"full"``, the *PostTarGroups* and *PostTarDirs* create
tar balls without deleting the original files/folders.  However, the user may
then choose to delete those files by using the *PostDeleteFiles* and
*PostDeleteDirs*.

    *PostDeleteFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 0 most recent

    *PostUpdateFiles*: {``[]``} | :class:`glob`
        List of file descriptors to delete, by default keeping 1 most recent
        
    *PostDeleteDirs*: {``[]``} | :class:`glob`
        List of folders to delete, by default keeping 0 most recent
        
    *PostTarGroups*: {``[]``} | :class:`dict` (:class:`glob`)
        List of groups of files to group into tar balls
        
    *PostTarDirs*: {``[]``} | :class:`dict` (:class:`glob`)
        List of folders to replace with tar balls


