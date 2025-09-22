----------------------------------------------------------------
``clean``: options for file clean-up while case is still running
----------------------------------------------------------------

**Option aliases:**

* *CopyFiles* → *ArchiveFiles*
* *DeleteDirs* → *PostDeleteDirs*
* *DeleteFiles* → *PostDeleteFiles*
* *TailFiles* → *PostTailFiles*
* *TarDirs* → *ArchiveTarDirs*
* *TarGroups* → *ArchiveTarGroups*

**Recognized options:**

*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*ArchiveTarDirs*: {``[]``} | :class:`object`
    folders to tar, e.g. ``fm/`` -> ``fm.tar``
*ArchiveTarGroups*: {``{}``} | :class:`dict`
    groups of files to archive as one tarball
*PostDeleteDirs*: {``[]``} | :class:`object`
    folders to delete after archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    files to delete after archiving
*PostTailFiles*: {``[]``} | :class:`object`
    files to replace with their tail after archiving
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar and then delete after archiving
*PostTarGroups*: {``{}``} | :class:`dict`
    groups of files to tar after archiving
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving

