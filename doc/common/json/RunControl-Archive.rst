-------------------------------
Options for ``Archive`` section
-------------------------------

**Recognized options:**

*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case
*ArchiveExtension*: ``'bz2'`` | {``'tar'``} | ``'tgz'`` | ``'zip'``
    archive file extension
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*ArchiveFormat*: ``''`` | ``'bz2'`` | {``'tar'``} | ``'tgz'`` | ``'zip'``
    format for case archives
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton

