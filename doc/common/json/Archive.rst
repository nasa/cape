-----------------
"Archive" section
-----------------

*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case
*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running
*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives
*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton

