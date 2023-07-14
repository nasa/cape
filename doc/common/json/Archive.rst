-----------------
"Archive" section
-----------------

*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case
*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension
*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives
*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings

