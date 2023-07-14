-----------------
"Archive" section
-----------------

*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running

