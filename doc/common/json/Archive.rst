-----------
ArchiveOpts
-----------

*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings
*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives
*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive

