-----------------
"Archive" section
-----------------

*PreUpdateFiles*: {``[]``} | :class:`object`
    files to keep *n* and delete older, b4 archiving
*ProgressUpdateFiles*: {``[]``} | :class:`object`
    files to delete old versions while running
*SkeletonDirs*: {``None``} | :class:`object`
    folders to **keep** during skeleton action
*ProgressDeleteDirs*: {``[]``} | :class:`object`
    folders to delete while still running
*SkeletonTailFiles*: {``[]``} | :class:`object`
    files to tail before deletion during skeleton
*RemoteCopy*: {``'scp'``} | :class:`str`
    command for archive remote copies
*ArchiveFiles*: {``[]``} | :class:`object`
    files to copy to archive
*PreTarGroups*: {``[]``} | :class:`object`
    file groups to tar before archiving
*SkeletonTarDirs*: {``[]``} | :class:`object`
    folders to tar before deletion during skeleton
*ArchiveFormat*: ``''`` | {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    format for case archives
*PostTarDirs*: {``[]``} | :class:`object`
    folders to tar after archiving
*ArchiveTemplate*: {``'full'``} | :class:`str`
    template for default archive settings
*ProgressArchiveFiles*: {``[]``} | :class:`object`
    files to archive at any time
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*ProgressDeleteFiles*: {``[]``} | :class:`object`
    files to delete while still running
*PostDeleteDirs*: {``[]``} | :class:`object`
    list of folders to delete after archiving
*ProgressTarGroups*: {``[]``} | :class:`object`
    list of file groups to tar while running
*PostUpdateFiles*: {``[]``} | :class:`object`
    globs: keep *n* and rm older, after archiving
*PreDeleteFiles*: {``[]``} | :class:`object`
    files to delete **before** archiving
*PreDeleteDirs*: {``[]``} | :class:`object`
    folders to delete **before** archiving
*PostDeleteFiles*: {``[]``} | :class:`object`
    list of files to delete after archiving
*PreTarDirs*: {``[]``} | :class:`object`
    folders to tar before archiving
*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*ArchiveExtension*: {``'tar'``} | ``'tgz'`` | ``'bz2'`` | ``'zip'``
    archive file extension
*PostTarGroups*: {``[]``} | :class:`object`
    groups of files to tar after archiving
*SkeletonFiles*: {``'case.json'``} | :class:`object`
    files to **keep** during skeleton action
*ProgressTarDirs*: {``[]``} | :class:`object`
    folders to tar while running
*ArchiveAction*: ``''`` | {``'archive'``} | ``'rm'`` | ``'skeleton'``
    action to take after finishing a case

