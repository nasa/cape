---------------------------------------
options for archiving and file clean-up
---------------------------------------

**Option aliases:**

* *progress* -> *clean*

**Recognized options:**

*ArchiveFolder*: {``''``} | :class:`str`
    path to the archive root
*ArchiveFormat*: ``''`` | ``'bz2'`` | ``'gz'`` | ``'lzma'`` | {``'tar'``} | ``'xz'`` | ``'zip'`` | ``'zst'``
    format for case archives
*ArchiveType*: {``'full'``} | ``'partial'``
    flag for single (full) or multi (sub) archive files
*SearchMethod*: {``'glob'``} | ``'regex'``
    method for declaring multiple files with one string
*TailLines*: {``{}``} | :class:`object`
    n lines to keep for any 'tail-ed' file, by regex/glob

**Subsections:**

.. toctree::
    :maxdepth: 1

    RunControl-Archive-clean
    RunControl-Archive-archive
    RunControl-Archive-skeleton
