r"""
``cape.cfdx.options.archiveopts``: Case archiving options
=========================================================

This module provides a class to access options relating to archiving
folders that were used to run CFD simulations.

The class provided in this module, :class:`ArchiveOpts`, is
loaded into the ``"RunControl"`` section of the main options interface.
"""

# Standard library
from typing import Union

# Local imports
from ...optdict import OptionsDict, INT_TYPES


# File extensions
TAR_EXTS = {
    ("", "tar"): ".tar",
    ("j", "bz2", "bzip2"): ".tar.bz2",
    ("z", "gz", "gzip"): ".tar.gz",
    ("J", "xz"): ".tar.xz",
    ("lzma",): ".tar.lzma",
    ("zst", "zstd"): ".tar.zst",
    ("zip",): ".zip",
}
# Constants
TAR_CMDS = {
    "bz2": ["tar", "-cjf"],
    "tar": ["tar", "-cf"],
    "gz": ["tar", "-czf"],
    "zip": ["zip", "-r"],
}
UNTAR_CMDS = {
    "bz2": ["tar", "-xjf"],
    "tar": ["tar", "-xf"],
    "gz": ["tar", "-xzf"],
    "zip": ["unzip", "-r"],
}

OLD_OPTMAP = {
    "ArchiveFiles": ("archive", "ArchiveFiles", 0),
    "ArchiveTarGroups": ("archive", "ArchiveTarGrups", 0),
    "PostDeleteDirs": ("archive", "PostDeleteDirs", 0),
    "PostDeleteFiles": ("archive", "PostDeleteFiles", 0),
    "PostTarDirs": ("archive", "PostTarDirs", 0),
    "PostTarGroups": ("archive", "PostTarGroups", 0),
    "PostUpdateFiles": ("archive", "PostDeleteFiles", 1),
    "PreDeleteDirs": ("archive", "PreDeleteDirs", 0),
    "PreDeleteFiles": ("archive", "PreDeleteFiles", 0),
    "PreUpdateFiles": ("archive", "PreDeleteFiles", 1),
    "ProgressArchiveFiles": ("clean", "ArchiveFiles", 0),
    "ProgressDeleteDirs": ("clean", "PostDeleteDirs", 0),
    "ProgressDeleteFiles": ("clean", "PostDeleteFiles", 0),
    "ProgressUpdateFiles": ("clean", "PostDeleteFiles", 1),
    "ProgressTarDirs": ("clean", "ArchiveTarDirs", 1),
    "ProgressTarGroups": ("clean", "ArchiveTarGroups", 0),
    "SkeletonDirs": ("skeleton", "PostDeleteDirs", 0),
    "SkeletonFiles": ("skeleton", "PostDeleteFiles", 0),
    "SkeletonTailFiles": ("skeleton", "PostTailFiles", 0),
}


# Class for options of one phase
class ArchivePhaseOpts(OptionsDict):
    r"""Options for a single archive action

    Actions include

        * ``--clean``
        * ``--archive``
        * ``--skeleton``

    The options in this class each describe a single action. The order
    of those sub-actions is

    1.  *PreDeleteDirs*
    2.  *PreDeleteFiles*
    3.  *ArchiveFiles*
    4.  *ArchiveTarGroups*
    5.  *ArchiveTarDirs*
    6.  *PostTarGroups*
    7.  *PostTarDirs*
    8.  *PostDeleteDirs*
    9.  *PostDeleteFiles*
    10. *PostTailFiles*
    """
    # No attributes
    __slots__ = ()

    # Recognized options
    _optlist = (
        "PreDeleteFiles",
        "PreDeleteDirs",
        "ArchiveFiles",
        "ArchiveTarGroups",
        "ArchiveTarDirs",
        "PostTarGroups",
        "PostTarDirs",
        "PostDeleteFiles",
        "PostDeleteDirs",
        "PostTailFiles",
    )

    # Alternate names
    _optmap = {
        "CopyFiles": "ArchiveFiles",
        "DeleteDirs": "PostDeleteDirs",
        "DeleteFiles": "PostDeleteFiles",
        "TailFiles": "PostTailFiles",
        "TarDirs": "ArchiveTarDirs",
        "TarGroups": "ArchiveTarGroups",
    }

    # Required types
    _opttypes = {
        "PreTarGroups": dict,
        "PostTarGroups": dict,
        "ArchiveTarGroups": dict,
        "_default_": (str, dict),
    }

    # Default values
    _rc = {
        "PreDeleteFiles": [],
        "PreDeleteDirs": [],
        "ArchiveFiles": [],
        "ArchiveTarGroups": {},
        "ArchiveTarDirs": [],
        "PostTarGroups": {},
        "PostTarDirs": [],
        "PostDeleteFiles": [],
        "PostDeleteDirs": [],
        "PostTailFiles": [],
    }

    # Descriptions
    _rst_descriptions = {
        "ArchiveFiles": "files to copy to archive",
        "ArchiveTarGroups": "groups of files to archive as one tarball",
        "ArchiveTarDirs": "folders to tar, e.g. ``fm/`` -> ``fm.tar``",
        "PostDeleteDirs": "folders to delete after archiving",
        "PostDeleteFiles": "files to delete after archiving",
        "PostTailFiles": "files to replace with their tail after archiving",
        "PostTailLines": "number of lines to keep in tail-ed files",
        "PostTarDirs": "folders to tar and then delete after archiving",
        "PostTarGroups": "groups of files to tar after archiving",
        "PreDeleteDirs": "folders to delete **before** archiving",
        "PreDeleteFiles": "files to delete **before** archiving",
    }

    # Apply defaults
    def init_post(self):
        self.update(self._rc)


# Class for "clean" phase
class ArchiveCleanOpts(ArchivePhaseOpts):
    r"""Options for the ``--clean`` archiving action

    The intent of the ``--clean`` phase is that it deletes files (and
    optionally archives files) in the case folder that don't prevent the
    continued running of the case or interfere with post-processing.
    """
    __slots__ = ()

    _name = "options for file clean-up while case is still running"


# Class for "archive" phase
class ArchiveArchiveOpts(ArchivePhaseOpts):
    r"""Options for the ``--archive`` archiving action

    The intent of the ``--archive`` phase is that it archives the
    critical contents of the case to long-term storage (of some kind)
    and only runs after a case has been marked ``PASS`` or ``ERROR``. In
    most cases, the ``--archive`` action will also delete many files to
    save space in the run folder. Ideally, post-processing should still
    be possible after archiving, but restarting the case may not be.
    """
    __slots__ = ()

    _name = "options for archiving and then cleanup case folders"


# Class for "skeleton" phase
class ArchiveSkeletonOpts(ArchivePhaseOpts):
    r"""Options for the ``--skeleton`` archiving action

    The ``--skeleton`` action first runs ``--archive``, and therefore it
    can only be done on a case marked PASS or FAIL. This is a final
    clean-up action that deletes many files and is one step short of
    deleting the entire case.

    The idea behind this capability is that it might preserve minimal
    information about each case such as how many iterations it ran, how
    many CPU hours it used, or just the residual history.

    There are no safety checks performed, so users may delete critical
    files.
    """
    __slots__ = ()

    _name = "options for further clean-up after archiving and post-processing"


class ArchiveOpts(OptionsDict):
    r"""Archiving options

    In addition to defining several generic properties of the archiving
    approach (format, full vs partial, glob vs regex, and location of
    archive), it defines three sections:

    * *clean*: archiving and folder clean-up while still running
    * *archive*: post-completion archiving, reports still generate
    * *skeleton*: post-archiving clean-up with no checks

    :See also:
        * :class:`ArchivePhaseOpts`
        * :class:`ArchiveCleanOpts`
        * :class:`ArchiveArchiveOpts`
        * :class:`ArchiveSkeletonOpts`
    """
    # No attributes
    __slots__ = ()

    # Description
    _name = "options for archiving and file clean-up"
    _subsec_name = "archive phase"

    # Allowed options
    _optlist = (
        "ArchiveFolder",
        "ArchiveFormat",
        "ArchiveType",
        "SearchMethod",
        "TailLines",
        "clean",
        "archive",
        "skeleton",
    )

    # Alternate options
    _optmap = {
        "progress": "clean",
    }

    # Section classes
    _sec_cls = {
        "clean": ArchiveCleanOpts,
        "archive": ArchiveArchiveOpts,
        "skeleton": ArchiveSkeletonOpts,
    }

    # Types
    _opttypes = {
        "ArchiveFolder": str,
        "ArchiveTemplate": str,
        "RemoteCopy": str,
        "SearchMethod": str,
    }

    # Limited allowed values
    _optvals = {
        "ArchiveFormat": (
            "",
            "tar",
            "gz",
            "bz2",
            "xz",
            "lzma",
            "zst",
            "zip"),
        "ArchiveType": ("full", "partial"),
        "SearchMethod": ("glob", "regex"),
    }

    # Parameters to avoid phasing
    _optlistdepth = {
        "_default_": 0,
    }

    # Default values
    _rc = {
        "ArchiveFolder": "",
        "ArchiveFormat": "tar",
        "ArchiveType": "full",
        "SearchMethod": "glob",
        "TailLines": {},
    }

    # Descriptions
    _rst_descriptions = {
        "ArchiveFolder": "path to the archive root",
        "ArchiveFormat": "format for case archives",
        "ArchiveType":  "flag for single (full) or multi (sub) archive files",
        "SearchMethod": "method for declaring multiple files with one string",
        "TailLines": "n lines to keep for any 'tail-ed' file, by regex/glob",
    }

    # Try to interpret "old" options
    def __init__(self, *args, **kw):
        # Read first arg
        a0 = {} if len(args) == 0 else args[0]
        # Check for any "old" options
        for opt in OLD_OPTMAP:
            # Check if found in these kwargs
            if opt in a0:
                a0 = self.read_old(**a0)
                break
        # Parent initialization method
        OptionsDict.__init__(self, a0, *args[1:], **kw)

    # Convert old options to new
    def read_old(self, **kw):
        r"""Convert CAPE 1.x *ArchiveOptions* to newer standard

        :Call:
            >>> kw_new = opts.read_old(**kw)
        :Inputs:
            *opts*: :class:`ArchiveOpts`
                Archiving options instance
            *kw*: :class:`dict`
                Options suitable for previous archiving options class
        :Outputs:
            *kw_new*: :class:`dict`
                Filtered and mapped version of *kw*
        :Versions:
            * 2024-09-17 ``@ddalle``: v1.0
        """
        # Read as "OldArchiveOpts"
        opts = OldArchiveOpts(**kw)
        # Initialize filtered options
        new_kw = dict(clean={}, archive={}, skeleton={})
        # Check for common options
        for opt in self.__class__._optlist:
            if opt in kw:
                new_kw[opt] = kw[opt]
        # Loop through old->new map
        for opt, (sec, subopt, n) in OLD_OPTMAP.items():
            # Check if found in *kw*
            if opt not in opts:
                continue
            # Get value
            rawval = kw[opt]
            # Check for empty
            if not rawval:
                continue
            # Expand
            if opt.endswith("TarGroups"):
                # Expand each
                val = {}
                if isinstance(rawval, list):
                    for _rawval in rawval:
                        for k, rawv in _rawval.items():
                            val[k] = expand_fileopt(rawv, n)
                else:
                    for k, rawv in rawval.items():
                        val[k] = expand_fileopt(rawv, n)
            else:
                # Expand whole
                val = expand_fileopt(rawval, n)
            # Initialize
            new_kw[sec].setdefault(subopt, {})
            new_kw[sec][subopt].update(val)
        # Output
        return new_kw

    # Archive extension
    def get_ArchiveExtension(self) -> str:
        r"""Get file extension for [compressed] archive groups

        :Call:
            >>> ext = opts.get_ArchiveExtension()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *ext*: :class:`str`
                File extension based on compression type
        :Versions:
            * 2024-09-14 ``@ddalle``: v1.0
            * 2024-09-17 ``@ddalle``: v1.1; fewer lines to test
        """
        # Get the format
        fmt = self.get_opt("ArchiveFormat")
        # Filter it
        for fmts, ext in TAR_EXTS.items():
            # Check if *fmt* is one of the formats for this extension
            if fmt in fmts:
                return ext
        # Default to ".tar"
        return ".tar"


# Add properties for limited options
_PROPS = (
    "ArchiveFolder",
    "ArchiveFormat",
    "ArchiveType",
    "SearchMethod",
)
ArchiveOpts.add_properties(_PROPS)
# Promote subsection commands
ArchiveOpts.promote_sections()


# Class for folder management and archiving
class OldArchiveOpts(OptionsDict):
    r"""Archive manangement options interface

    :Call:
        >>> opts = ArchiveOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of archive options
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Archive options interface
    :Versions:
        * 2016-30-02 ``@ddalle``: v1.0 (:class:`Archive`)
        * 2022-10-14 ``@ddalle``: v2.0; :class:`OptionsDict`
    """
    # List of recognized options
    _optlist = {
        "ArchiveAction",
        "ArchiveExtension",
        "ArchiveFiles",
        "ArchiveFolder",
        "ArchiveFormat",
        "ArchiveTarGroups",
        "ArchiveTemplate",
        "ArchiveType",
        "PostDeleteDirs",
        "PostDeleteFiles",
        "PostTarDirs",
        "PostTarGroups",
        "PostUpdateFiles",
        "PreDeleteDirs",
        "PreDeleteFiles",
        "PreTarDirs",
        "PreTarGroups",
        "PreUpdateFiles",
        "ProgressArchiveFiles",
        "ProgressDeleteDirs",
        "ProgressDeleteFiles",
        "ProgressUpdateFiles",
        "ProgressTarDirs",
        "ProgressTarGroups",
        "RemoteCopy",
        "SearchMethod",
        "SkeletonDirs",
        "SkeletonFiles",
        "SkeletonTailFiles",
        "SkeletonTarDirs",
        "TarAdapt",
        "TarViz",
        "nCheckPoint",
    }

    # Types
    _opttypes = {
        "ArchiveFolder": str,
        "ArchiveTemplate": str,
        "RemoteCopy": str,
        "SearchMethod": str,
        "TarAdapt": (bool, str),
        "TarViz": (bool, str),
        "nCheckPoint": INT_TYPES,
        "_default_": (str, dict),
    }

    # Limited allowed values
    _optvals = {
        "ArchiveAction": ("", "archive", "rm", "skeleton"),
        "ArchiveFormat": (
            "",
            "tar",
            "gz",
            "bz2",
            "xz",
            "lzma",
            "zst",
            "zip"),
        "ArchiveType": ("full", "partial"),
        "SearchMethod": ("glob", "regex"),
    }

    # Parameters to avoid phasing
    _optlistdepth = {
        "_default_": 0,
    }

    # Default values
    _rc = {
        "ArchiveAction": "archive",
        "ArchiveFolder": "",
        "ArchiveFormat": "tar",
        "ArchiveType": "full",
        "ArchiveTemplate": "full",
        "ArchiveFiles": [],
        "PostDeleteDirs": [],
        "PostDeleteFiles": [],
        "PostTarDirs": [],
        "PostTarGroups": [],
        "ProgressDeleteFiles": [],
        "ProgressDeleteDirs": [],
        "ProgressTarGroups": [],
        "ProgressTarDirs": [],
        "ProgressUpdateFiles": [],
        "ProgressArchiveFiles": [],
        "PreDeleteFiles": [],
        "PreDeleteDirs": [],
        "PreTarDirs": [],
        "PreTarGroups": [],
        "PreUpdateFiles": [],
        "PostUpdateFiles": [],
        "SearchMethod": "glob",
        "SkeletonFiles": ["case.json"],
        "SkeletonTailFiles": [],
        "SkeletonTarDirs": [],
        "RemoteCopy": "scp",
        "TarAdapt": True,
        "TarViz": True,
        "nCheckPoint": 1,
    }

    # Descriptions
    _rst_descriptions = {
        "ArchiveAction": "action to take after finishing a case",
        "ArchiveExtension": "archive file extension",
        "ArchiveFiles": "files to copy to archive",
        "ArchiveFolder": "path to the archive root",
        "ArchiveFormat": "format for case archives",
        "ArchiveTemplate": "template for default archive settings",
        "ArchiveType":  "flag for single (full) or multi (sub) archive files",
        "RemoteCopy": "command for archive remote copies",
        "PostDeleteDirs": "list of folders to delete after archiving",
        "PostDeleteFiles": "list of files to delete after archiving",
        "PostTarDirs": "folders to tar after archiving",
        "PostTarGroups": "groups of files to tar after archiving",
        "PostUpdateFiles": "globs: keep *n* and rm older, after archiving",
        "PreDeleteDirs": "folders to delete **before** archiving",
        "PreDeleteFiles": "files to delete **before** archiving",
        "PreTarGroups": "file groups to tar before archiving",
        "PreTarDirs": "folders to tar before archiving",
        "PreUpdateFiles": "files to keep *n* and delete older, b4 archiving",
        "ProgressArchiveFiles": "files to archive at any time",
        "ProgressDeleteDirs": "folders to delete while still running",
        "ProgressDeleteFiles": "files to delete while still running",
        "ProgressUpdateFiles": "files to delete old versions while running",
        "ProgressTarDirs": "folders to tar while running",
        "ProgressTarGroups": "list of file groups to tar while running",
        "SearchMethod": "method for declaring multiple files with one string",
        "SkeletonDirs": "folders to **keep** during skeleton action",
        "SkeletonFiles": "files to **keep** during skeleton action",
        "SkeletonTailFiles": "files to tail before deletion during skeleton",
        "SkeletonTarDirs": "folders to tar before deletion during skeleton",
    }

    # Archive extension
    def get_ArchiveExtension(self) -> str:
        r"""Get file extension for [compressed] archive groups

        :Call:
            >>> ext = opts.get_ArchiveExtension()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *ext*: :class:`str`
                File extension based on compression type
        :Versions:
            * 2024-09-14 ``@ddalle``: v1.0
        """
        # Get the format
        fmt = self.get_opt("ArchiveFormat")
        # Filter it
        if fmt == "":
            # No compression
            return ".tar"
        elif fmt in ("j", "bz2", "bzip2"):
            # bzip2
            return ".tar.bz2"
        elif fmt in ("z", "gz", "gzip"):
            # gzip
            return ".tar.gz"
        elif fmt in ("J", "xz"):
            # xz
            return ".tar.xz"
        elif fmt in ("lzma",):
            # lzma
            return ".tar.lzma"
        elif fmt in ("zst", "zstd"):
            # zstd
            return ".tar.zst"
        elif fmt in ("zip",):
            # zip
            return ".zip"
        # Default to ".tar"
        return ".tar"

    # Archive command
    def get_ArchiveCmd(self):
        r"""Get archiving command

        :Call:
            >>> cmd = opts.get_ArchiveCmd()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *cmd*: :class:`list`\ [:class:`str`]
                Tar command and appropriate flags
        :Versions:
            * 2016-03-01 ``@ddalle``: v1.0
        """
        # Get the format
        fmt = self.get_opt("ArchiveFormat")
        # Get command
        return TAR_CMDS.get(fmt, ["tar", "-cf"])

    # Unarchive command
    def get_UnarchiveCmd(self):
        r"""Get command to unarchive

        :Call:
            >>> cmd = opts.get_UnarchiveCmd()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *cmd*: :class:`list`\ [:class:`str`]
                Untar command and appropriate flags
        :Versions:
            * 2016-03-01 ``@ddalle``: v1.0
        """
        # Get the format
        fmt = self.get_opt("ArchiveFormat")
        # Get command
        return UNTAR_CMDS.get(fmt, ["tar", "-xf"])


# Turn dictionary into Archive options
def auto_Archive(opts, cls=ArchiveOpts):
    r"""Automatically convert :class:`dict` to :class:`ArchiveOpts`

    :Call:
        >>> opts = auto_Archive(opts)
    :Inputs:
        *opts*: :class:`dict`
            Dict of either global, "RunControl" or "Archive" options
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Instance of archiving options
    :Versions:
        * 2016-02-29 ``@ddalle``: v1.0
        * 2022-10-21 ``@ddalle``: v2.0, add *cls* input
    """
    # Check type
    if not isinstance(opts, dict):
        # Invalid type
        raise TypeError(
            "Expected input type 'dict'; got '%s'" % type(opts).__name__)
    # Downselect if appropriate
    opts = opts.get("RunControl", opts)
    opts = opts.get("Archive", opts)
    # Check if already initialized
    if isinstance(opts, cls):
        # Good; quit
        return opts
    else:
        # Convert to class
        return cls(**opts)


# Convert one of several deletion opts into common format
def expand_fileopt(rawval: Union[list, dict, str], vdef: int = 0) -> dict:
    r"""Expand *Archive* file name/list/dict to common format

    The output is a :class:`dict` where the key is the pattern of file
    names to process and the value is an :class:`int` that represents
    the number of most recent files matching that pattern to keep.

    :Call:
        >>> patdict = expand_fileopt(rawstr, vdef=0)
        >>> patdict = expand_fileopt(rawlist, vdef=0)
        >>> patdict = expand_fileopt(rawdict, vdef=0)
    :Inputs:
        *rawstr*: :class:`str`
            Pattern of file names to process
        *rawlist*: :class:`list`\ [:class:`str`]
            List of filee name patterns to process
        *rawdict*: :class:`dict`\ [:class:`int`]
            Dictionary of file name patterns to process and *n* to keep
    :Outputs:
        *patdict*: :class:`dict`\ [:class:`int`]
            File name patterns as desribed above
    :Versions:
        * 2024-09-02 ``@ddalle``: v1.0
    """
    # Check for dict
    if isinstance(rawval, dict):
        # Copy it
        optval = dict(rawval)
        # Remove any non-int
        for k, v in rawval.items():
            # Check type
            if not isinstance(v, INT_TYPES):
                optval.pop(k)
        # Output
        return optval
    # Check for string
    if isinstance(rawval, str):
        return {rawval: vdef}
    # Initialize output for list
    optval = {}
    # Loop through items of list
    for rawj in rawval:
        # Check type
        if not isinstance(rawj, (dict, str, list, tuple)):
            continue
        # Recurse
        valj = expand_fileopt(rawj, vdef=vdef)
        # Save to total dict
        optval.update(valj)
    # Output
    return optval
