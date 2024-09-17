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
from .util import OptionsDict
from ...optdict import INT_TYPES


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


# Class for folder management and archiving
class ArchiveOpts(OptionsDict):
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
    }

    # Types
    _opttypes = {
        "ArchiveFolder": str,
        "ArchiveTemplate": str,
        "RemoteCopy": str,
        "SearchMethod": str,
        "_default_": (str, dict),
    }

    # Limited allowed values
    _optvals = {
        "ArchiveAction": ("", "archive", "rm", "skeleton"),
        "ArchiveFormat": (
            "",
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

   # -----
   # Tools
   # -----
   # <
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
   # >


# Normal get/set options
_ARCHIVE_PROPS = (
    "ArchiveAction",
    "ArchiveFolder",
    "ArchiveFormat",
    "ArchiveTemplate",
    "ArchiveType",
    "RemoteCopy",
)
# Getters and extenders only
_GETTER_OPTS = (
    "ArchiveFiles",
    "PostDeleteDirs",
    "PostDeleteFiles",
    "PostTarDirs",
    "PostTarGroups",
    "PostUpdateFiles",
    "PreDeleteDirs",
    "PreDeleteFiles",
    "PreTarGroups",
    "PreTarDirs",
    "PreUpdateFiles",
    "ProgressArchiveFiles",
    "ProgressDeleteDirs",
    "ProgressDeleteFiles",
    "ProgressUpdateFiles",
    "ProgressTarDirs",
    "ProgressTarGroups",
    "SkeletonFiles",
    "SkeletonDirs",
    "SkeletonTailFiles",
    "SkeletonTarDirs",
)

# Add full options
ArchiveOpts.add_properties(_ARCHIVE_PROPS)
ArchiveOpts.add_properties(["SearchMethod"], prefix="Archive")
# Add getters only
ArchiveOpts.add_getters(_GETTER_OPTS, prefix="Archive")
ArchiveOpts.add_extenders(_GETTER_OPTS, prefix="Archive")


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
