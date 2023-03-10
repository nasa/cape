r"""
:mod:`cape.cfdx.archive.Archive`: Case archiving options
=========================================================

This module provides a class to access options relating to archiving
folders that were used to run CFD simulations.

The class provided in this module, :class:`ArchiveOpts`, is
loaded into the ``"RunControl"`` section of the main options interface.
"""

# Standard library
import os

# Local imports
from .util import OptionsDict


# Constants
TAR_CMDS = {
    "bz2": ["tar", "-cjf"],
    "tar": ["tar", "-cf"],
    "tgz": ["tar", "-czf"],
    "zip": ["zip", "-r"],
}
UNTAR_CMDS = {
    "bz2": ["tar", "-xjf"],
    "tar": ["tar", "-xf"],
    "tgz": ["tar", "-xzf"],
    "zip": ["unzip", "-r"],
}


# Class for folder management and archiving
class ArchiveOpts(OptionsDict):
    r"""Archive mamangement options interface

    :Call:
        >>> opts = ArchiveOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of archive options
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Archive options interface
    :Versions:
        * 2016-30-02 ``@ddalle``: Version 1.0 (:class:`Archive`)
        * 2022-10-14 ``@ddalle``: Version 2.0; :class:`OptionsDict`
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
        "_default_": (str, dict),
    }

    # Limited allowed values
    _optvals = {
        "ArchiveAction": ("", "archive", "rm", "skeleton"),
        "ArchiveExtension": ("tar", "tgz", "bz2", "zip"),
        "ArchiveFormat": ("", "tar", "tgz", "bz2", "zip"),
        "ArchiveType": ("full", "sub"),
    }

    # Parameters to avoid phasing
    _optlistdepth = {
        "_default_": 0,
    }

    # Default values
    _rc = {
        "ArchiveAction": "archive",
        "ArchiveExtension": "tar",
        "ArchiveFolder": "",
        "ArchiveFormat": "tar",
        "ArchiveProgress": True,
        "ArchiveType": "full",
        "ArchiveTemplate": "full",
        "ArchiveFiles": [],
        "ArchiveGroups": [],
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
        "SkeletonDirs": "folders to **keep** during skeleton action",
        "SkeletonFiles": "files to **keep** during skeleton action",
        "SkeletonTailFiles": "files to tail before deletion during skeleton",
        "SkeletonTarDirs": "folders to tar before deletion during skeleton",
    }

   # -----
   # Tools
   # -----
   # <
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
            * 2016-03-01 ``@ddalle``: Version 1.0
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
            * 2016-03-01 ``@ddalle``: Version 1.0
        """
        # Get the format
        fmt = self.get_opt("ArchiveFormat")
        # Get command
        return UNTAR_CMDS.get(fmt, ["tar", "-xf"])
   # >


# Normal get/set options
_ARCHIVE_PROPS = (
    "ArchiveAction",
    "ArchiveExtension",
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
        * 2016-02-29 ``@ddalle``: Version 1.0
        * 2022-10-21 ``@ddalle``: Version 2.0, add *cls* input
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
