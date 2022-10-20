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
from ...optdict import ARRAY_TYPES
from .util import OptionsDict


# Turn dictionary into Archive options
def auto_Archive(opts):
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
    if isinstance(opts, ArchiveOpts):
        # Good; quit
        return opts
    else:
        # Convert to class
        return ArchiveOpts(**opts)


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
        "ArchiveFolder",
        "ArchiveFormat",
        "ArchiveTemplate",
        "ArchiveType",
        "PreDeleteDirs",
        "PreDeleteFiles",
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
        "_default_": str,
    }

    # Limited allowed values
    _optvals = {
        "ArchiveAction": ("", "archive", "rm", "skeleton"),
        "ArchiveExtension": ("tar", "tgz", "bz2", "zip"),
        "ArchiveFormat": ("", "tar", "gzip", "bz2", "zip"),
        "ArchiveType": ("full", "sub"),
    }

    # Parameters to avoid phasing
    _optlistdepth = {
        "_default_": 1,
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
        "ProgressDeleteFiles": [],
        "ProgressDeleteDirs": [],
        "ProgressTarGroups": [],
        "ProgressTarDirs": [],
        "ProgressUpdateFiles": [],
        "ProgressArchiveFiles": [],
        "PreDeleteFiles": [],
        "PreDeleteDirs": [],
        "PreTarGroups": [],
        "PreTarDirs": [],
        "PreUpdateFiles": [],
        "PostDeleteFiles": [],
        "PostDeleteDirs": [],
        "PostTarGroups": [],
        "PostTarDirs": [],
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
        "ArchiveFolder": "path to the archive root",
        "ArchiveFormat": "format for case archives",
        "ArchiveTemplate": "template for default archive settings",
        "ArchiveType":  "flag for single (full) or multi (sub) archive files",
        "RemoteCopy": "command for archive remote copies",
        "PreDeleteDirs": "folders to delete **before** archiving",
        "PreDeleteFiles": "files to delete **before** archiving",
        "PreTarGroups": "file groups to tar before archiving",
        "PreTarDirs": "folders to tar before archiving",
        "PreUpdateFiles": "files to keep *n* and delete older, b4 archiving",
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

    # Get the umask
    def get_umask(self):
        r"""Get the current file permissions mask
        
        The default value is the read from the system
        
        :Call:
            >>> umask = opts.get_umask(umask=None)
        :Inputs:
            *opts* :class:`cape.cfdx.options.Options`
                Options interface
        :Outputs:
            *umask*: :class:`oct`
                File permissions mask
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Read the option.
        umask = self.get('umask')
        # Check if we need to use the default.
        if umask is None:
            # Get the value.
            umask = os.popen('umask', 'r').read()
            # Convert to value.
            umask = eval('0o' + umask.strip())
        elif type(umask).__name__ in ['str', 'unicode']:
            # Convert to octal
            umask = eval('0o' + str(umask).strip().lstrip('0o'))
        # Output
        return umask
        
    # Set the umask
    def set_umask(self, umask):
        r"""Set the current file permissions mask
        
        :Call:
            >>> umask = opts.get_umask(umask=None)
        :Inputs:
            *opts* :class:`cape.cfdx.options.Options`
                Options interface
        :Outputs:
            *umask*: :class:`oct`
                File permissions mask
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Default
        if umask is None:
            # Get the value.
            umask = os.popen('umask', 'r', 1).read()
            # Convert to value.
            self['umask'] = '0o' + umask.strip()
        elif type(umask).__name__ in ['str', 'unicode']:
            # Set the value as an octal number
            self['umask'] = '0o' + str(umask)
        else:
            # Convert to octal
            self['umask'] = '0o' + oct(umask)
        
    # Get the directory permissions to use
    def get_dmask(self):
        r"""Get the permissions to assign to new folders
        
        :Call:
            >>> dmask = opts.get_dmask()
        :Inputs:
            *opts* :class:`cape.cfdx.options.Options`
                Options interface
        :Outputs:
            *umask*: :class:`int`
                File permissions mask
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Get the umask
        umask = self.get_umask()
        # Subtract UMASK from full open permissions
        return 0o0777 - umask
        
    # Apply the umask
    def apply_umask(self):
        r"""Apply the permissions filter
        
        :Call:
            >>> opts.apply_umask()
        :Inputs:
            *opts* :class:`cape.cfdx.options.Options`
                Options interface
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        os.umask(self.get_umask())
            
    # Make a directory
    def mkdir(self, fdir):
        r"""Make a directory with the correct permissions
        
        :Call:
            >>> opts.mkdir(fdir)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fdir*: :class:`str`
                Directory to create
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Get umask
        umask = self.get_umask()
        # Apply umask
        dmask = 0o777 - umask
        # Make the directory.
        os.mkdir(fdir, dmask)
   # >
   
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
        # Process
        if fmt in ['gzip' ,'tgz']:
            # Gzip
            return ['tar', '-czf']
        elif fmt in ['zip']:
            # Zip
            return ['zip', '-r']
        elif fmt in ['bzip', 'bz', 'bzip2', 'bz2', 'tbz', 'tbz2']:
            # Bzip2
            return ['tar', '-cjf']
        else:
            # Default: tar
            return ['tar', '-cf']
            
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
        # Process
        if fmt in ['gzip', 'tgz']:
            # Gzip
            return ['tar', '-xzf']
        elif fmt in ['zip']:
            # Zip
            return ['unzip']
        elif fmt in ['bzip', 'bz', 'bzip2', 'bz2', 'tbz', 'tbz2']:
            # Bzip2
            return ['tar', '-xjf']
        else:
            # Default: tar
            return ['tar', '-xf']
   # >
    
   # ---------
   # Archiving
   # ---------
   # <
    # List of files to archive
    def get_ArchiveArchiveFiles(self):
        """Get list of files to copy to archive, in addition to tar balls
        
        :Call:
            >>> fglob = opts.get_ArchiveArchiveFiles()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of file wild cards to delete after archiving
        :Versions:
            * 2016-12-09 ``@ddalle``: Version 1.0
        """
        return self.get_key("ArchiveFiles")
        
    # Add to list of files to archive
    def add_ArchiveArchiveFiles(self, farch):
        """Add to the list of files to copy to archive
        
        :Call:
            >>> opts.add_ArchiveArchiveFiles(farch)
            >>> opts.add_ArchiveArchiveFiles(larch)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *farch*: :class:`str`
                File or file glob to archive
            *larch*: :class:`list`\ [:class:`str`]
                List of file or file globs to add to list
        :Versions:
            * 2016-12-09 ``@ddalle``: Version 1.0
        """
        self.add_to_key("ArchiveFiles", farch)
   # >
    
   # -------------------------
   # Post-Archiving Processing
   # -------------------------
   # <
    # List of files to delete
    def get_ArchivePostDeleteFiles(self):
        """Get list of files to delete after archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePreDeleteFiles()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of file wild cards to delete after archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        return self.get_key("PostDeleteFiles")
        
    # Add to list of files to delete
    def add_ArchivePostDeleteFiles(self, fpost):
        """Add to the list of files to delete after archiving
        
        :Call:
            >>> opts.add_ArchivePostDeleteFiles(fpost)
            >>> opts.add_ArchivePostDeleteFiles(lpost)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpost*: :class:`str`
                File or file glob to add to list
            *lpost*: :class:`list`\ [:class:`str`]
                List of files or file globs to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        self.add_to_key("PostDeleteFiles", fpost)
            
    # List of folders to delete
    def get_ArchivePostDeleteDirs(self):
        """Get list of folders to delete after archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostDeleteDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of globs of folders to delete after archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        return self.get_key("PostDeleteDirs")
        
    # Add to list of folders to delete
    def add_ArchivePostDeleteDirs(self, fpost):
        """Add to the list of folders to delete after archiving
        
        :Call:
            >>> opts.add_ArchivePostDeleteDirs(fpost)
            >>> opts.add_ArchivePostDeleteDirs(lpost)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpost*: :class:`str`
                Folder or file glob to add to list
            *lpost*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        self.add_to_key("PostDeleteDirs", fpost)
            
    # List of files to tar after archiving
    def get_ArchivePostTarGroups(self):
        """Get list of files to tar prior to archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostTarGroups()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: Version 1.0
        """
        return self.get_key("PostTarGroups")
        
    # Add to list of folders to delete
    def add_ArchivePostTarGroups(self, fpost):
        """Add to the list of groups to tar before archiving
        
        :Call:
            >>> opts.add_ArchivePostTarGroups(fpost)
            >>> opts.add_ArchivePostTarGroups(lpost)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpost*: :class:`str`
                File glob to add to list
            *lpost*: :class:`str`
                List of globs of files to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        self.add_to_key("PostTarGroups", fpost)
    
    # List of folders to tar before archiving
    def get_ArchivePostTarDirs(self):
        """Get list of folders to tar after archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostTarDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of globs of folders to delete fter archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: Version 1.0
        """
        return self.get_key("PostTarDirs")
        
    # Add to list of folders to delete
    def add_ArchivePostTarDirs(self, fpost):
        """Add to the folders of groups to tar after archiving
        
        :Call:
            >>> opts.add_ArchivePostTarDirs(fpost)
            >>> opts.add_ArchivePostTarDirs(lpost)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpost*: :class:`str`
                Folder or folder glob to add to list
            *lpost*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        self.add_to_key("PostTarDirs", fpost)
        
    # List of files to update before archiving
    def get_ArchivePostUpdateFiles(self):
        """Get :class:`dict` of files of which to keep only *n*
        
        :Call:
            >>> fglob = opts.get_ArchivePostUpdateFiles()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        return self.get_key("PostUpdateFiles")
        
    # Add to list of files to update before archiving
    def add_ArchivePostUpdateFiles(self, fpost):
        """Add to :class:`dict` of files of which to keep only *n*
        
        :Call:
            >>> opts.add_ArchivePostUpdateFiles(fpost)
            >>> opts.add_ArchivePostUpdateFiles(lpost)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpost*: :class:`str`
                Folder or folder glob to add to list
            *lpost*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: Version 1.0
        """
        self.add_to_key("PostUpdateFiles", fpost)
        
    # List of files to archive at end of each run
    def get_ArchiveProgressArchiveFiles(self):
        """Get :class:`dict` of files to archive at end of each run
        
        :Call:
            >>> fglob = opts.get_ArchiveProgressArchiveFiles()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list`\ [:class:`str`]
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-03-14 ``@ddalle``: Version 1.0
        """
        return self.get_key("ProgressArchiveFiles")
        
    # Add to list of files to archive at end of each run
    def add_ArchiveProgressArchiveFiles(self, fpro):
        """Add to :class:`dict` of files of which to keep only *n*
        
        :Call:
            >>> opts.add_ArchiveProgressArchiveFiles(fpro)
            >>> opts.add_ArchiveProgressArchiveFiles(lpro)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpro*: :class:`str`
                Folder or folder glob to add to list
            *lpo*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-03-14 ``@ddalle``: Version 1.0
        """
        self.add_to_key("ProgressArchiveFiles", fpro)
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
    "PreDeleteDirs",
    "PreDeleteFiles",
    "PreTarGroups",
    "PreTarDirs",
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
