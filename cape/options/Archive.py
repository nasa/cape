"""
Interface to case archiving options
===================================

This module provides a class to access options relating to archiving folders
that were used to run CFD analysis
"""

# Ipmort options-specific utilities
from util import rc0, odict, getel


# Class for folder management and archiving
class Archive(odict):
    """Dictionary-based interfaced for options specific to folder management"""
    
   
    # -------------
    # Basic Options
    # -------------
   # <
    # Archive base folder
    def get_ArchiveFolder(self):
        """Return the path to the archive
        
        :Call:
            >>> fdir = opts.get_ArchiveFolder()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fdir*: :class:`str`
                Path to root directory of archive
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveFolder')
        
    # Set archive base folder
    def set_ArchiveFolder(self, fdir=rc0('ArchiveFolder')):
        """Set the path to the archive
        
        :Call:
            >>> opts.set_ArchiveFolder(fdir)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fdir*: :class:`str`
                Path to root directory of archive
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveFolder', fdir)
        
    # Archive format
    def get_ArchiveFormat(self):
        """Return the format for folder archives
        
        :Call:
            >>> fmt = opts.get_ArchiveFormat()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveFormat')
        
    # Set archive format
    def set_ArchiveFormat(self, fmt=rc0('ArchiveFormat')):
        """Set the format for folder archives
        
        :Call:
            >>> opts.set_ArchiveFormat(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveFormat', fmt)
        
    # Get archive type
    def get_ArchiveType(self):
        """Return the archive type; whether or not to save as a single tar ball
        
            * ``"full"``: Archives all contents of the case as one tar ball
            * ``"sub"``: Archive case as group of tar balls/files
        
        :Call:
            >>> atype = opts.get_ArchiveType()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {"full"} | "sub"
                Name of archive type
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        return self.get_key('ArchiveType')
        
    # Set archive type
    def set_ArchiveType(self, atype=rc0('ArchiveType')):
        """Set the archive type; determines what is deleted before archiving
        
        :Call:
            >>> opts.set_ArchiveType(atype)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *atype*: {"full"} | "sub"
                Name of archive type
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        self.set_key('ArchiveType', atype)
        
    # Get archive type
    def get_ArchiveTemplate(self):
        """Return the archive type; determines what is deleted before archiving
        
        Note that all types except ``"full"`` will delete files from the working
        folder before archiving.  However, archiving actions, including the
        preliminary deletions, only take place if the case is marked *PASS*.
        
            * ``"full"``: Archives all contents of run directory
            * ``"best"``: Deletes all previous adaptation folders
            * ``"viz"``: Deletes all :file:`Mesh*.c3d` files and check files
            * ``"hist"``: Deletes all mesh, tri, and TecPlot files
        
        :Call:
            >>> atype = opts.get_ArchiveType()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {"full"} | "viz" | "best" | "hist"
                Name of archive type
        :Versions:
            * 2015-02-16 ``@ddalle``: First version
        """
        return self.get_key('ArchiveType')
        
    # Set archive type
    def set_ArchiveTemplate(self, atype=rc0('ArchiveTemplate')):
        """Set the archive type; determines what is deleted before archiving
        
        :Call:
            >>> opts.set_ArchiveTemplate(atype)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *atype*: {"full"} | "viz" | "best" | "hist"
                Name of archive template
        :Versions:
            * 2015-02-16 ``@ddalle``: First version
            * 2016-02-29 ``@ddalle``: Moved from previous *ArchiveType*
        """
        self.set_key('ArchiveTemplate', atype)
        
    # Get archiving action
    def get_ArchiveAction(self):
        """Return the action to take after finishing a case
        
        :Call:
            >>> fcmd = opts.get_ArchiveAction()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {""} | "skeleton" | "rm" | "archive"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('ArchiveAction')
        
    # Set archiving action
    def set_ArchiveAction(self, fcmd=rc0('ArchiveAction')):
        """Set the action to take after finishing a case
        
        :Call:
            >>> opts.set_ArchiveAction(fcmd)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fcmd*: {""} | "skeleton" | "rm" | "archive"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('ArchiveAction', fcmd)
        
    # Get archiving command
    def get_RemoteCopy(self):
        """Return the command used for remote copying
        
        :Call:
            >>> fcmd = opts.get_RemoteCopy()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fcmd*: {"scp"} | "rsync" | "shiftc --wait"
                Command to use for archiving
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('RemoteCopy')
        
    # Set archiving command
    def set_RemoteCopy(self, fcmd='scp'):
        """Set the command used for remote copying
        
        :Call:
            >>> opts.set_RemoteCopy(fcmd)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fcmd*: {"scp"} | "rsync" | "shiftc"
                Type of archiving action to take
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('RemoteCopy', fcmd)
        
    # Get the archive format for PBS output files
    def get_TarPBS(self):
        """Return the archive format for adapt folders
        
        :Call:
            >>> fmt = opts.get_TarPBS()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarPBS')
        
    # Set the archive format for visualization files
    def set_TarPBS(self, fmt=rc0('TarPBS')):
        """Set the archive format for adapt folders
        
        :Call:
            >>> opts.set_TarPBS(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarPBS', fmt)
   # >
    
    # -----------------------
    # Sub-Archive Definitions
    # -----------------------
   # <
   
   # >
   
    # ------------------------
    # Pre-Archiving Processing
    # ------------------------
   # <
    # List of files to delete
    def get_ArchivePreDeleteFiles(self):
        """Get list of files to delete **before** archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePreDeleteFiles()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        return self.get_key("PreDeleteFiles")
        
    # Add to list of files to delete
    def add_ArchivePreDeleteFiles(self, fpre):
        """Add to the list of files to delete before archiving
        
        :Call:
            >>> opts.add_ArchivePreDeleteFiles(fpre)
            >>> opts.add_ArchivePreDeleteFiles(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                File or file glob to add to list
            *lpre*: :class:`list` (:class:`str`)
                List of files or file globs to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get the current list
        fdel = self.get_key("PreDeleteFiles")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PreDeleteFiles"] = fdel
            
    # List of folders to delete
    def get_ArchivePreDeleteDirs(self):
        """Get list of folders to delete **before** archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePreDeleteDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        return self.get_key("PreDeleteDirs")
        
    # Add to list of folders to delete
    def add_ArchivePreDeleteDirs(self, fpre):
        """Add to the list of folders to delete before archiving
        
        :Call:
            >>> opts.add_ArchivePreDeleteDirs(fpre)
            >>> opts.add_ArchivePreDeleteDirs(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                Folder or file glob to add to list
            *lpre*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PreDeleteDirs")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PreDeleteDirs"] = fdel
    
    # List of files to tar before archiving
    def get_ArchivePreArchiveGroups(self):
        """Get list of files to tar prior to archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePreArchiveGroups()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: First version
        """
        return self.get_key("PreArchiveGroups")
        
    # Add to list of folders to delete
    def add_ArchivePreArchiveGroups(self, fpre):
        """Add to the list of groups to tar before archiving
        
        :Call:
            >>> opts.add_ArchivePreArchiveGroups(fpre)
            >>> opts.add_ArchivePreArchiveGroups(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                File glob to add to list
            *lpre*: :class:`str`
                List of globs of files to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PreArchiveGroups")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PreArchiveGroupss"] = fdel
    
    # List of folders to tar before archiving
    def get_ArchivePreArchiveDirs(self):
        """Get list of folders to tar prior to archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePreArchiveDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: First version
        """
        return self.get_key("PreArchiveDirs")
        
    # Add to list of folders to delete
    def add_ArchivePreArchiveDirs(self, fpre):
        """Add to the folders of groups to tar before archiving
        
        :Call:
            >>> opts.add_ArchivePreArchiveDirs(fpre)
            >>> opts.add_ArchivePreArchiveDirs(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                Folder or folder glob to add to list
            *lpre*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PreArchiveDirs")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PreArchiveDirs"] = fdel
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
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete after archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        return self.get_key("PostDeleteFiles")
        
    # Add to list of files to delete
    def add_ArchivePostDeleteFiles(self, fpre):
        """Add to the list of files to delete after archiving
        
        :Call:
            >>> opts.add_ArchivePostDeleteFiles(fpre)
            >>> opts.add_ArchivePostDeleteFiles(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                File or file glob to add to list
            *lpre*: :class:`list` (:class:`str`)
                List of files or file globs to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get the current list
        fdel = self.get_key("PostDeleteFiles")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PostDeleteFiles"] = fdel
            
    # List of folders to delete
    def get_ArchivePostDeleteDirs(self):
        """Get list of folders to delete after archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostDeleteDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of globs of folders to delete after archiving
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        return self.get_key("PostDeleteDirs")
        
    # Add to list of folders to delete
    def add_ArchivePostDeleteDirs(self, fpre):
        """Add to the list of folders to delete after archiving
        
        :Call:
            >>> opts.add_ArchivePostDeleteDirs(fpre)
            >>> opts.add_ArchivePostDeleteDirs(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                Folder or file glob to add to list
            *lpre*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PostDeleteDirs")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PostDeleteDirs"] = fdel
            
    # List of files to tar after archiving
    def get_ArchivePostArchiveGroups(self):
        """Get list of files to tar prior to archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostArchiveGroups()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of file wild cards to delete before archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: First version
        """
        return self.get_key("PostArchiveGroups")
        
    # Add to list of folders to delete
    def add_ArchivePostArchiveGroups(self, fpre):
        """Add to the list of groups to tar before archiving
        
        :Call:
            >>> opts.add_ArchivePostArchiveGroups(fpre)
            >>> opts.add_ArchivePostArchiveGroups(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                File glob to add to list
            *lpre*: :class:`str`
                List of globs of files to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PostArchiveGroups")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PostArchiveGroups"] = fdel
    
    # List of folders to tar before archiving
    def get_ArchivePostArchiveDirs(self):
        """Get list of folders to tar after archiving
        
        :Call:
            >>> fglob = opts.get_ArchivePostArchiveDirs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fglob*: :class:`list` (:class:`str`)
                List of globs of folders to delete fter archiving
        :Versions:
            * 2016-02-029 ``@ddalle``: First version
        """
        return self.get_key("PostArchiveDirs")
        
    # Add to list of folders to delete
    def add_ArchivePostArchiveDirs(self, fpre):
        """Add to the folders of groups to tar after archiving
        
        :Call:
            >>> opts.add_ArchivePostArchiveDirs(fpre)
            >>> opts.add_ArchivePostArchiveDirs(lpre)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fpre*: :class:`str`
                Folder or folder glob to add to list
            *lpre*: :class:`str`
                List of folders or globs of folders to add to list
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get Current list
        fdel = self.get_key("PostArchiveDirs")
        # Check input type
        if type(fpre).__name__ not in ['list', 'ndarray']:
            # Ensure list
            fpre = [fpre]
        # Append each file to the list
        for fi in fpre:
            # Check if the file is already there
            if fi not in fdel: fdel.append(fi)
        # Set the parameter
        self["PostArchiveDirs"] = fdel
   # >
            
# class Archive


