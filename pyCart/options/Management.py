"""Interface for options specific to running `adjointCart`"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class Management(odict):
    """Dictionary-based interfaced for options specific to folder management"""
    
    
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
    def set_ArchiveType(self, atype=rc0('ArchiveType')):
        """Set the archive type; determines what is deleted before archiving
        
        :Call:
            >>> opts.set_ArchiveType(atype)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *atype*: {"full"} | "viz" | "best" | "hist"
                Name of archive type
        :Versions:
            * 2015-02-16 ``@ddalle``: First version
        """
        self.set_key('ArchiveType', atype)
        
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
    def set_RemoteCopy(self):
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
        
    # Get number of check points to keep around
    def get_nCheckPoint(self):
        """Return the number of check point files to keep
        
        :Call:
            >>> nchk = opts.get_nCheckPoint()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('nCheckPoint')
        
    # Set the number of check point files to keep around
    def set_nCheckPoint(self, nchk=rc0('nCheckPoint')):
        """Set the number of check point files to keep
        
        :Call:
            >>> opts.set_nCheckPoint(nchk)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('nCheckPoint', nchk)
        
    # Get the archive format for visualization files
    def get_TarViz(self):
        """Return the archive format for visualization files
        
        :Call:
            >>> fmt = opts.get_TarViz()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarViz')
        
    # Set the archive format for visualization files
    def set_TarViz(self, fmt=rc0('TarViz')):
        """Set the archive format for visualization files
        
        :Call:
            >>> opts.set_TarViz(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarViz', fmt)
        
    # Get the archive format for visualization files
    def get_TarAdapt(self):
        """Return the archive format for adapt folders
        
        :Call:
            >>> fmt = opts.get_TarAdapt()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('TarAdapt')
        
    # Set the archive format for visualization files
    def set_TarAdapt(self, fmt=rc0('TarAdapt')):
        """Set the archive format for adapt folders
        
        :Call:
            >>> opts.set_TarAdapt(fmt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmt*: ``""`` | {``"tar"``} | ``"gzip"`` | ``"bz2"``
                Archive format
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('TarAdapt', fmt)
        
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
        
