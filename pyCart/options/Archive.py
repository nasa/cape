"""Case archiving options for Cart3D solutions"""

# Import options-specific utilities
from util import rc0
# Base module
import cape.options.Archive

# Turn dictionary into Archive options
def auto_Archive(opts):
    """Automatically convert dict to :mod:`pyCart.options.Archive.Archive`
    
    :Call:
        >>> opts = auto_Archive(opts)
    :Inputs:
        *opts*: :class:`dict`
            Dict of either global, "RunControl" or "Archive" options
    :Outputs:
        *opts*: :class:`pyCart.options.Archive.Archive`
            Instance of archiving options
    :Versions:
        * 2016-02-29 ``@ddalle``: First version
    """
    # Get type
    t = type(opts).__name__
    # Check type
    if t == "Archive":
        # Good; quit
        return opts
    elif t == "RunControl":
        # Get the sub-object
        return opts["Archive"]
    elif t == "Options":
        # Get the sub-sub-object
        return opts["RunControl"]["Archive"]
    elif t in ["dict", "odict"]:
        # Downselect if given parent class
        opts = opts.get("RunControl", opts)
        opts = opts.get("Archive",    opts)
        # Convert to class
        return Archive(**opts)
    else:
        # Invalid type
        raise TypeError("Unformatted input must be type 'dict', not '%s'" % t)
    

# Class for case management
class Archive(cape.options.Archive.Archive):
    """
    Dictionary-based interfaced for options specific to folder management
    
    :Call:
        >>> opts = Archive(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    # Apply template
    def apply_ArchiveTemplate(self):
        """Apply named template to set default files to delete/archive
        
        :Call:
            >>> opts.apply_ArchiveTemplate()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Versions:
            * 2016-02-29 ``@ddalle``: First version
        """
        # Get the template
        tmp = self.get_ArchiveTemplate().lower()
        # Check it
        if tmp in ["full"]:
            # 
            pass
            
        
    # Get number of check points to keep around
    def get_nCheckPoint(self, i=None):
        """Return the number of check point files to keep
        
        :Call:
            >>> nchk = opts.get_nCheckPoint(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        return self.get_key('nCheckPoint', i)
        
    # Set the number of check point files to keep around
    def set_nCheckPoint(self, nchk=rc0('nCheckPoint'), i=None):
        """Set the number of check point files to keep
        
        :Call:
            >>> opts.set_nCheckPoint(nchk)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nchk*: :class:`int`
                Number of check files to keep (all if ``0``)
            *i*: :class:`int`
                Phase number
        :Versions:
            * 2015-01-10 ``@ddalle``: First version
        """
        self.set_key('nCheckPoint', nchk, i)
        
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
# class Archive

