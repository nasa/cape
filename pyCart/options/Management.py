"""Interface for options specific to running `adjointCart`"""


# Import options-specific utilities
from util import rc0

# Import template module
import cape.options.Management

# Class for flowCart settings
class Management(cape.options.Management):
    """
    Dictionary-based interfaced for options specific to folder management
    
    :Call:
        >>> opts = Management(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
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
        
