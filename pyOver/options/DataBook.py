"""Interface for OVERFLOW data book configuration"""


# Import options-specific utilities
from util import rc0, getel, odict


# Import base class
import cape.options

# Class for DataBook options
class DataBook(cape.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyOver.options.DataBook
    :Versions:
        * 2015-12-29 ``@ddalle``: Subclassed from CAPE
    """
    
    pass

        
# Class for target data
class DBTarget(cape.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


