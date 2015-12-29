"""Interface for configuration control: :mod:`pyCart.options.Config`"""


# Import options-specific utilities
from util import rc0

# Import base class
import cape.options.Config

# Class for PBS settings
class Config(cape.options.Config):
    """
    Configuration options for OVERFLOW
    
    :Call:
        >>> opts = Config(**kw)
    :Versions:
        * 2015-12-29 ``@ddalle``: First version
    """
    
# class Config

