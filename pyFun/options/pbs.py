"""Interface for PBS script options: :mod:`pyFun.options.pbs`"""


# Import options-specific utilities
from .util import rc0

# Get PBS settings template
import cape.options.pbs

# Class for PBS settings
class PBS(cape.options.pbs.PBS):
    """Dictionary-based interface for PBS job control
    
    :Call:
        >>> opts = PBS(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
