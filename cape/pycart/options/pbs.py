"""
:mod:`cape.pycart.options.pbs`: pyCart PBS Job Options
======================================================

This module provides options for PBS jobs in pyCart.  It is based on the
:mod:`cape.options.pbs` module with no modifications.

:See Also:
    * :mod:`cape.options.pbs`
    * :mod:`cape.options`
    * :mod:`cape.pycart.options`
"""


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
