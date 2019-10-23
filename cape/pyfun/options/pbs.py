"""
:mod:`cape.pyfun.options.pbs`: pyFun PBS Job Options
=====================================================

This module provides options for PBS jobs in pyFun.  It is based on the
:mod:`cape.cfdx.options.pbs` module with no modifications.

:See Also:
    * :mod:`cape.cfdx.options.pbs`
    * :mod:`cape.cfdx.options`
    * :mod:`cape.pyfun.options`
"""


# Import options-specific utilities
from .util import rc0

# Get PBS settings template
import cape.cfdx.options.pbs

# Class for PBS settings
class PBS(cape.cfdx.options.pbs.PBS):
    """Dictionary-based interface for PBS job control
    
    :Call:
        >>> opts = PBS(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
