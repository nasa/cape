r"""
:mod:`cape.pyfun.util`: Utilities for pyFun 
============================================

This module imports the generic utilities using

    .. code-block:: python
    
        from cape.util import *
        
It also stores the absolute path to the folder containing the 
:mod:`cape.pyfun` module as the variable *pyFunFolder*.

:See also:
    * :mod:`cape.util`
"""

# Import everything from cape.util
from cape.util import *


# pyCart base folder
pyFunFolder = os.path.split(os.path.abspath(__file__))[0]


