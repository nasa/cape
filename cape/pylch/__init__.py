r"""

The :mod:`cape.pylch` module contains the top-level interface for
Loci/CHEM setup, execution, and post-processing. It loads some of the
most important methods from the various submodules so that they are
easier to access. Most tasks using the pyLCH API can be accessed by
loading this module and reading one instance of the
:class:`cape.pylch.cntl.Cntl` class.

    .. code-block:: python
    
        import cape.pylch
        
For example the following will read in a global settings instance
assuming that the present working directory contains the correct files.
(If not, various defaults will be used, but it is unlikely that the
resulting setup will be what you intended.)

    .. code-block:: python
        
        import cape.pylch
        cntl = cape.pylch.Cntl()

"""

# Standard library
import os

# Local imports
#from .cntl import Cntl


# Save version number
version = "1.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyLCHFolder = os.path.split(_fname)[0]

