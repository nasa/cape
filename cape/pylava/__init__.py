# Standard library
import os

# Local tools
from .cntl  import Cntl


# Save version number
version = "1.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyLavaFolder = os.path.split(_fname)[0]
