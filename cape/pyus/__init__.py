#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

The :mod:`cape.pyus` module contains the top-level interface for US3D setup,
execution, and post-processing. It loads some of the most important methods
from the various submodules so that they are easier to access. Most tasks using
the pyFun API can be accessed by loading this module and reading one instance
of the :class:`cape.pyus.cntl.Cntl` class.

    .. code-block:: python
    
        import cape.pyus
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files.  (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import cape.pyus
        cntl = cape.pyus.Cntl()
        
The following classes are imported in this module, so that code like
``pyFun.Cntl`` will work (although ``cape.pyfun.cntl.Cntl`` will also work).

    * :class:`cape.pyus.cntl.Cntl`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`cape.pyus.cntl`
        - :mod:`cape.pyus.casecntl`
        - :mod:`cape.pyus.options`
        
    * US3D and Cape files
        - :mod:`cape.pyus.inputinpfile`
        
    * Supporting modules
        - :mod:`cape.pyus.cmdgen`
        - :mod:`cape.pyus`

"""

# Standard library
import os

# Import Control module
from .cntl import Cntl, RunMatrix

# Save version number
version = "1.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyUSFolder = os.path.split(_fname)[0]

