#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

The :mod:`cape.pyfun` module contains the top-level interface for FUN3D setup,
execution, and post-processing. It loads some of the most important methods
from the various submodules so that they are easier to access. Most tasks using
the pyFun API can be accessed by loading this module and reading one instance
of the :class:`cape.pyfun.cntl.Cntl` class.

    .. code-block:: python
    
        import pyFun
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files.  (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import pyFun
        cntl = pyFun.Cntl()
        
Most of the pyFun submodules essentially contain a single class definition,
which is derived from a similarly named :mod:`cape` module.  For example,
:class:`pyFun.dataBook.DBComp` is subclassed to :class:`cape.dataBook.DBComp`,
but several functions are edited because their functionality needs
customization for FUN3D.  For example, reading iterative force & moment
histories require a customized method for each solver.

The following classes are imported in this module, so that code like
``pyFun.Cntl`` will work (although ``cape.pyfun.cntl.Cntl`` will also work).

    * :class:`pyUS.us3d.US3D`
    * :class:`pyUS.runmatrix.RunMatrix`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`pyUS.fun3d`
        - :mod:`pyUS.case`
        - :mod:`pyUS.manage`
        - :mod:`pyUS.dataBook`
        - :mod:`pyUS.options`
        
    * US3D and Cape files
        - :mod:`pyUS.inputInp`
        
    * Supporting modules
        - :mod:`pyUS.cmd`
        - :mod:`pyUS.bin`
        - :mod:`pyUS.report`
        - :mod:`pyUS.queue`
        - :mod:`pyUS.util`

"""

# System
import os

# Save version number
version = "0.8"

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyUSFolder = os.path.split(_fname)[0]

# Import Control module
from .us3d  import US3D, RunMatrix

