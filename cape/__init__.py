"""

The :mod:`cape` module contains the top-level interface setup of various
solvers. It loads the most important methods from the various submodules so that
they are easier to access. Most tasks using the Cape API can be accessed by
loading this module.

    .. code-block:: python
    
        import cape
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files.  (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import cape
        cntl = cape.Cntl()
        
Most of the pyCart submodules essentially contain a single class definition,
and many of these classes are accessible directly from the :mod:`cape` module.
The list of classes loaded directly in :mod:`cape`.

    * :class:`cape.cntl.Cntl`
    * :class:`cape.tri.Tri`
    * :class:`cape.tri.Triq`
    * :func:`cape.case.ReadCaseJSON`
    
Because Cape is a template module that has no specific solver, few modules are
loaded directly to :mod:`cape`.  The list of modules loaded are shown below.

    * :mod:`cape.manage`

A categorized list of modules available to the API are listed below.

    * Core modules
       - :mod:`cape.cntl`
       - :mod:`cape.options`
       - :mod:`cape.case`
       - :mod:`cape.trajectory`
       - :mod:`cape.dataBook`
       - :mod:`cape.pointSensor`
       - :mod:`cape.lineLoad`
       - :mod:`cape.report`
    
    * Primary supporting modules
       - :mod:`cape.bin`
       - :mod:`cape.cmd`
       - :mod:`cape.argread`
       - :mod:`cape.queue`
       - :mod:`cape.io`
       - :mod:`cape.manage`
        
    * File interfaces
       - :mod:`cape.fileCntl`
       - :mod:`cape.tri`
       - :mod:`cape.cgns`
       - :mod:`cape.msh`
       - :mod:`cape.namelist`
       - :mod:`cape.namelist2`
       - :mod:`cape.plot3d`
       - :mod:`cape.plt`
       - :mod:`cape.step`
       - :mod:`cape.tecplot`
       - :mod:`cape.tex`
    
    * Utilities
       - :mod:`cape.util`
       - :mod:`cape.atm`
       - :mod:`cape.config`
       - :mod:`cape.convert`
       - :mod:`cape.geom`
       - :mod:`cape.tar`
       - :mod:`cape.text`
       - :mod:`cape.volcomp`
    

:Versions:
    * Version 0.6: 2016-03-30
    * Version 0.8: 2018-03-23
"""

# File system and operating system management
import os

# Save version number
version = "0.8"
__version__ = version


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
CapeFolder = os.path.split(_fname)[0]
TemplateFolder = os.path.join(CapeFolder, "templates")


# Import classes and methods from the submodules
from tri    import Tri, Triq
from cntl   import Cntl
from case   import ReadCaseJSON

# Get the conversion tools directly.
from .convert import *

# Submodules
from . import manage

