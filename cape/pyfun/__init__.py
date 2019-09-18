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

    * :class:`cape.pyfun.cntl.Cntl`
    * :class:`pyFun.runmatrix.RunMatrix`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`cape.pyfun.cntl`
        - :mod:`pyFun.case`
        - :mod:`pyFun.manage`
        - :mod:`pyFun.dataBook`
        - :mod:`pyFun.lineLoad`
        - :mod:`pyFun.pointSensor`
        - :mod:`pyFun.options`
        
    * FUN3D and Cape files
        - :mod:`pyFun.faux`
        - :mod:`pyFun.mapbc`
        - :mod:`pyFun.namelist`
        - :mod:`pyFun.plt`
        - :mod:`pyFun.rubberData`
        - :mod:`pyFun.runmatrix`
        
    * Supporting modules
        - :mod:`pyFun.cmd`
        - :mod:`pyFun.bin`
        - :mod:`pyFun.report`
        - :mod:`pyFun.queue`
        - :mod:`pyFun.util`

"""

# System
import os

# Save version number
version = "0.8"

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(_fname)[0]

# Import Control module
from .fun3d  import Fun3d, RunMatrix

