"""

The :mod:`pyOver` module contains the top-level interface for OVERFLOW setup,
execution, and post-processing. It loads some of the most important methods
from the various submodules so that they are easier to access. Most tasks using
the pyOver API can be accessed by loading this module and reading one instance
of the :class:`pyOver.overflow.Overflow` class.

    .. code-block:: python
    
        import pyOver
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files.  (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import pyOver
        ofl = pyOver.Overflow()
        
Most of the pyOver submodules essentially contain a single class definition,
which is derived from a similarly named :mod:`cape` module.  For example,
:class:`pyOver.dataBook.DBComp` is subclassed to :class:`cape.dataBook.DBComp`,
but several functions are edited because their functionality needs
customization for OVERFLOW.  For example, reading iterative force & moment
histories require a customized method for each solver.

The following classes are imported in this module, so that code like
``pyOver.Overflow`` will work (although ``pyOver.overflow.Overflow`` will also
work).

    * :class:`pyOver.overflow.Overflow`
    * :class:`pyOver.trajectory.Trajectory`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`pyOver.overflow`
        - :mod:`pyOver.case`
        - :mod:`pyOver.manage`
        - :mod:`pyOver.dataBook`
        - :mod:`pyOver.lineLoad`
        - :mod:`pyOver.pointSensor`
        - :mod:`pyOver.options`
        
    * OVERFLOW and Cape files
        - :mod:`pyOver.overNamelist`
        - :mod:`pyOver.plot3d`
        - :mod:`cape.config`
        - :mod:`cape.step`
        - :mod:`cape.tri`
        
    * Supporting modules
        - :mod:`pyOver.cmd`
        - :mod:`pyOver.bin`
        - :mod:`pyOver.report`
        - :mod:`pyOver.queue`
        - :mod:`pyOver.util`

"""

# System
import os

# Save version number
version = "0.8"

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyOverFolder = os.path.split(_fname)[0]

# Import Control module
from overflow  import Overflow, Trajectory

