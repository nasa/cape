"""

The :mod:`cape.pyover` module contains the top-level interface for OVERFLOW
setup, execution, and post-processing. It loads some of the most important
methods from the various submodules so that they are easier to access. Most
tasks using the cape.pyover API can be accessed by loading this module and
reading one instance of the :class:`cape.pyover.cntl.Cntl` class.

    .. code-block:: python
    
        import cape.pyover
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files. (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import cape.pyover
        cntl = cape.pyover.Cntl()
        
Most of the cape.pyover submodules essentially contain a single class
definition, which is derived from a similarly named :mod:`cape` module. For
example, :class:`cape.pyover.dataBook.DBComp` is subclassed to
:class:`cape.dataBook.DBComp`, but several functions are edited because their
functionality needs customization for OVERFLOW. For example, reading iterative
force & moment histories require a customized method for each solver.

The following classes are imported in this module, so that code like
``cape.pyover.Cntl`` will work (although ``cape.pyover.cntl.Cntl`` will also
work).

    * :class:`cape.pyover.cntl.Cntl`
    * :class:`cape.pyover.runmatrix.RunMatrix`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`cape.pyover.cntl`
        - :mod:`cape.pyover.case`
        - :mod:`cape.pyover.manage`
        - :mod:`cape.pyover.dataBook`
        - :mod:`cape.pyover.lineLoad`
        - :mod:`cape.pyover.pointSensor`
        - :mod:`cape.pyover.options`
        
    * OVERFLOW and Cape files
        - :mod:`cape.pyover.overNamelist`
        - :mod:`cape.pyover.plot3d`
        - :mod:`cape.config`
        - :mod:`cape.step`
        - :mod:`cape.tri`
        
    * Supporting modules
        - :mod:`cape.pyover.cmd`
        - :mod:`cape.pyover.bin`
        - :mod:`cape.pyover.report`
        - :mod:`cape.pyover.queue`
        - :mod:`cape.pyover.util`

"""

# Local tools
from .cntl  import Cntl, RunMatrix

# Save version number
version = "1.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyOverFolder = os.path.split(_fname)[0]

