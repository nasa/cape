r"""
The :mod:`cape.pyfun` module contains the top-level interface for FUN3D
setup, execution, and post-processing. It loads some of the most
important methods from the various submodules so that they are easier
to access. Most tasks using the pyFun API can be accessed by loading
this module and reading one instance of the
:class:`cape.pyfun.cntl.Cntl` class.

    .. code-block:: python

        import cape.pyfun

For example the following will read in a global settings instance
assuming that the present working directory contains the correct files.
(If not, various defaults will be used, but it is unlikely that the
resulting setup will be what you intended.)

    .. code-block:: python

        import cape.pyfun
        cntl = cape.pyfun.Cntl()

Most of the pyFun submodules essentially contain a single class
definition, which is derived from a similarly named :mod:`cape` module.
For example, :class:`cape.pyfun.databook.DBComp` is subclassed to
:class:`cape.cfdx.databook.DBComp`, but several functions are edited
because their functionality needs customization for FUN3D.  For
example, reading iterative force & moment histories require a
customized method for each solver.

The following classes are imported in this module, so that code like
``cape.pyfun.Cntl`` will work (although ``cape.pyfun.cntl.Cntl`` will
also work).

    * :class:`cape.pyfun.cntl.Cntl`

Modules included within this one are outlined below.

    * Core modules:
        - :mod:`cape.pyfun.cntl`
        - :mod:`cape.pyfun.case`
        - :mod:`cape.pyfun.manage`
        - :mod:`cape.pyfun.dataBook`
        - :mod:`cape.pyfun.lineLoad`
        - :mod:`cape.pyfun.pointSensor`
        - :mod:`cape.pyfun.options`

    * FUN3D and Cape files
        - :mod:`cape.pyfun.faux`
        - :mod:`cape.pyfun.mapbc`
        - :mod:`cape.pyfun.namelist`
        - :mod:`cape.pyfun.plt`
        - :mod:`cape.pyfun.rubberData`

    * Supporting modules
        - :mod:`cape.pyfun.cmdgen`
        - :mod:`cape.pyfun.cmdrun`
        - :mod:`cape.pyfun.report`
        - :mod:`cape.pyfun.util`

"""

# Local imports
from .cntl import Cntl

