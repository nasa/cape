r"""
:mod:`cape.pyfun.bin`: FUN3D binary interface module
=====================================================

This module provides an interface to the various FUN3D executables and
other command-line utilities from :mod:`cape.bin`.  However, due to some
of the subtleties of how the main FUN3D executables are called,
:func:`pyFun.case.RunPhase` simply constructs the commands to run FUN3D
from :mod:`cape.pyfun.cmd` and runs them using :func:`cape.bin.callf`.
    
In addition, all the more universal executable interfaces provided in
:mod:`cape.bin`, including the Cart3D utilities ``intersect`` and
``verify``, are also imported.  These are imported directly,

    .. code-block:: python
    
        from cape.cfdx.bin import *
        
so no extra syntax is needed in order to access them from
:mod:`cape.pyfun.bin`

:See also:
    * :mod:`cape.cfdx.bin`
    * :mod:`cape.cfdx.cmd`
    * :mod:`cape.pyfun.cmd`

"""

# Import relevant tools
from ..cfdx.bin import *
from ..cfdx.bin import _assertfile, _upgradeDocString
