"""
:mod:`pyFun.bin`: FUN3D binary interface module
=================================================

This module provides an interface to the various FUN3D executables and other
command-line utilities from :mod:`cape.bin`.  However, due to some of the
subtleties of how the main FUN3D executables are called,
:func:`pyFun.case.RunPhase` simply constructs the commands to run FUN3D from
:mod:`pyFun.cmd` and runs them using :func:`cape.bin.callf`.
    
In addition, all the more universal executable interfaces provided in
:mod:`cape.bin`, including the Cart3D utilities ``intersect`` and ``verify``,
are also imported.  These are imported directly,

    .. code-block:: python
    
        from cape.bin import *
        
so no extra syntax is needed in order to access them from :mod:`pyFun.bin`

:See also:
    * :mod:`cape.bin`
    * :mod:`cape.cmd`
    * :mod:`pyFun.cmd`

"""

# Import relevant tools
from cape.bin import *
from cape.bin import _assertfile, _upgradeDocString



