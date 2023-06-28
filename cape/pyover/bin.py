"""
:mod:`cape.pyover.bin`: OVERFLOW binary interface module
=========================================================

This module provides an interface to the various OVERFLOW executables and other
command-line utilities from :mod:`cape.bin`. However, due to some of the
subtleties of how the main OVERFLOW executables are called,
:func:`pyOver.case.RunPhase` simply constructs the commands to run OVERFLOW
from :mod:`cape.pyover.cmd` and runs them using :func:`cape.bin.callf`.
    
In addition, all the more universal executable interfaces provided in
:mod:`cape.bin` are imported directly,

    .. code-block:: python
    
        from cape.bin import *
        
so no extra syntax is needed in order to access them from :mod:`cape.pyfun.bin`

:See also:
    * :mod:`cape.bin`
    * :mod:`cape.cmd`
    * :mod:`cape.pyfun.cmd`

"""

# Import relevant tools
from ..cfdx.bin import *
from ..cfdx.bin import _assertfile, _upgradeDocString
