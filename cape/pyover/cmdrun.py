r"""
:mod:`cape.pyover.bin`: OVERFLOW binary interface module
=========================================================

This module provides an interface to the various OVERFLOW executables
and other command-line utilities from :mod:`cape.bin`. However, due to
some of the subtleties of how the main OVERFLOW executables are called,
:func:`pyOver.case.RunPhase` simply constructs the commands to run
OVERFLOW from :mod:`cape.pyover.cmdgen` and runs them using
:func:`cape.cmdrun.callf`.

:See also:
    * :mod:`cape.cmdgen`
    * :mod:`cape.cmdrun`
    * :mod:`cape.pyover.cmdgen`

"""

# Import relevant tools
from ..cfdx.cmdrun import *
from ..cfdx.cmdrun import _assertfile, _upgradeDocString
