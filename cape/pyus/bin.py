#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyus.bin`: US3D binary interface module
=====================================================

This module provides an interface to the various US3D executables and
other command-line utilities from :mod:`cape.bin`.  However, due to some
of the subtleties of how the main US3D executables are called,
:func:`cape.pyus.case.RunPhase` constructs the commands to run US3D
from :mod:`cape.pyfun.cmd` locally and runs them using
:func:`cape.bin.callf`.

:See also:
    * :mod:`cape.cfdx.bin`
    * :mod:`cape.cfdx.cmd`
    * :mod:`cape.pyus.cmd`

"""

# CAPE modules
import capec.cfdx.bin as cbin

# Local modules
from . import cmd


# Execute ``us3d-prepar``
def us3d_prepar(opts, i=0, **kw):
    # Get command
    cmdi = cmd.us3d_prepar(opts, i=i, **kw)
    # Get verbosity option
    if opts:
        # Specified from "RunControl" section
        v = opts.get_Verbose(j)
    else:
        # Default is ``True``
        v = True
    # Check override
    v = kw.get("v", v)
    # Execute the command
    cbin.callf(cmdi, f="us3d-prepar.out", v=v)
    
