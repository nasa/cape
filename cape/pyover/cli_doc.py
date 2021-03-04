#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyover.cli_doc`: Create help message for ``pyover``
===============================================================

This module formats the output of ``pyover -h`` by filling in some slots
from :mod:`cape.cfdx.cli_doc`.

"""

# Standard library modules
import os
import sys

# CAPE modules
import cape.cfdx.cli_doc


# (Automatic) name of executable category
NAME = __name__.split(".")[1]
# (Suggested) tool name
TITLE = NAME[:2] + NAME[2].upper() + NAME[3:]
# Main module name
MODNAME = ".".join(__name__.split(".")[:2])
# reST of main module
MODULE_RST = ":mod:`%s`" % MODNAME
# Name of solver
SOLVER = "OVERFLOW"

# Detect which executable was used
if len(sys.argv) > 0:
    # Get whatever launched this python session
    _argv0 = sys.argv[0]
    # Get the executable name
    _cmd0 = os.path.split(_argv0)[-1]
else:
    # Not sure how this would happen, but let's just call it "python"
    _cmd0 = "python"

# Detect if specific executable was called (e.g. 'pycart3')
if _cmd0.startswith(NAME):
    # Use the specific calling function
    CMD = _cmd0
else:
    # Just use the main category
    CMD = NAME

# Check for specified version
if CMD.endswith("2"):
    # Forced Python 2.7
    PYVERSION = "2.7 "
elif CMD.endswith("3"):
    # Forced Python 3 (.6+)
    PYVERSION = "3 "
else:
    # System
    PYVERSION = ""

# Docstring values for PYCART
_fmt = {
    "cmd": CMD,
    "name": NAME,
    "mod": MODULE_RST,
    "modname": MODNAME,
    "pyver": PYVERSION,
    "solver": SOLVER,
    "title": TITLE,
}


# Get template
_template = cape.cfdx.cli_doc.template

# Create docstring for PYCART; add version info
PYOVER_HELP = _template % _fmt + r"""
    --stop, --stop STOP
        Stop one or more cases at iteration *STOP* (or immediately if
        iteration not specified) (currently broken)

:Versions:
    * 2014-10-06 ``@ddalle``: Version 1.0 (of ``pycart``)
    * 2015-10-16 ``@ddalle``: Version 1.0
    * 2021-03-04 ``@ddalle``: Version 1.1; use "entry_points"
"""


# Print help if appropriate.
if __name__ == "__main__":
    print(PYOVER_HELP)
