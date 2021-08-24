#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.cli`: Command-Line Interface to datakit tools
=================================================================


"""

# Standard library modules
import sys

# Local modules
from . import vendorutils
from . import writedb
from .. import argread
from .. import text as textutils


# Help message
HELP_DKIT = r"""
-------------------------------------------------
``dkit``: Command-Line Interface to datakit tools
-------------------------------------------------

"""


# main function
def main():
    # Process arguments
    a, kw = argread.readkeys(sys.argv)
    # Check first arg
    if len(a) == 0:
        # No command
        print(textutils.markdown(HELP_DKIT))
        return
    # Get first arg, the name of the command
    cmdname = a.pop(0)
    # Normalize
    cmd = normalize_cmd(cmdname)
    # Filter first arg
    if cmd in {"writedb", "write-db"}:
        # Call datakit writer
        return writedb.write_dbs(*a, **kw)
    elif cmd in {"vendorize"}:
        # Call vendorizer
        return vendorutils.vendorize_repo(*a, **kw)
    else:
        print(textutils.markdown(HELP_DKIT))
        print("")
        print("Unrecognized command '%s'" % cmdname)


# Normalize command names
def normalize_cmd(cmdname):
    r"""Normalize command names

    :Call:
        >>> cmd = normalize_cmd(cndname)
    :Inputs:
        *cmdname*: :class:`str`
            Name of ``dkit`` command
    :Outputs:
        *cmd*: :class:`str`
            Lower-case *cmdname* with ``_`` replaced with ``-``
    :Versions:
        * 2021-08-20 ``@ddalle``: Version 1.0
    """
    # Output
    return cmdname.lower().replace("_", "-")

