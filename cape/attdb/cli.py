#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.cli`: Command-Line Interface to datakit tools
=================================================================


"""

# Standard library modules
import sys

# Local modules
from . import quickstart
from . import vendorutils
from . import writedb
from .. import argread
from .. import text as textutils


# Help message
HELP_DKIT = r"""
-------------------------------------------------
``dkit``: Command-Line Interface to datakit tools
-------------------------------------------------

Perform actions on a DataKit package or package collection from the
command line interface.

:Usage:
    .. code-block:: console

        $ dkit CMD [ARGS] [OPTIONS]

:Arguments:
    * *CMD*: name of other command to run, one of:
        - ``write-db``: Process raw data into datakit files
        - ``vendorize``: Install local copies of packages
        - ``quicksart``: Create a template DataKit package

    * *ARGS*: arguments passed to individual commands

:Options:
    See options for specific commands

:Versions:
    * 2021-08-24 ``@ddalle``: Version 1.0
"""


# main function
def main():
    r"""Main ``dkit`` command-line interface function

    :Call:
        >>> main()
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
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
    elif cmd in {"quickstart"}:
        # Call quickstart method
        return quickstart.quickstart(*a, **kw)
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

