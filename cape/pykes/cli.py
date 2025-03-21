#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.cli`: Interface to ``pykes`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pykes`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from ..cfdx import cli


# Customized parser
class PykesFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pykes"
    _help_title = "Interact with Kestrel run matrix using CAPE"

    # Custom classes
    _cntl_mode = "cape.pykes.cntl"
    _casecntl_mod = "cape.pykes.casecntl"


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pykes``

    :Call:
        >>> main()
    :Versions:
        * 2021-10-19 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PykesFrontDesk, argv)


