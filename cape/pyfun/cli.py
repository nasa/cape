#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyfun.cli`: Interface to ``pyfun`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pyfun`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from .casecntl import CaseRunner
from .cntl import Cntl
from ..cfdx import cli


# Customized parser
class PyfunFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pyfun"

    # Custom classes
    _cntl_cls = Cntl
    _runner_cls = CaseRunner


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pyfun``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pyfun.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PyfunFrontDesk, argv)

