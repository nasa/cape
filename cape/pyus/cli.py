#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyus.cli`: Command-line interface to ``pyus``
===========================================================

This module provides the :func:`main` function that is used by the
executable called ``pyus``.
"""

# Standard library modules
from typing import Optional

# Local imports
from .casecntl import CaseRunner
from .cntl import Cntl
from ..cfdx import cli


# Customized parser
class PyusFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pyus"
    _help_title = "Interact with US3D run matrix using CAPE"

    # Custom classes
    _cntl_cls = Cntl
    _runner_cls = CaseRunner


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pyus``

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PyusFrontDesk, argv)


