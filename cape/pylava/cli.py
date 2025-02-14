
r"""
:mod:`cape.pylava.cli`: Interface to ``pylava`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pylava`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from ..cfdx import cli


# Customized parser
class PylavaFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pylava"
    _help_title = "Interact with LAVA run matrix using CAPE"

    # Custom classes
    _cntl_mod = "cape.pylava.cntl"
    _casecntl_mod = "cape.pylave.casecntl"


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pylava``

    :Call:
        >>> main()
    :Versions:
        * 2024-10-19 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PylavaFrontDesk, argv)


