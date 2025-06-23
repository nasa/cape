r"""
:mod:`cape.pylch.cli`: Interface to ``pylch`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pylch`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from ..cfdx import cli


# Customized parser
class PylchFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pyfun"
    _help_title = "Interact with Loci/CHEM run matrix using CAPE"

    # Custom classes
    _cntl_mod = "cape.pylch.cntl"
    _casecntl_mod = "cape.pylch.casecntl"


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pylch``

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PylchFrontDesk, argv)


