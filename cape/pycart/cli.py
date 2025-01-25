r"""
:mod:`cape.pycart.cli`: Interface to ``pycart`` executable
=============================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pycart`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from ..cfdx import cli


# Customized parser
class PycartFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pycart"
    _help_title = "Interact with Cart3D run matrix using CAPE"

    # Custom classes
    _cntl_mod = "cape.pycart.cntl"
    _casecntl_mod = "cape.pycart.casecntl"


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pycart``

    This turns ``sys.argv`` into Python arguments and calls
    :func:`cape.pyfun.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2021-03-03 ``@ddalle``: v1.0
        * 2024-12-30 ``@ddalle``: v2.0; use ``argread``
    """
    # Output
    return cli.main_template(PycartFrontDesk, argv)

