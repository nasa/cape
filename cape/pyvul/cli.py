r"""
:mod:`cape.pyvul.cli`: Interface to ``pyvul`` executable
===========================================================

This module provides the Python function :func:`main`, which is
executed whenever ``pyvul`` is used.
"""

# Standard library modules
from typing import Optional

# Local imports
from ..cfdx import cli


# Customized parser
class PyvulFrontDesk(cli.CfdxFrontDesk):
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "pyvul"
    _help_title = "Interact with VULCAN run matrix using CAPE"

    # Custom classes
    _cntl_mod = "cape.pyvul.cntl"
    _casecntl_mod = "cape.pyvul.casecntl"


# New-style CLI
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``pyfun``

    :Call:
        >>> ierr = main(argv=None)
    """
    # Output
    return cli.main_template(PyvulFrontDesk, argv)

