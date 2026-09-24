r"""
:mod:`cape.clidoc.dkit`: dkit help
===================================

Auto-generated help message for the dkit command-line interface.
"""

from ..dkit import cli


# Instantiate parser
parser = cli.DkitFrontDesk()
# Generate help
__doc__ = parser.genr8_help()
