r"""
:mod:`cape.clidoc.pycart`: pyCart command-line help
====================================================

Auto-generated help message for the pyCart command-line interface.
"""

from ..pycart import cli


# Instantiate a parser
parser = cli.PycartFrontDesk()

# Create help message
__doc__ = parser.genr8_help()

