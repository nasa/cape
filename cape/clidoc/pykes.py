r"""
:mod:`cape.clidoc.pykes`: pyKestrel command-line help
======================================================

Auto-generated help message for the pyKestrel command-line interface.
"""

from ..pykes import cli


# Instantiate a parser
parser = cli.PykesFrontDesk()

# Create help message
__doc__ = parser.genr8_help()


