r"""
:mod:`cape.clidoc.pyover`: pyOver command-line help
====================================================

Auto-generated help message for the pyOver command-line interface.
"""

from ..pyover import cli


# Instantiate a parser
parser = cli.PyoverFrontDesk()

# Create help message
__doc__ = parser.genr8_help()


