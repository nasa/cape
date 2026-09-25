r"""
:mod:`cape.clidoc.gruvoc`: gruvoc help
=======================================

Auto-generated help message for the gruvoc command-line interface.
"""

from ..gruvoc import cli


# Instantiate parser
parser = cli.GruvocFrontDesk()
# Generate help
__doc__ = parser.genr8_help()
