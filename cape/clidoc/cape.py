r"""
:mod:`cape.clidoc.cape`: CAPE command-line help
================================================

Auto-generated help message for the CAPE command-line interface.
"""

from ..cfdx import cli


# Instantiate parser
parser = cli.CfdxFrontDesk()
# Generate help
__doc__ = parser.genr8_help()


