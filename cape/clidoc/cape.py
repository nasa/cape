
from ..cfdx import cli


# Instantiate parser
parser = cli.CfdxFrontDesk()
# Generate help
__doc__ = parser.genr8_help()


