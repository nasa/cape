
# Standard library
import fnmatch
import glob
import sys
import os

# Local imports
from .cfdx import cli


def generate_completions(prefix: str) -> list:
    return glob.glob(f"{prefix}*")


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) else ""
    completions = generate_completions(prefix)
    print('\n'.join(completions))
