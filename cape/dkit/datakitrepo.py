r"""
:mod:`cape.dkit.datakitrepo`: Interact with DataKit collections/repos
======================================================================

"""

# Standard library
import os

# Third-party

# Local imports
from . import pkgutils


# Find repos matching regex-pattern name
def find_packages(*a) -> list:
    r"""Find packages in CWD whose names matches regex(s)

    :Call:
        >>> pkgs = find_packages(pat1)
        >>> pkgs = find_packages(*pats)
    :Inputs:
        *pat1*: :class:`str`
            Regular expression for package name to match
        *pats*: :class:`tuple`\ [:class:`str`]
            Multiple regex patterns; find pkgs matching at least one
    :Outputs:
        *pkgs*: :class:`list`\ [:class:`str`]
            List of packages matching at least one pattern
    :Versions:
        * 2025-06-12 ``@ddalle``: v1.0
    """
    # Change '/' to '.'
    a_normalized = tuple(ai.replace(os.sep, '.') for ai in a)
    # Initialize packages
    pkgs = []
    # Check inputs against list
    for j, aj in enumerate(a_normalized):
        # Find matching packages
        pkgsj = pkgutils.find_packages(regex=aj)
        # Check for errors
        if len(pkgsj) == 0:
            print("Found no packages for name %i, '%s'" % (j+1, aj))
            continue
        # Avoid duplicates
        for pkg in pkgsj:
            if pkg in pkgs:
                continue
            # Add to global list
            pkgs.append(pkg)
    # Output
    return pkgs
