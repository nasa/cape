r"""
:mod:`cape.dkit.datakitrepo`: Interact with DataKit collections/repos
======================================================================

"""

# Standard library
import json
import os
from typing import Optional

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


# Get list of requirements
def get_requirements(mod) -> list:
    r"""Get list of requirements, from file or local variable

    :Call:
        >>> reqs = get_requirements(module)
    :Inputs:
        *mod*: :class:`module`
            Module to read requirements from
    :Outputs:
        *reqs*: :class:`list`\ [:class:`str`]
            List of required databse names
    :Versions:
        * 2025-06-13 ``@ddalle``: v1.0
    """
    # Prioritize reading from file
    reqs = get_requirements_json(mod)
    # Check if that worked
    if isinstance(reqs, list):
        return reqs
    # Otherwise use the local *REQUIREMENTS* variable
    return getattr(mod, "REQUIREMENTS", [])


# Get list of requirements from file
def get_requirements_json(mod) -> Optional[list]:
    r"""Read list of requirements from JSON file, if applicable

    :Call:
        >>> reqs = ast.get_requirements_json()
    :Inputs:
        *ast*: :class:`DataKitAssistant`
            Tool for reading datakits for a specific module
    :Outputs:
        *reqs*: :class:`list`\ [:class:`str`] | ``None``
            Requirements read from file, if any
    :Versions:
        * 2025-06-13 ``@ddalle``: v1.0
    """
    # Get folder of module
    dirname = os.path.dirname(mod.__file__)
    # Get path to file
    fname = os.path.join(dirname, "requirements.json")
    # Check
    if not os.path.isfile(fname):
        return None
    try:
        with open(fname, 'r') as fp:
            return json.load(fp)
    except Exception:
        return None
