r"""
:mod:`cape.convert1to2`: Utilities to modify CAPE 1.x scripts to 2.0
=====================================================================

This module provides utilities to modify programs that used the CAPE 1.x
API to ones that will work with CAPE 2.0 and beyond. The primary change
between the two versions of CAPE is that many modules have been renamed
in order to simplify the structure of the package.

The upgrade from CAPE 1.2 to 2.0 does not actually entail a major change
in the underlying function and code of the package, but programs (e.g.
what CAPE calls "hooks") may fail without changing many import
statements.

"""

# Standard library
import re
from typing import Optional


# Module names to update
CAPE_MODNAME_MAP = {
    "cape.attdb": "cape.dkit",
    "cape.cntl": "cape.cfdx.cntl",
    "cape.cfdx.case": "cape.cfdx.casecntl",
    "cape.pycart.case": "cape.pycart.casecntl",
    "cape.pyfun.case": "cape.pyfun.casecntl",
    "cape.pykes.case": "cape.pykes.casecntl",
    "cape.pylch.case": "cape.pylch.casecntl",
    "cape.pyover.case": "cape.pyover.casecntl",
    "cape.pyus.dataBook": "cape.pyus.databook",
    "cape.cfdx.dataBook": "cape.cfdx.databook",
    "cape.pycart.dataBook": "cape.pycart.databook",
    "cape.pyfun.dataBook": "cape.pyfun.databook",
    "cape.pykes.dataBook": "cape.pykes.databook",
    "cape.pylch.dataBook": "cape.pylch.databook",
    "cape.pyover.dataBook": "cape.pyover.databook",
    "cape.pyus.dataBook": "cape.pyus.databook",
}


# Rewrite imports for one renamed module
def rewrite_imports(
        infile: str,
        modnames: dict,
        outfile: Optional[str] = None):
    r"""Rewrite imports for a collection of module name changes

    :Call:
        rewrite_imports(infile, modnames, outfile=None)
    """
    # Default name of the output file
    outfile = infile if outfile is None else outfile
    # Read the content of the file
    content = open(infile, 'r').read()
    # Loop through replacments
    for mod1, mod2 in modnames.items():
        # Make replacements
        content = sub_modname(content, mod1, mod2)
    # Write the modified content back to the file
    with open(outfile, 'w') as fp:
        fp.write(content)


# Rewrite imports for one renamed module
def sub_modname(content: str, mod1: str, mod2: str):
    r"""Replace imports of *mod1* with imports of *mod2*

    :Call:
        >>> new_content = sub_modname(content, mod1, mod2)
    :Inputs:
        *content*: :class:`str`
            Original contents before renaming
        *mod1*: :class:`str`
            Name of module to be replaced, e.g. ``"cape.attdb"``
        *mod2*: :class:`str`
            New name of module, e.g. ``"cape.dkit"``
    :Outputs:
        *new_content*: :class:`str`
            Revised *content* with rewritten imports
    """
    # Split both module names into parts
    parts1 = mod1.split('.')
    parts2 = mod2.split('.')
    # Full module name but escaped . -> \.
    full1 = r"\.".join(parts1)
    # Name of direct parent of each module
    parent1 = r"\.".join(parts1[:-1])
    parent2 = ".".join(parts2[:-1])
    # Base module names
    basename1 = parts1[-1]
    basename2 = parts2[-1]
    # Form regular exrpressions
    pat0 = re.compile(rf"import\s+{full1}")
    pat1 = re.compile(rf"from\s+{parent1}\s+import\s+{basename1}")
    pat2 = re.compile(rf"from\s+{full1}\s+import")
    # Replace name of module in subsequent calls
    # Careful:
    #     a = cape.attdb.f() -> a = cape.dkit.f()
    #     a = attdb.f() -> dkit.f()
    #     a = escape.attdb.f() !-> escape.dkit.f()
    #     a = mattdb.f() !-> mdkit.f()
    pat0b = re.compile(rf"(?<![a-zA-Z_]){full1}\.")
    pat1b = re.compile(rf"(?<![a-zA-Z_]){basename1}\.")
    # Replacements
    out0 = f"import {mod2}"
    out1 = f"from {parent2} import {basename2}"
    out2 = f"from {mod2} import"
    out0b = f"{mod2}."
    out1b = f"{basename2}."
    # Replace the matching import statements
    new_content = pat0.sub(out0, content)
    new_content = pat1.sub(out1, new_content)
    new_content = pat2.sub(out2, new_content)
    new_content = pat0b.sub(out0b, new_content)
    new_content = pat1b.sub(out1b, new_content)
    # Output
    return new_content
