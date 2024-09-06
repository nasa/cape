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
import fnmatch
import os
import re
from typing import Optional


# Module names to update
CAPE_MODNAME_MAP = {
    "_cape3": "_cape",
    "_ftypes3": "_cape",
    "cape.attdb": "cape.dkit",
    "cape.attdb.stats": "cape.dkit.statutils",
    "cape.tnakit.plot_mpl": "cape.plot_mpl",
    "cape.tnakit.modutils": "cape.dkit.modutils",
    "cape.tnakit.kwutils": "cape,kwutils",
    "cape.tnakit.statutils": "cape.attdb.statutils",
    "cape.cntl": "cape.cfdx.cntl",
    "cape.tri": "cape.trifile",
    "cape.plt": "cape.pltfile",
    "cape.cfdx.case": "cape.cfdx.casecntl",
    "cape.cfdx.dataBook": "cape.cfdx.databook",
    "cape.filecntl.tecplot": "cape.filecntl.tecfile",
    "cape.filecntl.tex": "cape.filecntl.texfile",
    "cape.pycart.aeroCsh": "cape.pycart.aerocshfile",
    "cape.pycart.case": "cape.pycart.casecntl",
    "cape.pycart.dataBook": "cape.pycart.databook",
    "cape.pycart.inputCntl": "cape.pycart.inputcntlfile",
    "cape.pycart.preSpecCntl": "cape.pycart.prespecfile",
    "cape.pyfun.case": "cape.pyfun.casecntl",
    "cape.pyfun.dataBook": "cape.pyfun.databook",
    "cape.pyfun.rubberData": "cape.pyfun.rubberdatafile",
    "cape.pykes.case": "cape.pykes.casecntl",
    "cape.pykes.dataBook": "cape.pykes.databook",
    "cape.pylch.case": "cape.pylch.casecntl",
    "cape.pylch.dataBook": "cape.pylch.databook",
    "cape.pyover.case": "cape.pyover.casecntl",
    "cape.pyover.dataBook": "cape.pyover.databook",
    "cape.pyover.overNamelist": "cape.pyover.overnmlfile",
    "cape.pyus.case": "cape.pyus.casecntl",
    "cape.pyus.dataBook": "cape.pyus.databook",
    "cape.pyus.inputInp": "cape.pyus.inputinpfile",
}


# Convert all functions
def upgrade1to2():
    r"""Recursively update imports in Python files found w/i current dir

    :Call:
        >>> upgrade1to2()
    """
    # Search through all folders
    for root, dirnames, fnames in os.walk("."):
        # Don't visit .git or __pycache__
        for ignored in (".git", "__pycache__", "_vendor"):
            if ignored in dirnames:
                dirnames.remove(ignored)
        # Loop through .py files
        for fname in fnmatch.filter(fnames, "*.py"):
            # Skip any links
            if os.path.islink(fname):
                continue
            # Full path
            ffull = os.path.join(root, fname)
            # Status update
            print(ffull)
            # Perform substitutions in place
            rewrite_imports(ffull, CAPE_MODNAME_MAP)


# Rewrite imports for one renamed module
def rewrite_imports(
        infile: str,
        modnames: dict,
        outfile: Optional[str] = None):
    r"""Rewrite imports for a collection of module name changes

    :Call:
        >>> rewrite_imports(infile, modnames, outfile=None)
    :Inputs:
        *infile*: :class:`str`
            Name of file to read
        *modnames*: :class:`dict`
            Key is old module name and value is new module name
        *outfile*: {``None``} | :class:`str`
            Name of file to write (defaults to *infile*)
    """
    # Default name of the output file
    outfile = infile if outfile is None else outfile
    # Read the content of the file
    content = open(infile, 'r').read()
    # Loop through replacments
    new_content = sub_imports(content, modnames)
    # Write the modified content back to the file
    with open(outfile, 'w') as fp:
        fp.write(new_content)


# Rewrite imports for collection of module renames
def sub_imports_1to2(content: str) -> str:
    r"""Rewrite imports for CAPE upgrade from 1 to 2

    :Call:
        >>> new_content = sub_imports_1to2(content)
    :Inputs:
        *content*: :class:`str`
            Original contents before renaming
    :Outputs:
        *new_content*: :class:`str`
            Revised *content* with rewritten imports
    """
    return sub_imports(content, CAPE_MODNAME_MAP)


# Rewrite imports for collection of module renames
def sub_imports(content: str, modnames: dict) -> str:
    r"""Rewrite imports for a collection of module name changes

    :Call:
        >>> new_content = sub_imports(content, modnames)
    :Inputs:
        *content*: :class:`str`
            Original contents before renaming
        *modnames*: :class:`dict`
            Key is old module name and value is new module name
    :Outputs:
        *new_content*: :class:`str`
            Revised *content* with rewritten imports
    """
    # Loop through replacments
    for mod1, mod2 in modnames.items():
        # Make replacements
        content = sub_modname(content, mod1, mod2)
    # Output
    return content


# Rewrite imports for one renamed module
def sub_modname(content: str, mod1: str, mod2: str) -> str:
    r"""Replace imports of *mod1* with imports of *mod2*

    There are some imports that will not get updated properly when
    *mod1* and *mod2* are not of the same depth. For example, updating
    ``cape.cntl`` -> ``cape.cfdx.cntl`` will work for most imports, but
    it won't proper change

    .. code-block:: python

        from ..cntl import Cntl

    to

    .. code-block:: python

        from ..cfdx.cntl import Cntl

    Such cases should be limited to CAPE-internal functions, so this
    limitation should not affect external users' hook functions or other
    scripts and modules.

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
    # Expression for "not followed by longer module name"
    c1 = "(?![a-zA-Z0-9_])"
    # Expression for "not preceded by longer module name"
    c2 = "(?<![a-zA-Z_])"
    # Form regular exrpressions
    pat0 = re.compile(rf"^import\s+{full1}{c1}", re.MULTILINE)
    pat1 = re.compile(rf"^from\s+{parent1}\s+import\s+{basename1}{c1}", re.M)
    pat2 = re.compile(rf"^from\s+{full1}([ .])", re.MULTILINE)
    pat3 = re.compile(rf"^from\s+(\.+)\s+import\s{basename1}{c1}", re.M)
    # Replace name of module in subsequent calls
    # Careful:
    #     a = cape.attdb.f() -> a = cape.dkit.f()
    #     a = attdb.f() -> dkit.f()
    #     a = escape.attdb.f() !-> escape.dkit.f()
    #     a = mattdb.f() !-> mdkit.f()
    pat0b = re.compile(rf"{c2}{full1}\.")
    pat1b = re.compile(rf"{c2}{basename1}\.")
    # Replacements
    out0 = f"import {mod2}"
    out1 = f"from {parent2} import {basename2}"
    out2 = f"from {mod2}\\1"
    out3 = f"from \\1 import {basename2}"
    out0b = f"{mod2}."
    out1b = f"{basename2}."
    # Replace the matching import statements
    new_content = pat0.sub(out0, content)
    new_content = pat1.sub(out1, new_content)
    new_content = pat2.sub(out2, new_content)
    new_content = pat3.sub(out3, new_content)
    new_content = pat0b.sub(out0b, new_content)
    new_content = pat1b.sub(out1b, new_content)
    # Output
    return new_content
