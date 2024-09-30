r"""
:mod:`cape.pylava.options.util`: Utilities for pyLava options module
===================================================================

This module provides tools to read, access, modify, and write settings
for :mod:`cape.pylava`.  It is based off of the
:mod:`cape.cfdx.options.util` module and provides a special class
:class:`cape.cfdx.options.odict` that is subclassed from the Python
built-in :class:`dict`.  

In addition, this module controls default values of each pyLava
parameter in a three-step process.  The precedence used to determine
what the value of a given parameter should be is below.

    #. Values directly specified in the input file, :file:`pyLava.json`

    #. Values specified in the default control file,
       :file:`$PYLAVA/settings/pyLava.default.json`

    #. Hard-coded defaults from this module

:See Also:
    * :mod:`cape.cfdx.options.util`
    * :mod:`cape.pylava.options`
"""

# Standard library
import os

# Local imports
from ...cfdx.options.util import applyDefaults, rc, getel, loadJSONFile


# Local folders
PYLAVA_OPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PYLAVA_FOLDER = os.path.dirname(PYLAVA_OPTS_FOLDER)

# Backup default settings
rc["project_rootname"] = "pylava"


# Function to get the defautl settings.
def getPyLavaDefaults():
    r"""Read ``pyLava.default.json`` default JSON file

    :Call:
        >>> defs = getPyLavaDefaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2024-08-05 ``@sneuhoff``: Version 1.0
    """
    # Fixed default file
    fname = os.path.join(PYLAVA_OPTS_FOLDER, "pyLava.default.json")
    # Process the default input file.
    return loadJSONFile(fname)

