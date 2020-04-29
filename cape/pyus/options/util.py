"""
:mod:`cape.pyus.options.util`: Utilities for pyUS options module
=================================================================

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  It is based off of the :mod:`cape.cfdx.options.util` module and
provides a special class :class:`cape.cfdx.options.odict` that is subclassed from
the Python built-in :class:`dict`.  Behavior, such as ``opts['Namelist']`` or 
``opts.get('Namelist')`` are also present.  In addition, many convenience
methods such as ``opts.get_InputInp()`` are provided.

In addition, this module controls default values of each pyFun parameter in a
three-step process.  The precedence used to determine what the value of a given
parameter should be is below.

    #. Values directly specified in the input file, :file:`pyFun.json`
    
    #. Values specified in the default control file,
       :file:`$CAPE/settings/pyUS.default.json`
    
    #. Hard-coded defaults from this module
    
:See Also:
    * :mod:`cape.cfdx.options.util`
    * :mod:`cape.pyus.options`
"""

# Import CAPE options utilities
from cape.cfdx.options.util import *

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyUSFolder = os.path.split(os.path.split(_fname)[0])[0]

# Backup default settings
rc["us3d_prepar_run"] = True
rc["us3d_prepar_grid"] = "pyus.cas"
rc["us3d_prepar_conn"] = "conn.h5"
rc["us3d_prepar_output"] = "grid.h5"
rc["us3d_input"] = "input.inp"
rc["us3d_grid"] = "grid.h5"
rc["us3d_gas"] = None
    

# Function to ensure scalar from above
def rc0(p):
    """
    Return default setting from ``pyCart.options.rc``, but ensure a scalar
    
    :Call:
        >>> v = rc0(s)
    :Inputs:
        *s*: :class:`str`
            Name of parameter to extract
    :Outputs:
        *v*: any
            Either ``rc[s]`` or ``rc[s][0]``, whichever is appropriate
    :Versions:
        * 2014-08-01 ``@ddalle``: First version
    """
    # Use the `getel` function to do this.
    return getel(rc[p], 0)
    
# Function to get a template file name
def getUS3DTemplate(fname):
    """Get full path to template with file name *fname*
    
    :Call:
        >>> fabs = getUS3DTemplate(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to the file, :file:`$CAPE/templates/us3d/$fname`
    :Versions:
        * 2016-04-27 ``@ddalle``: Copied from pyCart
    """
    # Construct the path relative to the Cape template folder
    ff3d = os.path.join('fun3d', fname)
    # Get the full path
    return getTemplateFile(ff3d)


# Function to get the default settings.
def getPyUSDefaults():
    """
    Read :file:`pyUS.default.json` default settings configuration file
    
    :Call:
        >>> defs = getPyUS()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: First version
        * 2014-07-28 ``@ddalle``: Moved to new options module
        * 2019-06-27 ``@ddalle``: US3D version
    """
    # Fixed default file
    fname = os.path.join(PyUSFolder, 
        "..", "..", "settings", "pyUS.default.json")
    # Process the default input file.
    return loadJSONFile(fname)


