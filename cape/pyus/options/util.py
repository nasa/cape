r"""
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


# Local folders
PYUS_OPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PYUS_FOLDER = os.path.dirname(PYUS_OPTS_FOLDER)


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
    r"""Return default from *cape.pyus.options.rc*; ensure a scalar
    
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
    r"""Get full path to template with file name *fname*
    
    :Call:
        >>> fabs = getUS3DTemplate(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to the file, :file:`$CAPE/templates/us3d/$fname`
    :Versions:
        * 2016-04-27 ``@ddalle``: Version 1.0
        * 2021-03-01 ``@ddalle``: Version 2.0; local templates
    """
    # Get the full path
    return get_template(fname)
    

# Function to get template
def get_template(fname):
    r"""Get the absolute path to a template file by name
    
    :Call:
        >>> fabs = get_template(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to file
    :Versions:
        * 2021-03-01 ``@ddalle``: Version 1.0
    """
    # Join with BaseFolder and 'templates'
    return os.path.join(PYUS_FOLDER, 'templates', fname)


# Function to get the default settings.
def getPyUSDefaults():
    r"""Read ``pyUS.default.json`` default settings file
    
    :Call:
        >>> defs = getPyUS()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2019-06-27 ``@ddalle``: Version 1.0 (US3D)
        * 2021-03-01 ``@ddalle``: Version 2.0; local JSON file
    """
    # Fixed default file
    fname = os.path.join(PYUS_OPTS_FOLDER, "pyUS.default.json")
    # Process the default input file.
    return loadJSONFile(fname)

