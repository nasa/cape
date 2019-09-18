"""
:mod:`cape.pyfun.options.util`: Utilities for pyFun options module
===================================================================

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  It is based off of the :mod:`cape.options.util` module and
provides a special class :class:`cape.options.odict` that is subclassed from
the Python built-in :class:`dict`.  Behavior, such as ``opts['Namelist']`` or 
``opts.get('Namelist')`` are also present.  In addition, many convenience
methods such as ``opts.get_FUN3DNamelist()`` are provided.

In addition, this module controls default values of each pyFun parameter in a
three-step process.  The precedence used to determine what the value of a given
parameter should be is below.

    #. Values directly specified in the input file, :file:`pyFun.json`
    
    #. Values specified in the default control file,
       :file:`$PYFUN/settings/pyFun.default.json`
    
    #. Hard-coded defaults from this module
    
:See Also:
    * :mod:`cape.options.util`
    * :mod:`cape.pyfun.options`
"""

# Import CAPE options utilities
from cape.options.util import *

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(os.path.split(_fname)[0])[0]

# Backup default settings
rc["project_rootname"]     ="pyfun"
rc["grid_format"]          = "aflr3"
rc["nodet_animation_freq"] = -1
# Solution mode settings
rc["KeepRestarts"] = False
# Mesh settings
rc["BCFile"] = "pyfun.mapbc"
# Other files
rc["RubberData"] = "rubber.data"
# Adaptation settings
rc["Adaptive"]   = False
rc["Dual"]       = False
rc["AdaptPhase"] = True
rc["DualPhase"]  = True
rc["PreMesh"]    = False
# Settings for ``dual``
rc["nIterAdjoint"] = 200
rc["dual_outer_loop_krylov"] = True
rc["dual_rad"] = True
rc["dual_adapt"] = True
# Namelist settings
rc["namelist_dist_tolerance"] = 1.0e-3
    

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
def getFun3DTemplate(fname):
    """Get full path to template with file name *fname*
    
    :Call:
        >>> fabs = getFun3DTemplate(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to the file, :file:`$PYCART/templates/cart3d/$fname`
    :Versions:
        * 2016-04-27 ``@ddalle``: Copied from pyCart
    """
    # Construct the path relative to the Cape template folder
    ff3d = os.path.join('fun3d', fname)
    # Get the full path
    return getTemplateFile(ff3d)


# Function to get the defautl settings.
def getPyFunDefaults():
    """
    Read :file:`pyCart.default.json` default settings configuration file
    
    :Call:
        >>> defs = getPyFunDefaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: First version
        * 2014-07-28 ``@ddalle``: Moved to new options module
    """
    # Fixed default file
    fname = os.path.join(PyFunFolder, 
        "..", "..", "settings", "pyFun.default.json")
    # Process the default input file.
    return loadJSONFile(fname)


