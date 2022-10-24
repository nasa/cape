r"""
:mod:`cape.pyfun.options.util`: Utilities for pyFun options module
===================================================================

This module provides tools to read, access, modify, and write settings
for :mod:`cape.pyfun`.  It is based off of the
:mod:`cape.cfdx.options.util` module and provides a special class
:class:`cape.cfdx.options.odict` that is subclassed from the Python
built-in :class:`dict`.  Behavior, such as ``opts['Namelist']`` or 
``opts.get('Namelist')`` are also present.  In addition, many
convenience methods such as ``opts.get_FUN3DNamelist()`` are provided.

In addition, this module controls default values of each pyFun
parameter in a three-step process.  The precedence used to determine
what the value of a given parameter should be is below.

    #. Values directly specified in the input file, :file:`pyFun.json`
    
    #. Values specified in the default control file,
       :file:`$PYFUN/settings/pyFun.default.json`
    
    #. Hard-coded defaults from this module
    
:See Also:
    * :mod:`cape.cfdx.options.util`
    * :mod:`cape.pyfun.options`
"""

# Standard library
import os

# Local imports
from ...cfdx.options.util import rc, getel, setel, loadJSONFile, odict


# Local folders
PYFUN_OPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PYFUN_FOLDER = os.path.dirname(PYFUN_OPTS_FOLDER)

# Backup default settings
rc["project_rootname"] = "pyfun"
rc["grid_format"] = "aflr3"
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
    r"""Get default from *cape.pyfun.options.rc*; ensure a scalar
    
    :Call:
        >>> v = rc0(s)
    :Inputs:
        *s*: :class:`str`
            Name of parameter to extract
    :Outputs:
        *v*: any
            Either ``rc[s]`` or ``rc[s][0]``, whichever is appropriate
    :Versions:
        * 2014-08-01 ``@ddalle``: Version 1.0
    """
    # Use the `getel` function to do this.
    return getel(rc[p], 0)

    
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
        * 2015-10-26 ``@ddalle``: Version 1.0; :func:`getTemplateFolder`
        * 2021-03-01 ``@ddalle``: Version 2.0
            - Moved to ``cape/pyfun/`` folder
            - Compatible with :mod:`setuptools`
    """
    # Join with BaseFolder and 'templates'
    return os.path.join(PYFUN_FOLDER, 'templates', fname)

    
# Function to get a template file name
def getFun3DTemplate(fname):
    r"""Get full path to template with file name *fname*
    
    :Call:
        >>> fabs = getFun3DTemplate(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to the template file
    :Versions:
        * 2016-04-27 ``@ddalle``: Version 1.0
        * 2021-03-01 ``@ddalle``: Version 2.0; see :func:`get_template`
    """
    # Get the full path
    return get_template(fname)


# Function to get the defautl settings.
def getPyFunDefaults():
    r"""Read ``pyFun.default.json`` default JSON file
    
    :Call:
        >>> defs = getPyFunDefaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: Version 1.0
        * 2014-07-28 ``@ddalle``: Version 1.1; :mod:`options` module
        * 2021-03-01 ``@ddalle``: Version 2.0; local JSON file
    """
    # Fixed default file
    fname = os.path.join(PYFUN_OPTS_FOLDER, "pyFun.default.json")
    # Process the default input file.
    return loadJSONFile(fname)

