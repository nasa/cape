"""
This module provides tools to read, access, modify, and write settings for
:mod:`cape.pycart`.  It is based off of the :mod:`cape.cfdx.options.util` module and
provides a special class :class:`cape.cfdx.options.odict` that is subclassed from
the Python built-in :class:`dict`.  Behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart parameter in a
two-step process.  The precedence used to determine what the value of a given
parameter should be is below.

    #. Values directly specified in the input file, :file:`pyCart.json`
    
    #. Values specified in the default control file,
       :file:`$PYCART/settings/pyCart.default.json`
    
    #. Hard-coded defaults from this module
    
:See Also:
    * :mod:`cape.cfdx.options.util`
    * :mod:`cape.pycart.options`
"""

# Import CAPE options utilities
from cape.cfdx.options.util import *


# Local folders
PYCART_OPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PYCART_FOLDER = os.path.dirname(PYCART_OPTS_FOLDER)


# Backup default settings (in case deleted from :file:`pyCart.defaults.json`)
rc["InputCntl"] = "input.cntl"
rc["Adaptive"] = False
rc["AeroCsh"] = "aero.csh"
rc["GroupMesh"] = False
rc["ConfigFile"] = "Config.xml"
rc["RefArea"] = 1.0
rc["RefLength"] = 1.0
rc["RefPoint"] = [0.0, 0.0, 0.0]
rc["Xslices"] = [0.0]
rc["Yslices"] = [0.0]
rc["Zslices"] = [0.0]
rc["PhaseSequence"] = [0]
rc["PhaseIters"] = [200]
rc["first_order"] = 0
rc["robust_mode"] = 0
rc["it_fc"] = 200
rc["clic"] = True
rc["cfl"] = 1.1
rc["cflmin"] = 0.8
rc["nOrders"] = 12
rc["mg_fc"] = 3
rc["RKScheme"] = None # This means don't change it.
rc["dt"] = 0.1
rc["unsteady"] = False
rc["it_avg"] = 0
rc["it_start"] = 100
rc["it_sub"] = 10
rc["jumpstart"] = False
rc["limiter"] = 2
rc["y_is_spanwise"] = True
rc["checkptTD"] = None
rc["vizTD"] = None
rc["fc_clean"] = False
rc["fc_stats"] = 0
rc["db_stats"] = 0
rc["db_min"] = 0
rc["db_max"] = 0
rc["db_dir"] = "data"
rc["db_nCut"] = 200
rc["Delimiter"] = ","
rc["binaryIO"] = True
rc["tecO"] = True
rc["fmg"] = True
rc["pmg"] = False
rc["nProc"] = 8
rc["tm"] = False
rc["buffLim"] = False
rc["mpicmd"] = "mpiexec"
rc["it_ad"] = 120
rc["mg_ad"] = 3
rc["adj_first_order"] = False
rc["n_adapt_cycles"] = 0
rc["etol"] = 1.0e-6
rc["max_nCells"] = 5e6
rc["ws_it"] = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100]
rc["mesh_growth"] = [1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5]
rc["apc"] = ["p", "a"]
rc["buf"] = 1
rc["final_mesh_xref"] = 0
rc["TriFile"] = "Components.i.tri"
rc["mesh2d"] = False
rc["pre"] = "preSpec.c3d.cntl"
rc["inputC3d"] = "input.c3d"
rc["BBox"] = []
rc["XLev"] = []
rc["r"] = 30.0
rc["verify"] = False
rc["intersect"] = False
rc["nDiv"] = 4
rc["maxR"] = 11
rc["pre"] = "preSpec.c3d.cntl"
rc["cubes_a"] = 10
rc["cubes_b"] = 2
rc["sf"] = 0
rc["reorder"] = True
rc["dC"] = 0.01
rc["nAvg"] = 100
rc["nPlot"] = None
rc["nRow"] = 2
rc["nCol"] = 2
rc["FigWidth"] = 8
rc["FigHeight"] = 6
rc["ulimit_s"] = 4194304
rc["ArchiveFolder"] = ""
rc["ArchiveFormat"] = "tar"
rc["ArchiveAction"] = "skeleton"
rc["ArchiveType"] = "full"
rc["ArchiveTemplate"] = "full"
rc["RemoteCopy"] = "scp"
rc["nCheckPoint"] = 2
rc["TarViz"] = "tar"
rc["TarAdapt"] = "tar"
rc["TarPBS"] = "tar"
    

# Function to ensure scalar from above
def rc0(p):
    r"""Get setting from *cape.pycart.options.rc*, but ensure a scalar
    
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
            - Moved to ``cape/pycart/`` folder
            - Compatible with :mod:`setuptools`
    """
    # Join with BaseFolder and 'templates'
    return os.path.join(PYCART_FOLDER, 'templates', fname)


# Function to get the default settings.
def get_pycart_defaults():
    r"""Read ``pyCart.default.json`` default settings file
    
    :Call:
        >>> defs = get_pycart_defaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: Version 1.0
        * 2014-07-28 ``@ddalle``: Version 1.1; :mod:`options` module
        * 2021-03-01 ``@ddalle``: Version 1.2
            - local JSON file
            - :mod:`setuptools` compatible
            - was :func:`getPyCartDefaults`
    """
    # Fixed default file
    fname = os.path.join(PYCART_OPTS_FOLDER, "pyCart.default.json")
    # Process the default input file
    return loadJSONFile(fname)


# Function to get a template file name
def getCart3DTemplate(fname):
    """Get full path to template with file name *fname*
    
    :Call:
        >>> fabs = getPyCartTemplate(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
    :Outputs:
        *fabs*: :class:`str`
            Full path to the file, :file:`$PYCART/templates/cart3d/$fname`
    :Versions:
        * 2015-10-26 ``@ddalle``: Version 1.0
        * 2021-03-01 ``@ddalle``: Version 2.0; see :func:`get_template`
    """
    # Get the full path
    return get_template(fname)
        
