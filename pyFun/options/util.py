"""
Utilities for pyFun Options module: :mod:`pyFun.options.util`
=============================================================

This module provides tools to read, access, modify, and write settings for
:mod:`pyFun`.  The class is based off of the built-int :class:`dict` class, so
its default behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`pyCart.json`
    
    *. Values specified in the default control file,
       :file:`$PYFUN/settings/pyFun.default.json`
    
    *. Hard-coded defaults from this module
"""

# Import CAPE options utilities
from cape.options.util import *

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(os.path.split(_fname)[0])[0]

# Backup default settings (in case deleted from :file:`pyCart.defaults.json`)
rc = {
    "nSubmit": 10,
    "Namelist": "fun3d.nml",
    "AeroCsh": "aero.csh",
    "GroupMesh": True,
    "ConfigFile": "Config.xml",
    "RefArea": 1.0,
    "RefLength": 1.0,
    "RefPoint": [0.0, 0.0, 0.0],
    "Xslices": [0.0],
    "Yslices": [0.0],
    "Zslices": [0.0],
    "nIter": 100,
    "InputSeq": [0],
    "IterSeq": [200],
    "first_order": 0,
    "robust_mode": 0,
    "cfl": 1.1,
    "cflmin": 0.8,
    "MPI": False,
    "unsteady": False,
    "qsub": True,
    "resub": False,
    "use_aero_csh": False,
    "it_avg": 0,
    "jumpstart": False,
    "limiter": 2,
    "y_is_spanwise": True,
    "nSteps": 10,
    "checkptTD": None,
    "vizTD": None,
    "fc_clean": False,
    "fc_stats": 0,
    "db_stats": 0,
    "db_min": 0,
    "db_max": 0,
    "db_dir": "data",
    "db_nCut": 200,
    "Delimiter": ",",
    "binaryIO": True,
    "tecO": True,
    "fmg": True,
    "pmg": False,
    "nProc": 8,
    "tm": False,
    "buffLim": False,
    "mpicmd": "mpiexec",
    "MeshFile": "pyfun.fgrid",
    "dC": 0.01,
    "nAvg": 100,
    "nPlot": None,
    "nRow": 2,
    "nCol": 2,
    "FigWidth": 8,
    "FigHeight": 6,
    "PBS_j": "oe",
    "PBS_r": "n",
    "PBS_S": "/bin/bash",
    "PBS_select": 1,
    "PBS_ncpus": 20,
    "PBS_mpiprocs": 20,
    "PBS_model": "ivy",
    "PBS_W": "",
    "PBS_q": "normal",
    "PBS_walltime": "8:00:00",
    "ArchiveFolder": "",
    "ArchiveFormat": "tar",
    "ArchiveAction": "skeleton",
    "ArchiveType": "full",
    "RemoteCopy": "scp",
    "nCheckPoint": 2,
    "TarViz": "tar",
    "TarAdapt": "tar",
    "TarPBS": "tar",
    "project_rootname": "pyfun",
    "grid_format": "fast"
}
    

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
    # Read the fixed default file.
    lines = open(os.path.join(PyFunFolder, 
        "..", "settings", "pyFun.default.json")).readlines()
    # Strip comments and join list into a single string.
    lines = expandJSONFile(lines)
    # Process the default input file.
    return json.loads(lines)


