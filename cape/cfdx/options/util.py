r"""
:mod:`cape.cfdx.options.util`: Utilities for options modules
=============================================================

This module provides utilities for the CAPE options module. It includes
the :class:`cape.options.util.odict` class upon which all CAPE options
classes are based, and it several basic methods useful to processing
options.

The :func:`getel` and :func:`setel` methods in particular play an
important role in the entire CAPE coding strategy.

"""

# Standard library modules
import copy
import functools
import io
import json
import os
import re

# Third-party modules
import numpy as np


# Local folders
CAPE_OPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CAPE_FOLDER = os.path.dirname(CAPE_OPTS_FOLDER)

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
_OptsFolder = os.path.split(_fname)[0]
_CFDXFolder = os.path.split(_OptsFolder)[0]
# Actual module home
CapeFolder = os.path.split(_CFDXFolder)[0]
# Parent folder
BaseFolder = os.path.split(CapeFolder)[0]

# Backup default settings
rc = {
    "nSubmit": 10,
    "Verbose": False,
    "GroupMesh": False,
    "ConfigFile": "Config.xml",
    "RefArea": 1.0,
    "RefLength": 1.0,
    "RefPoint": [[0.0, 0.0, 0.0]],
    "Xslices": [0.0],
    "Yslices": [0.0],
    "Zslices": [0.0],
    "nIter": 100,
    "PhaseSequence": [[0]],
    "PhaseIters": [[200]],
    "PythonExec": None,
    "cfl": 1.1,
    "cflmin": 0.8,
    "nOrders": 12,
    "dt": 0.1,
    "qsub": True,
    "sbatch": False,
    "Resubmit": False,
    "Continue": True,
    "PreMesh": False,
    "y_is_spanwise": True,
    "db_stats": 0,
    "db_min": 0,
    "db_max": 0,
    "db_dir": "data",
    "db_nCut": 200,
    "Delimiter": ",",
    "binaryIO": True,
    "tecO": True,
    "nProc": 8,
    "mpicmd": "mpiexec",
    "MPI": False,
    "TriFile": "Components.i.tri",
    "mesh2d": False,
    "dC": 0.01,
    "nAvg": 100,
    "nPlot": None,
    "nRow": 2,
    "nCol": 2,
    "FigWidth": 8,
    "FigHeight": 6,
    "PBS_j": "oe",
    "PBS_o": None,
    "PBS_e": None,
    "PBS_r": "n",
    "PBS_S": "/bin/bash",
    "PBS_select": 1,
    "PBS_ncpus": 20,
    "PBS_mpiprocs": 20,
    "PBS_model": "ivy",
    "PBS_aoe": None,
    "PBS_W": "",
    "PBS_q": "normal",
    "PBS_walltime": "8:00:00",
    "Slurm_A": "",
    "Slurm_gid": "",
    "Slurm_N": 1,
    "Slurm_n": 20,
    "Slurm_time": "8:00:00",
    "Slurm_shell": "/bin/bash",
    "Slurm_p": "normal",
    "ulimit_c": 0,
    "ulimit_d": "unlimited",
    "ulimit_e": 0,
    "ulimit_f": "unlimited",
    "ulimit_i": 127812,
    "ulimit_l": 64,
    "ulimit_m": "unlimited",
    "ulimit_n": 1024,
    "ulimit_p": 8,
    "ulimit_q": 819200,
    "uliimt_r": 0,
    "ulimit_s": 4194304,
    "ulimit_t": "unlimited",
    "ulimit_u": 127812,
    "ulimit_v": "unlimited",
    "ulimit_x": "unlimited",
    "ArchiveFolder": "",
    "ArchiveFormat": "tar",
    "ArchiveAction": "full",
    "ArchiveProgress": True,
    "ArchiveType": "full",
    "ArchiveTemplate": "full",
    "ArchiveFiles": [],
    "ArchiveGroups": [],
    "ProgressDeleteFiles": [],
    "ProgressDeleteDirs": [],
    "ProgressTarGroups": [],
    "ProgressTarDirs": [],
    "ProgressUpdateFiles": [],
    "ProgressArchiveFiles": [],
    "PreDeleteFiles": [],
    "PreDeleteDirs": [],
    "PreTarGroups": [],
    "PreTarDirs": [],
    "PreUpdateFiles": [],
    "PostDeleteFiles": [],
    "PostDeleteDirs": [],
    "PostUpdateFiles": [],
    "PostTarGroups": [],
    "PostTarDirs": [],
    "PostUpdateFiles": [],
    "SkeletonFiles": ["case.json"],
    "SkeletonTailFiles": [],
    "SkeletonTarDirs": [],
    "RemoteCopy": "scp",
    "TarPBS": "tar",
}

# TriMap settings
rc["atoldef"] = 1e-2
rc["rtoldef"] = 1e-4
rc["ctoldef"] = 1e-4
rc["ztoldef"] = 5e-2
rc["antoldef"] = 3e-2
rc["rntoldef"] = 1e-4
rc["cntoldef"] = 1e-3
rc["rztoldef"] = 1e-5

# Intersect options
rc['intersect_rm'] = False
rc['intersect_smalltri'] = 1e-4
rc['intersect_triged'] = True

# AFLR3 settings
rc['aflr3_cdfr']   = 1.1
rc['aflr3_cdfs']   = None
rc['aflr3_mdf']    = 2
rc['aflr3_mdsblf'] = 1
rc['aflr3_nqual']  = 2


ARRAY_TYPE_NAMES = {'list', 'tuple', 'array', 'ndarray'}


# Utility function to get elements sanely
def getel(x, i=None):
    r""" Return element *i* of an array if possible
    
    :Call:
        >>> x = getel(x)
        >>> xi = getel(x, i)
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: ``None`` | :class:`int`
            Index
    :Outputs:
        *xi*: scalar
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater
            than the length of *x*, or ``x`` if *x* is not a
            :class:`list` or :class:`numpy.ndarray` instance
    :Examples:
        >>> getel('abc', 2)
        'abc'
        >>> getel(1.4, 0)
        1.4
        >>> getel([200, 100, 300], 1)
        100
        >>> getel([200, 100, 300], 15)
        300
        >>> getel([200, 100, 300])
        200
    :Versions:
        * 2014-07-29 ``@ddalle``: Version 1.0
        * 2021-10-18 ``@ddalle``: Version 1.1; add :class:`tuple`
    """
    # Check the type.
    if i is None:
        return x
    if type(x).__name__ in ARRAY_TYPE_NAMES:
        # Check for empty input.
        if len(x) == 0:
            return None
        # Array-like
        if i:
            # Check the length.
            if i >= len(x):
                # Take the last element.
                return x[-1]
            else:
                # Take the *i*th element.
                return x[i]
        else:
            # Use the first element.
            return x[0]
    else:
        # Scalar
        return x


# Utility function to set elements sanely
def setel(x, i, xi):
    r"""Return element *i* of an array if possible
    
    :Call:
        >>> y = setel(x, i, xi)
        
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: :class:`int`
            Index. If *i* is ``None``, the output is reset to *xi*
        *xi*: scalar
            Value to set at scalar
    :Outputs:
        *y*: number-like or list-like
            Input *x* with ``y[i]`` set to ``xi`` unless *i* is ``None``
    :Examples:
        >>> setel(['a', 2, 'c'], 1, 'b')
        ['a', 'b', 'c']
        >>> setel(['a', 'b'], 2, 'c')
        ['a', 'b', 'c']
        >>> setel('a', 2, 'c')
        ['a', None, 'b']
        >>> setel([0, 1], None, 'a')
        'a'
    :Versions:
        * 2014-07-29 ``@ddalle``: Version 1.0
        * 2021-10-18 ``@ddalle``: Version 1.1; add :class:`tuple`
    """
    # Check the index input.
    if i is None:
        # Scalar output
        return xi
    # Ensure list
    if type(x).__name__ in ARRAY_TYPE_NAMES:
        # Already a list; make a copy
        y = list(x)
    else:
        # Create a singleton list
        y = [x]
    # Get default value
    if len(y) > 0:
        # Select the last value
        ydef = y[-1]
    else:
        # Set ``None`` until we get something
        ydef = None
    # Make sure *y* is long enough.
    for j in range(len(y), i):
        y.append(ydef)
    # Check if we are setting an element or appending it
    if i >= len(y):
        # Append
        y.append(xi)
    else:
        # Set the value
        y[i] = xi
    # Output
    return y
    

# Function to ensure scalar from above
def rc0(p):
    r"""Return default setting for named parameter
    
    :Call:
        >>> v = rc0(p)
    :Inputs:
        *p*: :class:`str`
            Name of parameter to extract
    :Outputs:
        *v*: any
            Either ``rc[p]`` or ``rc[p][0]``, whichever is appropriate
    :Versions:
        * 2014-08-01 ``@ddalle``: Version 1.0
    """
    # Use the :func:`getel` function to do this.
    return getel(rc.get(p), 0)


# Regular expression for JSON file inclusion
regex = re.compile(
    r'(?P<before>.*)' +
    r'(?P<cmd>JSONFile\("(?P<json>[-\w.+= /\\]+)"\))' +
    r'(?P<after>.*)')


# Function to expand CSV file inputs
def expandJSONFile(fname):
    r"""Expand contents of other JSON files
    
    :Call:
        >>> txt, fnames, linenos = expandJSONFile(fname)
    :Inputs:
        *fname*: :class:`str` | :class:`unicode`
            Name of JSON file to read
    :Outputs:
        *txt*: :class:`unicode`
            Full text with references to JSON file(s) expanded
        *fnames*: :class:`list` (:class:`str`)
            List of files read (can include the same file multiple
            times) including *fname* and any other expanded
            ``JSONFile()`` directives
        *linenos*: :class:`np.ndarray` (:class:`int`, ndim=2)
            Line numbers in original files; column *j* represents the
            line number of each line in file *j*; ``0`` for lines not
            from file *j*
    :Versions:
        * 2015-12-10 ``@ddalle``: Version 1.0
    """
    # Read the input file.
    txt = io.open(fname, mode="r", encoding="utf-8").read()
    # Split lines
    lines = txt.rstrip().split('\n')
    # Number of lines
    n = len(lines)
    # Initialize line numbers
    linenos = np.zeros((n,1), dtype="int")
    linenos[:,0] = 1 + np.arange(n)
    # Initialize list of file names
    fnames = [fname]
    # Number of JSON files read (can read same file more than once)
    nf = 1
    # Start with the first line.
    i = 0
    # Loop through lines.
    while i < len(lines):
        # Get the line
        line = lines[i]
        # Check if line starts with a comment
        if line.lstrip().startswith("//"):
            # Javascript-style comment
            lines[i] = ""
            continue
        elif line.lstrip().startswith("#"):
            # Python-style comment
            lines[i] = ""
            continue
        # Check for an inclusion
        match = regex.search(line)
        # If no match, move along
        if match is None:
            # Go to next line
            i += 1
            continue
        # Extract the file name.
        fjson = match.group("json")
        # Expand that JSON file
        t_j, F_j, ln_j = expandJSONFile(fjson)
        # Split text to lines
        lines_j = t_j.rstrip().split("\n")
        # Number of lines
        n_j = len(lines_j)
        # Number of included files
        nf_j = len(F_j)
        # Pad line counts with zeros for lines of new file
        if n_j > 1:
            # Split current line and insert n_j-1 zeros
            linenos = np.vstack((
                linenos[:i+1,:], np.zeros((n_j-2,nf), dtype="int"),
                linenos[i:,:]))
        # Create line counts for new file
        linenos_json = np.vstack((
            np.zeros((i,nf_j), dtype="int"), ln_j,
            np.zeros((n-i-1,nf_j), dtype="int")))
        # Accumulate line numbers and file list
        fnames += F_j
        linenos = np.hstack((linenos, linenos_json))
        # Modify line count and file count
        n  += (n_j-1)
        nf += nf_j
        # Update first and last line of expansion
        if n_j == 1:
            # One-line inclusion
            lines[i] = line.replace(match.group("cmd"), lines_j[0])
        else:
            # Update first line
            lines[i] = match.group("before") + lines_j[0]
            # Update last line of inclusion
            lines_j[-1] = lines_j[-1].rstrip() + match.group("after")
            # Update line set
            lines = lines[:i+1] + lines_j[1:] + lines[i+1:]
        # Check for multiple inclusions
        if regex.search(lines[i]):
            # Remain on this line
            pass
        else:
            # Move past expanded file
            i += n_j
    # Return the lines as one string.
    txt = "\n".join(lines) + "\n"
    # Output
    return txt, fnames, linenos


# Function to read JSON file with all the works
def loadJSONFile(fname):
    r"""Read JSON file w/ helpful error handling and comment stripping
    
    :Call:
        >>> d = loadJSONFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON file to read
    :Outputs:
        *d*: :class:`dict`
            JSON contents in Python form
    :Versions:
        * 2015-12-15 ``@ddalle``: Version 1.0
    """
    # Read the input file
    txt, fnames, linenos = expandJSONFile(fname)
    # Process into dictionary
    try:
        # Process into dictionary
        d = json.loads(txt)
    except Exception as e:
        # Get the line number
        try:
            # Read from the error message
            etxt = re.findall('line [0-9]+', e.args[0])[0]
            # Line number
            n = int(etxt.split()[1])
            # Get lines so we can print surrounding text by line number
            lines = txt.split("\n")
            # Start and end line number
            n0 = max(n-3, 0)
            n1 = min(n+2, len(lines))
            # Initialize message with 
            msg = "Error while reading JSON file '%s':\n" % fname
            # Add the exception's message
            msg += "\n".join(list(e.args)) + "\n"
            # Loop through individual files
            for i, fn in enumerate(fnames):
                # Get line number 
                lni = linenos[n-1,i]
                # Skip if ``0``
                if lni == 0: continue
                # Add to report
                msg += "    (line %i of file '%s')\n" % (lni, fn)
            # Additional header
            msg += "\nLines surrounding problem area (comments stripped):\n"
            # Print some lines around the problem
            for i in range(n0, n1):
                # Add line with line number
                if i+1 == n:
                    # Add special marker for reported line
                    msg += ("%4i> %s\n" % (i+1, lines[i]))
                else:
                    # Neighboring line
                    msg += ("%4i: %s\n" % (i+1, lines[i]))
            # Show the message
            raise ValueError(msg)
        except ValueError as e:
            # Raise the error we just made.
            raise e
        except Exception:
            # Unknown error
            raise e
    # Output
    return d


# Function to get the default settings.
def getDefaults(fname):
    r"""Read default settings configuration file
    
    :Call:
        >>> defs = getDefaults(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file with settings to read
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: Version 1.0
        * 2014-07-28 ``@ddalle``: Version 1.1; in options module
    """
    # Process the default input file.
    return loadJSONFile(fname)


# Function to get the default CAPE settings
def getCapeDefaults():
    r"""Read default CAPE settings configuration file
    
    :Call:
        >>> defs = getCapeDefaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2015-09-20 ``@ddalle``: Version 1.0
        * 2021-03-01 ``@ddalle``: Version 2.0; local JSON file
    """
    # File name
    fname = os.path.join(CAPE_OPTS_FOLDER, 'cape.default.json')
    # Read the settings.
    return getDefaults(fname)


# Function to get template
def getTemplateFile(fname):
    r"""Get the absolute path to a template file by name
    
    :Call:
        >>> fabs = getTemplateFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
        *fabs*: :class:`str`
            Full path to file
    :Versions:
        * 2015-10-26 ``@ddalle``: Version 1.0
    """
    # Join with BaseFolder and 'templates'
    return os.path.join(BaseFolder, 'templates', fname)


# Get the keys of the default dict.
def applyDefaults(opts, defs):
    r"""Recursively apply defaults for any missing options
    
    :Call:
        >>> opts = applyDefaults(opts, defs)
    :Inputs:
        *opts*: :class:`dict` | :class:`odict`
            Options dictionary with some options possibly missing
        *defs*: :class:`dict`
            Full dictionary of default settings
    :Outputs:
        *opts*: :class:`dict` | :class:`odict`
            Input dictionary with all of the fields of *defs*
    :Versions:
        * 2014-06-17 ``@ddalle``: Version 1.0
        * 2014-07-28 ``@ddalle``: Version 1.1; move to options module
    """
    # Loop through the keys in the options dict.
    for k in defs:
        # Check if the key is non-default.
        if k not in opts:
            # Assign the key.
            opts[k] = defs[k]
        elif (type(opts[k]) is dict) and (not k.startswith("Ref")):
            # Recurse for dictionaries.
            opts[k] = applyDefaults(opts[k], defs[k])
    # Output the modified defaults.
    return opts


# Test if a variable is "list-like"
def isArray(x):
    r"""Test if a variable is "list-like."
    
    :Call:
        >>> q = isArray(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            ``True`` if and only if *x* is a list or NumPy array
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0
    """
    return (type(x).__name__ in ARRAY_TYPE_NAMES)


# Test if a variable is "string-like"
def isStr(x):
    r"""Test if a variable is "string-like"
    
    :Call:
        >>> q = isArray(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            ``True`` if and only if *x* is a string or unicode
    :Versions:
        * 2014-12-17 ``@ddalle``: Version 1.0
    """
    # Get the type.
    typ = type(x).__name__
    # Test it.
    return typ.startswith('str') or (typ in ['unicode'])


# Dictionary derivative specific to options
class odict(dict):
    r"""Dictionary-based options module
    
    :Call:
        >>> opts = odict(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options
    :Outputs:
        *opts*: :class:`cape.options.util.odict`
            Dictionary-based options interface
    :Versions:
        * 2014-08-02 ``@ddalle``: Version 1.0
        * 2015-11-10 ``@ddalle``: More robust :func:`get_key` using *rck*
    """
    # General "get" function
    def get_key(self, k, i=None, rck=None):
        r"""Intelligently get option for index *i* of key *k*

        This is a two-step process.  The first is to get the dictionary
        value or the default if *k* is not in *opts*. The default is
        ``rc[k]``. Let *V* be the result of the process.

        The second step is to apply indexing. If *V* is a scalar or *i*
        is ``None``, then *V* is the output. Otherwise, the function
        will attempt to return ``V[i]``, but if *i* is too large,
        ``V[-1]`` is the output.

        :Call:
            >>> v = opts.get_key(k, i, rck=None)
        :Inputs:
            *k*: :class:`str`
                Name of key to get
            *i*: :class:`int` | ``None``
                Index to apply
            *rck*: :class:`str` | ``None``
                Name of *rc0* key to default to
        :Outputs:
            *v*: **any**
                Let ``V=opts.get(k,rc[k])``. Then *v* is either
                ``V[i]`` if possible, ``V[-1]`` if *V* is a list and *i*
                is not ``None``, or ``V`` otherwise
        :See also:
            * :func:`cape.options.util.getel`
        :Versions:
            * 2014-08-02 ``@ddalle``: Version 1.0
            * 2015-11-10 ``@ddalle``: Version 1.1; add *rck*
        """
        # Default key name
        if rck is None: rck = k
        # Get the value after applying defaults.
        v = self.get(k, rc.get(rck))
        # Apply intelligent indexing.
        return getel(v, i)
        
    # General "set" function
    def set_key(self, k, v=None, i=None, rck=None):
        r"""Set option for key *k*

        This sets the value for ``opts[k]`` or ``opts[k][i]`` if
        appropriate. If *i* is greater than the length of ``opts[k]``,
        then ``opts[k]`` is appended with its current last value enough
        times to make ``opts[k][i]`` exist.

        :Call:
            >>> opts.set_key(k, v=None, i=None, rck=None)
        :Inputs:
            *k*: :class:`str`
                Name of key to set
            *i*: :class:`int` | ``None``
                Index to apply
            *v*: any
                Value to set
            *rck*: :class:`str` | ``None``
                Name of key in *rc0* default option dictionary; defaults to *k*
        :See also:
            * :func:`cape.options.util.setel`
        :Versions:
            * 2014-08-02 ``@ddalle``: Version 1.0
            * 2015-11-10 ``@ddalle``: Version 1.1; add *rck*
        """
        # Check for default key name
        if rck is None: rck = k
        # Check for default value.
        if v is None:
            # Get the default, but ensure a scalar.
            v = rc0(rck)
        # Get the current full setting.
        V = self.get(k, rc.get(rck))
        # Assign the input value .
        self[k] = setel(V, i, v)
    
    # Copy
    def copy(self):
        r"""Create a copy of an options interface
        
        :Call:
            >>> opts1 = opts.copy()
        :Inputs:
            *opts*: :class:`odict`
                Options instance
        :Outputs:
            *opts1*: :class:`odict`
                Deep copy of options instance
        :Versions:
            * 2019-05-10 ``@ddalle``: Version 1.0
        """
        # Initialize copy
        opts = self.__class__()
        # Loop through keys
        for k, v in self.items():
            # Check the type
            if not isinstance(v, dict):
                # Save a copy of the key
                opts[k] = copy.copy(v)
            else:
                # Recurse
                opts[k] = v.copy()
        # Output
        return opts

    # Generic subsection
    def init_section(self, cls, sec=None, parent=None):
        r"""Initialize a generic section

        :Call:
            >>> opts.init_section(cls, sec=None, parent=None)
        :Inputs:
            *opts*: :class:`odict`
                Options interface
            *cls*: :class:`type`
                Class to use for *opts[sec]*
            *sec*: {*cls.__name__*} | :class:`str`
                Specific key name to use for subsection
            *parent*: {``None``} | :class:`str`
                Other subsection from which to inherit defaults
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Default name
        if sec is None:
            # Use the name of the class
            sec = cls.__name__
        # Check if present
        if sec not in self:
            # Create empty instance
            self[sec] = cls()
        # Otherwise get value
        v = self[sec]
        # Check its type
        if isinstance(v, cls):
            # Already good
            pass
        elif isinstance(v, dict):
            # Convert :class:`dict` to special class
            self[sec] = cls(**v)
        else:
            # Got something other than a mapping
            print("  Warning: could not convert options section '%s'," % sec)
            print("           which has type '%s'" % type(v).__name__)
            return
        # Check for *parent* to define default settings
        if parent:
            # Get the settings of parent
            vp = self.get(parent)
            # Ensure it's a dict
            if not isinstance(vp, dict):
                return
            # Loop through *vp*, but don't overwrite
            for k, vpk in vp.items():
                v.setdefault(k, vpk)


# Decorator to get function from subclass
def subsec_func(cls, sec=None, parent=None, init=True):
    r"""Decorator (w/ args) to apply a function from a subsection class
    
    :Call:
        >>> f = subsec_func(cls, sec=None, parent=None, init=True)
    :Inputs:
        *cls*::class:`type`
            Class to apply to subsection
        *sec*: {*cls.__name*} | :class:`str`
            Name of subsection
        *init*: {``True``} | ``False``
            If ``True`` and nontrivial *cls*, initialize subsection
        *parent*: {``None``} | :class:`str`
            Name of section from which to get default settings
    :Outputs:
        *f*: :class:`function`
            Decorator with arguments expanded
    :Examples:
        .. code-block:: python
        
            @subsec_func("RunControl", RunControl)
            def get_PhaseSequence(self, *a, **kw):
                pass
                
    :Versions:
        * 2019-01-10 ``@ddalle``: Version 1.0
        * 2021-10-18 ``@ddalle``: Version 1.1; default *sec*
    """
    # Default *sec*
    if sec is None:
        sec = cls.__name__
    # Decorator for the function
    def decorator_subsec(func):
        # Inherit metadata from func
        @functools.wraps(func)
        # The before and after function
        def wrapper(self, *a, **kw):
            # Initialize the section
            if init and (cls is not None):
                self.init_section(cls, sec, parent=parent)
            # Get the function from the subsection
            f = getattr(self[sec], func.__name__)
            # Call the function from the subsection
            v = f(*a, **kw)
            # Return value
            return v
        # Copy the docstring
        if cls is not None:
            wrapper.__doc__ = getattr(cls,func.__name__).__doc__
        # Output
        return wrapper
    # Return decorator
    return decorator_subsec


# Apply all methods of one subsection class to parent
def promote_subsec(cls1, cls2, sec=None, skip=[], **kw):
    r"""Promote all methods of a subsection class to parent options class

    Methods of parent class will not be overwritten

    :Call:
        >>> promote_subsec(cls1, cls2, sec=None, skip=[], **kw)
    :Inputs:
        *cls1*: :class:`type`
            Parent class
        *cls2*: :class:`type`
            Subsection class
        *skip*: {``[]``} | :class:`list`
            List of methods from *cls2* not to add to *cls1*
        *init*: {``True``} | ``False``
            If ``True``, initialize subsection when *cls1* methods used
        *parent*: {``None``} | :class:`str`
            Name of section from which to get default settings
    :Versions:
        * 2019-01-10 ``@ddalle``: Version 1.0
    """
    # Get property dictionaries
    dict1 = cls1.__dict__
    dict2 = cls2.__dict__
    # Create the decorator to promote each method (function)
    f_deco = subsec_func(cls2, sec, **kw)
    # Loop through methods of *cls2*
    for fn in dict2:
        # Manual skipping
        if fn in skip:
            continue
        # Get value of *cls2* attribute
        func = dict2[fn]
        # Skip if not a function
        if not callable(func):
            continue
        # Check if already present
        if fn in dict1:
            continue
        # Set attribute to decorated function
        setattr(cls1, fn, f_deco(func))

