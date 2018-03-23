"""
:mod:`cape.options.util`: Utilities for options modules
===========================================================

This module provides utilities for the Cape options module.  It includes the
:class:`cape.options.util.odict` class upon which all Cape options classes are
based, and it several basic methods useful to processing options.

The :func:`getel` and :func:`setel` methods in particular play an important
role in the entire Cape coding strategy.

"""

# Interaction with the OS
import os
# Text processing
import re, json

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
CapeFolder = os.path.split(os.path.split(_fname)[0])[0]
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
    "cfl": 1.1,
    "cflmin": 0.8,
    "nOrders": 12,
    "dt": 0.1,
    "qsub": True,
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

# Intersect options
rc['intersect_rm'] = False
rc['intersect_smalltri'] = 1e-4

# AFLR3 settings
rc['aflr3_cdfr']   = 1.1
rc['aflr3_cdfs']   = None
rc['aflr3_mdf']    = 2
rc['aflr3_mdsblf'] = 1
rc['aflr3_nqual']  = 2

# Utility function to get elements sanely
def getel(x, i=None):
    """
    Return element *i* of an array if possible
    
    :Call:
        >>> xi = getel(x)
        >>> xi = getel(x, i)
        
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: :class:`int`
            Index.  If not specified, treated as though *i* is ``0``
            
    :Outputs:
        *xi*: scalar
            Equal to ``x[i]`` if possible, ``x[-1]`` if *i* is greater than the
            length of *x*, or ``x`` if *x* is not a :class:`list` or
            :class:`numpy.ndarray` instance
    
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
        * 2014-07-29 ``@ddalle``: First version
    """
    # Check the type.
    if i is None:
        return x
    if type(x).__name__ in ['list', 'array', 'ndarray']:
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
    """
    Return element *i* of an array if possible
    
    :Call:
        >>> y = setel(x, i, xi)
        
    :Inputs:
        *x*: number-like or list-like
            A number or list or NumPy vector
        *i*: :class:`int`
            Index to set.  If *i* is ``None``, the output is reset to *xi*
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
        * 2014-07-29 ``@ddalle``: First version
    """
    # Check the index input.
    if i is None:
        # Scalar output
        return xi
    # Ensure list.
    if type(x).__name__ == 'ndarray':
        # NumPy array
        y = list(x)
    elif type(x).__name__ != 'list':
        # Scalar input
        y = [x]
    else:
        # Already a list
        y = x
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
    # Check if we are setting an element or appending it.
    if i >= len(y):
        # Append
        y.append(xi)
    else:
        # Set the value.
        y[i] = xi
    # Output
    return y
    

# Function to ensure scalar from above
def rc0(p):
    """Return default setting from ``cape.options.rc``, but ensure a scalar
    
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
    return getel(rc.get(p), 0)
    
    
# Function to delete comment lines.
def stripComments(lines, char='#'):
    """
    Delete lines that begin with a certain comment character.
    
    :Call:
        >>> txt = stripComments(lines, char='#')
    :Inputs:
        *lines*: :class:`list` of :class:`str`
            List of lines
        *char*: :class:`str`
            String that represents start of a comment
    :Outputs:
        *txt*: :class:`str`
            Lines joined into a single string but with comments removed
    :Versions:
        * 2014-06-03 ``@ddalle``: First version
    """
    # Start with the first line.
    i = 0
    # Check for combined lines.
    if type(lines) == str:
        # Split into lines.
        lines = lines.split('\n')
    # Loop until last line
    for i in range(len(lines)):
        # Get the line and strip leading and trailing white space.
        line = lines[i].strip()
        # Check it.
        if line.startswith(char):
            # Remove the content.
            lines[i] = ""
        else:
            # Remove the newline if appropriate
            lines[i] = lines[i].rstrip()
    # Return the remaining lines.
    return '\n'.join(lines)
    

# Function to expand CSV file inputs
def expandJSONFile(lines):
    """Expand contents of other JSON files
    
    :Call:
        >>> L = expandJSONFile(lines)
        >>> L = expandJSONFile(txt)
    :Inputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines from a file
        *txt*: :class:`str`
            Single string of text including newline characters
    :Outputs:
        *L*: (:class:`str`)
            Full text with references to JSON file(s) expanded
    :Versions:
        * 2015-10-14 ``@ddalle``: First version
    """
    # Strip comments
    lines = stripComments(lines, '#')
    lines = stripComments(lines, '//')
    # Split lines
    lines = lines.split('\n')
    # Start with the first line.
    i = 0
    # Loop through lines.
    while i < len(lines):
        # Get the line.
        line = lines[i]
        # Check for JSON file
        if 'JSONFile' not in line:
            # Go to next line.
            i += 1
            continue
        # Get the tag.
        txt = re.findall('JSONFile\([-"/\\w.= ]+\)', line)
        # Double-check match
        if len(txt) == 0:
            # Go to next line.
            i += 1
            continue
        # Split the line on the JSONFile() command
        j = line.index('JSONFile(')
        # Extract the file name.
        fjson = re.findall('"[-/\\w.= ]+"', txt[0])[0].strip('"')
        # Read that file.
        lines_json = open(fjson, 'r').readlines()
        # Strip that file of comments.
        lines_json = stripComments(lines_json, '#')
        lines_json = stripComments(lines_json, '//')
        # Return to list format
        lines_json = lines_json.split('\n')
        # Split the line on this command.
        line_0 = [line[:j] + lines_json[0]]
        line_1 = [lines_json[-1] + line[j+len(txt[0]):]]
        # Insert the new line set.
        lines = lines[:i] + line_0 + lines_json[1:-1] + line_1 + lines[i+1:]
    # Return the lines as one string.
    return '\n'.join(lines)
    
# Function to read JSON file with all the works
def loadJSONFile(fname):
    """Read a JSON file with helpful error handling and comment stripping
    
    :Call:
        >>> d = loadJSONFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON file to read
    :Outputs:
        *d*: :class:`dict`
            JSON contents in Python form
    :Versions:
        * 2015-12-15 ``@ddalle``: First version
    """
     # Read the input file.
    lines = open(fname).readlines()
    # Expand references to other JSON files and strip comments
    lines = expandJSONFile(lines)
    # Process into dictionary
    try:
        # Process into dictionary
        d = json.loads(lines)
    except Exception as e:
        # Get the line number
        try:
            # Read from the error message
            txt = re.findall('line [0-9]+', e.args[0])[0]
            # Line number
            n = int(txt.split()[1])
            # Check type
            if type(lines).__name__ in ['str', 'unicode']:
                # Not useful; split into list of lines
                lines = lines.split('\n')
            # Start and end line number
            n0 = max(n-3, 0)
            n1 = min(n+2, len(lines))
            # Initialize message with 
            msg = "Error while reading JSON file '%s':\n" % fname
            # Add the exception's message
            msg += "\n".join(list(e.args))
            msg += "\n\nLines surrounding problem area (comments stripped):\n"
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
    """Read default settings configuration file
    
    :Call:
        >>> defs = getDefaults(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file with settings to read
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2014-06-03 ``@ddalle``: First version
        * 2014-07-28 ``@ddalle``: Moved to new options module
    """
    # Read the fixed default file.
    lines = open(fname).readlines()
    # Strip comments and join list into a single string.
    lines = stripComments(lines, '#')
    lines = stripComments(lines, '//')
    # Process the default input file.
    return json.loads(lines)
    
# Function to get the default CAPE settings
def getCapeDefaults():
    """Read default CAPE settings configuration file
    
    :Call:
        >>> defs = getCapeDefaults()
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    :Versions:
        * 2015-09-20 ``@ddalle``: First version
    """
    # File name
    fname = os.path.join(BaseFolder, 'settings', 'cape.default.json')
    # Read the settings.
    return getDefaults(fname)
    
# Function to get template
def getTemplateFile(fname):
    """Get the absolute path to a template file by name
    
    :Call:
        >>> fabs = getTemplateFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file, such as :file:`input.cntl`
        *fabs*: :class:`str`
            Full path to file
    :Versions:
        * 2015-10-26 ``@ddalle``: First version
    """
    # Join with BaseFolder and 'templates'
    return os.path.join(BaseFolder, 'templates', fname)
    
    
# Get the keys of the default dict.
def applyDefaults(opts, defs):
    """Recursively apply defaults for any missing options
    
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
        * 2014-06-17 ``@ddalle``: First version
        * 2014-07-28 ``@ddalle``: Cleaned and moved to options module
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
    """Test if a variable is "list-like."
    
    :Call:
        >>> q = isArray(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            ``True`` if and only if *x* is a list or NumPy array
    :Versions:
        * 2014-12-17 ``@ddalle``: First version
    """
    return (type(x).__name__ in ['list', 'ndarray'])
    
# Test if a variable is "string-like"
def isStr(x):
    """Test if a variable is "string-like"
    
    :Call:
        >>> q = isArray(x)
    :Inputs:
        *x*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            ``True`` if and only if *x* is a string or unicode
    :Versions:
        * 2014-12-17 ``@ddalle``: First version
    """
    # Get the type.
    typ = type(x).__name__
    # Test it.
    return typ.startswith('str') or (typ in ['unicode'])
    
# Dictionary derivative specific to options
class odict(dict):
    """Dictionary-based options module
    
    :Call:
        >>> opts = odict(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options
    :Outputs:
        *opts*: :class:`cape.options.util.odict`
            Dictionary-based options interface
    :Versions:
        * 2014-08-02 ``@ddalle``: First version
        * 2015-11-10 ``@ddalle``: More robust :func:`get_key` using *rck*
    """
    
    # General "get" function
    def get_key(self, k, i=None, rck=None):
        """Intelligently get option for index *i* of key *k*
        
        This is a two-step process.  The first is to get the dictionary value
        or the default if *k* is not in *opts*.  The default is ``rc[k]``.  Let
        *V* be the result of the process.
        
        The second step is to apply indexing.  If *V* is a scalar or *i* is
        ``None``, then *V* is the output.  Otherwise, the function will attempt
        to return ``V[i]``, but if *i* is too large, ``V[-1]`` is the output.
        
        :Call:
            >>> v = opts.get_key(k, i, rck=None)
        :Inputs:
            *k*: :class:`str`
                Name of key to get
            *i*: :class:`int` or ``None``
                Index to apply
            *rck*: :class:`str` or ``None``
                Name of key in *rc0* default option dictionary; defaults to *k*
        :Outputs:
            *v*: any
                Let ``V=opts.get(k,rc[k])``.  Then *v* is either ``V[i]`` if
                possible, ``V[-1]`` if *V* is a list and *i* is not ``None``,
                or ``V`` otherwise
        :See also:
            * :func:`cape.options.util.getel`
        :Versions:
            * 2014-08-02 ``@ddalle``: First version
            * 2015-11-10 ``@ddalle``: Added *rck*
        """
        # Default key name
        if rck is None: rck = k
        # Get the value after applying defaults.
        v = self.get(k, rc.get(rck))
        # Apply intelligent indexing.
        return getel(v, i)
        
    # General "set" function
    def set_key(self, k, v=None, i=None, rck=None):
        """Set option for key *k*
        
        This sets the value for ``opts[k]`` or ``opts[k][i]`` if appropriate.
        If *i* is greater than the length of ``opts[k]``, then ``opts[k]`` is
        appended with its current last value enough times to make
        ``opts[k][i]`` exist.
        
        :Call:
            >>> opts.set_key(k, v=None, i=None, rck=None)
        :Inputs:
            *k*: :class:`str`
                Name of key to set
            *i*: :class:`int` or ``None``
                Index to apply
            *v*: any
                Value to set
            *rck*: :class:`str` or ``None``
                Name of key in *rc0* default option dictionary; defaults to *k*
        :See also:
            * :func:`cape.options.util.setel`
        :Versions:
            * 2014-08-02 ``@ddalle``: First version
            * 2015-11-10 ``@ddalle``: Added *rck*
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
# class odict

