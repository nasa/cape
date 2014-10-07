"""
Utilities for pyCart Options module: :mod:`pyCart.options.util`
===============================================================

This module provides tools to read, access, modify, and write settings for
:mod:`pyCart`.  The class is based off of the built-int :class:`dict` class, so
its default behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`pyCart.json`
    
    *. Values specified in the default control file,
       :file:`$PYCART/settings/pyCart.default.json`
    
    *. Hard-coded defaults from this module
"""

# Interaction with the OS
import os
# JSON module
import json

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(os.path.split(_fname)[0])[0]

# Backup default settings (in case deleted from :file:`pyCart.defaults.json`)
rc = {
    "InputCntl": "input.cntl",
    "AeroCsh": "aero.csh",
    "GroupMesh": True,
    "ConfigFile": "Config.xml",
    "RefArea": 1.0,
    "RefLength": 1.0,
    "RefPoint": [0.0, 0.0, 0.0],
    "InputSeq": [0],
    "IterSeq": [200],
    "first_order": 0,
    "it_fc": 200,
    "cfl": 1.1,
    "cflmin": 0.8,
    "mg_fc": 3,
    "mpi_fc": False,
    "qsub": True,
    "resub": False,
    "use_aero_csh": False,
    "limiter": 2,
    "y_is_spanwise": True,
    "binaryIO": True,
    "tecO": True,
    "nProc": 8,
    "tm": False,
    "mpicmd": "mpiexec",
    "it_ad": 120,
    "mg_ad": 3,
    "n_adapt_cycles": 0,
    "etol": 1.0e-6,
    "max_nCells": 5e6,
    "ws_it": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100],
    "mesh_growth": [1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5],
    "apc": ["p", "a"],
    "TriFile": "Components.i.tri",
    "mesh2d": False,
    "r": 8,
    "maxR": 11,
    "pre": "preSpec.c3d.cntl",
    "cubes_a": 10,
    "cubes_b": 2,
    "reorder": True,
    "PBS_j": "oe",
    "PBS_r": "n",
    "PBS_S": "/bin/bash",
    "PBS_select": 1,
    "PBS_ncpus": 20,
    "PBS_mpiprocs": 20,
    "PBS_model": "ivy",
    "PBS_W": "",
    "PBS_q": "normal",
    "PBS_walltime": "8:00:00"
}

# Utility function to get elements sanely
def getel(x, i=None):
    """
    Return the *i*th element of an array if possible
    
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
        * 2014.07.29 ``@ddalle``: First version
    """
    # Check the type.
    if i is None:
        return x
    if type(x).__name__ in ['list', 'array']:
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
    Return the *i*th element of an array if possible
    
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
        * 2014.07.29 ``@ddalle``: First version
    """
    # Check the index input.
    if i is None:
        # Scalar output
        return xi
    # Ensure list.
    if type(x).__name__ == 'array':
        # NumPy array
        y = list(x)
    elif type(x).__name__ != 'list':
        # Scalar input
        y = [x]
    else:
        # Already a list
        y = x
    # Make sure *y* is long enough.
    for j in range(len(y), i):
        y.append(y[-1])
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
        * 2014.08.01 ``@ddalle``: First version
    """
    # Use the `getel` function to do this.
    return getel(rc[p], 0)
    

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
        * 2014.06.03 ``@ddalle``: First version
    """
    # Start with the first line.
    i = 0
    # Check for combined lines.
    if type(lines) == str:
        # Split into lines.
        lines = lines.split('\n')
    # Loop until last line
    while i < len(lines):
        # Get the line and strip leading and trailing white space.
        line = lines[i].strip()
        # Check it.
        if line.startswith(char):
            # Remove it.
            lines.__delitem__(i)
        else:
            # Move to the next line.
            i += 1
    # Return the remaining lines.
    return "".join(lines)


# Function to get the defautl settings.
def getPyCartDefaults():
    """
    Read :file:`pyCart.default.json` default settings configuration file
    
    :Call:
        >>> defs = getPyCartDefaults()
        
    :Outputs:
        *defs*: :class:`dict`
            Dictionary of settings read from JSON file
    
    :Versions:
        * 2014.06.03 ``@ddalle``: First version
        * 2014.07.28 ``@ddalle``: Moved to new options module
    """
    # Read the fixed default file.
    lines = open(os.path.join(PyCartFolder, 
        "..", "settings", "pyCart.default.json")).readlines()
    # Strip comments and join list into a single string.
    lines = stripComments(lines, '#')
    lines = stripComments(lines, '//')
    # Process the default input file.
    return json.loads(lines)
    
    
# Get the keys of the default dict.
def applyDefaults(opts, defs):
    """
    Recursively apply defaults for any missing options
    
    :Call:
        >>> opts = applyDefaults(opts, defs)
        
    :Inputs:
        *opts*: :class:`dict`
            Options dictionary with some options possibly missing
        *defs*: :class:`dict`
            Full dictionary of default settings
            
    :Outputs:
        *opts*: :class:`dict`
            Input dictionary with all of the fields of *defs*
    
    :Versions:
        * 2014.06.17 ``@ddalle``: First version
        * 2014.07.28 ``@ddalle``: Cleaned and moved to options module
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
    
    
# Dictionary derivative specific to options
class odict(dict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
    # General "get" function
    def get_key(self, k, i=None):
        """Intelligently get option for index *i* of key *k*
        
        This is a two-step process.  The first is to get the dictionary value
        or the default if *k* is not in *opts*.  The default is ``rc[k]``.  Let
        *V* be the result of the process.
        
        The second step is to apply indexing.  If *V* is a scalar or *i* is
        ``None``, then *V* is the output.  Otherwise, the function will attempt
        to return ``V[i]``, but if *i* is too large, ``V[-1]`` is the output.
        
        :Call:
            >>> v = opts.get_key(k, i)
        :Inputs:
            *k*: :class:`str`
                Name of key to get
            *i*: :class:`int` or ``None``
                Index to apply
        :Outputs:
            *v*: any
                Let ``V=opts.get(k,rc[k])``.  Then *v* is either ``V[i]`` if
                possible, ``V[-1]`` if *V* is a list and *i* is not ``None``,
                or ``V`` otherwise
        :See also:
            * :func:`pyCart.options.util.getel`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        # Get the value after applying defaults.
        v = self.get(k, rc.get(k))
        # Apply intelligent indexing.
        return getel(v, i)
        
    # General "set" function
    def set_key(self, k, v=None, i=None):
        """Set option for key *k*
        
        This sets the value for ``opts[k]`` or ``opts[k][i]`` if appropriate.
        If *i* is greater than the length of ``opts[k]``, then ``opts[k]`` is
        appended with its current last value enough times to make
        ``opts[k][i]`` exist.
        
        :Call:
            >>> opts.set_key(k, v=None, i=None)
        :Inputs:
            *k*: :class:`str`
                Name of key to set
            *i*: :class:`int` or ``None``
                Index to apply
            *v*: any
                Value to set
        :See also:
            * :func:`pyCart.options.util.setel`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        # Check for default value.
        if v is None:
            # Get the default, but ensure a scalar.
            v = rc0(k)
        # Get the current full setting.
        V = self.get(k, rc[k])
        # Assign the input value .
        self[k] = setel(V, i, v)
        