"""
Cart3D and pyCart settings module: :mod:`pyCart.options`
========================================================

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
PyCartFolder = os.path.split(_fname)[0]

# Backup default settings (in case deleted from :file:`pyCart.defaults.json`)
rc = {
    "InputCntl": "input.cntl",
    "AeroCsh": "aero.csh",
    "it_fc": 200,
    "cfl": 1.1,
    "cflmin": 0.8,
    "mg_fc": 3,
    "limiter": 2,
    "y_is_spanwise": True,
    "binaryIO": True,
    "OMP_NUM_THREADS": 8,
    "tm": False,
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
    "reorder": True
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
        elif type(opts[k]) is dict:
            # Recurse for dictionaries.
            opts[k] = applyDefaults(opts[k], defs[k])
    # Output the modified defaults.
    return opts


    

# Class definition
class Options(dict):
    """
    Options structure, subclass of :class:`dict`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
        
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    
    :Versions:
        * 2014.07.28 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method with optional JSON input"""
        # Check for an input file.
        if fname:
            # Read the input file.
            lines = open(fname).readlines()
            # Strip comments and join list into a single string.
            lines = stripComments(lines, '#')
            lines = stripComments(lines, '//')
            # Get the equivalent dictionary.
            d = json.loads(lines)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getPyCartDefaults()
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
            
            
    # Method to get the input file
    
    # ===================
    # flowCart parameters
    # ===================
    
    # Number of iterations
    def get_it_fc(self, i=None):
        """Return the number of iterations for `flowCart`"""
        # Get the flowCart settings safely.
        opts_fc = self.get('flowCart', {})
        # Return the iteration number
        it_fc = opts_fc.get('it_fc', rc["it_fc"])
        # Safe indexing
        return getel(it_fc, i)
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        """Set the number of iterations for `flowCart`"""
        # Ensure flowCart settings
        self.setdefault('flowCart', {})
        # Get current setting safely
        IT_FC = self['flowCart'].get('it_fc', rc['it_fc']) 
        # Set the iteration count
        self['flowCart']['it_fc'] = setel(IT_FC, i, it_fc)
        
    # CFL number
    def get_cfl(self, i=None):
        """Return the CFL number for `flowCart`"""
        # Safe flowCart options.
        opts_fc = self.get('flowCart', {})
        # Get the CFL number and index if appropriate.
        cfl = opts_fc.get('cfl', rc['cfl'])
        # Safe indexing
        return getel(cfl, i)
    # Set nominal CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        """Set the CFL number for `flowCart`"""
        # Ensure flowCart settings
        self.setdefault('flowCart', {})
        # Get current setting safely
        CFL = self['flowCart'].get('cfl', rc['cfl'])
        # Safe indexing
        self['flowCart']['cfl'] = setel(CFL, i, cfl)
        
    # Minimum CFL number
    def get_cflmin(self, i=None):
        """Return the minimum CFL number for `flowCart`"""
        # Safe flowCart options.
        opts_fc = self.get('flowCart', {})
        # Get the CFL number and index if appropriate.
        cfl = opts_fc.get('cflmin', rc['cflmin'])
        # Safe indexing
        return getel(cfl, i)
    # Set minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        """Set the CFL number for `flowCart`"""
        # Ensure flowCart settings
        self.setdefault('flowCart', {})
        # Get current setting safely
        CFL = self['flowCart'].get('cflmin', rc['cflmin'])
        # Safe indexing
        self['flowCart']['cflmin'] = setel(CFL, i, cflmin)
        
    # Number of threads
    def get_OMP_NUM_THREADS(self, i=None):
        """Return number of threads to use"""
        # Safe flowCart settings
        opts_fc = self.get('flowCart', {})
        # Safe setting
        nThreads = opts_fc.get('OMP_NUM_THREADS', rc['OMP_NUM_THREADS'])
        # Safe indexing
        return getel(nThreads, i)
    # Number of threads
    def set_OMP_NUM_THREADS(self, nThreads=rc0('OMP_NUM_THREADS'), i=None):
        """Set the number of threads to use for each solution"""
        # Ensure flowCart settings.
        self.setdefault('flowCart', {})
        # Safe current settings.
        NTH = self['flowCart'].get('OMP_NUM_THREADS', rc['OMP_NUM_THREADS'])
        # Safe indexing
        self['flowCart']['OMP_NUM_THREADS'] = setel(NTH, i, nThreads)
    # Aliases for the above
    get_nThreads = get_OMP_NUM_THREADS
    set_nThreads = set_OMP_NUM_THREADS
    
    
    # ================
    # multigrid levels
    # ================
    
    # Get flowCart iteration levels
    def get_mg_fc(self, i=None):
        """Return the number of multigrid levels for `flowCart`"""
        # Get the settings for flowCart, which should exist.
        opts_fc = self.get('flowCart', {})
        # Return the value, applying the default.
        mg_fc = opts_fc.get('mg_fc', rc['mg_fc'])
        return getel(mg_fc, i)
    
    # Set flowCart iteration levels
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        """Set number of multigrid levels for `flowCart`"""
        # Ensure flowCart settings
        self.setdefault('flowCart', {})
        # Safe setting
        MG_FC = self['flowCart'].get('mg_fc', rc['mg_fc'])
        # Set the multigrid levels
        self['flowCart']['mg_fc'] = mg_fc
    
    # Get adjointCart iteration levels
    def get_mg_ad(self, i=None):
        """Return the number of multigrid levels for `adjointCart`"""
        # Get the settings for flowCart, which should exist.
        opts_ad = self.get('adjointCart', {})
        # Return the value, applying the default.
        mg_ad = opts_ad.get('mg_ad', rc['mg_ad'])
        # Safe indexing
        return getel(mg_ad, i)
        
    # Set the adjointCart iteration levels
    def set_mg_ad(self, mg_ad=3):
        """Set number of multigrid levels for `adjointCart`"""
        # Ensure flowCart settings
        self.setdefault('adjointCart', {})
        # Get current setting safely.
        MG_AD = self['adjointCart']
        # Set the multigrid levels
        self['adjointCart']['mg_ad'] = mg_ad
        
    # Method to get the number of multigrid levels
    def get_mg(self, mg=3):
        """
        Return the number of multigrid levels
        
        :Call:
            >>> mg = opts.get_mg(mg=3)
            
        :Inputs:
            *mg*: :class:`int`
                Default value for `mg_fc`
        
        :Outputs:
            *mg*: :class:`int`
                Maximum of `mg_fc` and `mg_ad`
        """
        # Get the two values.
        mg_fc = self.get_mg_fc(mg)
        mg_ad = self.get_mg_ad(mg_fc)
        # Check for valid settings.
        if mg_fc and mg_ad:
            # Both are defined, use maximum.
            return max(mg_fc, mg_ad)
        elif mg_fc:
            # Only one valid nonzero setting; use it.
            return mg_fc
        elif mg_ad:
            # Only one valid nonzero setting; use it.
            return mg_ad
        else:
            # Both either invalid or zero.  Return 0.
            return 0
    
    # Method to set both multigrid levels
    def set_mg(self, mg=3):
        """Set number of multigrid levels for `flowCart` and `adjointCart`"""
        self.set_mg_fc(mg)
        self.set_mg_ad(mg)
            
    
    # ========================
    # mesh creation parameters
    # ========================
    
    # Tri file
    def get_TriFile(self):
        """Get the input :file:`*.tri` file(s)"""
        # Get the mesh settings safely.
        opts_mesh = self.get('Mesh',{})
        # Get the .tri file safely
        TriFile = opts_mesh.get('TriFile', rc['TriFile'])
        return TriFile
    def set_TriFile(self, TriFile=rc['TriFile']):
        """Set the input :file:`*.tri` file(s)"""
        # Ensure the  'Mesh' key exists.
        self.setdefault('Mesh', {})
        # Apply the setting
        self['Mesh']['TriFile'] = TriFile
    
    # Mesh radius
    def get_r(self, r=8):
        """Get the value for the `autoInputs` mesh radius"""
        # Get to the autoInputs section safely.
        opts_ai = self.get('Mesh', {}).get('autoInputs', {})
        # Get the mesh radius value
        return opts_ai.get('r', r)
    def set_r(self, r=8):
        """Set the value for `autoInputs` mesh radius"""
        # Make sure the 'Mesh' and 'autoInputs' keys exist.
        self.setdefault('Mesh', {})
        self['Mesh'].setdefault('autoInptus', {})
        # Apply the setting.
        self['Mesh']['autoInputs'] = r
        
    # Refinement levels
    def get_maxR(self, maxR=10):
        """Get the value for `cubes` maximum refinement levels"""
        # Get to the `cubes` section safely.
        opts_cubes = self.get('Mesh',{}).get('cubes',{})
        # Get the refinement count.
        return opts_cubes.get('maxR')
    def set_maxR(self, maxR=10):
        """Set the value for `cubes` maximum refinement levels"""
        # Make sure 'Mesh' and 'cubes' keys exist.
        self.setdefault('Mesh', {})
        self['Mesh'].setdefault('cubes', {})
        # Apply the setting.
        self['Mesh']['cubes']['maxR'] = maxR
        
        
    # ==========
    # Adaptation
    # ==========
    
    # Number of adaptations
    def get_n_adapt_cycles(self, n_adapt_cycles=0):
        """Get the number of adaptation cycles"""
        # Get the adaptation settings.
        opts_a = self.get('Adaptation', {})
        # Get the number of adaptations.
        return opts_a.get('n_adapt_cycles', n_adapt_cycles)
    def set_n_adapt_cycles(self, n_adapt_cycles=0):
        """Set the number of adaptation cycles"""
        # Make sure the adaptation settings exist.
        self.setdefault('Adaptation', {})
        # Apply the setting.
        self['Adaptation']['n_adapt_cycles'] = n_adapt_cycles
    def get_nAdapt(self, n_adapt_cycles=0):
        """Alias for get_n_adapt_cycles"""
        return self.get_n_adapt_cycles(n_adapt_cycles)
    def set_nAdapt(self, n_adapt_cycles=0):
        """Alias for set_n_adapt_cycles"""
        self.set_n_adapt_cycles(n_adapt_cycles)
        
    

    


