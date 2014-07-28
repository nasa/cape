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


