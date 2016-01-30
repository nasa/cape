"""
Interface to Old-Style Fortran Namelists with Repeated Lists
============================================================

This is a module built off of the :mod:`cape.fileCntl` module customized for
manipulating :file:`input.cntl` files.  Such files are split into section by lines of
the format

    ``$__Post_Processing``
    
and this module is designed to recognize such sections.  The main feature of
this module is methods to set specific properties of the :file:`input.cntl` 
file, for example the Mach number or CFL number.
"""

# Import the base file control class.
from cape.fileCntl import FileCntl, re
# Use vector testing
from numpy import array

# Subclass off of the file control class
class Namelist2(FileCntl):
    
    
    # Initialization method
    def __init__(self, fname="overflow.inp"):
        """Initialization method
        
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Read the file.
        self.Read(fname)
        # Save the file name
        self.fname = fname
        # Get the lists of indices of each namelist
        self.UpdateNamelist()
        
        
    # Function to update the namelists
    def UpdateNamelist(self):
        """Update the line indices for each namelist
        
        :Call:
            >>> nml.UpdateNamelist()
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Find the lines that start the lists
        I = self.GetIndexSearch('\s+[&$]')
        # Ignore $END and &END
        J = [i for i in I if not self.lines[i].strip().endswith('END')]
        # Save the start indices
        self.ibeg = J
        # Save the end indices
        self.iend = self.GetIndexSearch('\s+[&$]END')
        # Save the names
        self.Names = [self.lines[i].strip().split()[0][1:] for i in J]
        
    
    # Turn a namelist into a dict
    def ReadListIndex(self, inml):
        """Read namelist *inml* and return a dictionary
        
        The output is a :class:`dict` such as the following
        
            ``{'FSMACH': '0.8', 'ALPHA': '2.0'}``
        
        :Call:
            >>> d = nml.ReadListIndex(inml)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *inml*: :class:`int`
                List index to read
        :Outputs:
            *d*: :class:`dict` (:class:`str`)
                Raw (uncoverted) values of the dict
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Initialize the dictionary
        d = {}
        # Get index of starting line
        ibeg = self.ibeg[inml]
        # Get index of end line
        if inml == len(self.ibeg):
            # Use the last line
            iend = len(self.lines)+1
        else:
            # Use the line before the start of the next line
            iend = self.ibeg[inml+1]
        # Get the lines
        lines = [line.strip() for line in self.lines[ibeg:iend]]
        # Process the first line to catch keys in the opening line
        vals = lines[0].split()
        # Check for multiple entries
        if len(vals) > 1:
            # Read the additional values
            line = ' '.join(vals[1:])
            # Process it
            di = self.ReadKeysFromLine(line)
            # Append the keys.
            for k in di: d[k] = di[k]
        # Loop through the lines.
        for line in lines[1:]:
            # Process the line
            di = self.ReadKeysFromLine(line)
            # Check for end
            if di == -1: break
            # Append the keys.
            for k in di: d[k] = di[k]
        # Output
        return d
        
    # Search for a specific key in a numbered section
    def GetKeyFromListIndex(self, inml, key):
        """Get the value of a key from a specific section
        
        :Call:
            >>> v = nml.GetKeyFromListIndex(inml, key)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *key*: :class:`str`
                Name of the key to search for
        :Outputs:
            *v*: :class:`str` | :class:`int` | :class:`float` | :class:`list`
                Evaluated value of the text for this key
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # -----------------
        # Get section index
        # -----------------
        # Get index of starting line
        ibeg = self.ibeg[inml]
        # Get index of end line
        if inml == len(self.ibeg):
            # Use the last line
            iend = len(self.lines)+1
        else:
            # Use the line before the start of the next line
            iend = self.ibeg[inml+1]
        # Initialize the boolean indicator of a match
        q = False
        # Loop through the lines
        for line in self.lines[ibeg:iend]:
            # Try to read the key from the line
            q, v = self.GetKeyFromLine(line, key)
            # Break if we found it.
            if q: break
        # Output
        return v
        
        
    # Try to read a key from a line
    def GetKeyFromLine(self, line, key):
        """Read the value of a key from a line
        
        :Call:
            >>> q, val = nml.GetKeyFromLine(line, key)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *line*: :class:`str`
                A line of text that may or may not contain the value of *key*
            *key*: :class:`str`
                Name of key
        :Outputs:
            *q*: :class:`bool`
                Whether or not the key was found in the line
            *val*: :class:`str` | :class:`float` | :class:`int` | :class:`bool`
                Value of the key, if found
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Check for the line
        if key not in line:
            # Key not read in this text
            return False, None
        # Otherwise, read the line as a dictionary.
        d = self.ReadKeysFromLine(line)
        # Now check for the key again.
        if key not in d:
            # The key name is hiding in a comment or string literal
            return False, None
        # Get the value.
        v = d[key]
        # Check for a list
        if v is None:
            # Don't convert a null value
            val = v
        elif v and v[0] not in ["'", '"'] and ',' in v:
            # Split into list
            V = v.split(',')
            # Convert each value
            val = [self.ConvertToVal(vi) for vi in V]
        else:
            # Convert the single value
            val = self.ConvertToVal(v)
        # Return the value (converted)
        return True, val
        
    # Set a key
    def SetKeyInListIndex(self, inml, key, val):
        
        return
            
    # Set a key
    def SetKeyInLine(self, line, key, val):
        """Set the value of a key in a line if the key is already in the line
        
        :Call:
            >>> q, txt = nml.SetKeyFromLine(line, key, val)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *line*: :class:`str`
                A line of text that may or may not contain the value of *key*
            *key*: :class:`str`
                Name of key
            *val*: :class:`str` | :class:`float` | :class:`int` | :class:`bool`
                Value of the key, if found
        :Outputs:
            *q*: :class:`bool`
                Whether or not the key was found in the line
            *txt*: :class:`str`
                New version of the line with *key* reset to *val*
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        return False, line
    
            
    # Pop line
    def PopLine(self, line):
        """Read the left-most key from a namelist line of text
        
        :Versions:
            * 201-01-29 ``@ddalle``: First version
        """
        # Strip line
        txt = line.strip()
        # Check for comment
        if txt.startswith('!'):
            return '', None, None
        # Check for start of namelist
        if txt and txt[0] in ["$", "&"]:
            # Get the stuff.
            V = line.split()
            # Check for remaining stuff to process.
            if len(V) > 1:
                # Process the rest of the line
                txt = ' '.join(V[1:])
            else:
                # Nothing else in the line
                txt = ''
        # Check for empty key
        if txt == "":
            return txt, None, None
        # Split on the equals signs
        vals = txt.split("=")
        # Remaining text
        txt = '='.join(vals[1:])
        # Get the name of the key.
        key = vals[0].strip()
        # Deal with quotes or no quotes
        if len(vals) == 1:
            # No value, last key in the line
            txt = ''
            val = None
        elif len(vals) == 2:
            # Last value in the line
            txt = ''
            val = vals[1].rstrip(',').strip()
            # Check for trivial value
            if val == "": val = None
        elif txt.startswith('"'):
            # Check for a second quote
            if '"' not in txt[1:]:
                # Unterminated string
                raise ValueError(
                    "Namelist line '%s' could not be interpreted" 
                    % line)
            # Split of at this point
            val = txt[:iq+1]
            # Remaining text (?)
            if len(txt) > iq+1:
                txt = txt[iq+1:]
            else:
                txt = ''
        elif txt.startswith("'"):
            # Check for a second quote
            if "'" not in txt[1:]:
                # Unterminated string
                raise ValueError(
                    "Namelist line '%s' could not be interpreted" 
                    % line)
            # Split of at this point
            val = txt[:iq+1]
            # Remaining text (?)
            if len(txt) > iq+1:
                txt = txt[iq+1:]
            else:
                txt = ''
        else:
            # Read until just before the next '='
            subvals = vals[1].split(',')
            # Rejoin but without name of next key
            val = ','.join(subvals[:-1])
            # Remaining text
            txt = subvals[-1] + '=' + '='.join(vals[2:])
        # Ouptut
        return txt, key, val
        
            
        
    # Conversion
    def ConvertToVal(self, val):
        """Convert a text file value to Python based on a series of rules
        
        :Call:
            >>> v = nml.ConvertToVal(val)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *val*: :class:`str` | :class:`unicode`
                Text of the value from file
        :Outputs:
            *v*: :class:`str` | :class:`int` | :class:`float` | :class:`list`
                Evaluated value of the text
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2016-01-29 ``@ddalle``: Added boolean shortcuts, ``.T.``
        """
        # Check inputs.
        if type(val).__name__ not in ['str', 'unicode']:
            # Not a string; return as is.
            return val
        # Split to parts
        V = val.split()
        # Check the value.
        try:
            # Check the value.
            if ('"' in val) or ("'" in val):
                # It's a string.  Remove the quotes.
                return eval(val)
            elif val.lower() in [".false.", ".f."]:
                # Boolean
                return False
            elif val.lower() in [".true.", ".t."]:
                # Boolean
                return True
            elif len(V) == 0:
                # Nothing here.
                return None
            elif len(V) == 1:
                # Convert to float/integer
                return eval(val)
            else:
                # List
                return [eval(v) for v in V]
        except Exception:
            # Give it back, whatever it was.
            return val
            
    # Conversion to text
    def ConvertToText(self, v):
        """Convert a value to text to write in the namelist file
        
        :Call:
            >>> val = nml.ConvertToText(v)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *v*: :class:`str` | :class:`int` | :class:`float` | :class:`list`
                Evaluated value of the text
        :Outputs:
            *val*: :class:`str` | :class:`unicode`
                Text of the value from file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Get the type
        t = type(v).__name__
        # Form the output line.
        if t in ['str', 'unicode']:
            # Force quotes
            return '"%s"' % v
        elif t in ['bool'] and v:
            # Boolean
            return ".T."
        elif t in ['bool']:
            # Boolean
            return ".F."
        elif type(v).__name__ in ['list', 'ndarray']:
            # List (convert to string first)
            V = [str(vi) for vi in v]
            return " ".join(V)
        else:
            # Use the built-in string converter
            return str(v)
        
    # Function to process a single line
    def ReadKeysFromLine(self, line):
        """Read zero or more keys from a single text line
        
        :Call:
            >>> d = nml.ReadKeysFromLine(line)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *line*: :class:`str`
                One line from a namelist file
        :Outputs:
            *d*: :class:`dict` (:class:`str`)
                Unconverted values of each key
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Initialize dictionary
        d = {}
        # Initialize remaining text
        txt = line.strip()
        # Loop until line is over
        while txt != '':
            # Read the keys
            txt, key, val = self.PopLine(txt)
            # Check for relevant key
            if key is not None: d[key] = val
        # Output
        return d
        
# class Namelist2

