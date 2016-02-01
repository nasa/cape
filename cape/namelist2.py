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
import numpy as np

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
        self.Groups = [self.lines[i].strip().split()[0][1:] for i in J]
        
    # Add a group
    def InsertGroup(self, igrp, grp):
        """Insert a group as group number *igrp*
        
        :Call:
            >>> nml.InsertGroup(igrp, grp)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *igrp*: :class:`int`
                Index of location at which to insert group
            *grp*: :class:`str`
                Name of the group to insert
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the index of the group
        ibeg = self.ibeg[igrp]
        # Query the current starting character
        gchar = self.lines[ibeg].lstrip()[0]
        # Insert the lines in reverse order
        self.lines.insert(ibeg, "    %sEND\n" % gchar)
        self.lines.insert(ibeg, " %s%s\n" % (gchar, grp))
        # Update the namelist info
        self.UpdateNamelist()
        
    # Find a list by name (and index if repeated)
    def GetGroupByName(self, grp, igrp=0):
        """Get index of group with a specific name
        
        :Call:
            >>> i = nml.GetGroupByName(grp, igrp=0)
        :Inputs:
            *nml*: :class:`cape.name.ist2.Namelist2`
                Old-style namelist interface
            *grp*: :class:`str`
                Name of namelist group
            *igrp*: :class:`int`
                If namelist contains multiple copies, return match number *igrp*
        :Outputs:
            *i*: :class:`int` | :class:`np.ndarray` (:class:`int`)
                Group index of requested match
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Search based on lower-case names
        grps = np.array([gi.lower() for gi in self.Groups])
        # Find the all indices that match
        I = np.where(grps == grp.lower())[0]
        # Process output
        if igrp is None:
            # Return all matches
            return I
        elif len(I) == 0:
            # No match
            return KeyError("Namelist '%s' has no list '%s'" % 
                (self.fname, grp))
        elif len(I) < igrp:
            # Not enough matches
            return ValueError("Namelist '%s' has fewer than %i lists named '%s'"
                % (self.fname, igrp, grp))
        else:
            # Return the requested match
            return I[igrp]
    
    # Turn a namelist into a dict
    def ReadGroupIndex(self, igrp):
        """Read group *igrp* and return a dictionary
        
        The output is a :class:`dict` such as the following
        
            ``{'FSMACH': '0.8', 'ALPHA': '2.0'}``
        
        :Call:
            >>> d = nml.ReadGroupIndex(igrp)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *igrp*: :class:`int`
                Group index to read
        :Outputs:
            *d*: :class:`dict` (:class:`str`)
                Raw (uncoverted) values of the dict
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Initialize the dictionary
        d = {}
        # Get index of starting line
        ibeg = self.ibeg[igrp]
        # Get index of end line
        if igrp == len(self.ibeg):
            # Use the last line
            iend = len(self.lines)+1
        else:
            # Use the line before the start of the next line
            iend = self.ibeg[igrp+1]
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
    def GetKeyFromGroupIndex(self, igrp, key):
        """Get the value of a key from a specific section
        
        :Call:
            >>> v = nml.GetKeyFromGroupIndex(igrp, key)
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
        # Get index of starting line
        ibeg = self.ibeg[igrp]
        # Get index of end line
        if igrp == len(self.ibeg):
            # Use the last line
            iend = len(self.lines)+1
        else:
            # Use the line before the start of the next line
            iend = self.ibeg[igrp+1]
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
        
    # Search for a specific key by name
    def GetKeyFromGroupName(self, grp, key, igrp=0):
        """Get the value of a key from a section by group name
        
        :Call:
            >>> v = nml.GetKeyFromGroupName(grp, key, igrp=0)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style Fortran namelist interface
            *grp*: :class:`str`
                Group name to search for
            *key*: :class:`str`
                Name of key to search for
            *igrp*: :class:`int`
                If multiple sections have same name, use match number *igrp*
        :Outputs:
            *v*: :class:`any`
                Converted value
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Find matches
        i = self.GetGroupByName(grp, igrp)
        # Get the key from that list
        return self.GetKeyFromGroupIndex(i, key)
        
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
            * 2016-01-30 ``@ddalle``: Case-insensitive
        """
        # Check for the line
        if key.lower() not in line.lower():
            # Key not read in this text
            return False, None
        # Initialize text remaining.
        tend = line
        # Read the keys from this line one-by-one
        q = False
        while tend != "":
            # Read the first key in the remaining text.
            tend, ki, vi = self.PopLine(tend)
            # Check for a match.
            if ki.lower() == key.lower():
                # Use the value from this key.
                return True, self.ConvertToVal(vi)
        # If this point is reached, the key name is hiding in a comment or str
        return False, None
        
    # Set a key
    def SetKeyInGroupName(self, grp, key, val, igrp=0):
        """Set the value of a key from a group by name
        
        :Call:
            >>> nml.SetKeyInGroupName(grp, key, val, igrp=0)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style Fortran namelist interface
            *grp*: :class:`str`
                Group name to search for
            *key*: :class:`str`
                Name of key to search for
            *val*: :class:`any`
                Converted value
            *igrp*: :class:`int`
                If multiple sections have same name, use match number *igrp*
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Find matches
        i = self.GetGroupByName(grp, igrp)
        # Get the key from that list
        return self.SetKeyInGroupIndex(i, key, val)
        
    # Set a key
    def SetKeyInGroupIndex(self, igrp, key, val):
        """Set the value of a key in a group by index
        
        If the key is not set in the present text, add it as a new line.  The
        contents of the file control's text (in *nml.lines*) will be edited, and
        the list indices will be updated if a line is added.
        
        :Call:
            >>> nml.SetKeyInGroupIndex(igrp, key, val)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                File control instance for old-style Fortran namelist
            *igrp*: :class:`int`
                Index of namelist to edit
            *key*: :class:`str`
                Name of key to alter or set
            *val*: :class:`any`
                Value to use for *key*
        :Versions:
            * 2015-01-30 ``@ddalle``: First version
        """
        # Get index of starting and end lines
        ibeg = self.ibeg[igrp]
        iend = self.iend[igrp]
        # Initialize the boolean indicator of a match in existing text
        q = False
        # Loop through the lines
        for i in range(ibeg, iend):
            # Get the line.
            line = self.lines[i]
            # Try to set the key in this line
            q, line = self.SetKeyInLine(line, key, val)
            # Check for match.
            if q:
                # Set this line in the FC's text and exit
                self.lines[i] = line
                return
        # If no match found in existing text, add a line.
        line = '    %s = %s,\n' % (key, self.ConvertToText(val))
        # Insert the line.
        self.lines = self.lines[:iend] + [line] + self.lines[iend:]
        # Update the namelist indices.
        self.UpdateNamelist()
    
    # Set a key
    def SetKeyInLine(self, line, key, val):
        """Set the value of a key in a line if the key is already in the line
        
        :Call:
            >>> q, line = nml.SetKeyInLine(line, key, val)
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
            *line*: :class:`str`
                New version of the line with *key* reset to *val*
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Check if the key is present in the line of the text.
        if key not in line:
            return False, line
        # Initialize prior and remaining text
        tbeg = ""
        tend = line
        # Loop through keys in this line
        while True:
            # Read the first key in the remaining line.
            txt, ki, vi = self.PopLine(tend)
            # Check if the key matches the target.
            if ki.lower() == key.lower():
                # Match found; exit and remember remaining text
                tbeg += tend[:tend.index(ki)]
                tend = txt
                break
            # Check if the line is empty.
            if txt == "":
                # No match in this line.
                return False, line
            # Otherwise, append to the prefix text and keep looking.
            tbeg += tend[:tend.index(txt)]
            # Update the text remaining
            tend = txt
        # If the value is ``None``, delete the entry.
        if val is None:
            # Use the beginning and remaining text Only
            line = "%s%s\n" % (tbeg.rstrip(), tend)
        else:
            # Convert value to text
            sval = self.ConvertToText(val)
            line = "%s%s = %s,%s\n" % (tbeg, key, sval, tend)
        return True, line
    
            
    # Pop line
    def PopLine(self, line):
        """Read the left-most key from a namelist line and return rest of line
        
        :Call:
            >>> txt, key, val = nml.PopLine(line)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                Old-style namelist interface
            *line*: :class:`str`
                One line of namelist text
        :Outputs:
            *txt*: :class:`str`
                Remaining text in *line* after first key has been read
            *key*: :class:`str`
                Name of first key read from *line*
            *val*: ``None`` | :class:`str`
                Raw (unconverted) value of *key*
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
        V = val.split(',')
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
        """Convert a scalar value to text to write in the namelist file
        
        :Call:
            >>> val = nml.ConvertToText(v)
        :Inputs:
            *nml*: :class:`cape.namelist2.Namelist2`
                File control instance for old-style Fortran namelist
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
            return "'%s'" % v
        elif t in ['bool'] and v:
            # Boolean
            return ".T."
        elif t in ['bool']:
            # Boolean
            return ".F."
        elif type(v).__name__ in ['list', 'ndarray']:
            # List (convert to string first)
            V = [str(vi) for vi in v]
            return ", ".join(V)
        else:
            # Use the built-in string converter
            return str(v)
    
    
# class Namelist2

