"""
:mod:`cape.namelist`: Fortran namelists module 
================================================

This is a module built off of the :mod:`cape.fileCntl.FileCntl` module
customized for manipulating Fortran namelists.  Such files are split into
sections which are called "name lists."  Each name list has syntax similar to
the following.

    .. code-block:: none
    
        &project
            project_rootname = "pyfun"
            case_title = "Test case"
        /
    
and this module is designed to recognize such sections.  The main feature of
this module is methods to set specific properties of a namelist file, for
example the Mach number or CFL number.

This function provides a class :class:`cape.namelist.Namelist` that can both
read and set values in the namelist.  The key functions are

    * :func:`Namelist.SetVar`
    * :func:`Namelist.GetVar`
    
The conversion from namelist text to Python is handled by
:func:`Namelist.ConvertToText`, and the reverse is handled by
:func:`Namelist.ConvertToVal`.  Conversions cannot quite be performed just by
the Python functions :func:`print` and :func:`eval` because delimiters are not
used in the same fashion.  Some of the conversions are tabulated below.

    +----------------------+------------------------+
    | Namelist             | Python                 |
    +======================+========================+
    | ``val = "text"``     | ``val = "text"``       |
    +----------------------+------------------------+
    | ``val = 'text'``     | ``val = 'text'``       |
    +----------------------+------------------------+
    | ``val = 3``          | ``val = 3``            |
    +----------------------+------------------------+
    | ``val = 3.1``        | ``val = 3.1``          |
    +----------------------+------------------------+
    | ``val = .false.``    | ``val = False``        |
    +----------------------+------------------------+
    | ``val = .true.``     | ``val = True``         |
    +----------------------+------------------------+
    | ``val = .f.``        | ``val = False``        |
    +----------------------+------------------------+
    | ``val = .t.``        | ``val = True``         |
    +----------------------+------------------------+
    | ``val = 10.0 20.0``  | ``val = [10.0, 20.0]`` |
    +----------------------+------------------------+
    | ``val = 1, 100``     | ``val = [1, 100]``     |
    +----------------------+------------------------+
    | ``val(1) = 1.2``     | ``val = [1.2, 1.5]``   |
    +----------------------+------------------------+
    | ``val(2) = 1.5``     |                        |
    +----------------------+------------------------+
    | ``val = _mach_``     | ``val = "_mach_"``     |
    +----------------------+------------------------+

In most cases, the :class:`Namelist` will try to interpret invalid values for
any namelist entry as a string with missing quotes.  The reason for this is
that users often create template namelist with entries like ``_mach_`` that can
be safely replaced with appropriate values using ``sed`` commands or something
similar.

There is also a function :func:`Namelist.ReturnDict` to access the entire
namelist as a :class:`dict`.  Similarly, :func:`Namelist.ApplyDict` can be used
to apply multiple settings using a :class:`dict` as input.

See also:

    * :mod:`cape.namelist2`
    * :mod:`pyFun.namelist`
    * :mod:`pyOver.overNamelist`
    * :func:`pyFun.case.GetNamelist`
    * :func:`pyFun.fun3d.Fun3d.ReadNamelist`
    * :func:`pyOver.overflow.Overflow.ReadNamelist`

"""

# Import the base file control class.
from .fileCntl import FileCntl

# Base this class off of the main file control class.
class Namelist(FileCntl):
    """
    File control class for Fortran namelists
    
    This class is derived from the :class:`cape.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> nml = cape.Namelist()
        >>> nml = cape.Namelist(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'fun3d.nml'``
    :Outputs:
        *nml*: :class:`cape.namelist.Namelist`
            Namelist file control instance
        *nml.Sections*: :class:`dict` (:class:`list` (:class:`str`))
            Dictionary of sections containing contents of each namelist
        *nml.SectionNames*: :class:`list` (:class:`str`)
            List of section names
    :Version:
        * 2015-10-15 ``@ddalle``: Started
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="fun3d.nml"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\&([\w_]+)")
        
    # Copy the file
    def Copy(self, fname):
        """Copy a file interface
        
        :Call:
            >>> nml2 = nml.Copy()
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
        :Outputs:
            *nml2*: :class:`cape.namelist.Namelist`
                Duplicate file control instance for :file:`fun3d.nml`
        :Versions:
            * 2015-06-12 ``@ddalle``: First version
        """
        # Create empty instance.
        nml = Namelist(fname=None)
        # Copy the file name.
        nml.fname = self.fname
        nml.lines = self.lines
        # Copy the sections
        nml.Section = self.Section
        nml.SectionNames = self.SectionNames
        # Update flags.
        nml._updated_sections = self._updated_sections
        nml._updated_lines = self._updated_lines
        # Output
        return nml
        
    # Function to set generic values, since they have the same format.
    def SetVar(self, sec, name, val, k=None):
        """Set generic :file:`fun3d.nml` variable value
        
        :Call:
            >>> nml.SetVar(sec, name, val)
            >>> nml.SetVar(sec, name, val, k)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
            *sec*: :class:`str`
                Name of section in which to set variable
            *name*: :class:`str`
                Name of variable as identified in 'aero.csh'
            *val*: any
                Value to which variable is set in final script
            *k*: :class:`int`
                Namelist index
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
            * 2015-10-20 ``@ddalle``: Added Fortran index
        """
        # Check sections
        if sec not in self.SectionNames:
            # Add the section
            self.AddSection(sec)
        # Check format
        if k is None:
            # Format: '   component = "something"'
            # Line regular expression: "XXXX=" but with white spaces
            reg = '^\s*%s\s*[=\n]' % name
            # Form the output line.
            line = '    %s = %s\n' % (name, self.ConvertToText(val))
        else:
            # Format: '   component(1) = "something"'
            # Format: '   component(1,3) = "something"'
            # Format: '   component(:,1) = "something"'
            # Index type
            tk = type(k).__name__
            # Convert index to string
            if tk in ['tuple', 'list', 'ndarray']:
                # Convert to comma-separated list
                lk = [':' if ki is None else str(ki) for ki in k]
                # Join list of indices via comma
                sk = ','.join(lk)
            else:
                # Convert to string as appropriate
                sk = str(k)
            # Line regular expression: "XXXX([0-9]+)=" but with white spaces
            reg = '^\s*%s\(%s\)\s*[=\n]' % (name, sk)
            # Form the output line.
            line = '    %s(%s) = %s\n' % (name, sk, self.ConvertToText(val))
        # Replace the line; prepend it if missing
        self.ReplaceOrAddLineToSectionSearch(sec, reg, line, -1)
        
    # Function to get the value of a variable
    def GetVar(self, sec, name, k=None):
        """Get value of a variable
        
        :Call:
            >>> val = nml.GetVar(sec, name)
            >>> val = nml.GetVar(sec, name, k)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
            *sec*: :class:`str`
                Name of section in which to set variable
            *name*: :class:`str`
                Name of variable as identified in 'aero.csh'
            *k*: :class:`int`
                Namelist index
        :Outputs:
            *val*: any
                Value to which variable is set in final script
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
            * 2015-10-20 ``@ddalle``: Added Fortran index
        """
        # Check sections
        if sec not in self.SectionNames:
            return None
        # Check for index
        if k is None:
            # Line regular expression: "XXXX=" but with white spaces
            reg = '^\s*%s\s*[=\n]' % name
        else:
            # Index type
            tk = type(k).__name__
            # Convert index to string
            if tk in ['tuple', 'list', 'ndarray']:
                # Convert to comma-separated list
                lk = [':' if ki is None else str(ki) for ki in k]
                # Join list of indices via comma
                sk = ','.join(lk)
            else:
                # Convert to string as appropriate
                sk = str(k)
            # Index: "XXXX(k)=" but with white spaces
            reg = '^\s*%s\(%s\)\s*[=\n]' % (name, sk)
        # Find the line.
        lines = self.GetLineInSectionSearch(sec, reg, 1)
        # Exit if no match
        if len(lines) == 0: return None
        # Split on the equal sign
        vals = lines[0].split('=')
        # Check for a match
        if len(vals) < 1: return None
        # Convert to Python value
        return self.ConvertToVal(vals[1])
    
    
    # Return a dictionary
    def ReturnDict(self):
        """Return a dictionary of options that mirrors the namelist
        
        :Call:
            >>> opts = nml.ReturnDict()
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
        :Outputs:
            *opts*: :class:`dict`
                Dictionary of namelist options
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Initialize dictionary
        opts = {}
        # Loop through sections
        for sec in self.SectionNames[1:]:
            # Initialize the section dictionary
            o = {}
            # Loop through the lines
            for line in self.Section[sec]:
                # Split the line to values
                vals = line.split('=')
                # Check for a parameter.
                if len(vals) < 2: continue
                # Get the name.
                key = vals[0].strip()
                val = vals[1].strip()
                # Set the value.
                o[key] = self.ConvertToVal(val)
            # Set the section dictionary
            opts[sec] = o
        # Output
        return opts
        
    # Apply a whole bunch of options
    def ApplyDict(self, opts):
        """Apply a whole dictionary of settings to the namelist
        
        :Call:
            >>> nml.ApplyDict(opts)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
            *opts*: :class:`dict`
                Dictionary of namelist options
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Loop through major keys.
        for sec in opts.keys():
            # Loop through the keys in this subnamelist/section
            for k in opts[sec].keys():
                # Set the value.
                self.SetVar(sec, k, opts[sec][k])
                
    # Add a section
    def AddSection(self, sec):
        """Add a section to the namelist interface
        
        :Call:
            >>> nml.AddSection(sec)
        :Inputs:
            *sec*: :class:`str`
                Name of section
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Escape if already present
        if sec in self.SectionNames: return
        # Append the section
        self.SectionNames.append(sec)
        # Add the lines
        self.Section[sec] = [
            ' &%s\n' % sec,
            ' /\n',
            '\n'
        ]
    
    # Conversion
    def ConvertToVal(self, val):
        """Convert a text file value to Python based on a series of rules
        
        :Call:
            >>> v = nml.ConvertToVal(val)
        :Inputs:
            *nml*: :class:`cape.namelist.Namelist`
                Namelist file control instance
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
                Namelist file control instance
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
            return ".true."
        elif t in ['bool']:
            # Boolean
            return ".false."
        elif type(v).__name__ in ['list', 'ndarray']:
            # List (convert to string first)
            V = [str(vi) for vi in v]
            return " ".join(V)
        else:
            # Use the built-in string converter
            return str(v)
        
        
# class Namelist

        
