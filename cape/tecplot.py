"""
Module to interface with Tecplot scripts: :mod:`pyCart.tecplot`
===============================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating Tecplot layout files and macros.
"""

# Import the base file control class.
from cape.fileCntl import FileCntl
# Import command to actually run Tecplot
from .bin import tecmcr
# Import Tecplot folder
from .util import TecFolder

# Numerics
import numpy as np
# Regular expressions
import re
# System interface
import os, shutil

# Stand-alone function to run a Tecplot layout file
def ExportLayout(lay="layout.lay", fname="export.png", fmt="PNG", w=None):
    """Stand-alone function to open a layout and export an image
    
    :Call:
        >>> ExportLayout(lay="layout.lay", fname="export.png", fmt="PNG", w=None)
    :Inputs:
        *lay*: {``"layout.lay"``} | :class:`str`
            Name of Tecplot layout file
        *fname*: {``"export.png"``} | :class:`str`
            Name of image file to export
        *fmt*: {``"PNG"``} | ``"JPG"`` | :class:`str`
            Valid image format for Tecplot export
        *w*: {``None``} | :class:`float`
            Image width in pixels
    :Versions:
        * 2015-03-10 ``@ddalle``: First version
    """
    # Macro file name
    fmcr = "export-lay.mcr"
    fsrc = os.path.join(TecFolder, fmcr)
    # Open the macro interface.
    tec = TecMacro(fsrc)
    # Set the layout file name
    tec.SetLayout(lay)
    # Check for options
    if fname is not None: tec.SetExportFileName(fname)
    if fmt   is not None: tec.SetExportFormat(fmt)
    if w     is not None: tec.SetImageWidth(w)
    # Write the customized macro
    tec.Write(fmcr)
    # Run the macro
    tecmcr(mcr=fmcr)
    # Remove the macro
    os.remove(fmcr)

# Base this class off of the main file control class.
class Tecscript(FileCntl):
    """
    File control class for Tecplot script files
    
    :Call:
        >>> tec = cape.tecplot.Tecscript()
        >>> tec = cape.tecplot.Tecscript(fname="layout.lay")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
            Instance of Tecplot script base class
    :Versions:
        * 2015-02-26 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="layout.lay"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Get the command list
        self.UpdateCommands()
        
    # Set parameter on header line
    def SetPar(self, cmd, val, i):
        """Set a parameter value on the header line of a command
        
        :Call:
            >>> tec.SetPar(key, val, i)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecsript` or derivative
                Instance of Tecplot script
            *cmd*: :class:`str`
                Name of command
            *val*: :class:`str`
                String to set on the header line
            *i*: :class:`int`
                Alter the instdance *i* of this command
        :Versions:
            * 2016-10-04 ``@ddalle``: First version
        """
        # Get the indices of this command
        I = self.GetIndexStartsWith('$!'+cmd)
        # Make sure there are at least *i* matches
        if i >= len(I):
            raise ValueError(
                ("Requested to alter instance %s of command '%s'"%(i+1,cmd)) +
                ("but layout only contains %i instances" % len(I)))
        # Get the line number
        j = I[i]
        # Set the line
        self.lines[j] = "$!%s %s\n" % (cmd, val)
        
    # Set variable
    def SetVar(self, key, val):
        """Set a variable to a particular value
        
        :Call:
            >>> tec.SetVar(key, val)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *key*: :class:`str`
                Name of variable
            *val*: any
                Value to set the variable, converted via :func:`str`
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Form the command
        cmd = 'VarSet'
        # Form the text to replace
        reg = '\|%s\|' % key
        # Form the text to insert
        txt = '|%s| = %s' % (key, val)
        # Replace or insert the command
        self.ReplaceCommand('VarSet', txt=txt, reg=reg)
        
    # Set the freestream Mach number
    def SetMach(self, mach):
        """Set the freestream Mach number
        
        :Call:
            >>> tec.SetMach(mach)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *mach*: :class:`float`
                Freestream Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Set the variable
        self.SetVar('Minf', mach)
        
    # Create contour map
    def CreateColorMap(self, name, cmap, k=2):
        """Create a new color map
        
        :Call:
            >>> tec.CreateColorMap(name, cmap, k=2)
            >>> tec.CreateColorMap(name, {f0: c0, f1:f1}, k=2
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script
            *cmap*
        :Versions:
            * 2016-10-31 ``@ddalle``: First version
        """
        # Initialize command
        cmd = 'CREATECOLORMAP'
        lines = ["  NAME = '%s'\n" % name]
        # Get cmap fraction keys
        V = cmap.keys()
        # Make sure all are floats
        for v in V:
            # Get control point type and check it
            t = type(v).__name__
            if t not in ['float', 'int']:
                raise TypeError(("COLORMAPFRACTION value '%s' " % v) +
                    ("must be a float or int (found: '%s')" % t))
        # Sort
        V.sort()
        # Number of control points
        n = len(V)
        lines.append("  NUMCONTROLPOINTS = %i\n" % n)
        # Loop through control points
        for i in range(n):
            # Get the color value
            c = cmap[V[i]]
            t = type(c).__name__
            # Unpack if two colors
            if t in ['list', 'ndarray']:
                # Separate lead and trail RGB values
                cL = c[0]
                cR = c[0]
            else:
                # One color
                cL = c
                cR = c
            # Lead, trail RGB types
            tL = type(cL).__name__
            tR = type(cR).__name__
            # Process the type
            if (t not in ['list','ndarray']):
                raise TypeError("Color '%s' does not have valid type" % V[i])
            elif len(V) != 3:
                raise ValueError("Color '%s' must have three values" % V[i])
            # Append the color
            lines.append("  CONTROLPOINT %i\n" % i)
            lines.append("    {\n")
            lines.append("    LEADRGB\n")
            lines.append("      {\n")
            lines.append("      R = %i\n" % cL[0])
            lines.append("      G = %i\n" % cL[1])
            lines.append("      B = %i\n" % cL[2])
            lines.append("      }\n")
            lines.append("    TRAILRGB\n")
            lines.append("      {\n")
            lines.append("      R = %i\n" % cR[0])
            lines.append("      G = %i\n" % cR[1])
            lines.append("      B = %i\n" % cR[2])
            lines.append("      }\n")
            lines.append("    }\n")
        # Insert the command
        self.InsertCommand(k, cmd, lines=lines)
        
        
    # Set group stuff
    def SetFieldMap(self, grps):
        """Set active zones for a Tecplot layout, mostly for Overflow
        
        :Call:
            >>> tec.SetFieldMap(grps)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript`
                Instance of Tecplot script interface
            *grps*: :class:`list` (:class:`int`)
                List of last zone number in each ``FIELDMAP`` section
        :Versions:
            * 2016-10-04 ``@ddalle``: First version
        """
        # Number of groups of field maps
        n = len(grps)
        # Loop through groups
        for i in range(n-1,-1,-1):
            # Construct entry: [1-171], [172-340], etc.
            if i == 0:
                gmin = 1
            else:
                gmin = grps[i-1]+1
            # End index
            gmax = grps[i]
            # Check for null group
            if gmin > gmax:
                # Delete the command
                self.DeleteCommandN('FIELDMAP', i)
                continue
            # Set value
            self.SetPar('FIELDMAP', "[%s-%s]" % (gmin, gmax), i)
        # Set the total number of maps
        self.SetPar('ACTIVEFIELDMAPS', "= [1-%s]" % grps[-1], 0)
        
    # Function to get command names and line indices
    def UpdateCommands(self):
        """Find lines that start with '$!' and report their indices
        
        :Call:
            >>> tec.UpdateCommands()
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
        :Effects:
            *tec.icmd*: :class:`list` (:class:`int`)
                Indices of lines that start commands
            *tec.cmds*: :class:`list` (:class:`str`)
                Name of each command
        :Versions:
            * 2015-02-28 ``@ddalle``: First version
        """
        # Find the indices of lines starting with '$!'
        self.icmd = self.GetIndexStartsWith('$!')
        # Get those lines
        lines = [self.lines[i] for i in self.icmd]
        # Isolate the first word of the command.
        self.cmds = [line[2:].split()[0] for line in lines]
        
    # Function to delete a command.
    def DeleteCommand(self, cmd, txt=None, lines=None):
        """Delete text for a specific command or commands and update text
        
        :Call:
            >>> kcmd = tec.DeleteCommand(cmd, txt=None, lines=None)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to delete
            *txt*: :class:`str`
                Regular expression for text after the command
            *lines*: :class:`list` (:class:`str`)
                Additional lines to filter for (regular expressions)
        :Outputs:
            *kcmd*: :class:`int`
                Index of earliest deleted command or ``None`` if no deletions
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Initialize output
        kcmd = None
        # Loop through commands in reverse order.
        for k in range(len(self.cmds)-1, -1, -1):
            # Check the command.
            if self.cmds[k] != cmd: continue
            # Check for additional text to match
            if txt:
                # Get the line of the command.
                line = self.lines[self.icmd[k]]
                # Extract the part after the command.
                line = line[len(cmd)+3:].strip()
                # Check the line
                if not re.search(txt, line): continue
            # Check for additional lines to filter
            if lines is not None:
                # Check for a single line
                if type(lines).__name__ in ['str', 'unicode']:
                    # Make it a list.
                    lines = [lines]
                # Initialize match flag
                qlines = True
                # Loop through the lines to filter.
                for j in range(len(lines)):
                    # Get the line to filter.
                    line = self.lines[self.icmd[k]+j+1].strip()
                    # Get the filter.
                    reg = lines[j]
                    # Check for match.
                    if not re.search(reg, line):
                        # Set flag
                        qlines = False
                        break
                # Check for a failed match.
                if not qlines: continue
            # If this point is reached, all criteria are met.
            kcmd = k
            # Delete the lines.
            if k == len(self.cmds):
                # Delete to the end
                del self.lines[self.icmd[k]:]
            else:
                # Delete until the next command starts(ed).
                del self.lines[self.icmd[k]:self.icmd[k+1]]
        # Update commands.
        self.UpdateCommands()
        # Report
        return kcmd
        
    # Function to get lines of a command
    def GetCommand(self, cmd, n=0):
        """Get the start and end line numbers in the *n*th instance of *cmd*
        
        This allows the user to get the lines of text in the command to be
        ``tec.lines[ibeg:iend]``.
        
        :Call:
            >>> ibeg, iend = tec.GetCommand(cmd, n=0)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *n*: {``0``} | :class:`int` >= 0
                Instance of command to find
        :Outputs:
            *ibeg*: ``None`` | :class:`int`
                Index of start of command (or ``None`` if less than *n*
                instances of commands named *cmd*)
            *iend*: ``None`` | :class:`int`
                Index of start of next command
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Find instances of command
        Kcmd = np.where(np.array(self.cmds) == cmd)[0]
        # Check for possible patch
        if n >= len(Kcmd):
            return None, None
        # Get the global index of the command
        k = Kcmd[n]
        # Get the line indices
        if k == len(self.cmds):
            # Last command; use number of lines for the end
            iend = len(self.lines) + 1
        else:
            # Use the start of the next command as the end of this one
            iend = self.icmd[k+1]
        # Start line
        ibeg = self.icmd[k]
        # Output
        return ibeg, iend
        
    # Function to get key from a command
    def GetKey(self, cmd, key, n=0):
        """Get the value of a key from the *n*th instance of a command
        
        :Call:
            >>> val = tec.GetKey(cmd, key, n=0)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *n*: {``0``} | :class:`int` >= 0
                Instance of command to find
        :Outputs:
            *ibeg*: ``None`` | :class:`int`
                Index of start of command (or ``None`` if less than *n*
                instances of commands named *cmd*)
            *iend*: ``None`` | :class:`int`
                Index of start of next command
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Get the lines in the command
        ibeg, iend = self.GetCmd(cmd, n=n)
        # Loop through lines
        i = ibeg + 1
        while i < iend:
            # Try the next line
            line = self.lines[i]
            # Get the key for this line
            k = line.split()[0]
            # Test for a match
            
            
    # Convert text to value
    def ConvertToVal(self, val):
        """Convert a text string to a scalar Python value
        
        :Call:
            >>> v = tec.ConvertToval(val)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script interface
            *val*: :class:`str` | :class:`unicode`
                Text of the value from file
        :Outputs:
            *v*: :class:`str` | :class:`int` | :class:`float`
                Evaluated value of the text
        :Versions:
            * 2017-01-05 ``@ddalle``: First version
        """
        # Check inputs
        if type(val).__name__ not in ['str', 'unicode']:
            # Not a string; return as is
            return val
        # Be safe; some of these conversions may fail
        try:
            # Check the contents
            if ('"' in val) or ("'" in val):
                # It's a string; For Tecplot we do not remove the quotes
                return val
            elif len(val.strip()) == 0:
                # Nothing here
                return None
            else:
                # Convert to float/integer
                return eval(val)
        except Exception:
            # Give back the string
            return val
            
    # Read a value into a key
    def ReadKey(self, i):
        """Read a key by converting text to a value
        
        :Call:
            >>> key, val, m = tec.ReadKey(i)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *i*: :class:`int`
                Line number on which to start
        :Outputs:
            *key*: :class:`str`
                Name of the key whose definition starts on this line
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`dict`
                    | :class:`np.ndarray` (:class:`float`)
                Value for that line
            *m*: :class:`int`
                Number of lines used for definition of this key
        :Versions:
            * 2016-01-05 ``@ddalle``: First version
        """
        # Get the line
        line = self.lines[i]
        # Split by '='
        V = [v.strip() for v in line.split('=')]
        # Check for valid line
        if len(V) == 0:
            return None, None, 0
        # Initialize line count
        m = 1
        # Number of lines
        nline = len(self.lines)
        # Read the name of the key
        key = V[0]
        # Check if there is an '=' sign
        if len(V) > 1:
            # We have a scalar value
            t = 's'
            val = self.ConvertToVal(V[1])
        else:
            # Initialize value
            val = []
            t = 'l'
            # Loop through lines
            while True:
                # We must go to a new line
                if i+m >= nline:
                    # No more lines
                    break
                # Read the next line
                line = self.lines[i+m].strip()
                # Check what kind of marker we have
                if line == "{":
                    # Start of a dictionary
                    m += 1
                    val = {}
                    t = 'd'
                elif line == "}":
                    # End of the dictionary
                    m += 1
                    break
                elif line.startswith('!'):
                    # Comment
                    m += 1
                    continue
                elif line.startswith('$!'):
                    # Start of next command
                    break
                elif t == 'l':
                    # New entry to a list
                    m += 1
                    # Check if it's the number of entries (ignore)
                    if m == 2: continue
                    # Read the value
                    try:
                        # Should be a scalar
                        v = self.ConvertToVal(line)
                        # Add to the list
                        val.append(v)
                    except Exception:
                        # Failed
                        continue
                else:
                    # Read a dictionary key
                    ki, vi, mi = self.ReadKey(i+m)
                    # Save the dictionary key
                    val[ki] = vi
                    # Move *mi* lines
                    m += mi
        # Output
        if t == 'l':
            # Convert to array
            return key, np.array(val), m
        else:
            # Return dictionary
            return key, val, m
        
        
    
    # Function to delete a command.
    def DeleteCommandN(self, cmd, n=0):
        """Delete the *n*th instance of a command
        
        :Call:
            >>> kcmd = tec.DeleteCommandN(cmd, n=0)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to delete
            *n*: {``0``} | :class:`int` >= 0
                Instance of command to delete
        :Outputs:
            *kcmd*: :class:`int`
                Index of deleted command or ``None`` if no deletions
        :Versions:
            * 2016-10-05 ``@ddalle``: First version
        """
        # Find instances of command
        Kcmd = np.where(np.array(self.cmds) == cmd)[0]
        # Check for possible match
        if n >= len(Kcmd):
            return
        # Get global index of the command to delete
        k = Kcmd[n]
        # Delete the lines.
        if k == len(self.cmds):
            # Delete to the end
            del self.lines[self.icmd[k]:]
        else:
            # Delete until the next command starts(ed).
            del self.lines[self.icmd[k]:self.icmd[k+1]]
        # Report
        return k
    
    # Function to insert a command at a certain location
    def InsertLines(self, i, lines):
        """Insert a list of lines starting at a certain location
        
        :Call:
            >>> tec.InsertLines(i, lines)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *i*: :class:`int`
                Index at which to insert the first line
            *lines*: :class:`list` (:class:`str`)
                Lines to insert, *lines[0]* is inserted at line *i*
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Check for a single line.
        if type(lines).__name__ in ['str', 'unicode']:
            # Make it a list.
            lines = [lines]
        # Loop through the lines.
        for j in range(len(lines)):
            # Insert the line.
            self.lines.insert(i+j, lines[j])
        # Update the commands
        self.UpdateCommands()
            
    # Insert a command
    def InsertCommand(self, k, cmd, txt="", lines=[]):
        """Insert a command
        
        :Call:
            >>> tec.InsertCommand(k, cmd, txt="", lines=[])
        :Inputs:
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *k*: :class:`int`
                Default command index at which to insert command
            *cmd*: :class:`str`
                Title of the command to insert
            *txt*: :class:`str`
                Text to add after the command on the same line
            *lines*: :class:`list` (:class:`str`)
                Additional lines to add to the command
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Create the lines to add.
        if txt is None:
            # Create simple command title
            L = ["$!%s\n" % cmd]
        else:
            # Merge the command and additional text
            L = ["$!%s %s\n" % (cmd, txt)]
        # Additional lines
        if lines is None:
            # No lines to add
            pass
        elif type(lines).__name__ in ["str", "unicode"]:
            # Add a single line.
            L.append(lines)
        else:
            # Add list of lines.
            L += lines
        # Get the line index based on the command index.
        i = self.icmd[k]
        # Insert the lines.
        self.InsertLines(i, L)
        
    
    # Replace command
    def ReplaceCommand(self, cmd, txt="", lines=[], k=1, reg=None, regs=None):
        """Replace a command
        
        :Call:
            >>> tec.ReplaceCommand(cmd,txt="",lines=[],k=1,reg=None,regs=None)
        :Inputs:
            *tec*: :class:`cape.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to replace
            *txt*: :class:`str`
                Text to add after the command on the same line
            *lines*: :class:`list` (:class:`str`)
                Additional lines to add to the command
            *k*: :class:`int`
                Default command index at which to insert command
            *reg*: :class:`str`
                Regular expression for text after the command
            *regs*: :class:`list` (:class:`str`)
                Additional lines to filter for (regular expressions)
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Delete the command
        kcmd = self.DeleteCommand(cmd, txt=reg, lines=regs)
        # Get the default command index.
        if kcmd is None: kcmd = k
        # Insert the command.
        self.InsertCommand(kcmd, cmd, txt, lines)
            

# Tecplot macro
class TecMacro(Tecscript):
    """
    File control class for Tecplot macr files
    
    :Call:
        >>> tec = pyCart.tecplot.TecMacro()
        >>> tec = pyCart.tecplot.TecMacro(fname="export.mcr")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`cape.tecplot.TecMacro`
            Instance of Tecplot macro interface
    :Versions:
        * 2015-03-10 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="export.mcr"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Get the command list
        self.UpdateCommands()
        
    # Set the export format
    def SetExportFormat(self, fmt="PNG"):
        """Set Tecplot macro export format
        
        :Call:
            >>> tec.SetExportFormat(fmt="PNG")
        :Inputs:
            *tec*: :class:`cape.tecplot.TecMacro`
                Instance of Tecplot macro interface
            *fmt*: :class:`str`
                Export format
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Form the export format code
        txt = 'EXPORTFORMAT = %s' % fmt
        # Do the replacement
        self.ReplaceCommand('EXPORTSETUP', txt, k=2, reg='EXPORTFORMAT')
        
    # Set the layout file
    def SetLayout(self, lay="layout.lay"):
        """Set the Tecplot layout file name
        
        :Call:
            >>> tec.SetLayout(lay="layout.lay")
        :Inputs:
            *tec*: :class:`cape.tecplot.TecMacro`
                Instance of Tecplot macro interface
            *lay*: :class:`str`
                Tecplot layout file name
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Form the layout file name code
        txt = ' "%s"' % lay
        # Do the replacement
        self.ReplaceCommand('OPENLAYOUT', txt, k=1)
        
    # Set the export file name
    def SetExportFileName(self, fname="export.png"):
        """Set the name of the exported image file
        
        :Call:
            >>> tec.SetExportFileName(fname="export.png")
        :Inputs:
            *tec*: :class:`cape.tecplot.TecMacro`
                Instance of Tecplot macro interface
            *fname*: :class:`str`
                Export image file name
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Form the layout file name code
        txt = 'EXPORTFNAME = "%s"' % fname
        # Do the replacement
        self.ReplaceCommand('EXPORTSETUP', txt, k=-3, reg='EXPORTFNAME')
        
    # Set the export image width
    def SetImageWidth(self, w=1024):
        """Set the export image width
        
        Call:
            >>> tec.SetImageWidth(w=1024)
        :Inputs:
            *tec*: :class:`cape.tecplot.TecMacro`
                Instance of Tecplot macro interface
            *w*: :class:`int`
                Image width in pixels
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Form the layout file name code
        txt = 'IMAGEWIDTH = %i' % w
        # Do the replacement
        self.ReplaceCommand('EXPORTSETUP', txt, k=-3, reg='IMAGEWIDTH')
# class TecMacro

    
