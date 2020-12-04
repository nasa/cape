"""
:mod:`cape.filecntl.tecplot`: Interface to Tecplot scripts 
===========================================================

This is a module built off of the :mod:`cape.filecntl` module customized for
manipulating Tecplot layout files and macros.

It allows users to edit quantities of any layout command in addition to
declaring and adding layout variables.  In addition, the :func:`ExportLayout`
function provides a utility to open a layout using Tecplot in batch mode to
export an image.

The class provides two classes, the first of which is the generic version
typically used for layout files.  The second class has a few extra methods for
handling Tecplot macros specifically.

    * :class:`cape.filecntl.tecplot.Tecsript`
    * :class:`cape.filecntl.tecplot.TecMacro`

:See also:
    * :mod:`cape.filecntl`
    * :mod:`cape.cfdx.report`
"""

# Standard library
import os
import re
import shutil

# Third-party
import numpy as np

# CAPE modules
from cape.cfdx.bin import tecmcr
from cape.color    import ToRGB, Hex2RGB
from cape.util     import TecFolder

# Local modules
from .filecntl import FileCntl


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
# def ExportLayout

# Base this class off of the main file control class.
class Tecscript(FileCntl):
    """
    File control class for Tecplot script files
    
    :Call:
        >>> tec = cape.filecntl.tecplot.Tecscript()
        >>> tec = cape.filecntl.tecplot.Tecscript(fname="layout.lay")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`pyCart.tecplot.Tecscript`
            Instance of Tecplot script base class
    :Versions:
        * 2015-02-26 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """
  # =============
  # Configuration
  # =============
  # <
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="layout.lay"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Get the command list
        self.UpdateCommands()
    
    # Function to get command names and line indices
    def UpdateCommands(self):
        """Find lines that start with '$!' and report their indices
        
        :Call:
            >>> tec.UpdateCommands()
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
    
    # Function to insert a command at a certain location
    def InsertLines(self, i, lines):
        """Insert a list of lines starting at a certain location
        
        :Call:
            >>> tec.InsertLines(i, lines)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
  # >
    
  # ===============
  # Text Conversion
  # ===============
  # <
    # Convert text to value
    def ConvertToVal(self, val):
        """Convert a text string to a scalar Python value
        
        :Call:
            >>> v = tec.ConvertToval(val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
        
    # Create text for a key based on a value
    def KeyToText(self, key, val, m=0):
        """Create text for a key and value pair
        
        :Call:
            >>> lines = tec.KeyToText(key, val, m=0)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script/layout interface
            *key*: :class:`str`
                Name of the key
            *val*: :class:`any`
                Value to write
            *m*: {``2``} | nonnegative :class:`int`
                Number of leading spaces
        :Versions:
            * 2016-01-05 ``@ddalle``: First version
        """
        # Initialize text
        lines = []
        # Leading spaces
        s = ' '*m
        # Get the type of the data to write
        t = type(val).__name__
        # Check the type
        if t in ['dict', 'odict']:
            # Initialize a dictionary
            lines.append('%s%s\n' % (s, key))
            lines.append('%s{\n'  % (' '*(m+2)))
            # Sort the keys
            keys = val.keys()
            keys.sort()
            # Loop through the keys
            for k in keys:
                # Recurse
                lines_k = self.KeyToText(k, val[k], m=m+2)
                # Append the lines for that key
                lines += lines_k
            # Close the dictionary
            lines.append('%s}\n'  % (' '*(m+2)))
        elif t in ['list', 'ndarray']:
            # Initialize an array
            lines.append('%s%s\n' % (s, key))
            # Write the length of the array
            lines.append('%i\n' % len(val))
            # Loop through the values
            for v in val:
                lines.append('%s\n' % v)
        else:
            # Convert value to string
            vs = '%s' % val
            # Check for empty string, which Tecplot writes as ''
            if len(vs) == 0: vs = "''"
            # Write as a string
            lines = ["%s%s = %s\n" % (s, key, vs)]
        # Output
        return lines
  # >
    
  # =========
  # Variables
  # =========
  # <
    # Set variable
    def SetVar(self, key, val):
        """Set a variable to a particular value
        
        :Call:
            >>> tec.SetVar(key, val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *mach*: :class:`float`
                Freestream Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Set the variable
        self.SetVar('Minf', mach)
  # >
    
  # ========
  # Commands
  # ========
  # <
   # -------------
   # Find Commands
   # -------------
   # [
    # Find a command by name
    def GetCommandIndex(self, cmd, nmax=1):
        r"""Find indices of command by name

        :Call:
            >>> Kcmd = tec.GetCommandIndex(cmd, nmax=1)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *nmax*: {``1``} | :class:`int` > 0
                Maximum finds to locate
        :Outputs:
            *Kcmd*: :class:`list`\ [:class:`int`]
                List of indices of *tec.cmds* that match *cmd*
        :Versions:
            * 2020-01-28 ``@ddalle``: First version
        """
        # Initialize indices
        Kcmd = []
        # Number of finds
        n = 0
        # Loop through lines
        for (j, cmdj) in enumerate(self.cmds):
            # Check for case-insensitive match
            if cmdj.lower() == cmd.lower():
                # Increase find
                n += 1
                # Save find
                Kcmd.append(j)
                # Check count
                if nmax and n >= nmax:
                    break
        # Output
        return Kcmd
        
    # Function to get lines of a command
    def GetCommand(self, cmd, n=0):
        """Get the start and end line numbers in the *n*\ th instance of *cmd*
        
        This allows the user to get the lines of text in the command to be
        ``tec.lines[ibeg:iend]``.
        
        :Call:
            >>> ibeg, iend = tec.GetCommand(cmd, n=0)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *n*: {``0``} | ``None`` | :class:`int` >= 0
                Instance of command to find; if ``None`` return all
        :Outputs:
            *ibeg*: ``None`` | :class:`int` | :class:`list` (:class:`int`)
                Index of start of command (or ``None`` if less than *n*
                instances of commands named *cmd*)
            *iend*: ``None`` | :class:`int` | :class:`list` (:class:`int`)
                Index of start of next command
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Find instances of command
        if n is None:
            # Unlimited finds
            Kcmd = self.GetCommandIndex(cmd, None)
        else:
            # Modify limit
            Kcmd = self.GetCommandIndex(cmd, n + 1)
        # Append total line count to icmd
        icmd = self.icmd + [len(self.lines)+1]
        # Check for possible match
        if n is None:
            # Return all matches
            # Create arrays
            ibeg = [icmd[k]   for k in Kcmd]
            iend = [icmd[k+1] for k in Kcmd]
            # Output
            return ibeg, iend
        elif n >= len(Kcmd):
            # Did not find command at least *n* times
            return None, None
        # Get the global index of the command
        k = Kcmd[n]
        # Start line
        ibeg = icmd[k]
        # Use the start of the next command as the end of this one
        iend = icmd[k+1]
        # Output
        return ibeg, iend
    
    # Get command using a parameter value
    def GetCommandByPar(self, cmd, val):
        """Search for a command based on name and parameter
        
        A 'parameter' is a value printed on the same line as the command name
        
        :Call:
            >>> ibeg, iend = tec.GetCommandByPar(cmd, val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *val*: ``None`` | :class:`str` | :class:`int` | :class:`float`
                Target value for the parameter, often an index number
        :Outputs:
            *ibeg*: ``None`` | :class:`int`
                Index of start of command (or ``None`` if less than *n*
                instances of commands named *cmd*)
            *iend*: ``None`` | :class:`int`
                Index of start of next command
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Find all commands matching name *cmd*
        Ibeg, Iend = self.GetCommand(cmd, n=None)
        # Loop through those until a matching parameter is found.
        for n in range(len(Ibeg)):
            # Check the parameter
            par = self.GetPar(cmd, n)
            # Check for match
            if par == val:
                # Output
                return Ibeg[n], Iend[n]
        # If no commands matched, return empty result
        return None, None
        
    # Get command using a key value
    def GetCommandByKey(self, cmd, key, val):
        """Search for a command based on a key and value
        
        :Call:
            >>> ibeg, iend = tec.GetCommandByKey(cmd, key, val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *key*: :class:`str`
                Name of key to filter
            *val*: ``None`` | :class:`str` | :class:`int` | :class:`float`
                Target value for the key
        :Outputs:
            *ibeg*: ``None`` | :class:`int`
                Index of start of command (or ``None`` if less than *n*
                instances of commands named *cmd*)
            *iend*: ``None`` | :class:`int`
                Index of start of next command
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Find all commands matching name *cmd*
        Ibeg, Iend = self.GetCommand(cmd, n=None)
        # Loop through those commands until a match is found
        for n in range(len(Ibeg)):
            # Check the key
            v = self.GetKey(cmd, key, n=n)
            # Check for a match
            if v == val:
                # Output
                return Ibeg[n], Iend[n]
        # If no commands matched, return empty result
        return None, None
    
   # ]
    
   # -------
   # Editing
   # -------
   # [
    
   # ]
   
   # ------------
   # Bulk Actions
   # ------------
   # [
    # Insert a command
    def InsertCommand(self, k, cmd, txt="", lines=[]):
        """Insert a command
        
        :Call:
            >>> tec.InsertCommand(k, cmd, txt="", lines=[])
        :Inputs:
            *tec*: :class:`pyCart.tecplot.Tecscript`
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
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
        if kcmd is None:
            kcmd = k
        # Insert the command.
        self.InsertCommand(kcmd, cmd, txt, lines)
   
   # ]
    
   # --------
   # Deletion
   # --------
   # [
    # Function to delete a command.
    def DeleteCommand(self, cmd, txt=None, lines=None):
        """Delete text for a specific command or commands and update text
        
        :Call:
            >>> kcmd = tec.DeleteCommand(cmd, txt=None, lines=None)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
            if self.cmds[k].lower() != cmd.lower():
                continue
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
        
    # Function to delete a command.
    def DeleteCommandN(self, cmd, n=0):
        """Delete the *n*\ th instance of a command
        
        :Call:
            >>> kcmd = tec.DeleteCommandN(cmd, n=0)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
        Kcmd = self.GetCommandIndex(cmd, n+1)
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
   # ]
  # >
    
    
  # ==========
  # Parameters
  # ==========
  # <
    # Set parameter on header line
    def SetPar(self, cmd, val, n):
        """Set a parameter value on the header line of a command
        
        :Call:
            >>> tec.SetPar(cmd, val, n)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecsript`
                Instance of Tecplot script
            *cmd*: :class:`str`
                Name of command
            *val*: :class:`str`
                String to set on the header line
            *n*: :class:`int`
                Alter the instance *n* of this command
        :Versions:
            * 2016-10-04 ``@ddalle``: First version
            * 2017-01-05 ``@ddalle``: Changed *i* -> *n*
        """
        # Get the indices of this command
        I = self.GetIndexStartsWith('$!'+cmd)
        # Make sure there are at least *i* matches
        if n >= len(I):
            raise ValueError(
                ("Requested to alter instance %s of command '%s'"%(n+1,cmd)) +
                ("but layout only contains %i instances" % len(I)))
        # Get the line number
        j = I[n]
        # Set the line
        self.lines[j] = "$!%s %s\n" % (cmd, val)
        
    # Read a parameter on header line
    def GetPar(self, cmd, n=0):
        """Read a parameter value on the header line of a command
        
        :Call:
            >>> val = tec.GetPar(cmd, n=0)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecsript`
                Instance of Tecplot script
            *cmd*: :class:`str`
                Name of command
            *n*: :class:`int`
                Alter the instance *n* of this command
        :Outputs:
            *val*: ``None`` | :class:`str` | :class:`int` | :class:`float`
                Value of the parameter on that line, if any
        :Versions:
            * 2017-01-05 ``@ddalle``: First version
        """
        # Get the line indices for this command
        ibeg, iend = self.GetCommand(cmd, n=n)
        # Split
        V = self.lines[ibeg].split()
        # Check for multiple entries
        if len(V) == 1:
            # No parameter
            return None
        else:
            # Single parameter
            return self.ConvertToVal(V[1])
            
        
  # >
    
  # ====
  # Keys
  # ====
  # <
    # Read a value into a key
    def ReadKey(self, i):
        """Read a key by converting text to a value
        
        :Call:
            >>> key, val, m = tec.ReadKey(i)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *i*: :class:`int`
                Line number on which to start
        :Outputs:
            *key*: :class:`str`
                Name of the key whose definition starts on this line
            *val*: :class:`any`
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
        
    # Overwrite a key
    def WriteKey(self, i, key, val):
        """Replace a key with a new value
        
        :Call:
            >>> tec.WriteKey(i, key, val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *i*: :class:`int`
                Line number on which to start
            *key*: :class:`str`
                Name of the key
            *val*: :class:`any`
                Value for that line
        :Versions:
            * 2016-01-05 ``@ddalle``: First version
        """
        # Get the current value of the key starting on line *i*
        key, v, m = self.ReadKey(i)
        # Check for valid key
        if key is None:
            raise ValueError(
                ("Cannot write key '%s' at line %i " % (key, i)) + 
                ("because it is not the start of an existing key"))
        # Check the indentation
        S = re.findall('^\s*', self.lines[i])
        ns = len(S[0])
        # Create the new text
        lines = self.KeyToText(key, val, m=ns)
        # Substitute the new lines
        self.lines = self.lines[:i] + lines + self.lines[i+m:]
        # Update command indices
        self.UpdateCommands()
        
    # Insert a new key
    def InsertKey(self, i, key, val):
        """Insert a new key
        
        :Call:
            >>> tec.InsertKey(i, key, val)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *i*: :class:`int`
                Line number on which to start
            *key*: :class:`str`
                Name of the key
            *val*: :class:`any`
                Value for that line
        :Versions:
            * 2018-03-29 ``@ddalle``: First version
        """
        # Check the indentation
        S = re.findall('^\s*', self.lines[i])
        ns = len(S[0])
        # Create the new text
        lines = self.KeyToText(key, val, m=ns)
        # Substitute the new lines
        self.lines = self.lines[:i] + lines + self.lines[i:]
        # Update command indices
        self.UpdateCommands()
        
    # Function to get key from a command
    def GetKey(self, cmd, key, n=0, par=None, k=None, v=None):
        """Get the value of a key from the *n*\ th instance of a command
        
        :Call:
            >>> val = tec.GetKey(cmd, key, n=0, par=None, k=None, v=None)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *key*: :class:`str`
                Name of the key to find within the command
            *n*: {``0``} | :class:`int` >= 0
                Instance of command to find
            *par*: {``None``} | :class:`int` | :class:`str`
                Optional parameter value for which to search
            *k*: {``None``} | :class:`str`
                Optional key to use as search value
            *v*: {``None``} | :class:`str` | :class:`int`
                If *k* is used, value to test for search key
        :Outputs:
            *val*: :class:`any` | ``None``
                Value of the key if present
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Check for a parameter value\
        if k is not None:
            # Search by key name
            ibeg, iend = self.GetCommandByKey(cmd, k, v)
        elif par is not None:
            # Search for command by parameter
            ibeg, iend = self.GetCommandByPar(cmd, par)
        else:
            # Search for command by count
            ibeg, iend = self.GetCommand(cmd, n=n)
        # If no command, don't continue
        if ibeg is None:
            return None
        # Loop through lines
        i = ibeg + 1
        while i < iend:
            # Try the next key
            k, val, m = self.ReadKey(i)
            # Test for a match
            if k is not None and k.lower() == key.lower():
                return val
            # Move *m* lines to read the next key
            i += m
        # If reached here, no match
        return None
    
    # Replace the contents of a key
    def SetKey(self, cmd, key, val, n=0, par=None, k=None, v=None):
        """Find a key in a specified command and rewrite it
        
        :Call:
            >>> tec.SetKey(cmd, key, val, n=0, par=None, k=None, v=None)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *cmd*: :class:`str`
                Title of the command to find
            *key*: :class:`str`
                Name of the key to edit
            *val*: :class:`any`
                New value of the key to write
            *n*: {``0``} | :class:`int` >= 0
                Instance of command to find
            *par*: {``None``} | :class:`int` | :class:`str`
                Optional parameter value for which to search
            *k*: {``None``} | :class:`str`
                Optional key to use as search value
            *v*: {``None``} | :class:`str` | :class:`int`
                If *k* is used, value to test for search key
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Check for a parameter value
        if k is not None:
            # Search by key name
            ibeg, iend = self.GetCommandByKey(cmd, k, v)
        elif par is not None:
            # Search for command by parameter
            ibeg, iend = self.GetCommandByPar(cmd, par)
        else:
            # Search for command by count
            ibeg, iend = self.GetCommand(cmd, n=n)
        # If no command, don't continue
        if ibeg is None:
            return None
        # Loop through lines
        i = ibeg + 1
        while i < iend:
            # Try the next key
            k, v, m = self.ReadKey(i)
            # Test for a match
            if k is not None and k.lower() == key.lower():
                break
            # Move *m* lines
            i += m
            # If we reached the end, give up
            if i >= iend:
                # Insert new key
                self.InsertKey(i, key, val)
                return
        # Create text for the key
        self.WriteKey(i, key, val)
  # >
    
  # ==============
  # Custom Methods
  # ==============
  # <
    # Reset contour levels
    def SetContourLevels(self, n, V):
        """Set contour levels for global contour map *n*
        
        :Call:
            >>> tec.SetContourLevels(n, V)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script base class
            *n*: :class:`int`
                Contour map number (as labeled in layout file)
            *V*: :class:`np.ndarray` (:class:`float`)
                List of contour levels
        :Versions:
            * 2017-10-05 ``@ddalle``: First version
        """
        # Number of contour levels
        nlev = len(V)
        # Write the number of levels
        self.SetKey('GLOBALCONTOUR', 'DEFNUMLEVELS', nlev, par=n)
        # Write the levels
        self.SetKey('CONTOURLEVELS', 'RAWDATA', V, k='CONTOURGROUP', v=n)
        
    # Rewrite a color map
    def EditColorMap(self, name, cmap, vmin=None, vmax=None, **kw):
        """Replace the contents of a color map
        
        :Call:
            >>> tec.EditColorMap(name, cmap, vmin=None, vmax=None, **kw)
            >>> tec.EditColorMap(name, {f0:c0, f1:c1, f2:c2, ...}, ... )
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script
            *cmap*: :class:`dict` (:class:`list`)
                Dictionary of color map fractions and colors
            *f0*: :class:`float`
                Level for the first color, either 0 to 1 or *vmin* to *vmax*
            *c0*: :class:`str` | :class:`list`
                Either color name, hex code, or RGB list
            *vmin*: {``None``} | :class:`float`
                Minimum value to use for determining COLORMAPFRACTION values
            *vmax*: {``None``} | :class:`float`
                Maximum value to use for determining COLORMAPFRACTION values
            *nContour*: {``None``} | :class:`int`
                Global contour number to edit
            *nColorMap*: {``None``} | :class:`int`
                Number of color map to edit
        :Versions:
            * 2017-01-05 ``@ddalle``: First version
        """
        # Initialize command
        cmd = "CREATECOLORMAP"
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
        nV = len(V)
        lines = ["  NUMCONTROLPOINTS = %i\n" % nV]
        # Default min/max values
        if vmin is None or vmax is None:
            # Assume the levels go from 0 to 1
            vmin = min(V)
            vmax = max(V)
        # Loop through control points
        for i in range(nV):
            # Get the color value
            c = cmap[V[i]]
            t = type(c).__name__
            # Unpack if two colors
            if (t in ['list', 'ndarray']) and (len(c) == 2):
                # Separate lead and trail RGB values
                cL = ToRGB(c[0])
                cR = ToRGB(c[1])
            else:
                # One color
                cL = ToRGB(c)
                cR = ToRGB(c)
            # Convert value to level
            v = (V[i] - vmin) / (vmax - vmin)
            # Append the color
            lines.append("  CONTROLPOINT %i\n" % (i+1))
            lines.append("    {\n")
            lines.append("    COLORMAPFRACTION = %s\n" % v)
            lines.append("    LEADRGB\n")
            lines.append("      {\n")
            lines.append("      R = %i\n" % cR[0])
            lines.append("      G = %i\n" % cR[1])
            lines.append("      B = %i\n" % cR[2])
            lines.append("      }\n")
            lines.append("    TRAILRGB\n")
            lines.append("      {\n")
            lines.append("      R = %i\n" % cL[0])
            lines.append("      G = %i\n" % cL[1])
            lines.append("      B = %i\n" % cL[2])
            lines.append("      }\n")
            lines.append("    }\n")
        # Get contour and colormap numbers
        nContour  = kw.get('nContour', None)
        nColorMap = kw.get('nColorMap', None)
        # Make sure *name* starts and ends with quotes
        if name is not None:
            # Strip any quotes and add exactly one set back
            name = "'%s'" % name.strip('"').strip("'")
        # Check for a search command
        if nContour is not None:
            # Get the search name from the *n*th contour
            sname = self.GetKey('GLOBALCONTOUR', 'COLORMAPNAME', par=nContour)
            # Use this name as default output name
            if name is None:
                # Use color map name from the contour specification
                name = sname
            else:
                # Change the name of the color map
                self.SetKey('GLOBALCONTOUR', 'COLORMAPNAME', name,
                    par=nContour)
        else:
            # Search for the input name
            sname = name
        # Add the name of the color map
        lines.insert(0, "  NAME = %s\n" % name)
        # Find the index for the existing CREATECOLORMAP command
        if nColorMap is not None:
            # Use the *n*th color map directly
            ibeg, iend = self.GetCommand(cmd, nColorMap-1)
        elif sname is not None:
            # Use the name
            ibeg, iend = self.GetCommandByKey(cmd, 'NAME', sname)
        else:
            # Default: no match
            ibeg, iend = None, None
        # Edit the text
        if ibeg is None:
            # Create a new color map
            k = kw.get('k', 5)
            self.InsertCommand(k, cmd, lines=lines)
        else:
            # Prepend the command name to lines
            lines.insert(0, '$!%s\n' % cmd)
            # Replace those lines
            self.lines = self.lines[:ibeg] + lines + self.lines[iend:]
        # Update line numbers of commands
        self.UpdateCommands()
        
    # Set group stuff
    def SetFieldMap(self, grps):
        """Set active zones for a Tecplot layout, mostly for Overflow
        
        :Call:
            >>> tec.SetFieldMap(grps)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
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
            # Check for null group, single zone, or multiple zone
            if gmin > gmax:
                # Null group; delete the command
                self.DeleteCommandN('FIELDMAP', i)
                continue
            elif gmin == gmax:
                # Single group
                self.SetPar('FIELDMAP', "[%s]" % gmin, i)
            else:
                # Range
                self.SetPar('FIELDMAP', "[%s-%s]" % (gmin, gmax), i)
        # Set the total number of maps
        self.SetPar('ACTIVEFIELDMAPS', "= [1-%s]" % grps[-1], 0)
        
    # Set slice locations
    def SetSliceLocation(self, n=1, **kw):
        """Set slice location
        
        :Call:
            >>> tec.SetSlice(n=1, **kw)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.Tecscript`
                Instance of Tecplot script interface
            *n*: {``1``} | positive :class:`int`
                Slice number to edit
            *x*: {``None``} | :class:`float`
                *x*-coordinate of slice
            *y*: {``None``} | :class:`float`
                *y*-coordinate of slice
            *z*: {``None``} | :class:`float`
                *z*-coordinate of slice
            *i*: {``None``} | :class:`int`
                Index of *I* slice to plot
            *j*: {``None``} | :class:`int`
                Index of *J* slice to plot
            *k*: {``None``} | :class:`int`
                Index of *K* slice to plot
        :Versions:
            * 2017-02-03 ``@ddalle``: First version
        """
        # Get the existing coordinate
        pos = self.GetKey('SLICEATTRIBUTES', 'PRIMARYPOSITION', par=n)
        # Default POSITION if none found
        if pos is None:
            pos = {"X":0, "Y":0, "Z":0, "Z":0, "I":1, "J":1, "K":1}
        # Set parameters given as inputs
        if "x" in kw: pos["X"] = kw["x"]
        if "y" in kw: pos["Y"] = kw["y"]
        if "z" in kw: pos["Z"] = kw["z"]
        if "i" in kw: pos["I"] = kw["i"]
        if "j" in kw: pos["J"] = kw["j"]
        if "k" in kw: pos["K"] = kw["k"]
        # Set parameter
        self.SetKey('SLICEATTRIBUTES', 'PRIMARYPOSITION', pos, par=n)
  # >
    
# class Tecscript

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
        *tec*: :class:`cape.filecntl.tecplot.TecMacro`
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
            *tec*: :class:`cape.filecntl.tecplot.TecMacro`
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
            *tec*: :class:`cape.filecntl.tecplot.TecMacro`
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
            *tec*: :class:`cape.filecntl.tecplot.TecMacro`
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
        
        :Call:
            >>> tec.SetImageWidth(w=1024)
        :Inputs:
            *tec*: :class:`cape.filecntl.tecplot.TecMacro`
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

    
