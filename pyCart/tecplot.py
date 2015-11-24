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
    #os.remove(fmcr)

# Base this class off of the main file control class.
class Tecscript(FileCntl):
    """
    File control class for Tecplot script files
    
    :Call:
        >>> tec = pyCart.tecplot.Tecscript()
        >>> tec = pyCart.tecplot.Tecscript(fname="layout.lay")
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
        
    # Set variable
    def SetVar(self, key, val):
        """Set a variable to a particular value
        
        :Call:
            >>> tec.SetVar(key, val)
        :Inputs:
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
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
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
                Instance of Tecplot script base class
            *mach*: :class:`float`
                Freestream Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Set the variable
        self.SetVar('Minf', mach)
        
    # Function to get command names and line indices
    def UpdateCommands(self):
        """Find lines that start with '$!' and report their indices
        
        :Call:
            >>> tec.UpdateCommands()
        :Inputs:
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
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
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
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
    
    # Function to insert a command at a certain location
    def InsertLines(self, i, lines):
        """Insert a list of lines starting at a certain location
        
        :Call:
            >>> tec.InsertLines(i, lines)
        :Inputs:
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
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
            *cmd*: :class:`str`
                Title of the command to insert
            *txt*: :class:`str`
                Text to add after the command on the same line
            *lines*: :class:`list` (:class:`str`)
                Additional lines to add to the command
            *k*: :class:`int`
                Default command index at which to insert command
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Create the lines to add.
        if txt is None:
            # Create simple command title
            L = ["$! %s\n" % cmd]
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
            *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
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
        *tec*: :class:`pyCart.tecplot.TecMacro`
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
            *tec*: :class:`pyCart.tecplot.TecMacro`
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
            *tec*: :class:`pyCart.tecplot.TecMacro`
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
            *tec*: :class:`pyCart.tecplot.TecMacro`
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
            *tec*: :class:`pyCart.tecplot.TecMacro`
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

    
