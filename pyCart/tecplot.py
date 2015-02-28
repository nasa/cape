"""
Module to interface with Tecplot scripts: :mod:`pyCart.tecplot`
======================================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating Tecplot layout files and macros.
"""

# Import the base file control class.
from fileCntl import FileCntl, _num, _float

# Numerics
import numpy as np

# Base this class off of the main file control class.
class Tecscript(FileCntl):
    """
    File control class for Tecplot script files
    
    :Call:
        >>> tec = pyCart.tecplot.Tecscript()
        >>> tec = pyCart.tecplot.Tecscript(fname="tecploy.lay")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`pyCart.tecplot.Tecscript` or derivative
            Instance of Tecplot script base class
    :Versions:
        * 2015-02-26 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="aero.csh"):
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
