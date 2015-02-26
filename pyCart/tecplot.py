"""
Module to interface with Tecplot scripts: :mod:`pyCart.tecplot`
======================================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating Tecplot layout files and macros.
"""

# Import the base file control class.
from fileCntl import FileCntl, _num, _float

# Numerics
inmport numpy as np

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
        *tec*: :class:`pyCart.tecplot.Tecscript`
            Instance of Tecplot script
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="aero.csh"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        return None
