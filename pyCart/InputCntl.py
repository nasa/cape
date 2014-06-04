"""
Module to Interface with 'input.cntl' Files
===========================================

"""

# Import the base file control class.
from FileCntl import FileCntl

# Base this class off of the main file control class.
class InputCntl(FileCntl):
    """
    File control class for "input.cntl" files.
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="input.cntl"):
        """
        File control class for "input.cntl" files.
        
        :Call:
            >>> cntl = pyCart.InputCntl.InputCntl()
            >>> cntl = pyCart.InputCntl.InputCntl(fname)
            
        :Inputs:
            *fname*: :class:`str`
                Name of CNTL file to read, defaults to ``'input.cntl'``
        """
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\$__([\w_]+)")
        return None
        
    # 
