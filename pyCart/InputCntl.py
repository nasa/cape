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
        
    # Function set the Mach number.
    def SetMach(self, Mach):
        """
        Set the freestream Mach number
        
        :Call:
            >>> IC.SetMach(Mach)
        
        :Inputs:
            *IC*: :class:`pyCart.InputCntl.InputCntl`
                File control instance for "input.cntl"
            *Mach*: :class:`float`
                Mach number
            
        :Effects:
            Replaces or adds a line to the "Case_Information" section.
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Case_Information',
            'Mach ', 'Mach     %12.8f   # Mach number\n' % Mach)
        return None
        
    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        """
        Set the angle of attack
        
        :Call:
            >>> IC.SetAlpha(alpha)
            
        :Inputs:
            *IC*: :class:`pyCart.InputCntl.InputCntl`
                File control instance for "input.cntl"
            *alpha*: :class:`float`
                Angle of attack
                
        :Effects:
            Replaces or adds a line to the "Case_Information" section.
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Case_Information',
            'alpha ', 'alpha    %+12.8f   # angle of attack\n' % alpha)
        return None
        
    # Function to set the sideslip angle
    def SetBeta(self, beta):
        """
        Set the sideslip angle
        
        :Call:
            >>> IC.SetAlpha(alpha)
            
        :Inputs:
            *IC*: :class:`pyCart.InputCntl.InputCntl`
                File control instance for "input.cntl"
            *beta*: :class:`float`
                Sideslip angle
                
        :Effects:
            Replaces or adds a line to the "Case_Information" section.
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Case_Information',
            'beta ', 'beta     %+12.8f   # sideslip angle\n' % beta)
        return None
        
        
    
    
    
    
    
    
    
    
    
