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
        
    # Function to set a surface boundary condition (e.g. nozzle condition)
    def SetSurfBC(self, compID, u):
        """
        Set a surface boundary condition, for example on a nozzle surface
        
        :Call:
            >>> IC.SetSurfBC(compID, u)
        
        :Inputs:
            *IC*: :class:`pyCart.InputCntl.InputCntl`
                File control instance for "input.cntl"
            *compID*: :class:`int`
                Component number to apply boundary condition to
            *u*: :class:`numpy.ndarray`, *shape*=(5,)
                Vector of density, velocity, pressure on surface
        
        :Effects:
            Writes a line with appropriate "SurfBC i ..." syntax to "input.cntl"
            file.
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Line starts with "SurfBC", has some amount of white space, and then
        # has the component number.
        reg = 'SurfBC\s+' + str(compID)
        # Create the output line.
        line = 'SurfBC %7i      %.8f %.8f %.8f %.8f %.8f' % (
            compID, u[0], u[1], u[2], u[3], u[4])
        # Replace the line or add it if necessary. The amount of white space can
        # vary, so we need to use regular expressions.
        self.ReplaceOrAddLineToSectionSearch('Boundary_Conditions', reg, line)
        return None
        
    # Function to get Cart3D to report the forces on a component
    def RequestForce(self, compID):
        """
        Request the force coefficients on a particular component.
        
        :Call:
            >>> IC.RequestForce(compID)
            
        :Inputs:
            *compID*: :class:`str` or :class:`int`
                Name of component to log or ``"all"`` or ``"entire"``
        
        :Effects:
            Adds a line to 'input.cntl' that looks like "Force entire", if it
            is not already present.
        """
        # Versions:
        #  2014.06.09 @ddalle  : First version
        
        # Line starts looks like "Force $compID", but arbitrary white space.
        reg = 'Force\s+' + str(compID)
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Force ' + str(compID))
        return None
        
    # Function to get Cart3D to report the moments on a component
    def RequestMoment(self, compID, MRP=None):
        """
        Request the moment coefficients on a particular component.
        
        :Call:
            >>> IC.RequestMoment(compID, MRP)
            
        :Inputs:
            *compID*: :class:`str` or :class:`int`
                Name of component to log or ``"all"`` or ``"entire"``
            *MRP*: *array_like*
                Reference point (defaults to ``[0,0,0]``)
        
        :Effects:
            Adds a line to 'input.cntl' that tells Cart3D to calculate the
            moment coefficients using a specific reference point.
        """
        # Versions:
        #  2014.06.09 @ddalle  : First version
        
        # Process reference point.
        if MRP is None:
            # Default reference points.
            x = 0.0
            y = 0.0
            z = 0.0
        else:
            # Get values from input.
            x = MRP[0]
            y = MRP[1]
            z = MRP[2]
        # Regular expression for "Moment_Point[anything]$comp_ID"
        reg = 'Moment_Point.*' + str(compID)
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing', reg,
            'Moment_Point  %s %s %s  %s' % (x,y,z,compID))
    
    
    
    
    
    
    
    
    
