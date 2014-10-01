"""
Module to interface with "input.cntl" files: :mod:`pyCart.inputCntl`
====================================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating :file:`input.cntl` files.  Such files are split into section by lines of
the format

    ``$__Post_Processing``
    
and this module is designed to recognize such sections.  The main feature of
this module is methods to set specific properties of the :file:`input.cntl` 
file, for example the Mach number or CFL number.
"""

# Import the base file control class.
from fileCntl import FileCntl, _num, _float

# Base this class off of the main file control class.
class InputCntl(FileCntl):
    """
    File control class for :file:`input.cntl` files
    
    :Call:
        >>> cntl = pyCart.InputCntl()
        >>> cntl = pyCart.InputCntl(fname)
        
    :Inputs:
        *fname*: :class:`str`
            Name of CNTL file to read, defaults to ``'input.cntl'``
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="input.cntl"):
        """Initialization method"""
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\$__([\w_]+)")
        return None
        
    # Function to set to first-order mode
    def SetFirstOrder(self):
        """
        Set the solver to first-order mode
        
        :Call:
            >>> IC.SetFirstOrder()
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            
        :Effects:
            Sets the gradient evaluation to ``0`` for the first RK line
        """
        # Versions:
        #  2014.06.17 @ddalle  : First version
        
        # Name of the section
        sec = 'Solver_Control_Information'
        # Find the line that is sought.
        L = self.GetLineInSectionStartsWith(sec, 'RK', 1)
        # Form the new line.
        line = L[0].replace(' 1 ', ' 0 ')
        # Now write the updated line back.
        self.ReplaceLineInSectionStartsWith(sec, 'RK', [line])
        
    # Function to set to second-order mode
    def SetSecondOrder(self):
        """
        Set the solver to second-order mode
        
        :Call:
            >>> IC.SetSecondOrder()
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            
        :Effects:
            Sets the gradient evaluation to ``1`` for the first RK line
        """
        # Versions:
        #  2014.06.17 @ddalle  : First version
        
        # Name of the section
        sec = 'Solver_Control_Information'
        # Find the line that is sought.
        L = self.GetLineInSectionStartsWith(sec, 'RK', 1)
        # Form the new line.
        line = L[0].replace(' 0 ', ' 1 ')
        # Now write the updated line back.
        self.ReplaceLineInSectionStartsWith(sec, 'RK', [line])
        
    # Function set the Mach number.
    def SetMach(self, Mach):
        """
        Set the freestream Mach number
        
        :Call:
            >>> IC.SetMach(Mach)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
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
        
    # Function to get the current Mach number.
    def GetMach(self):
        """
        Find the current Mach number
        
        :Call:
            >>> M = IC.GetMach()
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
                
        :Outputs:
            *M*: :class:`float` (or :class:`str`)
                Mach number specified in :file:`input.cntl`
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Find the line.
        lines = self.GetLineInSectionStartsWith('Case_Information', 'Mach', 1)
        # Convert.
        vals = lines[0].split()
        # Check for a match.
        if len(vals)==0 or len(vals[0])<2:
            # Not enough info.
            return ''
        else:
            # Attempt to convert the string.
            return _float(vals[1])
        
    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        """
        Set the angle of attack
        
        :Call:
            >>> IC.SetAlpha(alpha)
            
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
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
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
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
        
    # Function to set the CFL number
    def SetCFL(self, CFL):
        """
        Set the CFL number
        
        :Call:
            >>> IC.SetCFL(CFL)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *CFL*: :class:`float`
                Value of the CFL number to use
                
        :Effects:
            Replaces or adds a line to the "Solver_Control_Information" section
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Solver_Control_Information',
            'CFL ', 'CFL%11s%s\n' % ('', CFL))
        return None
        
        
    # Function to set the list of x-slices
    def SetXSlices(self, x):
        """
        Set the list of *x*-coordinates at which to form cut planes
        
        :Call:
            >>> IC.SetXSlices(x)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *x*: *array_like* (:class:`float`)
                List or vector of *x*-coordinates at which to make cut planes
            
        :Effects:
            Replaces the current list of *x* cut planes with the specified list.
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Initialize the output line.
        line = 'Xslices'
        # Add the cuts.
        for xi in x:
            # Append two spaces and the coordinate.
            line += "  %s" % xi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Xslices', line + '\n')
        return None
        
    # Function to set the list of x-slices
    def SetYSlices(self, y):
        """
        Set the list of *x*-coordinates at which to form cut planes
        
        :Call:
            >>> IC.SetYSlices(y)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *y*: *array_like* (:class:`float`)
                List or vector of *y*-coordinates at which to make cut planes
            
        :Effects:
            Replaces the current list of *y* cut planes with the specified list.
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Initialize the output line.
        line = 'Yslices'
        # Add the cuts.
        for yi in y:
            # Append two spaces and the coordinate.
            line += "  %s" % yi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Yslices', line + '\n')
        return None
        
    # Function to set the list of x-slices
    def SetZSlices(self, z):
        """
        Set the list of *x*-coordinates at which to form cut planes
        
        :Call:
            >>> IC.SetZSlices(z)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *z*: *array_like* (:class:`float`)
                List or vector of *z*-coordinates at which to make cut planes
            
        :Effects:
            Replaces the current list of *z* cut planes with the specified list.
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Initialize the output line.
        line = 'Zslices'
        # Add the cuts.
        for zi in z:
            # Append two spaces and the coordinate.
            line += "  %s" % zi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Zslices', line + '\n')
        return None
        
    # Function to set the reference area
    def SetReferenceArea(self, Aref, compID='all'):
        """
        Set the reference area in an :file:`input.cntl` file.
        
        :Call:
            >>> IC.SetReferenceArea(Aref)
            >>> IC.SetReferenceArea(Aref, compID)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Aref*: :class:`float`
                Reference area value
            *compID*: :class:`str`
                Component to which reference area applies (default is ``'all'``)
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Regular expression for this line.
        reg = 'Reference_Area.*%s' % compID
        # Replace or add the line.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Reference_Area    %s   %s\n' % (Aref, compID))
        return None
        
    # Function to set the reference area
    def SetReferenceLength(self, Lref, compID='all'):
        """
        Set the reference length in an :file:`input.cntl` file.
        
        :Call:
            >>> IC.SetReferenceLength(Lref)
            >>> IC.SetReferenceLength(Lref, compID)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Lref*: :class:`float`
                Reference length value
            *compID*: :class:`str`
                Component to which reference area applies (default is ``'all'``)
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Regular expression for this line.
        reg = 'Reference_Length.*%s' % compID
        # Replace or add the line.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Reference_Length  %s   %s\n' % (Lref, compID))
        return None
        
    # Function to set a surface boundary condition (e.g. nozzle condition)
    def SetSurfBC(self, compID, u):
        """
        Set a surface boundary condition, for example on a nozzle surface
        
        :Call:
            >>> IC.SetSurfBC(compID, u)
        
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *compID*: :class:`int`
                Component number to apply boundary condition to
            *u*: :class:`numpy.ndarray`, *shape* = (5,)
                Vector of density, velocity, pressure on surface
        
        :Effects:
            Writes a line with appropriate "SurfBC i ..." syntax to 
            :file:`input.cntl` file.
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # Line starts with "SurfBC", has some amount of white space, and then
        # has the component number.
        reg = 'SurfBC\s+' + str(compID)
        # Create the output line.
        line = 'SurfBC %7i      %.8f %.8f %.8f %.8f %.8f\n' % (
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
            Adds a line to :file:`input.cntl` that looks like "Force entire",
            if it is not already present.
        """
        # Versions:
        #  2014.06.09 @ddalle  : First version
        
        # Line starts looks like "Force $compID", but arbitrary white space.
        reg = 'Force\s+' + str(compID) + '$'
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Force %s\n' % compID)
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
            Adds a line to :file:`input.cntl` that tells Cart3D to calculate the
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
        reg = 'Moment_Point.*' + str(compID) + '$'
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing', reg,
            'Moment_Point  %s %s %s  %s\n' % (x,y,z,compID))
    
    
    
    
    
    
    
    
    
