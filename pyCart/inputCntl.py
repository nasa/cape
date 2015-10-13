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
from cape.fileCntl import FileCntl, _num, _float

# Base this class off of the main file control class.
class InputCntl(FileCntl):
    """
    File control class for :file:`input.cntl`
    =========================================
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> cntl = pyCart.InputCntl()
        >>> cntl = pyCart.InputCntl(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of CNTL file to read, defaults to ``'input.cntl'``
    :Version:
        * 2014-06-04 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="input.cntl"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\$__([\w_]+)")
        
    # Copy the file
    def Copy(self, fname):
        """Copy a file interface
        
        :Call:
            >>> IC2 = IC.Copy()
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
        :Outputs:
            *IC2*: :class:`pyCart.inputCntl.InputCntl`
                Duplicate file control instance for :file:`input.cntl`
        :Versions:
            * 2015-06-12 ``@ddalle``: First version
        """
        # Create empty instance.
        IC = InputCntl(fname=None)
        # Copy the file name.
        IC.fname = self.fname
        IC.lines = self.lines
        # Copy the sections
        IC.Section = self.Section
        IC.SectionNames = self.SectionNames
        # Update flags.
        IC._updated_sections = self._updated_sections
        IC._updated_lines = self._updated_lines
        # Output
        return IC
        
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
        :Versions:
            * 2014-06-17 ``@ddalle``: First version
        """
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
        :Versions:
            * 2014-06-17 ``@ddalle``: First version
        """
        # Name of the section
        sec = 'Solver_Control_Information'
        # Find the line that is sought.
        L = self.GetLineInSectionStartsWith(sec, 'RK', 1)
        # Form the new line.
        line = L[0].replace(' 0 ', ' 1 ')
        # Now write the updated line back.
        self.ReplaceLineInSectionStartsWith(sec, 'RK', [line])
        
    # Function to set to second-order mode
    def SetRobustMode(self):
        """
        Set gradient flag in all stages
        
        :Call:
            >>> IC.SetRobustMode()
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
        :Effects:
            Sets the gradient evaluation to ``1`` for each RK line
        :Versions:
            * 2014-11-21 ``@ddalle``: First version
        """
        # Name of the section
        sec = 'Solver_Control_Information'
        # Find the line that is sought.
        L = self.GetLineInSectionStartsWith(sec, 'RK')
        # Loop through lines.
        for line in L[1:]:
            # Form the new line.
            li = line.replace(' 0 ', ' 1 ')
            # Now write the updated line back.
            self.ReplaceLineInSectionStartsWith(sec, line, [li])
        
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
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
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
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
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
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Case_Information',
            'alpha ', 'alpha    %+12.8f   # angle of attack\n' % alpha)
        return None
        
    # Function to set the sideslip angle
    def SetBeta(self, beta):
        """Set the sideslip angle
        
        :Call:
            >>> IC.SetBeta(beta)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *beta*: :class:`float`
                Sideslip angle
        :Effects:
            Replaces or adds a line to the "Case_Information" section.
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Case_Information',
            'beta ', 'beta     %+12.8f   # sideslip angle\n' % beta)
        return None
        
    # Function to set the CFL number
    def SetCFL(self, CFL):
        """Set the CFL number
        
        :Call:
            >>> IC.SetCFL(CFL)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *CFL*: :class:`float`
                Value of the CFL number to use
        :Effects:
            Replaces or adds a line to the "Solver_Control_Information" section
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith('Solver_Control_Information',
            'CFL ', 'CFL%11s%s\n' % ('', CFL))
        return None
        
    # Function to set the number of orders of convergence to terminate at
    def SetNOrders(self, nOrders):
        """
        Set the early termination criterion in number of orders of magnitude
        decrease in the global L1 residual
        
        :Call:
            >>> IC.SetNOrders(nOrders)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *nOrders*: :class:`int`
                Number of orders of convergence at which to terminate early
        :Effects:
            Replaces a line in "Convergence_History_reporting"
        :Versions:
            * 2014-12-12 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionStartsWith(
            'Convergence_History_reporting',
            'nOrders ', 'nOrders   %2i\n' % nOrders)
        
        
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
            Replaces the current list of *x* cut planes with the input list.
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Initialize the output line.
        line = 'Xslices'
        # Add the cuts.
        for xi in x:
            # Append two spaces and the coordinate.
            line += "  %s" % xi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Xslices', line + '\n')
        
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
            Replaces the current list of *y* cut planes with the input list.
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Initialize the output line.
        line = 'Yslices'
        # Add the cuts.
        for yi in y:
            # Append two spaces and the coordinate.
            line += "  %s" % yi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Yslices', line + '\n')
        
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
            Replaces the current list of *z* cut planes with the input list.
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Initialize the output line.
        line = 'Zslices'
        # Add the cuts.
        for zi in z:
            # Append two spaces and the coordinate.
            line += "  %s" % zi
        # Write the line.
        self.ReplaceOrAddLineToSectionStartsWith('Post_Processing',
            'Zslices', line + '\n')
        
    # Function to write a line sensor
    def AddLineSensor(self, name, X):
        """Write a line sensor
        
        :Call:
            >>> IC.AddLineSensor(name, X)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *name*: :class:`str`
                Name of the line sensor
            *X*: :class:`list` (:class:`double`)
                List of start x,y,z and end x,y,z
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Check input length.
        if len(X) < 6:
            raise IOError(
                "Line sensor '%s' needs start x,y,z and end x,y,z" % X)
        # Initialize the output line.
        line = "lineSensor %s " % name
        # Add the start and end coordinates.
        for x in X[:6]:
            line += (" %s" % x)
        # Regular expression of existing line sensor to search for
        reg = 'lineSensor\s*%s' % name
        # Write the line
        self.ReplaceOrAddLineToSectionSearch('Post_Processing',
            reg, line + "\n")
        
    # Set list of line sensors
    def SetLineSensors(self, LS):
        """Write all line sensors
        
        :Call:
            >>> IC.SetLineSensors(LS)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *LS*: :class:`dict`
                Dictionary of line sensors
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Loop through line sensors.
        for name in LS:
            self.AddLineSensor(name, LS[name])
        
    # Function to write a point sensor
    def AddPointSensor(self, name, X):
        """Write a point sensor
        
        :Call:
            >>> IC.AddPointSensor(name, X)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *name*: :class:`str`
                Name of the line sensor
            *X*: :class:`list` (:class:`double`)
                List of point x,y,z coordinates
        :Versions:
            * 2015-05-07 ``@ddalle``: First version
        """
        # Check input length.
        if len(X) < 3:
            raise IOError(
                "Point sensor '%s' has less than three coordinates" % X)
        # Initialize the output line.
        line = "pointSensor %s " % name
        # Add the start and end coordinates.
        for x in X[:3]:
            line += (" %s" % x)
        # Regular expression of existing line sensor to search for
        reg = 'pointSensor\s*%s' % name
        # Write the line
        self.ReplaceOrAddLineToSectionSearch('Post_Processing',
            reg, line + "\n")
        
    # Set list of point sensors
    def SetPointSensors(self, PS):
        """Write all point sensors
        
        :Call:
            >>> IC.SetPointSensors(PS)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *PS*: :class:`dict`
                Dictionary of point sensors
        :Versions:
            * 2015-05-07 ``@ddalle``: First version
        """
        # Loop through line sensors.
        for name in PS:
            self.AddPointSensor(name, PS[name])
        
        
    # Function to set the reference area(s)
    def SetReferenceArea(self, A):
        """Set all moment reference points according to an input :class:`dict`
        
        :Call:
            >>> IC.SetReferenceArea(A)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *A*: :class:`dict`(:class:`float`) or :class:`float`
                Dictionary of reference areas by component or universal ARef
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Filter type.
        if type(A).__name__ == "dict":
            # Loop through the components.
            for ki in A:
                # Set the point for that component.
                self.SetSingleReferenceArea(A[ki], ki)
        else:
            # Just set it.
            self.SetSingleReferenceArea(A)
            
    # Function to set a single reference area
    def SetSingleReferenceArea(self, Aref, compID='all'):
        """
        Set the reference area in an :file:`input.cntl` file.
        
        :Call:
            >>> IC.SetSingleReferenceArea(Aref)
            >>> IC.SetSingleReferenceArea(Aref, compID)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Aref*: :class:`float`
                Reference area value
            *compID*: :class:`str`
                Component to which reference applies (default is ``'all'``)
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
            * 2014-10-08 ``@ddalle``: Demoted to "single"
        """
        # Regular expression for this line.
        reg = 'Reference_Area.*%s\s*$' % compID
        # Replace or add the line.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Reference_Area    %s   %s\n' % (Aref, compID))
    
    # Function to set the reference length(s)
    def SetReferenceLength(self, L):
        """Set all moment reference points according to an input :class:`dict`
        
        :Call:
            >>> IC.SetReferenceLength(L)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *L*: :class:`dict`(:class:`float`) or :class:`float`
                Dictionary of reference length by component or universal LRef
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Filter type.
        if type(L).__name__ == "dict":
            # Loop through the components.
            for ki in A:
                # Set the point for that component.
                self.SetSingleReferenceLength(L[ki], ki)
        else:
            # Just set it.
            self.SetSingleReferenceLength(L)
            
    # Function to set a single reference length
    def SetSingleReferenceLength(self, Lref, compID='all'):
        """Set the reference length in an :file:`input.cntl` file
        
        :Call:
            >>> IC.SetSingleReferenceLength(Lref)
            >>> IC.SetSingleReferenceLength(Lref, compID)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Lref*: :class:`float`
                Reference length value
            *compID*: :class:`str`
                Component to which reference applies (default is ``'all'``)
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
            * 2014-10-08 ``@ddalle``: Demoted to "single"
        """
        # Regular expression for this line.
        reg = 'Reference_Length.*%s\s*$' % compID
        # Replace or add the line.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Reference_Length  %s   %s\n' % (Lref, compID))
        
    # Function to set the moment reference point(s)
    def SetMomentPoint(self, xMRP):
        """Set all moment reference points according to an input :class:`dict`
        
        :Call:
            >>> IC.SetMomentPoint(xMRP)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *xMRP*: :class:`dict`(:class:`list`) or :class:`list`
                Dictionary of reference points by component or universal MRP
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Filter type.
        if type(xMRP).__name__ == "dict":
            # Loop through the components.
            for ki in xMRP:
                # Set the point for that component.
                self.SetSingleMomentPoint(xMRP[ki], ki)
        else:
            # Just set it.
            self.SetSingleMomentPoint(xMRP)
        
    # Function to a single moment reference point
    def SetSingleMomentPoint(self, x, compID='all'):
        """Set the moment reference point in an :file:`input.cntl` file
        
        :Call:
            >>> IC.SetSingleMomentPoint(x)
            >>> IC.SetSingleMomentPoint(x, compID)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *x*: :class:`list`(:class:`float`)
                List of three coordinates of moment reference point
            *compID*: :class:`str`
                Component to which reference applies (default is ``'all'``)
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
            * 2014-10-08 ``@ddalle``: Downgraded to "single" function
        """
        # Regular expression for this line.
        reg = 'Moment_Point.*%s\s*$' % compID
        # Form the output line.
        line = 'Moment_Point    '
        # Loop through entries of x.
        for xi in x:
            line += ("%s " % xi)
        # Add the component ID.
        line += ("  %s\n" % compID)
        # Replace or add the line.
        self.ReplaceOrAddLineToSectionSearch(
            'Force_Moment_Processing', reg, line)
        
    # Function to get a reference point
    def GetSingleMomentPoint(self, compID='all'):
        """Get the moment reference point of a component in :file:`input.cntl`
        
        :Call:
            >>> x = IC.GetSingleMomentPoint(compID)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *compID*: :class:`str`
                Component to which reference applies (default is ``'all'``)
        :Outputs:
            *x*: :class:`list` (:class:`float`)
                List of three coordinates of moment reference point
        :Versions:
            * 2015-03-02 ``@ddalle``: First version
        """
        # Regular expression to look for.
        reg = 'Moment_Point.*%s\s*$' % compID
        # Find the line.
        line = self.GetLineInSectionSearch('Force_Moment_Processing', reg, 1)
        # Check for a match.
        if len(line) == 0: return [0.0, 0.0, 0.0]
        # Split into values.
        v = line[0].split()
        # Try to process the coordinates.
        try:
            # Form the vector.
            x = [float(v[1]), float(v[2]), float(v[3])]
            # Output.
            return x
        except Exception:
            # Give up.
            return [0.0, 0.0, 0.0]
        
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
            *u*: :class:`numpy.ndarray`, *shape* = (5,) or ``None``
                Vector of density, velocity, pressure on surface
        :Effects:
            Writes a line with appropriate "SurfBC i ..." syntax to 
            :file:`input.cntl` file.
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Line starts with "SurfBC", has some amount of white space, and then
        # has the component number.
        reg = 'SurfBC\s+' + str(compID)
        # Create the output line.
        if u is None:
            # Turn off the BC; make it a commented line
            line = '# SurfBC %7i\n' % compID
        else:
            # Specify the full state.
            line = 'SurfBC %7i      %.8f %.8f %.8f %.8f %.8f\n' % (
                compID, u[0], u[1], u[2], u[3], u[4])
        # Replace the line or add it if necessary. The amount of white space can
        # vary, so we need to use regular expressions.
        self.ReplaceOrAddLineToSectionSearch('Boundary_Conditions', reg, line)
        
    # Function to set an output functional force
    def SetOutputForce(self, Name, **kwargs):
        """Request a force be added to the output functional
        
        :Call:
            >>> IC.SetOutputForce(Name, **kwargs)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Name*: :class:`str`
                Name of the force (required)
            *force*: :class:`int` [ {0} | 1 | 2 | None]
                Force axis, e.g. ``0`` for axial force. If ``Force=None``, this
                force is not used in the output.
            *frame*: :class:`int` [ {0} | 1 ]
                Body frame (``0``) or velocity frame (``1``)
            *weight*: :class:`float` [ {1.0} | :class:`float` ]
                Linear weight on term in overall functional
            *compID*: :class:`str` [ {entire} | :class:`str` | :class:`int` ]
                Component to use for calculating the force
            *J*: :class:`int` [ {0} | 1 ]
                Modifier of force, not normally used
            *N*: :class:`int` [ {1} | :class:`int` ]
                Exponent on force coefficient
            *target*: :class:`float` [ {0.0} | :class:`float` ]
                Target value for the functional; irrelevant if *N*\ =1
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        # Line looks like "optForce  CY_L 1 0 0 1 0. 1. 0  Core"
        reg = 'optForce\s+' + str(Name) + '\s'
        # Process the other inputs (with defaults)
        Force = kwargs.get('force', 0)
        Frame = kwargs.get('frame', 1)
        Weight = kwargs.get('weight', 1.0)
        CompID = kwargs.get('compID', 'entire')
        # Less likely inputs
        Target = kwargs.get('target', 0.0)
        J = kwargs.get('J', 0)
        N = kwargs.get('N', 1)
        # Form the line.
        if Force is None:
            # Use this to delete the line.
            line = '# optForce %12s' % Name
        else:
            # Full line
            line = 'optForce %12s %7s %7i %6i %6i %9s %8s   0   %s\n' % (
                Name, Force, Frame, J, N, Target, Weight, CompID)
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Design_Info', reg, line)
        
    # Function to set an output functional force
    def SetOutputMoment(self, Name, **kwargs):
        """Request a force be added to the output functional
        
        :Call:
            >>> IC.SetOutputMoment(Name, **kwargs)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Name*: :class:`str`
                Name of the force (required)
            *index*: :class:`int` [ {0} | :class:`int` ]
                Index of which MRP to use for named component
            *moment*: :class:`int` [ {0} | 1 | 2 | None]
                Force axis, e.g. ``0`` for axial force. If ``moment=None``, this
                component is not used in the output.
            *frame*: :class:`int` [ {0} | 1 ]
                Body frame (``0``) or velocity frame (``1``)
            *weight*: :class:`float` [ {1.0} | :class:`float` ]
                Linear weight on term in overall functional
            *compID*: :class:`str` [ {entire} | :class:`str` | :class:`int` ]
                Component to use for calculating the force
            *J*: :class:`int` [ {0} | 1 ]
                Modifier of force, not normally used
            *N*: :class:`int` [ {1} | :class:`int` ]
                Exponent on force coefficient
            *target*: :class:`float` [ {0.0} | :class:`float` ]
                Target value for the functional; irrelevant if *N*\ =1
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        # Line looks like "optForce  CY_L 1 0 0 1 0. 1. 0  Core"
        reg = 'optMoment_Point\s+' + str(Name) + '\s'
        # Process the other inputs (with defaults)
        Index = kwargs.get('index', 0)
        Force = kwargs.get('moment', 0)
        Frame = kwargs.get('frame', 1)
        Weight = kwargs.get('weight', 1.0)
        CompID = kwargs.get('compID', 'entire')
        # Less likely inputs
        Target = kwargs.get('target', 0.0)
        J = kwargs.get('J', 0)
        N = kwargs.get('N', 1)
        # Form the line.
        if Force is None:
            # Use this to delete the line.
            line = '# optMoment_Point %12s' % Name
        else:
            # Full line
            line = (
                'optMoment_Point %12s %7s %7s %7i %6i %6i %9s %8s   0   %s\n'
                % (Name, Index, Force, Frame, J, N, Target, Weight, CompID))
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Design_Info', reg, line)
        
    # Function to set an output functional line or point sensor
    def SetOutputSensor(self, Name, **kwargs):
        """Request a line or point sensor
        
        :Call:
            >>> IC.SetOutputSensor(Name, **kwargs)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *Name*: :class:`str`
                Name of the sensor (required)
            *J*: :class:`int` [ {0} | 1 ]
                Modifier of sensor, not normally used
            *N*: :class:`int` [ {2} | :class:`int` ]
                Exponent on sensor, usually 2 for line sensors
            *target*: :class:`float` [ {0.0} | :class:`float` ]
                Target value for the functional
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Line looks like "optSensor Line1 0 2 0.0 1.0 0"
        reg = 'optSensor\s+' + str(Name) + '\s'
        # Process the other inputs (with defaults)
        Weight = kwargs.get('weight', 1.0)
        Target = kwargs.get('target', 0.0)
        J = kwargs.get('J', 0)
        N = kwargs.get('N', 2)
        # Form the output line.
        if Weight == 0.0:
            # Use this to delete the line.
            line = '# optSensor %12s' % Name
        else:
            # Full line
            line = 'optSensor %12s %7i %6i %6s %9s   0\n' % (
                Name, J, N, Target, Weight)
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Design_Info', reg, line)
        
        
    # Function to get Cart3D to report the forces on several components
    def RequestForce(self, comps):
        """Request the force coefficients on a component or list of components
        
        :Call:
            >>> IC.RequestForce(comps)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *comps*: :class:`str` | :class:`int` | :class:`list`
                Name of component to log or ``"all"`` or ``"entire"``
        :Effects:
            Adds a line to :file:`input.cntl` that looks like "Force entire",
            if it is not already present for each entry in *comps*
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Check the input type.
        if type(comps).__name__ in ['list', 'array']:
            # Loop through entries.
            for compID in comps:
                # Ensure that the force is present.
                self.RequestSingleForce(compID)
        else:
            # Request the specified single component.
            self.RequestSingleForce(comps)
        
    # Function to get Cart3D to report the forces on a component
    def RequestSingleForce(self, compID):
        """Request the force coefficients on a particular component
        
        :Call:
            >>> IC.RequestSingleForce(compID)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *compID*: :class:`str` or :class:`int`
                Name of component to log or ``"all"`` or ``"entire"``
        :Effects:
            Adds a line to :file:`input.cntl` that looks like "Force entire",
            if it is not already present.
        :Versions:
            * 2014-06-09 ``@ddalle``: First version
            * 2014-12-08 ``@ddalle``: Renamed from `RequestForce`
        """
        # Line looks like "Force $compID", but arbitrary white space.
        reg = 'Force\s+' + str(compID) + '$'
        # Replace the line or add it if necessary.
        self.ReplaceOrAddLineToSectionSearch('Force_Moment_Processing',
            reg, 'Force %s\n' % compID)
        return None
        
    # Function to get Cart3D to report the moments on a component
    def RequestMoment(self, compID, MRP=None):
        """Request the moment coefficients on a particular component.
        
        :Call:
            >>> IC.RequestMoment(compID, MRP)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *compID*: :class:`str` or :class:`int`
                Name of component to log or ``"all"`` or ``"entire"``
            *MRP*: *array_like*
                Reference point (defaults to ``[0,0,0]``)
        :Effects:
            Adds a line to :file:`input.cntl` that tells Cart3D to calculate the
            moment coefficients using a specific reference point.
        :Versions:
            * 2014-06-09 ``@ddalle``: First version
        """
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
        
    # Function to set Runge-Kutta inputs
    def SetRungeKutta(self, RK):
        """Set the Runge-Kutta time step coefficients
        
        The input can be a list of lists or a string or ``None``.  If it's a
        string, the the function will attempt to use one of the following known
        sets of Runge-Kutta inputs.  The first column is the stage coefficient,
        and the second column is whether or not to use a gradient evaluation in
        that stage.
        
            * ``'van Leer 5-stage' | 'VL5' | 2 | '2' | 'default'``
                *RK* = [
                    [0.0695, 1],
                    [0.1602, 0],
                    [0.2898, 0],
                    [0.5060, 0],
                    [1.0,    0]]
            * ``'first-order' | 1 | '1'``
                *RK* = [
                    [0.0695, 0],
                    [0.1602, 0],
                    [0.2898, 0],
                    [0.5060, 0],
                    [1.0,    0]]
            * ``'robust'``
                *RK* = [
                    [0.0695, 1],
                    [0.1602, 1],
                    [0.2898, 1],
                    [0.5060, 1],
                    [1.0,    1]]
            * ``'VL3-1'``
                *RK* = [
                    [0.1481, 1],
                    [0.4,    0],
                    [1.0,    0]]
            * ``'van Leer 3-stage' | 'VL3-2' | 'VL3' ``
                *RK* = [
                    [0.1918, 1],
                    [0.4929, 0],
                    [1.0,    0]]
            * ``'van Leer 4-stage' | 'VL4'``
                *RK* = [
                    [0.1084, 1], 
                    [0.2602, 1],
                    [0.5052, 1],
                    [1.0,    0]]
        
        :Call:
            >>> IC.SetRungeKutta(RK)
        :Inputs:
            *IC*: :class:`pyCart.inputCntl.InputCntl`
                File control instance for :file:`input.cntl`
            *RK*: :class:`str` or :class:`list` ([:class:`float`,:class:`int`])
                Named Runge-Kutta scheme or list of coefficients and gradient
                evaluation flags
        :Effects:
            Deletes current lines beginning with ``RK`` in the
            `Solver_Control_Information` section and replaces them with the
            specified values
        :Versions:
            * 2014-12-17 ``@ddalle``: First version
        """
        # Check for recognized inputs.
        if (RK is None) or (RK=='None'):
            # Do nothing
            return
        elif type(RK).__name__ not in ['list', 'ndarray']:
            # Not a list; check for recognition
            if RK in ['van Leer 5-stage', 'VL5', 2, '2', 'default']:
                # Default inputs.
                RK = [[0.0695, 1], [0.1602, 0],
                    [0.2898, 0], [0.5060, 0], [1.0, 0]]
            elif RK in ['van Leer 4-stage', 'VL4']:
                # 4-stage inputs
                RK = [[0.1084, 1], [0.2602, 1], [0.5052, 1], [1.0, 0]]
            elif RK in ['van Leer 3-stage', 'VL3', 'VL3-2']:
                # 3-stage inputs
                RK = [[0.1918, 1], [0.4929, 0], [1.0,    0]]
            elif RK in ['VL3-1']:
                # 3-stage cheapo
                RK = [[0.1481, 1], [0.4, 0], [1.0, 0]]
            elif RK in ['first-order']:
                # VL5 with no grad evals
                RK = [[0.0695, 0], [0.1602, 0],
                    [0.2898, 0], [0.5060, 0], [1.0, 0]]
            elif RK in ['robust']:
                # VL5 with all grad evals
                RK = [[0.0695, 1], [0.1602, 1],
                    [0.2898, 1], [0.5060, 1], [1.0, 1]]
            else:
                # Oh no!
                raise IOError("Runge-Kutta scheme '%s' not recognized." % RK)
        # Name of the relevant section
        sec = 'Solver_Control_Information'
        # Clear the lines.
        self.DeleteLineInSectionStartsWith(sec, 'RK', count=None)
        # Add the new ones. (Loop through stages backwards)
        for RKi in RK[::-1]:
            # Create the line.
            line = 'RK       %7.4f    %i\n' % (RKi[0], RKi[1])
            # Add it to front of section.
            self.PrependLineToSection(sec, line) 
    
    
    
