"""Interface for configuration control: :mod:`pyCart.options.Config`"""


# Import options-specific utilities
from util import rc0, odict

# Need 

# Class for PBS settings
class Config(odict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
    # Get configuration file name
    def get_ConfigFile(self):
        """Return the configuration file name
        
        :Call:
            >>> fname = opts.get_ConfigFile()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Configuration file name, usually ``'Config.xml'``
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        return self.get_key('ConfigFile',0)
        
    # Set configuration file name
    def set_ConfigFile(self, fname):
        """Set the configuration file name
        
        :Call:
            >>> opts.set_ConfigFile(fname)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fname*: :class:`str`
                Configuration file name, usually ``'Config.xml'``
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        self.set_key('ConfigFile', fname)
    
    
    # Get reference area for a given component.
    def get_RefArea(self, comp=None):
        """Return the global reference area or reference area of a component
        
        The component index can only be used if the 'RefArea' option is defined
        as a :class:`list`.  Similarly, the component name can only be used if
        the 'RefArea' option is a :class:`dict`.
        
        :Call:
            >>> A = opts.get_RefArea()
            >>> A = opts.get_RefArea(comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Outputs:
            *A*: :class:`float`
                Global reference area or reference area for a component.
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Get the specified value.
        RefA = self.get('RefArea')
        # Check the type.
        if type(RefA).__name__ == 'dict':
            # Check the component input.
            if (comp in RefA):
                # Return the specific component.
                A = RefA[comp]
            else:
                # Get the default.
                A = RefA.get('default', 1.0)
        elif type(RefA).__name__ == 'list':
            # Check the component input.
            if comp and (comp < len(RefA)):
                # Return the value by CompID.
                A = RefA[comp]
            else:
                # Return the first entry.
                A = RefA[0]
        else:
            # It's just a number.
            A = RefA
        # Output
        return A
        
    # Set the reference area for a given component.
    def set_RefArea(self, A, comp=None):
        """Set the global reference area or reference area of a component
        
        :Call:
            >>> opts.set_RefArea(A)
            >>> opts.set_RefArea(A, comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *A*: :class:`float`
                Global reference area or reference area for a component.
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Check the component input.
        if comp is None:
            # Global assignment.
            self['RefArea'] = A
        elif type(comp).__name__ == "int":
            # Set the index.
            self.set_key('RefArea', A, comp)
        else:
            # Get the current value.
            RefA = self.get('RefArea')
            # Make sure that the value is a dict.
            if RefA is None:
                # Initialize it.
                self['RefArea'] = {}
            elif type(RefA).__name__ != 'dict':
                # Use current value as default.
                self['RefArea'] = {"default": RefA}
            # Assign the specified value.
            self['RefArea'][comp] = A
            
            
    # Get reference length for a given component.
    def get_RefLength(self, comp=None):
        """Return the global reference length or that of a component
        
        The component index can only be used if the 'RefLength' option is
        defined as a :class:`list`.  Similarly, the component name can only be
        used if the 'RefLength' option is a :class:`dict`.
        
        :Call:
            >>> L = opts.get_RefLength()
            >>> L = opts.get_RefLength(comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Outputs:
            *L*: :class:`float`                        
                Global reference length or reference length for a component
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Get the specified value.
        RefL = self.get('RefLength')
        # Check the type.
        if type(RefL).__name__ == 'dict':
            # Check the component input.
            if (comp in RefL):
                # Return the specific component.
                L = RefL[comp]
            else:
                # Get the default.
                L = RefL.get('default', 1.0)
        elif type(RefL).__name__ == 'list':
            # Check the component input.
            if comp and (comp < len(RefL)):
                # Return the value by CompID.
                L = RefL[comp]
            else:
                # Return the first entry.
                L = RefL[0]
        else:
            # It's just a number.
            L = RefL
        # Output
        return L
        
    # Set the reference length for a given component.
    def set_RefLength(self, L, comp=None):
        """Set the global reference length or that of a component
        
        :Call:
            >>> opts.set_RefLength(L)
            >>> opts.set_RefLength(L, comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *L*: :class:`float`
                Global reference length or reference length for a component 
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Check the component input.
        if comp is None:
            # Global assignment.
            self['RefLength'] = L      
        elif type(comp).__name__ == "int":
            # Set the index.
            self.set_key('RefLength', L, comp)
        else:
            # Get the current value.
            RefL = self.get('RefLength')
            # Make sure that the value is a dict.
            if RefL is None:
                # Initialize it.
                self['RefLength'] = {}
            elif type(RefA).__name__ != 'dict':
                # Use current value as default.
                self['RefLength'] = {"default": RefL}
            # Assign the specified value.
            self['RefLength'][comp] = L
            
            
    # Get moment reference point for a given component.
    def get_RefPoint(self, comp=None):
        """Return the global moment reference point or that of a component
        
        The component index can only be used if the 'RefPoint' option is
        defined as a :class:`list`.  Similarly, the component name can only be
        used if the 'RefPoint' option is a :class:`dict`.
        
        :Call:
            >>> x = opts.get_RefPoint()
            >>> x = opts.get_RefPoint(comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Outputs:
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Global moment reference point or that for a component
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Get the specified value.
        RefP = self.get('RefPoint')
        # Check the type.
        if type(RefP).__name__ == 'dict':
            # Check the component input.
            if (comp in RefP):
                # Return the specific component.
                x = RefP[comp]
            elif comp == 'default':
                # Default value.
                x = RefP.get('default', [0.0, 0.0, 0.0])
            else:
                # Return the whole dict.
                x = RefP
        elif type(RefP[0]).__name__ == 'list':
            # Check the component input.
            if comp and (comp < len(RefP)):
                # Return the value by CompID.
                x = RefP[comp]
            else:
                # Return the first entry.
                x = Refp
        else:
            # It's just a number.
            x = RefP
        # Output
        return x
        
    # Set the reference length for a given component.
    def set_RefPoint(self, x, comp=None):
        """Set the global moment reference point or that of a component
        
        :Call:
            >>> opts.set_RefPoint(x)
            >>> opts.set_RefPoint(x, comp=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Global moment reference point or that for a component 
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Check the component input.
        if comp is None:
            # Global assignment.
            self['RefPoint'] = x      
        elif type(comp).__name__ == "int":
            # Set the index.
            self.set_key('RefPoint', x, comp)
        else:
            # Get the current value.
            RefP = self.get('RefPoint')
            # Make sure that the value is a dict.
            if RefP is None:
                # Initialize it.
                self['RefPoint'] = {}
            elif type(RefP).__name__ != 'dict':
                # Use current value as default.
                self['RefPoint'] = {"default": list(RefP)}
            # Assign the specified value.
            self['RefPoint'][comp] = list(x)
        
    
    # Get the list of forces to report
    def get_ClicForces(self, i=None):
        """Get force or list of forces requested to track using `clic`
        
        :Call:
            >>> comp = opts.get_ClicForces(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: ``None`` | :class:`int` | :class:`list`(:class:`int`)
                Index of component in list to return
        :Outputs:
            *comp*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components to track with `clic`
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Get the values.
        comp = self.get_key('Force', i)
        # Check to make sure it's a list.
        if (i is None) and (type(comp).__name__!="list"):
            # Create simple list.
            comp = [comp]
        # Output
        return comp
        
    # Set a clic force
    def set_ClicForces(self, comp='entire', i=None):
        """Set force or list of forces by index to track using `clic`
        
        :Call:
            >>> opts.set_ClicForces(comp, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components to track with `clic`
            *i*: ``None`` | :class:`int` | :class:`list`(:class:`int`)
                Index of component in list to return
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        self.set_key('Force', comp, i)
    
    # Add an additional clic force
    def add_ClicForce(self, comp):
        """Add a component to track using `clic`
        
        :Call:
            >>> opts.add_ClicForce(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Get the current list.
        comps = self.get_ClicForce()
        # Set the input value as an addendum to the list.
        self.set_key('Force', comp, len(comps))
    
    
    # Get cut plane extraction coordinate(s)
    def get_Xslices(self, i=None):
        """Return the list of Xslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.get_Xslices()
            >>> x = opts.get_Xslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *x*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the values.
        Xslices = self.get_key('Xslices', i)
        # Check for required list.
        if (i is None) and (type(Xslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Xslices = [Xslices]
        # Output
        return Xslices
        
    # Set cut plane extraction coordinate(s)
    def set_Xslices(self, x, i=None):
        """Return the list of Xslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.set_Xslices(x)
            >>> x = opts.set_Xslices(x, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *x*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Xslices', x, i)
        
    # Add an additional cut plane
    def add_Xslice(self, x):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Xslice(x)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *x*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Xslices = self.get_key('Xslices')
        # Set the input value as an addendum to the list.
        self.set_key('Xslices', x, len(Xslices))
        
        
    # Get cut plane extraction coordinate(s)
    def get_Yslices(self, i=None):
        """Return the list of Yslices for extracting solution cut planes
        
        :Call:
            >>> y = opts.get_Yslices()
            >>> y = opts.get_Yslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *y*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        Yslices = self.get_key('Yslices', i)
        # Check for required list.
        if (i is None) and (type(Yslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Yslices = [Yslices]
        # Output
        return Yslices
        
    # Set cut plane extraction coordinate(s)
    def set_Yslices(self, y, i=None):
        """Return the list of Yslices for extracting solution cut planes
        
        :Call:
            >>> y = opts.set_Yslices(x)
            >>> y = opts.set_Yslices(x, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *y*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Yslices', y, i)
        
    # Add an additional cut plane
    def add_Yslice(self, y):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Yslice(x)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *y*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Yslices = self.get_key('Yslices')
        # Set the input value as an addendum to the list.
        self.set_key('Yslices', y, len(Yslices))
        
        
    # Get cut plane extraction coordinate(s)
    def get_Zslices(self, i=None):
        """Return the list of Zslices for extracting solution cut planes
        
        :Call:
            >>> z = opts.get_Zslices()
            >>> z = opts.get_Zslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *z*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        Zslices = self.get_key('Zslices', i)
        # Check for required list.
        if (i is None) and (type(Zslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Zslices = [Zslices]
        # Output
        return Zslices
        
    # Set cut plane extraction coordinate(s)
    def set_Zslices(self, z, i=None):
        """Return the list of Zslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.set_Zslices(z)
            >>> x = opts.set_Zslices(z, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *z*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Zslices', z, i)
        
    # Add an additional cut plane
    def add_Zslice(self, z):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Zslice(z)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *z*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Zslices = self.get_key('Zslices')
        # Set the input value as an addendum to the list.
        self.set_key('Zslices', z, len(Zslices))
        
        
