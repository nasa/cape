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
            * 2014.09.29 ``@ddalle``: First version
        """
        return self.get_key('ConfigFile',0)
        
    # Set configuration file name
    def set_ConfigFile(self, fname):
        """Set the configuration file name
        
        :Call:
            opts.set_ConfigFile(fname)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fname*: :class:`str`
                Configuration file name, usually ``'Config.xml'``
        :Versions:
            * 2014.09.29 ``@ddalle``: First version
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
            * 2014.09.29 ``@ddalle``: First version
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
            * 2014.09.29 ``@ddalle``: First version
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
            * 2014.09.29 ``@ddalle``: First version
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
            * 2014.09.29 ``@ddalle``: First version
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
            * 2014.09.29 ``@ddalle``: First version
        """
        # Get the specified value.
        RefP = self.get('RefPoint')
        # Check the type.
        if type(RefP).__name__ == 'dict':
            # Check the component input.
            if (comp in RefP):
                # Return the specific component.
                x = RefP[comp]
            else:
                # Get the default.
                x = RefP.get('default', [0.0, 0.0, 0.0])
        elif type(RefP[0]).__name__ == 'list':
            # Check the component input.
            if comp and (comp < len(RefP)):
                # Return the value by CompID.
                x = RefP[comp]
            else:
                # Return the first entry.
                x = Refp[0]
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
            * 2014.09.29 ``@ddalle``: First version
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
                self['RefPoint'] = {"default": RefP}
            # Assign the specified value.
            self['RefPoint'][comp] = x
            
        
        
