"""Interface for Cart3D meshing settings"""


# Import options-specific utilities
from util import rc0, odict


# Class for autoInputs
class autoInputs(odict):
    """Dictionary-based interface for `autoInputs` options"""
    
    # Get the nominal mesh radius
    def get_r(self, i=None):
        """Get the nominal mesh radius
        
        :Call:
            >>> r = opts.get_r(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *r*: :class:`float` or :class:`list`(:class:`float`)
                Nominal mesh radius
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('r', i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        """Set the nominal mesh radius
        
        :Call:
            >>> opts.set_r(r, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *r*: :class:`float` or :class:`list`(:class:`float`)
                Nominal mesh radius
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('r', r, i)
        
        
# Class for cubes
class cubes(odict):
    """Dictionary-based interface for `cubes` options"""
    
    # Get the maximum number of refinements
    def get_maxR(self, i=None):
        """Get the number of refinements
        
        :Call:
            >>> maxR = opts.get_maxR(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *maxR*: :class:`int` or :class:`list`(:class:`int`)
                (Maximum) number of refinements for initial mesh
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('maxR', i)
        
    # Set the maximum number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        """Get the number of refinements
        
        :Call:
            >>> opts.set_maxR(maxR, i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *maxR*: :class:`int` or :class:`list`(:class:`int`)
                (Maximum) number of refinements for initial mesh
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('maxR', maxR, i)
        
    # Set the mesh prespecification file
    def get_pre(self, i=None):
        """Get the mesh prespecification input file
        
        :Call:
            >>> pre = opts.get_pre(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *pre*: :class:`str` or :class:`list`(:class:`str`)
                Mesh prespecification file name
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('pre', i)
    
    # Set the mesh prespecification file
    def set_pre(self, pre=rc0('pre'), i=None):
        """Set the mesh prespecification input file
        
        :Call:
            >>> opts.set_pre(pre, i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *pre*: :class:`str` or :class:`list`(:class:`str`)
                Mesh prespecification file name
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('pre', pre, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        """Get the "cubes_a" parameter
        
        :Call:
            >>> cubes_a = opts.get_cubes_a(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cubes_a*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('cubes_a', i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        """Set the "cubes_a" parameter
        
        :Call:
            >>> opts.set_cubes_a(cubes_a, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cubes_a*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('cubes_a', cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        """Get the "cubes_b" parameter
        
        :Call:
            >>> cubes_b = opts.get_cubes_b(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cubes_b*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('cubes_b', i)
        
    # Set the 'cubes_b' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        """Set the "cubes_b" parameter
        
        :Call:
            >>> opts.set_cubes_b(cubes_b, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cubes_b*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('cubes_b', cubes_b, i)
        
    # Get the reorder setting
    def get_reorder(self, i=None):
        """Get the `cubes` reordering status
        
        :Call:
            >>> reorder = opts.get_reorder(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *reorder*: :class:`bool` or :class:`list`(:class:`bool`)
                Reorder status
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('reorder', i)
        
    # Set the reorder setting
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        """Set the `cubes` reordering status
        
        :Call:
            opts.set_reorder(reorder, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *reorder*: :class:`bool` or :class:`list`(:class:`bool`)
                Reorder status
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('reorder', reorder, i)
        

# Class for Cart3D mesh settings
class Mesh(odict):
    """Dictionary-based interfaced for options for Cart3D meshing"""
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._autoInputs()
    
    # Initialization and confirmation for autoInputs options
    def _autoInputs(self):
        """Initialize `autoInputs` options if necessary"""
        # Check for missing entirely.
        if 'autoInputs' not in self:
            # Empty/default
            self['autoInputs'] = autoInputs()
        elif type(self['autoInputs']).__name__ == 'dict':
            # Convert to special class.
            self['autoInputs'] = autoInputs(**self['autoInputs'])
    
    # Initialization and confirmation for cubes options
    def _cubes(self):
        """Initialize `cubes` options if necessary"""
        # Check for missing entirely.
        if 'cubes' not in self:
            # Empty/default
            self['cubes'] = cubes()
        elif type(self['cubes']).__name__ == 'dict':
            # Convert to special class.
            self['cubes'] = cubes(**self['cubes'])
            
    
    # Get the triangulation file(s)
    def get_TriFile(self, i=None):
        """Return the surface triangulation file
        
        :Call:
            >>> TriFile = opts.get_TriFile(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *TriFile*: :class:`str` or :class:`list`(:class:`str`)
                Surface triangulation file
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('TriFile', i)
        
    # Set the triangulation file(s)
    def set_TriFile(self, TriFile=rc0('TriFile'), i=None):
        """Set the surface triangulation file(s)
        
        :Call:
            >>> opts.set_n_adaptation_cycles(nAdapt, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *TriFile*: :class:`str` or :class:`list`(:class:`str`)
                Surface triangulation file
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('TriFile', TriFile, i)
        
        
    # Get the mesh 2D status
    def get_mesh2d(self, i=None):
        """Get the two-dimensional mesh status
        
        :Call:
            >>> mesh2d = opts.get_mesh2d(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mesh2d*: :class:`bool` or :class:`list`(:class:`bool`)
                Two-dimensional if ``True``, three-dimensional otherwise
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        return self.get_key('mesh2d', i)
    
    # Set the mesh 2D status
    def set_mesh2d(self, mesh2d=rc0('mesh2d'), i=None):
        """Set the two-dimensional mesh status
        
        :Call:
            >>> opts.set_mesh2d(mesh2d, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mesh2d*: :class:`bool` or :class:`list`(:class:`bool`)
                Two-dimensional if ``True``, three-dimensional otherwise
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.03 ``@ddalle``: First version
        """
        self.set_key('mesh2d', mesh2d, i)
    
    
    # ==========
    # autoInputs
    # ==========
    
    # Get the nominal mesh radius
    def get_r(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_r(i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        self._autoInputs()
        self['autoInputs'].set_r(r, i)
        
    # Copy over the documentation.
    for k in ['r']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(autoInputs,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(autoInputs,'set_'+k).__doc__
        
    
    # =====
    # cubes
    # =====
    
    # Get the number of refinements
    def get_maxR(self, i=None):
        self._cubes()
        return self['cubes'].get_maxR(i)
        
    # Set the number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        self._cubes()
        self['cubes'].set_maxR(maxR, i)
        
    # Get the prespecification file
    def get_pre(self, i=None):
        self._cubes()
        return self['cubes'].get_pre(i)
        
    # Set the prespecification file
    def set_pre(self, pre=rc0('pre'), i=None):
        self._cubes()
        self['cubes'].set_pre(pre, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_a(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        self._cubes()
        self['cubes'].set_cubes_a(cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_b(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        self._cubes()
        self['cubes'].set_cubes_b(cubes_b, i)
        
    # Get the mesh reordering status
    def get_reorder(self, i=None):
        self._cubes()
        return self['cubes'].get_reorder(i)
        
    # Set the mesh reordering status
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        self._cubes()
        self['cubes'].set_reorder(reorder, i)


    # Copy over the documentation.
    for k in ['maxR', 'pre', 'cubes_a', 'cubes_b', 'reorder']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(cubes,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(cubes,'set_'+k).__doc__



