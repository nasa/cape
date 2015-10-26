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
            >>> r = opts.get_r(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *r*: :class:`float` or :class:`list`(:class:`float`)
                Nominal mesh radius
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('r', r, i)
        
    # Get the number of initial divisions
    def get_nDiv(self, i=None):
        """Get the number of divisions in background mesh
        
        :Call:
            >>> nDiv = opts.get_nDiv(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nDiv*: :class:`int` or :class:`list`(:class:`int`)
                Number of background mesh divisions
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        return self.get_key('nDiv', i)
        
    # Set the number of initial mesh divisions
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        """Set the number of divisions in background mesh
        
        :Call:
            >>> opts.set_nDiv(nDiv, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nDiv*: :class:`int` or :class:`list`(:class:`int`)
                Number of background mesh divisions
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        self.set_key('nDiv', nDiv, i)
# class autoInputs

        
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('maxR', maxR, i)
        
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('reorder', i)
        
    # Set the reorder setting
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        """Set the `cubes` reordering status
        
        :Call:
            >>> opts.set_reorder(reorder, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *reorder*: :class:`bool` or :class:`list`(:class:`bool`)
                Reorder status
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('reorder', reorder, i)
        
    # Get the number of initial refinements at sharp edges
    def get_sf(self, i=None):
        """Get the number of additional refinements around sharp edges
        
        :Call:
            >>> sf = opts.get_sf(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *sf*: :class:`int` or :class:`list`(:class:`int`)
                Number of additional refinements at sharp edges
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        return self.get_key('sf', i)
        
    # Set the number of additional refinements at sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        """Set the number of additional refinements around sharp edges
        
        :Call:
            >>> opts.set_sf(sf, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sf*: :class:`int` or :class:`list` (:class:`int`)
                Number of additional refinements at sharp edges
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        self.set_key('sf', sf, i)
# class cubes
        

# Class for Cart3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for options for Cart3D meshing"""
    
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
            
    
    # Get verify status
    def get_verify(self):
        """Determine whether or not to call `verify` before running `cubes`.
        
        :Call:
            >>> q = opts.get_verify()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *q*: :class:`bool`
                Whether or not to run `verify`
        :Versions:
            * 2015-02-13 ``@ddalle``: First version
        """
        return self.get_key('verify')
        
    # Set verify status
    def set_verify(self, q=rc0('verify')):
        """Set whether or not to call `verify` before running `cubes`.
        
        :Call:
            >>> opts.get_verify(q)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *q*: :class:`bool`
                Whether or not to run `verify`
        :Versions:
            * 2015-02-13 ``@ddalle``: First version
        """
        return self.get_key('verify')
            
    
    # Get intersect status
    def get_intersect(self):
        """Determine whether or not to call `intersect` before running `cubes`.
        
        :Call:
            >>> q = opts.get_intersect()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *q*: :class:`bool`
                Whether or not to run `intersect`
        :Versions:
            * 2015-02-13 ``@ddalle``: First version
        """
        return self.get_key('intersect')
        
    # Set verify status
    def set_intersect(self, q=rc0('intersect')):
        """Set whether or not to call `intersect` before running `cubes`.
        
        :Call:
            >>> opts.set_intersect(q)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *q*: :class:`bool`
                Whether or not to run `intersect`
        :Versions:
            * 2015-02-13 ``@ddalle``: First version
        """
        return self.get_key('intersect')
        
        
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('TriFile', TriFile, i)
            
    
    # Get the mesh prespecification file
    def get_preSpecCntl(self):
        """Return the template :file:`preSpec.c3d.cntl` file
        
        :Call:
            >>> fpre = opts.get_preSpecCntl(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fpre*: :class:`str`
                Mesh prespecification file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the value
        self._cubes()
        fpre = self['cubes'].get_key('pre')
        # Check for ``None``
        if fpre is None:
            # Use default
            return rc0('preSpecCntl')
        else:
            # Specified value
            return fpre

        
    # Set the mesh prespecification file
    def set_preSpecCntl(self, fpre=rc0('preSpecCntl')):
        """Set the template :file:`preSpec.c3d.cntl` file
        
        :Call:
            >>> opts.set_preSpecCntl(fpre)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fpre*: :class:`str`
                Mesh prespecification file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self._cubes()
        self['cubes'].set_key('pre', fpre)
        
        
    # Get the mesh input file
    def get_inputC3d(self):
        """Return the template :file:`input.c3d` file
        
        :Call:
            >>> fpre = opts.get_inputC3d(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fpre*: :class:`str`
                Mesh prespecification file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get value
        fc3d = self.get_key('inputC3d')
        # Check for default.
        if fc3d is None:
            # Code default
            return rc0('inputC3d')
        else:
            # Specified value
            return fc3d
        
    # Set the mesh input file
    def set_inputC3d(self, fc3d=rc0('inputC3d')):
        """Set the template :file:`input.c3d` file
        
        :Call:
            >>> opts.set_inputC3d(fpre)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fc3d*: :class:`str`
                Mesh input file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('inputC3d', fc3d)
        
        
    # Return the list of bounding boxes.
    def get_BBox(self):
        """Return the list of bounding boxes from the master input file
        
        :Call:
            >>> BBox = opts.get_BBox()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *BBox*: :class:`list`(:class:`dict`)
                List of bounding box specifications
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the bounding boxes.
        BBox = self.get_key('BBox')
        # Make sure it's a list.
        if type(BBox).__name__ == 'dict':
            # Single component
            return [BBox]
        elif BBox is None:
            # Return empty list
            return []
        else:
            # Given as list.
            return BBox
        
    # Set the list of bounding boxes.
    def set_BBox(self, BBox=rc0('BBox')):
        """Set the list of bounding boxes
        
        :Call:
            >>> opts.set_BBox(BBox)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *BBox*: :class:`list`(:class:`dict`)
                List of bounding box specifications
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('BBox', BBox)
        
        
    # Return the list of XLev specifications.
    def get_XLev(self):
        """Return the list of XLev specifications
        
        :Call:
            >>> XLev = opts.get_XLev()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *XLev*: :class:`list`(:class:`dict`)
                List of surface refinement specifications
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the bounding boxes.
        XLev = self.get_key('XLev')
        # Make sure it's a list.
        if type(XLev).__name__ == 'dict':
            # Single component
            return [XLev]
        elif XLev is None:
            # Empty list
            return []
        else:
            # List
            return XLev
        
    # Set the list of XLev specifications.
    def set_XLev(self, XLev=rc0('XLev')):
        """Set the list of XLev specifications
        
        :Call:
            >>> opts.set_BBox(BBox)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *XLev*: :class:`list`(:class:`dict`)
                List of surface refinement specifications
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('XLev', XLev)
        
        
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
            * 2014-08-03 ``@ddalle``: First version
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
            * 2014-08-03 ``@ddalle``: First version
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
        
    # Get the background mesh divisions
    def get_nDiv(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_nDiv(i)
    
    # Set the background mesh divisions
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        self._autoInputs()
        self['autoInputs'].set_nDiv(nDiv, i)
        
    # Copy over the documentation.
    for k in ['r', 'nDiv']:
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
        
    # Get the additional refinements around sharp edges
    def get_sf(self, i=None):
        self._cubes()
        return self['cubes'].get_sf(i)
        
    # Set the additional refinements around sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        self._cubes()
        self['cubes'].set_sf(sf, i)


    # Copy over the documentation.
    for k in ['maxR', 'cubes_a', 'cubes_b', 'reorder', 'sf']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(cubes,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(cubes,'set_'+k).__doc__



