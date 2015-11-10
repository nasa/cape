"""Interface for Cart3D meshing settings"""


# Import options-specific utilities
from util import rc0, odict
        

# Class for Cart3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for options for Cart3D meshing"""
            
    
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
    

