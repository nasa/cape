"""
:mod:`cape.pycart.options.Mesh`: pyCart Meshing Options
========================================================

This module provides options for creating volume meshes in Cart3D.  This
consists of three parts, with the latter two being the generally more useful
capabilities.

    * Provides the name of a template :file:`input.c3d` file; overrides the 
      file created by ``autoInputs``
      
    * Specifies instructions for Cart3D bounding boxes (``"BBox"``) with an
      automated interface to specify boxes by giving the name of a component
      along with a margin, in addition to the capability to manually give the
      coordinates
      
    * Specifies instructions for Cart3D surface refinement options (``"XLev"``)
      by name of component(s) or component index
      
These ``BBox`` and ``XLev`` instructions edit the file
:file:`preSpec.c3d.cntl`.  The ``BBox`` instructions are applied via the method
:func:`cape.pycart.cntl.Cntl.PreparePreSpecCntl`.  This file is an input file to
``cubes`` and affects the resolution of the volume created.

:See Also:
    * :mod:`cape.options.Mesh`
    * :mod:`cape.pycart.preSpecCntl.PreSpecCntl`
"""


# Import options-specific utilities
from .util import rc0, odict
# Import Cape template
import cape.options.Mesh

# Class for Cart3D mesh settings
class Mesh(cape.options.Mesh):
    """Dictionary-based interface for options for Cart3D meshing"""
        
        
    # Return the list of bounding boxes.
    def get_BBox(self):
        """Return the list of bounding boxes for volume mesh creation
        
        There are two methods to specify a ``"BBox"``.  The first is to give
        the name of a component taken from a Cart3d ``Config.xml`` file.  The
        :class:`cape.tri.Tri` class automatically finds the smallest bounding
        box that contains this component, and then the user can specify
        additional margins beyond this box (margins can also be negative).  In
        addition, separate margins (or "pads") on the minimum and maximum
        coordinates can be given following the convention ``"xm"`` (short for
        *x-minus*) on the minimum coordinate and ``"xp"`` for the maximum
        coordinate.
        
            .. code-block:: python
            
                {
                    "compID": "fin2",
                    "n": 9,
                    "pad": 2.0,
                    "xpad": 3.0,
                    "ym": -1.0,
                    "yp": 3.0
                }
        
        The other method is to simply give the limits as a list of *xmin*,
        *xmax*, *ymin*, ...  In both methods *n* is the number of refinements
        to specify within the bounding ox.
        
            .. code-block:: python
            
                {
                    "n": 9,
                    "xlim": [10.0, 13.5, 3.0, 3.7, -0.8, 0.8]
                }
                
        :Call:
            >>> BBox = opts.get_BBox()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *BBox*: :class:`list` (:class:`dict`)
                List of bounding box specifications
        :See Also:
            * :func:`cape.tri.TriBase.GetCompBBox`
            * :class:`pyCart.options.runControl.cubes`
            * :class:`pyCart.preSpecCntl.PreSpecCntl`
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
        """Return the list of *XLev* specifications for surface refinements
        
        An *XLev* specification tells ``cubes`` to perform a number of
        additional refinements on any cut cells that intersect a triangle from
        a named component.  This refinement can violate the *maxR* command-line
        input to ``cubes`` and is very useful for ensuring that small features
        of the surface have adequate resolution in the initial volume mesh.
        
        The following example specifies two additional refinements (after the
        initial run-through by ``cubes``) on all triangles in the component
        ``"fins"``.  These instructions are then written to
        :file:`preSpec.c3d.cntl`.
        
            .. code-block:: python
            
                {
                    "compID": "fins",
                    "n": 2
                }
        
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
        
    # Get input.c3d file name (Not using autoInputs)
    def get_inputC3d(self):
        """Get the ``cubes`` input file name if not using ``autoInputs``
        
        :Call:
            >>> fc3d = opts.get_inputC3d()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fc3d*: :class:`str`
                Name of input file
        :Versions:
            * 2015-11-08 ``@ddalle``: Redone
        """
        return self.get_key('inputC3d', 0)
        
    # Set the input.c3d file name
    def set_inputC3d(self, fc3d=rc0('inputC3d')):
        """Set the ``cubes`` input file name if not using ``autoInputs``
        
        :Call:
            >>> opts.set_inputC3d(fc3d)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fc3d*: :class:`str`
                Name of input file
        :Versions:
            * 2015-11-08 ``@ddalle``: Redone
        """
        self['inputC3d'] = fc3d
    

