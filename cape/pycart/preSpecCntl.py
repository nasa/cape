"""
:mod:`cape.pycart.preSpecCntl`: Cart3D *preSpec.c3d.cntl* Interface
===================================================================

This is a module built off of the :class:`cape.filecntl.FileCntl` class
customized for manipulating :file:`preSpec.c3d.cntl` files.  Such files are
split into section by lines of the format

    ``$__Prespecified_Adaptation_Regions``
    
and this module is designed to recognize such sections, although this is the
only section.  The main feature of this module is to add or remove additional
refinement boxes and additional *XLev* surface refinements.

:See Also:
    * :mod:`cape.filecntl`
    * :mod:`cape.pycart.cntl`
    * :mod:`cape.pycart.options.Mesh`
"""

# Import the base file control class.
from cape.filecntl.filecntl import FileCntl, _num, _float

# Base this class off of the main file control class.
class PreSpecCntl(FileCntl):
    """File control class for :file:`preSpec.c3d.cntl` files
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> preSpec = pyCart.PreSpecCntl()
        >>> preSpec = pyCart.PreSpecCntl(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of CNTL file to read, defaults to ``'preSpec.c3d.cntl'``
    :Versions:
        * 2014-06-16 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="preSpec.c3d.cntl"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\$__([\w_]+)")
        return None
        
    # Function to add an additional BBox
    def AddBBox(self, n, xlim):
        """
        Add an additional bounding box to the :file:`cubes` input control file
        
        :Call:
            >>> preSpec.AddBBox(n, xlim)
        :Inputs:
            *preSpec*: :class:`pyCart.preSpecCntl.PreSpecCntl`
                Instance of the :file:`preSpec.c3d.cntl` interface
            *n*: :class:`int`
                Number of refinements to use in the box
            *xlim*: :class:`numpy.ndarray` or :class:`list` (:class:`float`)
                List of *xmin*, *xmax*, *ymin*, *ymax*, *zmin*, *zmax*
        :Effects:
            Adds a bounding box line to the existing boxes
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
        """
        # Compose the line.
        line = "BBox: %-2i %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n" % (
            n, xlim[0], xlim[1], xlim[2], xlim[3], xlim[4], xlim[5])
        # Add the line.
        self.PrependLineToSection('Prespecified_Adaptation_Regions', line)
        
    # Function to clear all existing bounding boxes.
    def ClearBBox(self):
        """Delete all existing bounding boxes
        
        :Call:
            >>> preSpec.ClearBBox()
        :Inputs:
            *preSpec*: :class:`pyCart.preSpecCntl.PreSpecCntl`
                Instance of the :file:`preSpec.c3d.cntl` interface
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
        """
        # Delete the lines.
        self.DeleteLineInSectionStartsWith(
            'Prespecified_Adaptation_Regions', 'BBox')
        
    # Function to add an additional XLev line
    def AddXLev(self, n, compID):
        """
        Add a refinement level on a component or list of components
        
        :Call:
            >>> preSpec.AddXLev(n, compID)
        :Inputs:
            *preSpec*: :class:`pyCart.preSpecCntl.PreSpecCntl`
                Instance of the :file:`preSpec.c3d.cntl` interface
            *n*: :class:`int`
                Number of refinements to use in the box
            *compID*: :class:`int` | :class:`list` (:class:`int`)
                List of component IDs
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Check the input.
        if type(compID).__name__ in ['int', 'float']:
            # Ensure list.
            compID = [compID]
        # Initialize the line.
        line = "XLev: %i" % n
        # Loop through components.
        for c in compID:
            # Add the component to the line.
            line += (" %i" % c)
        # Make sure to end the line.
        line += "\n"
        # Add the line.
        self.AppendLineToSection('Prespecified_Adaptation_Regions', line)
        
    # Function to clear out all XLev specifications
    def ClearXLev(self):
        """Delete all existing XLev specifications
        
        :Call:
            >>> preSpec.ClearXLev()
        :Inputs:
            *preSpec*: :class:`pyCart.preSpecCntl.PreSpecCntl`
                Instance of the :file:`preSpec.c3d.cntl` interface
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Delete the lines.
        self.DeleteLineInSectionStartsWith(
            'Prespecified_Adaptation_Regions', 'XLev')
        
        
        
        
        
