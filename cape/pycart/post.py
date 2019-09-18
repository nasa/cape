"""
Post-processing module: :mod:`cape.pycart.post`
===============================================

This is the :mod:`cape.pycart` post-processing module.  Primarily it is focused on
reading :file:`loadsCC.dat` and :file:`loadsTRI.dat` files.
"""

# Import NumPy arrays.
import numpy as np
import os


# Class to store forces and moments
class LoadsDat:
    """
    Class to store force and moment coefficients
    
    :Call:
        >>> FM = pyCart.LoadsDat(cart3d=None, fname="loadsCC.dat")
    
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Overall :mod:`cape.pycart` control instance
        *fname*: :class:`str`
            Name of individual files to read
    
    :Outputs:
        *FM*: :class:`pyCart.post.LoadsDat`
            Instance of this class
            
    :Data members:
        *Aref*: :class:`float`
            Reference area
        *Lref*: :class:`float`
            Reference length
        *MRP*: :class:`numpy.ndarray`, shape=(3,)
            Moment reference point
        *fname*: :class:`str`
            Name of individual files to read
        *Components*: :class:`list` (:class:`str`)
            List of component names found
        *C_A*: :class:`dict`
            Dictionary of axial force coefficient for each component
            
    """
    # Initialization method
    def __init__(self, cart3d=None, fname="loadsCC.dat"):
        """Initialization method"""
        # Versions:
        #  2014.06.02 @ddalle  : First version
        #  2014.06.04 @ddalle  : General for "loadsTRI.dat" or "loadsCC.dat"
        
        # Initialize the common fields.
        #  (These will come from pyCart.json.)
        self.Aref = 1.0
        self.Lref = 1.0
        self.MRP = np.array([0., 0., 0.])
        # Save the file name.
        self.fname = fname
        # Check for an input.
        if cart3d is None:
            # Quit.
            return None
        # Get the folder names.
        fnames = cntl.RunMatrix.GetFullFolderNames()
        # Initialize the component list.
        self.Components = []
        # Initialize the force coefficients
        self.C_A = {}; self.C_Y = {}; self.C_N = {}
        self.C_D = {}; self.C_S = {}; self.C_L = {}
        # Initialize the moment coefficients
        self.C_l = {}; self.C_m = {}; self.C_n = {}
        self.C_M_x = {}; self.C_M_y = {}; self.C_M_z = {}
        # Number of cases.
        nCase = cntl.RunMatrix.nCase
        # Loop through the files.
        for i in range(nCase):
            # Create the file name.
            fi = os.path.join(fnames[i], fname)
            # Process the line.
            self.ReadLoadsFile(fi, i, nCase)
                
        
    # Function to make a new component
    def NewComponent(self, compID, nCase):
        """
        Create a new component with coefficient vector of correct size.
        
        :Call:
            >>> FM.NewComponent(compID, nCase)
        
        :Inputs:
            *FM*: :class:`pyCart.post.LoadsDat`
                Instance of force-and-moment object
            *compID*: :class:`str`
                Name of component to add
            *nCase*: :class:`int`
                Number of cases in trajectoy (needed to initialize new comps)
        
        :Outputs:
            ``None``
        
        This function creates a vector of ``NaN``  if *comp* is not in the list
        of existing components.
        """
        # Versions:
        #  2014.06.02 @ddalle  : First version
        
        # Check if the component is present.
        if compID in self.Components: return None
        # Add the component.
        self.Components.append(compID)
        # Initialize the coefficients.
        self.C_A[compID] = np.nan * np.ones(nCase)
        self.C_Y[compID] = np.nan * np.ones(nCase)
        self.C_N[compID] = np.nan * np.ones(nCase)
        self.C_D[compID] = np.nan * np.ones(nCase)
        self.C_S[compID] = np.nan * np.ones(nCase)
        self.C_L[compID] = np.nan * np.ones(nCase)
        self.C_l[compID] = np.nan * np.ones(nCase)
        self.C_m[compID] = np.nan * np.ones(nCase)
        self.C_n[compID] = np.nan * np.ones(nCase)
        self.C_M_x[compID] = np.nan * np.ones(nCase)
        self.C_M_y[compID] = np.nan * np.ones(nCase)
        self.C_M_z[compID] = np.nan * np.ones(nCase)
        # Done
        return None
            
        
    # Function to read a 'loads*.dat' file
    def ReadLoadsFile(self, fname, i, nCase):
        """
        Read a "loads*.dat" file and process each line
        
        :Call:
            >>> FM.ReadLoadsFile(fname, i, nCase)
        
        :Inputs:
            *FM*: :class:`pyCart.post.LoadsDat`
                Instance of force-and-moment object
            *fname*: :class:`str`
                Path to file to read
            *i*: :class:`int`
                Case number
            *nCase*: :class:`int`
                Number of cases in trajectoy (needed to initialize new comps)
        
        :Outputs:
            ``None``
        
        This function updates the ``FM.C_A``, ``FM.C_Y``, etc. fields and adds
        components to ``FM.Components`` if necessary.
        """
        # Versions:
        #  2014.06.02 @ddalle  : First version
        
        # Check if the file exists.
        if not os.path.isfile(fname): return None
        # Open the file and read the contents.
        lines = open(fname, 'r').readlines()
        # Loop through the lines.
        for line in lines:
            # Strip the leading and trailing spaces.
            line = line.strip()
            # Check for comments or empty lines.
            if line.startswith('#') or len(line)==0: continue
            # Split into components.
            lc = line.split()
            # Break out the components (words) for readability.
            comp = lc[0]
            coeff = lc[-2][1:-2]
            c = float(lc[-1])
            # Check the component.
            self.NewComponent(comp, nCase)
            # Save the coefficient.
            getattr(self, coeff)[comp][i] = c
        # Done
        return None
        
        
    # Function to write forces and moments
    def Write(self, T, compID=None):
        """
        Write loads data to files.
        
        A comma-separated values file is generated.  An example name of the file
        is "loadsCC.csv".
        
        :Call:
            >>> FM.Write(T, compID=None)
            
        :Inputs:
            *T*: :class:`pyCart.runmatrix.RunMatrix`
                RunMatrix matching the conditions saved
            *compID*: :class:`list`
                List of components to process, all are written by default
        """
        # Versions:
        #  2014.06.04 @ddalle  : First version
        
        # List of coefficient names.
        coeffs = ['C_A', 'C_Y', 'C_N', 'C_D', 'C_S', 'C_L',
            'C_l', 'C_m', 'C_n', 'C_M_x', 'C_M_y', 'C_M_z']
        # Component list
        if compID is None or compID=="all":
            # Default: use all
            compID = self.Components
        # Output file name (e.g., "loadsCC.csv")
        fout = self.fname.rstrip(".dat") + ".csv"
        # Open the (new) file.
        f = open(fout, 'w')
        # Write the first header.
        f.write("Case, ")
        # Write the headers for each variable name in trajectory.
        f.write(", ".join(T.keys))
        # Loop through the components.
        for comp in compID:
            # Loop through the coefficients.
            for c in coeffs:
                # Write the coefficient name.
                f.write(', %s_%s' % (c, comp))
        # End of header line.
        f.write('\n')
        # Loop through the conditions.
        for i in range(T.nCase):
            # Write the case number.
            f.write('%i' % i)
            # Loop through trajectory keywords.
            for k in T.keys:
                # Write the value.
                f.write(', %.8f' % getattr(T,k)[i])
            # Loop through the components.
            for comp in compID:
                # Loop through the coefficients.
                for c in coeffs:
                    # Write the value.
                    f.write(', %.8e' % getattr(self,c)[comp][i])
            # New line
            f.write('\n')
        # Close the file and quit.
        f.close()
        return None

