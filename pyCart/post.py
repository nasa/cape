"""
Read Forces and Moments, etc.
=============================

This is the :mod:`pyCart` post-processing module.
"""

# Import NumPy arrays.
import numpy as np
import os


# Class to store forces and moments
class LoadsCC:
    """
    Store forces and moments in a class from "loadsCC.dat" files
    """
    # Initialization method
    def __init__(self, cntl=None):
        """
        Class to store force and moment coefficients
        
        :Call:
            >>> FM = pyCart.LoadsCC(cntl=None)
        
        :Inputs:
            *cntl*: :class:`pyCart.cntl.C_ntl`
                Overall :mod:`pyCart` control instance
        
        :Outputs:
            *FM*: :class:`pyCart.post.LoadsCC`
                Instance of this class
                
        :Data members:
            *Aref*: :class:`float`
                Reference area
            *Lref*: :class:`float`
                Reference length
            *Components*: :class:`list` (:class:`str`)
                List of component names found
        """
        # Versions:
        #  2014.06.02 @ddalle  : First version
        
        # Initialize the common fields.
        #  (These will come from pyCart.json.)
        self.Aref = 1.0
        self.Lref = 1.0
        self.MRP = np.array([0., 0., 0.])
        # Check for an input.
        if cntl is None:
            # Quit.
            return None
        # Get the folder names.
        dnames = cntl.GetFolderNames()
        # Initialize the component list.
        self.Components = []
        # Initialize the force coefficients
        self.C_A = {}; self.C_Y = {}; self.C_N = {}
        self.C_D = {}; self.C_S = {}; self.C_L = {}
        # Initialize the moment coefficients
        self.C_l = {}; self.C_m = {}; self.C_n = {}
        self.C_M_x = {}; self.C_M_y = {}; self.C_M_z = {}
        # Number of cases.
        nCase = cntl.Trajectory.nCase
        # Loop through the files.
        for i in range(nCase):
            # Create the file name.
            fname = os.path.join('Grid', dnames[i], 'loadsCC.dat')
            # Process the line.
            self.ReadLoadsCC(fname, i, nCase)
                
        
    # Function to make a new component
    def NewComponent(self, comp, nCase):
        """
        Create a new component with coefficient vector of correct size.
        
        :Call:
            >>> FM.NewComponent(cmop, nCase)
        
        :Inputs:
            *FM*: :class:`pyCart.post.LoadsCC`
                Instance of force-and-moment object
            *comp*: :class:`str`
                Name of component to add
            *nCase*: :class:`int`
                Number of cases in trajectoy (needed to initialize new comps)
        
        :Outputs:
            ``None``
        
        This function creates a vector of ``NaN``s  if *comp* is not in the list
        of existing components.
        """
        # Versions:
        #  2014.06.02 @ddalle  : First version
        
        # Check if the component is present.
        if comp in self.Components: return None
        # Add the component.
        self.Components.append(comp)
        # Initialize the coefficients.
        self.C_A[comp] = np.nan * np.ones(nCase)
        self.C_Y[comp] = np.nan * np.ones(nCase)
        self.C_N[comp] = np.nan * np.ones(nCase)
        self.C_D[comp] = np.nan * np.ones(nCase)
        self.C_S[comp] = np.nan * np.ones(nCase)
        self.C_L[comp] = np.nan * np.ones(nCase)
        self.C_l[comp] = np.nan * np.ones(nCase)
        self.C_m[comp] = np.nan * np.ones(nCase)
        self.C_n[comp] = np.nan * np.ones(nCase)
        self.C_M_x[comp] = np.nan * np.ones(nCase)
        self.C_M_y[comp] = np.nan * np.ones(nCase)
        self.C_M_z[comp] = np.nan * np.ones(nCase)
        # Done
        return None
            
        
    # Function to read a 'loadsCC.dat' file
    def ReadLoadsCC(self, fname, i, nCase):
        """
        Read a "loadsCC.dat" file and process each line
        
        :Call:
            >>> FM.ReadLoadsCC(fname, i, nCase)
        
        :Inputs:
            *FM*: :class:`pyCart.post.LoadsCC`
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

