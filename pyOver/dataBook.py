"""
Data Book Module: :mod:`pyOver.dataBook`
========================================

This module contains functions for reading and processing forces, moments, and
other statistics from cases in a trajectory.

:Versions:
    * 2016-02-02 ``@ddalle``: Started
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re
# Date processing
from datetime import datetime

# Use this to only update entries with newer iterations.
from .case import GetCurrentIter, GetPrefix
# Utilities or advanced statistics
from . import util
from . import bin

# Template module
import cape.dataBook

# Placeholder variables for plotting functions.
plt = 0

# Radian -> degree conversion
deg = np.pi / 180.0

# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    """Import :mod:`matplotlib.pyplot` if not loaded
    
    :Call:
        >>> pyOver.dataBook.ImportPyPlot()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
        * 2016-01-02 ``@ddalle``: Copied from pyCart
    """
    # Make global variables
    global plt
    global tform
    global Text
    # Check for PyPlot.
    try:
        plt.gcf
    except AttributeError:
        # Load the modules.
        import matplotlib.pyplot as plt
        import matplotlib.transforms as tform
        from matplotlib.text import Text
# def ImportPyPlot

# Read component names from a fomoco file
def ReadFomocoComps(fname):
    """Get list of components in an OVERFLOW fomoco file
    
    :Call:
        >>> comps = pyOver.dataBook.ReadFomocoComps(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of the file to read
    :Outputs:
        *comps*: :class:`list` (:class:`str`)
            List of components
    :Versions:
        * 2016-02-03 ``@ddalle``: First version
    """
    # Initialize components
    comps = []
    # Open the file
    f = open(fname, 'r')
    # Read the first line and first component.
    comp = f.readline().strip()
    # Loop until a component repeats
    while comp not in comps:
        # Add the component
        comps.append(comp)
        # Move to the next component
        f.seek(569, 1)
        # Read the next component.
        comp = f.readline().strip()
    # Close the file
    f.close()
    # Output
    return comps
    
# Read basic stats from a fomoco file
def ReadFomocoNIter(fname, nComp=None):
    """Get number of iterations in an OVERFLOW fomoco file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadFomocoNIter(fname)
        >>> nIter = pyOver.dataBook.ReadFomocoNIter(fname, nComp)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *nComp*: :class:`int` | ``None``
            Number of components in each record
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations in the file
    :Versions:
        * 2016-02-03 ``@ddalle``: First version
    """
    # If no number of comps, get list
    if nComp is None:
        # Read component list
        comps = ReadFomocoComps(fname)
        # Number of components
        nComp = len(comps)
    # Open file to get number of iterations
    f = open(fname)
    # Go to end of file to get length of file
    f.seek(0, 2)
    # Position at EOF
    L = f.tell()
    # Close the file.
    f.close()
    # Save number of iterations
    return int(np.ceil(L / (nComp*650.0)))
# def ReadFomocoComps

# Aerodynamic history class
class DataBook(cape.dataBook.DataBook):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyFun.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyFun.trajectory.Trajectory`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`pyFun.dataBook.DataBook`
            Instance of the pyFun data book class
    :Versions:
        * 2015-10-20 ``@ddalle``: Started
    """
    pass

# class DataBook

# Force/moment history
class CaseFM(cape.dataBook.CaseFM):
    """
    This class contains methods for reading data about an the history of an
    individual component for a single case.  It reads the Tecplot file
    :file:`$proj_fm_$comp.dat` where *proj* is the lower-case root project name
    and *comp* is the name of the component.  From this file it determines
    which coefficients are recorded automatically.
    
    :Call:
        >>> FM = pyOver.dataBook.CaseFM(proj, comp)
    :Inputs:
        *proj*: :class:`str`
            Root name of the project
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *FM*: :class:`pyOver.dataBook.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list` (:class:`str`)
            List of coefficients
        *FM.i*: :class:`numpy.ndarray` shape=(0,)
            List of iteration numbers
        *FM.CA*: :class:`numpy.ndarray` shape=(0,)
            Axial force coefficient at each iteration
        *FM.CY*: :class:`numpy.ndarray` shape=(0,)
            Lateral force coefficient at each iteration
        *FM.CN*: :class:`numpy.ndarray` shape=(0,)
            Normal force coefficient at each iteration
        *FM.CLL*: :class:`numpy.ndarray` shape=(0,)
            Rolling moment coefficient at each iteration
        *FM.CLM*: :class:`numpy.ndarray` shape=(0,)
            Pitching moment coefficient at each iteration
        *FM.CLN*: :class:`numpy.ndarray` shape=(0,)
            Yaw moment coefficient at each iteration
    :Versions:
        * 2016-02-02 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, proj, comp):
        """Initialization method"""
        # Save component name
        self.comp = comp
        # Get the project rootname
        self.proj = proj
        # Expected name of the component history file
        ftmp = 'fomoco.tmp'
        fout = 'fomoco.out'
        frun = '%s.fomoco' % proj
        # Read stats from these files
        i_t, nc_t, ni_t = self.GetFomocoInfo(ftmp, comp)
        i_o, nc_o, ni_o = self.GetFomocoInfo(fout, comp)
        i_r, nc_r, ni_r = self.GetFomocoInfo(frun, comp)
        # Number of iterations
        ni = ni_t + ni_o + ni_r
        # Return empty if no data
        if ni == 0:
            self.MakeEmpty()
            return
        # Initialize data
        self.data = np.nan*np.ones((ni, 38))
        # Read the data.
        self.ReadFomocoData(frun, i_r, nc_r, ni_r, 0)
        self.ReadFomocoData(fout, i_o, nc_o, ni_o, ni_r)
        self.ReadFomocoData(fout, i_t, nc_t, ni_t, ni_r+ni_o)
        # Save data as attributes
        self.SaveAttributes()
        
        
    # Get stats from a named FOMOCO file
    def GetFomocoInfo(self, fname, comp):
        """Get basic stats about an OVERFLOW fomoco file
        
        :Call:
            >>> ic, nc, ni = FM.GetFomocoInfo(fname, comp)
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Force and moment iterative history
            *fname*: :class:`str`
                Name of file to query
            *comp*: :class:`str`
                Name of component to find
        :Outputs:
            *ic*: :class:`int` | ``None``
                Index of component in the list of components
            *nc*: :class:`int` | ``None``
                Number of components
            *ni*: :class:`int`
                Number of iterations
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Check for the file
        if os.path.isfile(fname):
            # Get list of components
            comps = ReadFomocoComps(fname)
            # Number of components
            nc = len(comps)
            # Check if our component is present
            if comp in comps:
                # Index of the component.
                ic = comps.index(comp)
                # Number of (relevant) iterations
                ni = ReadFomocoNIter(fname, nc)
            else:
                # No useful iterations
                ic = 0
                # Number of (relevant) iterations
                ni = 0
            # Output
            return ic, nc, ni
        else:
            # No file
            return None, None, 0
            
    # Function to make empty one.
    def MakeEmpty(self, n=0):
        """Create empty *CaseFM* instance
        
        :Call:
            >>> FM.MakeEmpty()
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Template entry
        A = np.nan * np.zeros(n)
        # Iterations
        self.i = A.copy()
        # Time
        self.t = A.copy()
        # Force coefficients
        self.CA = A.copy()
        self.CY = A.copy()
        self.CN = A.copy()
        # Moment coefficients
        self.CLL = A.copy()
        self.CLM = A.copy()
        self.CLN = A.copy()
        # Force pressure contributions
        self.CA_p = A.copy()
        self.CY_p = A.copy()
        self.CN_p = A.copy()
        # Force viscous contributions
        self.CA_v = A.copy()
        self.CY_v = A.copy()
        self.CN_v = A.copy()
        # Force momentum contributions
        self.CA_m = A.copy()
        self.CY_m = A.copy()
        self.CN_m = A.copy()
        # Moment pressure contributions
        self.CLL_p = A.copy()
        self.CLM_p = A.copy()
        self.CLN_p = A.copy()
        # Moment viscous contributions
        self.CLL_v = A.copy()
        self.CLM_v = A.copy()
        self.CLN_v = A.copy()
        # Moment momentum contributions
        self.CLL_m = A.copy()
        self.CLM_m = A.copy()
        self.CLN_m = A.copy()
        # Mass flow
        self.mdot = A.copy()
        # Areas
        self.A = A.copy()
        self.Ax = A.copy()
        self.Ay = A.copy()
        self.Az = A.copy()
        # Save a default list of columns and components.
        self.coeffs = [
            'CA',   'CY',   'CN',   'CLL',  'CLM',  'CLN',
            'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
            'CA_m', 'CY_m', 'CN_m', 'CLL_p','CLM_p','CLN_p',
            'CLL_v','CLM_v','CLN_v','CLL_m','CLM_v','CLN_v',
            'mdot', 'A',    'Ax',   'Ay',   'Az'
        ]
        self.cols = ['i', 't'] + self.coeffs
        
    # Read data from a FOMOCO file
    def ReadFomocoData(self, fname, ic, nc, ni, n0=0):
        """Read data from a FOMOCO file with known indices and size
        
        :Call:
            >>> FM.ReadFomocoData(fname, ic, nc, ni, n0)
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Force and moment history
            *fname*: :class:`str`
                Name of fomoco file
            *ic*: :class:`int`
                Index of *FM.comp* in list of components in *fname*
            *nc*: :class:`int`
                Number of components in *fname*
            *ni*: :class:`int`
                Number of iterations in *fname*
            *n0*: :class:`int`
                Number of iterations already read into *FM.data*
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Exit if nothing to do
        if ni == 0: return
        # Check for file (in case any changes occurred before getting here)
        if not os.path.isfile(fname): return
        # Open the file
        f = open(fname)
        # Skip to start of first iteration
        f.seek(650*ic+81)
        # Loop through iterations
        for i in range(ni):
            # Read data
            self.data[n0+i] = np.fromfile(f, sep=" ", count=38)
            # Skip to next iteration
            f.seek(650*(nc-1)+81, 1)
        # Close the file
        f.close()
    
    # Function to make empty one.
    def SaveAttributes(self):
        """Save columns of *FM.data* as named attributes
        
        :Call:
            >>> FM.SaveAttributes()
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Iterations
        self.i = self.data[:,0]
        # Time
        self.t = self.data[:,28]
        # Force pressure contributions
        self.CA_p = self.data[:,6]
        self.CY_p = self.data[:,7]
        self.CN_p = self.data[:,8]
        # Force viscous contributions
        self.CA_v = self.data[:,9]
        self.CY_v = self.data[:,10]
        self.CN_v = self.data[:,11]
        # Force momentum contributions
        self.CA_m = self.data[:,12]
        self.CY_m = self.data[:,13]
        self.CN_m = self.data[:,14]
        # Force coefficients
        self.CA = self.CA_p + self.CA_v + self.CA_m
        self.CY = self.CY_p + self.CY_v + self.CY_m
        self.CN = self.CN_p + self.CN_v + self.CN_m
        # Moment pressure contributions
        self.CLL_p = self.data[:,29]
        self.CLM_p = self.data[:,30]
        self.CLN_p = self.data[:,31]
        # Moment viscous contributions
        self.CLL_v = self.data[:,32]
        self.CLM_v = self.data[:,33]
        self.CLN_v = self.data[:,34]
        # Moment momentum contributions
        self.CLL_m = self.data[:,35]
        self.CLM_m = self.data[:,36]
        self.CLN_m = self.data[:,37]
        # Moment coefficients
        self.CLL = self.CLL_p + self.CLL_v + self.CLL_m
        self.CLM = self.CLM_p + self.CLM_v + self.CLM_m
        self.CLN = self.CLN_p + self.CLN_v + self.CLN_m
        # Mass flow
        self.mdot = self.data[:,27]
        # Areas
        self.A  = self.data[:,2]
        self.Ax = self.data[:,3]
        self.Ay = self.data[:,4]
        self.Az = self.data[:,5]
        # Save a default list of columns and components.
        self.coeffs = [
            'CA',   'CY',   'CN',   'CLL',  'CLM',  'CLN',
            'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
            'CA_m', 'CY_m', 'CN_m', 'CLL_p','CLM_p','CLN_p',
            'CLL_v','CLM_v','CLN_v','CLL_m','CLM_v','CLN_v',
            'mdot', 'A',    'Ax',   'Ay',   'Az'
        ]
        self.cols = ['i', 't'] + self.coeffs
        
# class CaseFM


# Residual class
class CaseResid(cape.dataBook.CaseResid):
    """OVERFLOW iterative residual history class
    
    This class provides an interface to residuals for a given case by reading
    the files ``resid.out``, ``resid.tmp``, ``run.resid``, ``turb.out``,
    ``species.out``, etc.
    
    :Call:
        >>> R = pyOver.dataBook.CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *R*: :class:`pyOver.databook.CaseResid`
            Instance of the residual histroy class
    :Versions:
        * 2016-02-03 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, proj):
        """Initialization method
        
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Save the prefix
        self.proj = proj
        

