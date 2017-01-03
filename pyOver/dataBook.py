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
from .case import GetCurrentIter, GetPrefix, ReadCaseJSON, GetPhaseNumber
# Utilities or advanced statistics
from . import util
from . import bin
# Data modules
from . import pointSensor

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
        # Check for empty line
        if comp == "": break
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
# def ReadFomoco

# Read grid names from a resid file
def ReadResidGrids(fname):
    """Get list of grids in an OVERFLOW residual file
    
    :Call:
        >>> grids = pyOver.dataBook.ReadResidGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *grids*: :class:`list` (:class:`str`)
            List of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Initialize grids
    comps = []
    # Open the file
    f = open(fname, 'r')
    # Read the first line and last component
    comp = f.readline().split()[-1]
    # Loop until a component repeates
    while comp not in comps:
        # Add the component
        comps.append(comp)
        # Read the next grid name
        comp = f.readline().split()[-1]
    # Close the file
    f.close()
    # Output
    return comps
    
# Read number of grids from a resid file
def ReadResidNGrids(fname):
    """Get number of grids from an OVERFLOW residual file
    
    :Call:
        >>> nGrid = pyOver.dataBook.ReadResidNGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *nGrid*: :class:`int`
            Number of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Initialize number of grids
    nGrid = 0
    # Open the file
    f = open(fname, 'r')
    # Read the first grid number
    iGrid = int(f.readline().split()[0])
    # Loop until grid number decreases
    while iGrid > nGrid:
        # Update grid count
        nGrid += 1
        # Read the next line.
        line = f.readline().split()
        # Check for EndOfFile
        if len(line) == 0: break
        # Read the next grid number.
        iGrid = int(line[0])
    # Close the file
    f.close()
    # Output
    return nGrid
    
# Read the first iteration number from a resid file.
def ReadResidFirstIter(fname):
    """Read the first iteration number in an OVERFLOW residual file
    
    :Call:
        >>> iIter = pyOver.dataBook.ReadResidFirstIter(fname)
        >>> iIter = pyOver.dataBook.ReadResidFirstIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *iIter*: :class:`int`
            Iteration number from first line
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Check input type
    if type(fname).__name__ == "file":
        # Already a file.
        f = fname
        # Check if it's open already
        qf = True
        # Get current location
        ft = f.tell()
    else:
        # Open the file.
        f = open(fname, 'r')
        # Not open
        qf = False
    # Read the second entry from the first line
    iIter = int(f.readline().split()[1])
    # Close the file.
    if qf:
        # Return to original location
        f.seek(ft)
    else:
        # Close the file
        f.close()
    # Output
    return iIter
    
# Read the first iteration number from a resid file.
def ReadResidLastIter(fname):
    """Read the first iteration number in an OVERFLOW residual file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadResidLastIter(fname)
        >>> nIter = pyOver.dataBook.ReadResidLastIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *nIter*: :class:`int`
            Iteration number from last line
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Check input type
    if type(fname).__name__ == "file":
        # Already a file.
        f = fname
        # Check if it's open already
        qf = True
        # Get current location
        ft = f.tell()
    else:
        # Open the file.
        f = open(fname, 'r')
        # Not open
        qf = False
    # Go to last line
    f.seek(-218, 2)
    # Read the second entry from the last line
    iIter = int(f.readline().split()[1])
    # Close the file.
    if qf:
        # Return to original location
        f.seek(ft)
    else:
        # Close the file
        f.close()
    # Output
    return iIter
    
# Get number of iterations from a resid file
def ReadResidNIter(fname):
    """Get number of iterations in an OVERFLOW residual file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadResidNIter(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Get the number of grids.
    nGrid = ReadResidNGrids(fname)
    # Open the file
    f = open(fname, 'r')
    # Go to the end of the file
    f.seek(0, 2)
    # Use the position to determine the number of lines
    nIter = f.tell() / nGrid / 218
    # close the file.
    f.close()
    # Output
    return nIter
# def ReadResid

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
    # Initialize a DBComp object
    def InitDBComp(self, comp, x, opts, targ=None):
        """Initialize data book for one component
        
        :Call:
            >>> DB.InitDBComp(comp, x, opts)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *x*: :class:`pyCart.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *opts*: :class:`pyCart.options.Options`
                Global pyCart options instance
            *targ*: {``None``} | :class:`str`
                If used, read a duplicate data book as a target named *targ*
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
            * 2016-06-27 ``@ddalle``: Added *targ* keyword
        """
        self[comp] = DBComp(comp, x, opts, targ=targ)
        
    # Update data book
    def UpdateDataBook(self, I=None):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DB.UpdateDataBook()
            >>> DB.UpdateDataBook(I)
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list` (:class:`int`) or ``None``
                List of trajectory indices or update all cases in trajectory
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Default.
        if I is None:
            # Use all trajectory points.
            I = range(self.x.nCase)
        # Loop through indices.
        for i in I:
            self.UpdateCase(i)

    # Update or add an entry.
    def UpdateCase(self, i):
        """Update or add a case to a data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> DB.UpdateCase(i)
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Trajectory index
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the first data book component.
        DBc = self.GetRefComponent()
        # Try to find a match existing in the data book.
        j = DBc.FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Status update.
        print(frun)
        # Go home.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            # Nothing to do.
            return
        # Go to the folder.
        os.chdir(frun)
        # Get the current iteration number.
        nIter = GetCurrentIter()
        # Get the number of iterations used for stats.
        nStats = self.opts.get_nStats()
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_nMin()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations (or zero iterations)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry.
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif DBc['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i."
                % (self[c0]['nIter'][j], nIter))
            q = True
        elif DBc['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Get the phase number
        rc = ReadCaseJSON()
        k = GetPhaseNumber(rc)
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Maximum number of iterations allowed.
        nMax = min(nIter-nMin, self.opts.get_nMaxStats())
        # Loop through components.
        for comp in self.Components:
            # Ensure proper type
            tcomp = self.opts.get_DataBookType(comp)
            if tcomp not in ['Force', 'Moment', 'FM']: continue
            # Read the iterative history for that component.
            FM = CaseFM(proj, comp)
            # Extract the component databook.
            DBc = self[comp]
            # List of transformations
            tcomp = self.opts.get_DataBookTransformations(comp)
            # Special transformation to reverse *CLL* and *CLN*
            tflight = {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}
            # Check for ScaleCoeffs
            if tflight not in tcomp:
                # Append a transformation to reverse *CLL* and *CLN*
                tcomp.append(tflight)
            # Loop through the transformations.
            for topts in tcomp:
                # Apply the transformation.
                FM.TransformFM(topts, self.x, i)
                
            # Process the statistics.
            s = FM.GetStats(nStats, nMax)
            
            # Save the data.
            if np.isnan(j):
                # Add to the number of cases.
                DBc.n += 1
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DBc[k] = np.append(DBc[k], getattr(self.x,k)[i])
                # Append values.
                for c in DBc.DataCols:
                    DBc[c] = np.hstack((DBc[c], [s[c]]))
                # Append iteration counts.
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DBc.DataCols:
                    DBc[c][j] = s[c]
                # Update the other statistics.
                DBc['nIter'][j]   = nIter
                DBc['nStats'][j]  = s['nStats']
        # Go back.
        os.chdir(self.RootDir)
    
    # Read point sensor (group)
    def ReadPointSensor(self, name):
        """Read a point sensor group if it is not already present
        
        :Call:
            >>> DB.ReadPointSensor(name)
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of the pycart data book class
            *name*: :class:`str`
                Name of point sensor group
        :Versions:
            * 2015-12-04 ``@ddalle``: Copied from pyCart
        """
        # Initialize if necessary.
        try: 
            self.PointSensors
        except Exception:
            self.PointSensors = {}
        # Initialize the group if necessary
        try:
            self.PointSensors[name]
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read the point sensor.
            self.PointSensors[name] = pointSensor.DBPointSensorGroup(
                self.x, self.opts, name, RootDir=self.RootDir)
            # Return to starting locaiton
            os.chdir(fpwd)
    
# class DataBook

# Component data book
class DBComp(cape.dataBook.DBComp):
    """Individual component data book
    
    This class is derived from :class:`cape.dataBook.DBBase`. 
    
    :Call:
        >>> DBc = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyOver.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`pyOver.options.Options`
            Global pyCart options instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
    :Outputs:
        *DBc*: :class:`pyOver.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2016-09-15 ``@ddalle``: First version
    """
    pass
# class DBComp


# Data book target instance
class DBTarget(cape.dataBook.DBTarget):
    """
    Class to handle data from data book target files.  There are more
    constraints on target files than the files that data book creates, and raw
    data books created by pyCart are not valid target files.
    
    :Call:
        >>> DBT = DBTarget(targ, x, opts)
    :Inputs:
        *targ*: :class:`pyOver.options.DataBook.DBTarget`
            Instance of a target source options interface
        *x*: :class:`pyOver.trajectory.Trajectory`
            Run matrix interface
        *opts*: :class:`pyOver.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyOver.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    pass
# class DBTarget

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
        self.ReadFomocoData(ftmp, i_t, nc_t, ni_t, ni_r+ni_o)
        # Find non-NaN rows
        I = np.logical_not(np.isnan(self.data[:,0]))
        # Downselect
        self.data = self.data[I,:]
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
        # Number of iterations stored
        j = 0
        # Loop through iterations
        for i in range(ni):
            # Read data
            A = np.fromfile(f, sep=" ", count=38)
            # Check for iteration
            if len(A)==38 and (n0 == 0 or A[0] > self.data[n0-1,0]):
                # Save the data
                self.data[n0+j] = A
                # Increase count
                j += 1
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
        >>> H = pyOver.dataBook.CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *H*: :class:`pyOver.databook.CaseResid`
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
        # Initialize arrays.
        self.i = np.array([])
        self.L2   = np.array([])
        self.LInf = np.array([])
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Display
        return "<pyOver.dataBook.CaseResid n=%i, prefix='%s'>" % (
            len(self.i), self.proj)
    # Copy the function
    __str__ = __repr__
        
    # Number of orders of magnitude of residual drop
    def GetNOrders(self, nStats=1):
        """Get the number of orders of magnitude of residual drop
        
        :Call:
            >>> nOrders = hist.GetNOrders(nStats=1)
        :Inputs:
            *hist*: :class:`pyCart.dataBook.CaseResid`
                Instance of the DataBook residual history
            *nStats*: :class:`int`
                Number of iterations to use for averaging the final residual
        :Outputs:
            *nOrders*: :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: First versoin
        """
        # Process the number of usable iterations available.
        i = max(self.nIter-nStats, 0)
        # Get the maximum residual.
        L2Max = np.log10(np.max(self.L2))
        # Get the average terminal residual.
        L2End = np.log10(np.mean(self.L2[i:]))
        # Return the drop
        return L2Max - L2End
    
    # Read entire global residual history
    def ReadGlobalL2(self):
        """Read entire global L2 history
        
        The file ``history.L2.dat`` is also updated.
        
        :Call:
            >>> H.ReadGlobalL2()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.L2 = self.ReadGlobalHist('history.L2.dat')
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2")
        
    # Read entire L-inf residual
    def ReadGlobalLInf(self):
        """Read entire L-infinity norm history
        
        The file ``history.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadGlobalLInf()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.LInf = self.ReadGlobalHist('histroy.LInf.dat')
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf")
        
    # Read turbulence L2 residual
    def ReadTurbResidL2(self):
        """Read the entire L2 norm of the turbulence residuals
        
        The file ``history.turb.L2.dat`` is also updated
        
        :Call:
            >>> H.ReadTurbResidL2()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.L2 = self.ReadglobalHist('history.turb.L2.dat')
        # OVERFLOW file names
        frun = '%.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.turb.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2")
        
    # Read turbulence LInf residual
    def ReadTurbResidLInf(self):
        """Read the global L-infinity norm of the turbulence residuals
        
        The file ``history.turb.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadTurbResidLInf()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.L2 = self.ReadglobalHist('history.turb.LInf.dat')
        # OVERFLOW file names
        frun = '%.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.turb.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf")
        
    # Read species L2 residual
    def ReadSpeciesResidL2(self):
        """Read the global L2 norm of the species equations
        
        The file ``history.species.L2.dat`` is also updated
        
        :Call:
            >>> H.ReadSpeciesResidL2()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.L2 = self.ReadglobalHist('history.species.L2.dat')
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.species.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2")
        
    # Read species LInf residual
    def ReadSpeciesResidLInf(self):
        """Read the global L-infinity norm of the species equations
        
        The file ``history.species.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadSpeciesResidLInf()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Read the global history file
        self.i, self.L2 = self.ReadglobalHist('history.species.LInf.dat')
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = self.i[-1]
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf")
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf")
        # Write the updated history (tmp file not safe to write here)
        self.WriteGlobalHist('history.species.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf")
    
    # Read a consolidated history file
    def ReadGlobalHist(self, fname):
        """Read a condensed global residual file for faster read times
        
        :Call:
            >>> i, L = H.ReadGlobalHist(fname)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *i*: :class:`numpy.ndarray` (:class:
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Check for file.
        if not os.path.isfile(fname):
            # Return empty arrays
            return np.array([]), np.array([])
        # Try to read the file
        try:
            # Read the file.
            A = np.loadtxt(fname)
            # Split into columns
            return A[:,0], A[:,1]
        except Exception:
            # Reading file failed
            return np.array([]), np.array([])
    
    # Write a consolidated history file
    def WriteGlobalHist(self, fname, i, L, n=None):
        """Write a condensed global residual file for faster read times
        
        :Call:
            >>> H.WriteGlobalHist(fname, i, L, n=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *i*: :class:`np.ndarray` (:class:`float` | :class:`int`)
                Vector of iteration numbers
            *L*: :class:`np.ndarray` (:class:`float`)
                Vector of residuals to write
            *n*: :class:`int` | ``None``
                Last iteration already written to file.
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Default number of lines to skip
        if n is None:
            # Query the file.
            if os.path.isfile(fname):
                try:
                    # Read the last line of the file
                    line = bin.tail(fname)
                    # Get the iteration number
                    n = float(line.split()[0])
                except Exception:
                    # File exists but has some issues
                    n = 0
            else:
                # Start at the beginning of the array
                n = 0
        # Find the index of the first iteration greater than *n*
        I = np.where(i > n)[0]
        # If no hits, nothing to write
        if len(I) == 0: return
        # Index to start at
        istart = I[0]
        # Append to the file
        f = open(fname, 'a')
        # Loop through the lines
        for j in range(istart, len(i)):
            # Write iteration
            f.write('%8i %14.7E\n' % (i[j], L[j]))
        # Close the file.
        f.close()

    # Read a global residual file
    def ReadResidGlobal(self, fname, coeff="L2", n=None):
        """Read a global residual using :func:`numpy.loadtxt` from one file
        
        :Call:
            >>> i, L2 = H.ReadResidGlobal(fname, coeff="L2", n=None)
            >>> i, LInf = H.ReadResidGlobal(fname, coeff="LInf", n=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *fname*: :class:`str`
                Name of file to process
            *coeff*: :class:`str`
                Name of coefficient to read
            *n*: :class:`int` | ``None``
                Number of last iteration that's already processed
        :Outputs:
            *i*: :class:`np.ndarray` (:class:`float`)
                Array of iteration numbers
            *L2*: :class:`np.ndarray` (:class:`float`)
                Array of weighted global L2 norms
            *LInf*: :class:`np.ndarray` (:class:`float`)
                Array of global L-infinity norms
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname): return
        # First iteration
        i0 = ReadResidFirstIter(fname)
        # Number of iterations
        nIter = ReadResidNIter(fname)
        self.nIter = nIter
        # Number of grids
        nGrid = ReadResidNGrids(fname)
        # Process current iteration number
        if n is None:
            # Use last known iteration
            if len(self.i) == 0:
                # No iterations
                n = 0
            else:
                # Use last current iter
                n = max(self.i)
        # Number of iterations to skip
        nIterSkip = max(0, n-i0+1)
        # Skip *nGrid* rows for each iteration
        nSkip = int(nIterSkip * nGrid)
        # Number of iterations to be read
        nIterRead = nIter - nIterSkip
        # Check for something to read
        if nIterRead <= 0:
            return np.array([]), np.array([])
        # Process columns to read
        if coeff.lower() == "linf":
            # Read the iter, L-infinity norm
            cols = (1,3)
            nc = 2
            # Coefficient
            c = 'LInf'
        else:
            # Read the iter, L2 norm, nPts
            cols = (1,2,13)
            nc = 3
            # Field name
            c = 'L2'
        # Read the file
        A = np.loadtxt(fname, skiprows=nSkip, usecols=cols)
        # Reshape the data
        B = np.reshape(A[:nIterRead*nGrid,:], (nIterRead, nGrid, nc))
        # Get iterations
        i = B[:,0,0]
        # Filter iterations greater than *n*
        I = i > n
        i = i[I]
        # Exit if no iterations
        if len(i) == 0: return
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I,:,1]*B[I,:,2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I,:,2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.L2 = np.hstack((self.L2, L))
        else:
            # Get the maximum value
            L = np.max(B[I,:,1], axis=1)
            # Append to data
            self.LInf = np.hstack((self.LInf, L))
        # Check for issues
        if np.any(np.diff(i) < 0):
            # Warning
            print("  Warning: file '%s' contains non-ascending iterations" %
                fname)
        # Append to data
        self.i = np.hstack((self.i, i))
        # Output
        return i, L
    
    # Plot L2 norm
    def PlotL2(self, n=None, nFirst=None, nLast=None, **kw):
        """Plot the L2 residual
        
        :Call:
            >>> h = hist.PlotL2(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: Referred to :func:`PlotResid`
        """
        # Y-label option
        ylbl = kw.get('YLabel', 'L2 Residual')
        # Plot 'L2Resid'
        return self.PlotResid('L2', 
            n=n, nFirst=nFirst, nLast=nLast, YLabel='L2 Residual')
# class CaseResid

