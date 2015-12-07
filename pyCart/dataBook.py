"""
Data Book Module: :mod:`pyCart.dataBook`
========================================

This module contains functions for reading and processing forces, moments, and
other statistics from cases in a trajectory.

:Versions:
    * 2014-12-20 ``@ddalle``: Started
    * 2015-01-01 ``@ddalle``: First version
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re
# Date processing
from datetime import datetime

# Utilities or advanced statistics
from . import util
# Line loads and other data types
from . import lineLoad
from . import pointSensor

# Template module
import cape.dataBook

# Placeholder variables for plotting functions.
plt = 0

# Radian -> degree conversion
deg = np.pi / 180.0


# Aerodynamic history class
class DataBook(cape.dataBook.DataBook):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyCart.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyCart.trajectory.Trajectory`
            The current pyCart trajectory (i.e. run matrix)
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
    :Outputs:
        *DB*: :class:`pyCart.dataBook.DataBook`
            Instance of the pyCart data book class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2015-01-03 ``@ddalle``: First version
        * 2015-10-16 ``@ddalle``: Subclassed to :mod:`cape.dataBook.DataBook`
    """
        
    # Initialize a DBComp object
    def InitDBComp(self, comp, x, opts):
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
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        self[comp] = DBComp(comp, x, opts)
    
    # Read line load
    def ReadLineLoad(self, comp):
        """Read a line load data book target if it is not already present
        
        :Call:
            >>> DB.ReadLineLoad(comp)
        :Inputs:
            *DB*: :class:`pycart.dataBook.DataBook`
                Instance of the pycart data book class
            *comp*: :class:`str`
                Line load component group
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Try to access the line load
        try:
            self.LineLoads[comp]
        except Exception:
            # Read the file.
            self.LineLoads.append(
                lineLoad.DBLineLoad(self.x, self.opts, comp))
            
    # Read point sensor (group)
    def ReadPointSensor(self, name):
        """Read a point sensor group if it is not already present
        
        :Call:
            >>> DB.ReadPointSensor(name)
        :Inputs:
            *DB*: :class:`pycart.dataBook.DataBook`
                Instance of the pycart data book class
            *name*: :class:`str`
                Name of point sensor group
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
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
            # Read the point sensor.
            self.PointSensors[name] = pointSensor.DBPointSensorGroup(
                self.x, self.opts, name, RootDir=self.RootDir)
    
    
    # Update data book
    def UpdateDataBook(self, I=None):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DB.UpdateDataBook()
            >>> DB.UpdateDataBook(I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
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
            
    # Update line load data book
    def UpdateLineLoadDataBook(self, comp, I=None):
        """Update a line load data book for a list of cases
        
        :Call:
            >>> DB.UpdateLineLoadDataBook(comp)
            >>> DB.UpdateLineLoadDataBook(comp, I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list` (:class:`int`) or ``None``
                List of trajectory indices or update all cases in trajectory
        :Versions:
            * 2015-09-17 ``@ddalle``: First version
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Loop through indices.
        for i in I:
            self.UpdateLineLoadCase(comp, i)
            
    # Update point sensor group
    def UpdatePointSensor(self, name, I=None):
        """Update a point sensor group data book for a list of cases
        
        :Call:
            >>> DB.UpdatePointSensorGroup(name)
            >>> DB.UpdatePointSensorGroup(name, I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list` (:class:`int`) or ``None``
                List of trajectory indices or update all cases in trajectory
        :Versions:
            * 2015-10-04 ``@ddalle``: First version
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Read the point sensors if necessary
        self.ReadPointSensor(name)
        # Loop through cases.
        for i in I:
            # Update the point sensors for that case
            self.PointSensors[name].UpdateCase(i)
        
    # Function to delete entries by index
    def Delete(self, I):
        """Delete list of cases from data book
        
        :Call:
            >>> DB.Delete(I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list` (:class:`int`)
                List of trajectory indices or update all cases in trajectory
        :Versions:
            * 2015-03-13 ``@ddalle``: First version
        """
        # Get the first data book component.
        DBc = self[self.Components[0]]
        # Number of cases in current data book.
        nCase = DBc.n
        # Initialize data book index array.
        J = []
        # Loop though indices to delete.
        for i in I:
            # Find the match.
            j = DBc.FindMatch(i)
            # Check if one was found.
            if np.isnan(j): continue
            # Append to the list of data book indices.
            J.append(j)
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Loop through components.
        for comp in self.Components:
            # Extract data book component.
            DBc = self[comp]
            # Loop through data book columns.
            for c in DBc.keys():
                # Apply the mask
                DBc[c] = DBc[c][mask]
            # Update the number of entries.
            DBc.n = len(DBc['nIter'])
        
            
    # Update one line load case
    def UpdateLineLoadCase(self, comp, i):
        """Update one line load case if necessary
        
        :Call:
            >>> DB.UpdateLineLoadCase(comp, i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of line load group
            *i*: :class:`int`
                Case number
        :Versions:
            * 2015-09-17 ``@ddalle``: First version
        """
        # Read the line loads if necessary
        self.ReadLineLoad(comp)
        # Data book directory
        fdat = self.cart3d.opts.get_DataBookDir()
        flls = 'lineloads-%s' % comp
        fldb = os.path.join(fdat, flls)
        # Expected seam cut file
        fsmy = os.path.join(self.cart3d.Rootdir, fldb, '%s.smy'%comp)
        fsmz = os.path.join(self.cart3d.Rootdir, fldb, '%s.smz'%comp)
        # Extract line load
        DBL = self.LineLoad[comp]
        # Try to find a match existing in the data book.
        j = DBL.FindMatch(i)
        # Get the name of the folder.
        frun = self.cart3d.x.GetFullFolderNames(i)
        # Status update.
        print(frun)
        # Go home
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.join(fldb):
            os.mkdir(flds, 0027)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Determine minimum number of iterations required.
        nAvg = self.opts.get_nStats()
        nMin = self.opts.get_nMin()
        # Get the number of iterations
        ftriq, nStats, n0, nIter = lineLoad.GetTriqFile()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations (or zero iterations)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry.
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif DBL['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i."
                % (self[c0]['nIter'][j], nIter))
            q = True
        elif DBL['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Read the new line load
        LL = lineLoad.CaseLL(self.cart3d, i, comp)
        # Calculate it.
        LL.CalculateLineLoads()
        # Check if the seam cut file exists.
        if not os.path.isfile(fsmy):
            # Collect seam cuts.
            q_seam = True
            # Read the seam curves.
            LL.ReadSeamCurves()
        else:
            # Seam cuts already present.
            q_seam = False
        # Save the data.
        if np.isnan(j):
            # Add the the number of cases.
            DBL.n += 1
            # Append trajectory values.
            for k in self.x.keys:
                # I found a better way to append in NumPy.
                DBL[k] = np.append(DBL[k], getattr(self.cart3d.x,k)[i])
            # Save parameters.
            DBL['Mach'] = np.append(DBL['Mach'], LL.Mach)
            DBL['Re']   = np.append(DBL['Re'],   LL.Re)
            DBL['XMRP'] = np.append(DBL['XMRP'], LL.MRP[0])
            DBL['YMRP'] = np.append(DBL['YMRP'], LL.MRP[1])
            DBL['ZMRP'] = np.append(DBL['ZMRP'], LL.MRP[2])
            # Append iteration counts.
            DBL['nIter']  = np.append(DBL['nIter'],  nIter)
            DBL['nStats'] = np.append(DBL['nStats'], nStats)
        else:
            # No need to update trajectory values.
            # Update the other statistics.
            DBL['nIter'][j]   = nIter
            DBL['nStats'][j]  = nStats
        # Go into the databook folder
        os.chdir(self.cart3d.RootDir)
        os.chdir(fldb)
        # Lineloads file name
        flds = frun.replace(os.sep, '-')
        # Write the loads
        lineload.WriteLDS(flds)
        # Write the seam curves if appropriate
        if q_seam:
            # Write both
            lineLoad.WriteSeam(fsmy, LL.smy)
            lineLoad.WriteSeam(fsmz, LL.smz)
        # Go back.
        os.chdir(fpwd)
    
    # Update or add an entry.
    def UpdateCase(self, i):
        """Update or add a trajectory to a data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> DB.UpdateCase(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Trajectory index
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the first data book component.
        c0 = self.Components[0]
        # Try to find a match existing in the data book.
        j = self[c0].FindMatch(i)
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
        nIter = int(util.GetTotalHistIter())
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
        elif self[c0]['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i."
                % (self[c0]['nIter'][j], nIter))
            q = True
        elif self[c0]['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Maximum number of iterations allowed.
        nMax = min(nIter-nMin, self.opts.get_nMaxStats())
        # Read residual
        H = CaseResid()
        # Loop through components.
        for comp in self.Components:
            # Read the iterative history for that component.
            FM = CaseFM(comp)
            # Extract the component databook.
            DBc = self[comp]
            # Loop through the transformations.
            for topts in self.opts.get_DataBookTransformations(comp):
                # Apply the transformation.
                FM.TransformFM(topts, self.x, i)
                
            # Process the statistics.
            s = FM.GetStats(nStats, nMax)
            # Get the corresponding residual drop
            nOrders = H.GetNOrders(s['nStats'])
            
            # Save the data.
            if np.isnan(j):
                # Add the the number of cases.
                DBc.n += 1
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DBc[k] = np.hstack((DBc[k], [getattr(self.x,k)[i]]))
                # Append values.
                for c in DBc.DataCols:
                    DBc[c] = np.hstack((DBc[c], [s[c]]))
                # Append residual drop.
                DBc['nOrders'] = np.hstack((DBc['nOrders'], [nOrders]))
                # Append iteration counts.
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DBc.DataCols:
                    DBc[c][j] = s[c]
                # Update the other statistics.
                DBc['nOrders'][j] = nOrders
                DBc['nIter'][j]   = nIter
                DBc['nStats'][j]  = s['nStats']
        # Go back.
        os.chdir(self.RootDir)
# class DataBook
        
            
# Function to automatically get inclusive data limits.
def get_ylim(ha, pad=0.05):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    :Call:
        >>> ymin, ymax = get_ylim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    return cape.get_ylim(ha, pad=pad)
    
# Function to automatically get inclusive data limits.
def get_xlim(ha, pad=0.05):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    return cape.get_xlim(ha, pad=pad)
# DataBook Plot functions


# Individual component data book
class DBComp(cape.dataBook.DBComp):
    """
    Individual component data book
    
    :Call:
        >>> DBi = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyCart.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
    :Outputs:
        *DBi*: :class:`pyCart.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
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
        >>> DBT = pyCart.dataBook.DBTarget(targ, x, opts)
    :Inputs:
        *targ*: :class:`pyCart.options.DataBook.DBTarget`
            Instance of a target source options interface
        *x*: :class:`pyCart.trajectory.Trajectory`
            Run matrix interface
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyCart.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    pass
# class DBTarget

    
        
        
# Individual component force and moment
class CaseFM(cape.dataBook.CaseFM):
    """
    This class contains methods for reading data about an the history of an
    individual component for a single case.  It reads the file :file:`$comp.dat`
    where *comp* is the name of the component.  From this file it determines
    which coefficients are recorded automatically.  If some of the comment lines
    from the Cart3D output file have been deleted, it guesses at the column
    definitions based on the number of columns.
    
    :Call:
        >>> FM = pyCart.dataBook.CaseFM(comp)
    :Inputs:
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *FM*: :class:`pyCart.aero.FM`
            Instance of the force and moment class
        *FM.coeffs*: :class:`list` (:class:`str`)
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
        * 2014-11-12 ``@ddalle``: Starter version
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.FM`
        * 2015-10-16 ``@ddalle``: Self-contained version
    """
    # Initialization method
    def __init__(self, comp):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2015-10-16 ``@ddalle``: Eliminated reliance on pyCart.Aero
        """
        # Save component name
        self.comp = comp
        # Get the working folder.
        fdir = util.GetWorkingFolder()
        # Expected name of the component history file
        fname = os.path.join(fdir, comp+'.dat')
        # Check if it exists.
        if not os.path.isfile(fname):
            # Make an empty CaseFM
            self.MakeEmpty()
            return
        # Otherwise, read the file.
        lines = open(fname).readlines()
        # Process the column meanings.
        self.ProcessColumnNames(lines)
        # Filter comments
        lines = [l for l in lines if not l.startswith('#')]
        # Convert all values to floats
        # (This is not guaranteed to be rectangular yet.)
        V = [[float(v) for v in l.split()] for l in lines]
        # Number of coefficients.
        n = len(self.coeffs)
        # Create an array with the original data
        A = np.array([v[0:1] + v[-n:] for v in V])
        # Get number of values in each raw data row.
        L = np.array([len(v) for v in V])
        # Check for columns without an extra column.
        if np.any(L == n+1):
            # At least one steady-state iteration.
            n0 = np.max(A[L==n+1,0])
            # Add that iteration number to the time-accurate steps.
            A[L!=n+1,0] += n0
        # Save the values.
        for k in range(n+1):
            # Set the values from column *k* of the data
            setattr(self,self.cols[k], A[:,k])
        
    # Function to make empty one.
    def MakeEmpty(self):
        """Create empty *CaseFM* instance
        
        :Call:
            >>> FM.MakeEmpty()
        :Inputs:
            *FM*: :class:`pyCart.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Make all entries empty.
        self.i = np.array([])
        self.CA = np.array([])
        self.CY = np.array([])
        self.CN = np.array([])
        self.CLL = np.array([])
        self.CLM = np.array([])
        self.CLN = np.array([])
        # Save a default list of columns and components.
        self.coeffs = ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']
        self.cols = ['i'] + self.coeffs
        
    # Process the column names
    def ProcessColumnNames(self, lines):
        """Determine column names
        
        :Call:
            >>> FM.ProcessColumnNames(lines)
        :Inputs:
            *FM*: :class:`pyCart.dataBook.CaseFM`
                Case force/moment history
            *lines*: :class:`list` (:class:`str`)
                List of lines from the data file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Get the lines from the file that explain the contents.
        lines = [l for l in lines if l.startswith('# cycle')]
        # Check for lines
        if len(lines) == 0:
            # Alert to the status of this file.
            print("Warning: no header found for component '%s'" % self.comp)
            # Use a data line.
            lines = [l for l in lines if not l.startswith('#')]
            # Check for lines.
            if len(lines) == 0:
                print("Warning: no data found for component '%s'" % self.comp)
                # Empty
                self.MakeEmpty()
                return
            # Split into values
            vals = lines[0].split()
            # Guess at the uses from contents.
            if len(vals) > 6:
                # Full force-moment
                self.C = ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']
                self.txt = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
            elif len(vals) in [4,5]:
                # Force only (ambiguous with 2D F&M
                self.C = ['CA', 'CY', 'CN']
                self.txt = ['Fx', 'Fy', 'Fz']
            else:
                # Guess at 2D force
                self.C = ['CA', 'CN']
                self.txt = ['Fx', 'Fz']
            # Add iteration to column list.
            self.cols = ['i'] + self.C
            self.txt.prepend('cycle')
            return
        # Read the contents
        self.txt = lines[0].lstrip('#').strip().split()
        self.cols = []
        self.coeffs = []
        # Loop through columns.
        for i in range(len(self.txt)):
            # Get the raw column name.
            col = self.txt[i]
            # Filter its name
            if col == 'cycle':
                # Iteration number
                self.cols.append('i')
            elif col == 'i':
                # Iteration number
                self.cols.append('i')
            elif col == 'Fx':
                # Axial force coefficient
                self.cols.append('CA')
                self.coeffs.append('CA')
            elif col == 'Fy':
                # Side force coefficient
                self.cols.append('CY')
                self.coeffs.append('CY')
            elif col == 'Fz':
                # Normal force coefficient
                self.cols.append('CN')
                self.coeffs.append('CN')
            elif col == 'Mx':
                # Rolling moment
                self.cols.append('CLL')
                self.coeffs.append('CLL')
            elif col == 'My':
                # Pitching moment
                self.cols.append('CLM')
                self.coeffs.append('CLM')
            elif col == 'Mz':
                # Yawing moment
                self.cols.append('CLN')
                self.coeffs.append('CLN')
            else:
                # Something else
                self.cols.append(col)
                self.coeffs.append(col)
        
    # Write a pure file.
    def Write(self, fname):
        """Write contents to force/moment file
        
        :Call:
            >>> FM.Write(fname)
        :Inputs:
            *FM*: :class:`pyCart.dataBook.CaseFM`
                Instance of the force and moment class
            *fname*: :class:`str`
                Name of file to write.
        :Versions:
            * 2015-03-02 ``@ddalle``: First version
        """
        # Open the file for writing.
        f = open(fname, 'w')
        # Start the header.
        f.write('# ')
        # Write the raw column titles
        f.write(' '.join(self.txt))
        # End the header.
        f.write('\n')
        # Initialize the data.
        A = np.array([self.i])
        # Loop through coefficients.
        for c in self.coeffs:
            # Append the data.
            A = np.vstack((A, [getattr(self,c)]))
        # Transpose.
        A = A.transpose()
        # Form the string flag.
        flg = '%i' + (' %s'*len(self.coeffs)) + '\n'
        # Loop through iterations.
        for v in A:
            # Write the line.
            f.write(flg % tuple(v))
        # Close the file.
        f.close()
# class CaseFM
    

# Aerodynamic history class
class CaseResid(cape.dataBook.CaseResid):
    """
    Iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = pyCart.dataBook.CaseResid()
    :Outputs:
        *hist*: :class:`pyCart.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Process the best data folder.
        fdir = util.GetWorkingFolder()
        # History file name.
        fhist = os.path.join(fdir, 'history.dat')
        # Read the file.
        lines = open(fhist).readlines()
        # Filter comments.
        lines = [l for l in lines if not l.startswith('#')]
        # Convert all the values to floats.
        A = np.array([[float(v) for v in l.split()] for l in lines])
        # Get the indices of steady-state iterations.
        # (Time-accurate iterations are marked with decimal step numbers.)
        i = np.array(['.' not in l.split()[0] for l in lines])
        # Check for steady-state iterations.
        if np.any(i):
            # Get the last steady-state iteration.
            n0 = np.max(A[i,0])
            # Add this to the time-accurate iteration numbers.
            A[np.logical_not(i),0] += n0
        else:
            # No steady-state iterations.
            n0 = 0
        # Process unsteady iterations if any.
        if A[-1,0] > n0:
            # Get the integer values of the iteration indices.
            # For *ni0*, 2000.000 --> 1999; 2000.100 --> 2000
            ni0 = np.array(A[n0:,0]-1e-4, dtype=int)
            # For *ni0*, 2000.000 --> 2000; 1999.900 --> 1999
            ni1 = np.array(A[n0:,0], dtype=int)
            # Look for iterations where the index crosses an integer.
            i0 = np.insert(np.where(ni0[1:] > ni0[:-1])[0]+1, 0, 0) + n0
            i1 = np.where(ni1[1:] > ni1[:-1])[0] + 1 + n0
        else:
            # No unsteady iterations.
            i0 = np.array([], dtype=int)
            i1 = np.array([], dtype=int)
        # Indices of steady-state iterations
        if n0 > 0:
            # Get the steady-state iterations from the '.' test above.
            i2 = np.where(i)[0]
        else:
            # No steady-state iterations
            i2 = np.arange(0)
        # Prepend the steady-state iterations.
        i0 = np.hstack((i2, i0))
        i1 = np.hstack((i2, i1))
        # Make sure these stupid things are ints.
        i0 = np.array(i0, dtype=int)
        i1 = np.array(i1, dtype=int)
        # Save the initial residuals.
        self.L1Resid0 = A[i0, 3]
        # Rewrite the history.dat file without middle subiterations.
        if not os.path.isfile('RUNNING'):
            # Iterations to keep.
            i = np.union1d(i0, i1)
            # Write the integer iterations and the first subiterations.
            open(fhist, 'w').writelines(np.array(lines)[i])
        # Eliminate subiterations.
        A = A[i1]
        # Save the number of iterations.
        self.nIter = int(A[-1,0])
        # Save the iteration numbers.
        self.i = A[:,0]
        # Save the CPU time per processor.
        self.CPUtime = A[:,1]
        # Save the maximum residual.
        self.maxResid = A[:,2]
        # Save the global residual.
        self.L1Resid = A[:,3]
        # Process the CPUtime used for steady cycles.
        if n0 > 0:
            # At least one steady-state cycle.
            # Find the index of the last steady-state iter.
            i0 = np.where(self.i==n0)[0] + 1
            # Get the CPU time used up to that point.
            t = self.CPUtime[i0-1]
        else:
            # No steady state cycles.
            i0 = 0
            t = 0.0
        # Process the unsteady cycles.
        if self.nIter > n0:
            # Add up total CPU time for unsteady cycles.
            t += np.sum(self.CPUtime[i0:])
        # Check for a 'user_time.dat' file.
        if os.path.isfile('user_time.dat'):
            # Loop through lines.
            for line in open('user_time.dat').readlines():
                # Check comment.
                if line.startswith('#'): continue
                # Add to the time everything except flowCart time.
                t += np.sum([float(v) for v in line.split()[2:]])
        # Save the time.
        self.CPUhours = t / 3600.
        
# class CaseResid

        
