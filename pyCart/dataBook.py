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

# Use this to only update entries with newer iterations.
from .case import GetCurrentIter, GetWorkingFolder
# Finer control of dicts
from .options import odict
# Utilities or advanced statistics
from . import util
# Line loads
from . import lineLoad

# Template module
import cape.dataBook

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# ---------------------------------
#-->

# Placeholder variables for plotting functions.
plt = 0

# Radian -> degree conversion
deg = np.pi / 180.0

# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    """Import :mod:`matplotlib.pyplot` if not loaded
    
    :Call:
        >>> pyCart.dataBook.ImportPyPlot()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
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
    """
    
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
                lineLoad.DBLineLoad(self.cart3d, comp))
    
    
            
    
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
        nIter = int(GetCurrentIter())
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
        # Read the history.
        A = Aero(self.Components)
        # Maximum number of iterations allowed.
        nMax = min(nIter-nMin, self.opts.get_nMaxStats())
        # Loop through components.
        for comp in self.Components:
            # Extract the component history and component databook.
            FM = A[comp]
            DC = self[comp]
            # Loop through the transformations.
            for topts in self.opts.get_DataBookTransformations(comp):
                # Apply the transformation.
                FM.TransformFM(topts, self.x, i)
                
            # Process the statistics.
            s = FM.GetStats(nStats, nMax)
            # Get the corresponding residual drop
            nOrders = A.Residual.GetNOrders(s['nStats'])
            
            # Save the data.
            if np.isnan(j):
                # Add the the number of cases.
                DC.n += 1
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DC[k] = np.hstack((DC[k], [getattr(self.x,k)[i]]))
                # Append values.
                for c in DC.DataCols:
                    DC[c] = np.hstack((DC[c], [s[c]]))
                # Append residual drop.
                DC['nOrders'] = np.hstack((DC['nOrders'], [nOrders]))
                # Append iteration counts.
                DC['nIter']  = np.hstack((DC['nIter'], [nIter]))
                DC['nStats'] = np.hstack((DC['nStats'], [s['nStats']]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DC.DataCols:
                    DC[c][j] = s[c]
                # Update the other statistics.
                DC['nOrders'][j] = nOrders
                DC['nIter'][j]   = nIter
                DC['nStats'][j]  = s['nStats']
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
        
        
# Aerodynamic history class
class Aero(cape.dataBook.Aero):
    """
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> aero = pyCart.dataBook.Aero(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class, similar to dictionary of force
            and/or moment histories
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.Aero`
    """
    
    # Initialization method
    def __init__(self, comps=[]):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Process the best data folder.
        fdir = GetWorkingFolder()
        # Read the loadsCC.dat file to see what components are requested.
        self.ReadLoadsCC()
        # Read the residuals.
        self.Residual = CaseResid()
        # Default component list.
        if (type(comps).__name__ in ["str", "unicode", "int"]):
            # Make a singleton list.
            comps = [comps]
        elif len(comps) < 1:
            # Extract keys from dictionary.
            comps = self.Components.keys()
        # Loop through components.
        for comp in comps:
            # Expected name of the history file.
            fname = os.path.join(fdir, comp+'.dat')
            # Check if it exists.
            if not os.path.isfile(fname):
                # Warn and got to the next component.
                print("Warning: Component '%s' was not found." % comp)
                continue
            # Otherwise, read the file.
            lines = open(fname).readlines()
            # Filter comments
            lines = [l for l in lines if not l.startswith('#')]
            # Convert all the values to floats
            # Can't make this an array yet because it's not rectangular.
            V = [[float(v) for v in l.split()] for l in lines]
            # Columns to use: 0 and {-6,-3}
            try:
                # Read from loadsCC.dat
                n = len(self.Components[comp]['C'])
            except Exception:
                # Infer; could fail for some 2D cases
                n = (len(V[0])/3) * 3
            # Create an array with the original data.
            A = np.array([v[0:1] + v[-n:] for v in V])
            # Get the number of entries in each row.
            # This will be one larger if a time-accurate iteration.
            # It's a column of zeros, and it's the second column.
            L = np.array([len(v) for v in V])
            # Check for steady-state iterations.
            if np.any(L == n+1):
                # At least one steady-state iteration
                n0 = np.max(A[L==n+1,0])
                # Add that iteration number to the time-accurate steps.
                A[L!=n+1,0] += n0
            # Extract info from components for readability
            try:
                # Read from loadsCC.dat
                d = self.Components[comp]
            except Exception:
                # No MRP known.
                d = {"MRP": [0.0, 0.0, 0.0]}
                # Decide force or moment
                if n == 3:
                    # Apparently a force
                    d['C'] = ["CA", "CY", "CN"]
                else:
                    # Apparemntly moment
                    d['C'] = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
            # Make the component.
            self[comp] = CaseFM(d['C'], MRP=d['MRP'], A=A)
            
    # Function to calculate statistics and select ideal nStats
    def GetStats(self, nStats=0, nMax=0, nLast=None):
        """
        Get statistics for all components and decide how many iterations to use
        for calculating statistics.
        
        The number of iterations to use is selected such that the sum of squares
        of all errors (all coefficients of each component) is minimized.  Only
        *nStats*, *nMax*, and integer multiples of *nStats* are considered as
        candidates for the number of iterations to use.
        
        :Call:
            >>> S = A.GetStats(nStats, nMax=0, nLast=None)
        :Inputs:
            *nStats*: :class:`int`
                Nominal number of iterations to use in statistics
            *nMax*: :class:`int`
                Maximum number of iterations to use for statistics
            *nLast*: :class:`int`
                Specific iteration at which to get statistics
        :Outputs:
            *S*: :class:`dict` (:class:`dict` (:class:`float`))
                Dictionary of statistics for each component
        :See also:
            :func:`pyCart.dataBook.CaseFM.GetStats`
        :Versions:
            * 2015-02-28 ``@ddalle``: First version
        """
        # Initialize statistics for this count.
        S = {}
        # Loop through components.
        for comp in self:
            # Get the statistics.
            S[comp] = self[comp].GetStats(nStats, nMax=nMax, nLast=nLast)
        # Output
        return S
    
    # Function to read 'loadsCC.dat'
    def ReadLoadsCC(self):
        """Read forces and moments from a :file:`loadsCC.dat` file if possible
        
        :Call:
            >> A.ReadLoadsCC()
        :Inputs:
            *A*: :class:`pyCart.aero.Aero`
                Instance of the aero history class
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize list of components.
        self.Components = {}
        # Get working directory.
        fdir = GetWorkingFolder()
        # Path to the file.
        fCC = os.path.join(fdir, 'loadsCC.dat')
        # Check for the file.
        if not os.path.isfile(fCC):
            # Change the loadsTRI.dat
            fCC = os.path.join(fdir, 'loadsTRI.dat')
        # Try again.
        if not os.path.isfile(fCC):
            # Change to common directory.
            fCC = os.path.join('..', '..', 'inputs', 'loadsCC.dat')
        # Check for the last time.
        if not os.path.isfile(fCC):
            # Nothing to do.
            return None
        # Read the file.
        linesCC = open(fCC).readlines()
        # Loop through the lines.
        for line in linesCC:
            # Strip line.
            line = line.strip()
            # Check for empty line or comment.
            if (not line) or line.startswith('#'): continue
            # Get name of component.
            comp = line.split()[0]
            # Add line to dictionary if necessary.
            if comp not in self.Components:
                self.Components[comp] = {'C':[], 'MRP':None}
            # Try to get the coefficient name.
            try:
                # Find text like '(C_A)' and return 'C_A'.
                c = re.search('\(([A-Za-z_]+)\)', line).group(1)
            except Exception:
                # Failed to find expected text.
                continue
            # Filter the coefficient.
            if c == 'C_A':
                # Axial force
                self.Components[comp]['C'].append('CA')
                continue
            elif c == 'C_Y': 
                # Lateral force
                self.Components[comp]['C'].append('CY')
                continue
            elif c == 'C_N':
                # Normal force
                self.Components[comp]['C'].append('CN')
                continue
            elif c == 'C_M_x':
                # Rolling moment
                self.Components[comp]['C'].append('CLL')
            elif c == 'C_M_y':
                # Pitching moment
                self.Components[comp]['C'].append('CLM')
            elif c == 'C_M_z':
                # Yaw moment
                self.Components[comp]['C'].append('CLN')
            else:
                # Extra coefficient such as lift, drag, etc.
                continue
            # Only process reference point once.
            if self.Components[comp]['MRP'] is not None: continue
            # Try to find reference point.
            try:
                # Search for text like '(17.0, 0, 0)'.
                txt = re.search('\(([0-9EeDd., +-]+)\)', line).group(1)
                # Split into coordinates.
                MRP = np.array([float(v) for v in txt.split(',')])
                # Save it.
                self.Components[comp]['MRP'] = MRP
            except Exception:
                # Failed to find expected text.
                print("Warning: no reference point in line:\n  '%s'" % line)
                # Function to plot a single coefficient.
    
    
            
    # Function to plot several coefficients.
    def Plot(self, comp, C, d={}, **kw):
        """Plot one or several component histories
        
        :Call:
            >>> h = AP.Plot(comp, C, d={}, n=1000, nAvg=100, **kw)
        :Inputs:
            *AP*: :class:`pyCart.aero.Plot`
                Instance of the force history plotting class
            *comp*: :class:`str`
                Name of component to plot
            *nRow*: :class:`int`
                Number of rows of subplots to make
            *nCol*: :class:`int`
                Number of columns of subplots to make
            *C*: :class:`list` (:class:`str`)
                List of coefficients or ``'L1'`` to plot
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                First iteration to plot
            *nLast*: :class:`int`
                Last iteration to plot
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *d*: :class:`dict`
                Dictionary of deltas for each component
            *tag*: :class:`str` 
                Tag to put in upper corner, for instance case number and name
            *restriction*: :class:`str`
                Type of data, e.g. ``"SBU - ITAR"`` or ``"U/FOUO"``
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
            * 2015-03-04 ``@ddalle``: Added *nFirst* and *nLast*
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Read inputs
        nRow = kw.get('nRow', 2)
        nCol = kw.get('nCol', 2)
        n    = kw.get('n', 1000)
        nAvg = kw.get('nAvg', 100)
        nBin = kw.get('nBin', 20)
        d0   = kw.get('d0', 0.01)
        # Window control
        nFirst = kw.get('nFirst')
        nLast  = kw.get('nLast')
        # Check for single input.
        if type(C).__name__ == "str": C = [C]
        # Number of components
        nC = len(C)
        # Check inputs.
        if nC > nRow*nCol:
            raise IOError("Too many components for %i rows and %i columns" 
                % (nRow, nCol))
        # Initialize handles.
        h = CasePlot()
        # Loop through components.
        for i in range(nC):
            # Get coefficient.
            c = C[i]
            # Pull up the subplot.
            plt.subplot(nRow, nCol, i+1)
            # Check if residual was requested.
            if c == 'L1':
                # Plot it.
                h[c] = self.PlotL1(n=n, nFirst=nFirst, nLast=nLast)
            elif c.endswith('hist'):
                # Get the coeff name.
                ci = c[:-4]
                # Plot histogram
                h[c] = self.PlotCoeffHist(comp, ci, nAvg=nAvg, nBin=nBin, 
                    nLast=nLast)
            else:
                # Get the delta
                di = d.get(c, d0)
                # Plot
                h[c] = self.PlotCoeff(comp, c, n=n, nAvg=nAvg, d=di,
                    nFirst=nFirst, nLast=nLast)
            # Turn off overlapping xlabels for condensed plots.
            if (nCol==1 or nRow>2) and (i+nCol<nC):
                # Kill the xlabel and xticklabels.
                h[c]['ax'].set_xticklabels(())
                h[c]['ax'].set_xlabel('')
        # Max of number 
        n0 = max(nCol, nRow)
        # Determine target font size.
        if n0 == 1:
            # Font size (default)
            fsize = 12
        elif n0 == 2:
            # Smaller
            fsize = 9
        else:
            # Really small
            fsize = 8
        # Loop through the text labels.
        for h_t in plt.gcf().findobj(Text):
            # Apply the target font size.
            h_t.set_fontsize(fsize)
        # Add tag.
        tag = kw.get('tag', '')
        h['tag'] = plt.figtext(0.015, 0.985, tag, verticalalignment='top')
        # Add restriction.
        txt = kw.get('restriction', '')
        h['restriction'] = plt.figtext(0.5, 0.01, txt,
            horizontalalignment='center')
        # Add PASS label (empty but handle is useful)
        h['pass'] = plt.figtext(0.99, 0.97, "", color="#00E500",
            horizontalalignment='right')
        # Add iteration label
        h['iter'] = plt.figtext(0.99, 0.94, "%i/" % self[comp].i[-1],
            horizontalalignment='right', size=9)
        # Attempt to use the tight_layout() utility.
        try:
            # Add room for labels with *rect*, and tighten up other margins.
            plt.gcf().tight_layout(pad=0.2, w_pad=0.5, h_pad=0.7,
                rect=(0.01,0.015,0.99,0.91))
        except Exception:
            pass
        # Save the figure.
        h['fig'] = plt.gcf()
        # Output
        return h
        
    # Function to add plot restriction label
    
            
# class Aero
    
    
# Individual component force and moment
class CaseFM(cape.dataBook.CaseFM):
    """
    This class contains methods for reading data about an the histroy of an
    individual component for a single case.  The list of available components
    comes from a :file:`loadsCC.dat` file if one exists.
    
    :Call:
        >>> FM = pyCart.dataBook.CaseFM(C, MRP=None, A=None)
    :Inputs:
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *FM*: :class:`pyCart.aero.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list` (:class:`str`)
            List of coefficients
        *FM.MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
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
    """
        
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
        f.write('# cycle')
        # Check for basic force coefficients.
        if 'CA' in self.coeffs:
            f.write(' Fx Fy')
        # Check for side force.
        if 'CY' in self.coeffs:
            f.write(' Fz')
        # Check for 3D moments.
        if 'CLN' in self.coeffs:
            # 3D moments
            f.write(' CLL CLM CLN')
        elif 'CLM' in self.coeffs:
            # 2D, only pitching moment
            f.write(' CLM')
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
        fdir = GetWorkingFolder()
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

        
