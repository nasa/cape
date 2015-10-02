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
class DataBook(dict):
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
    
    # Initialization method
    def __init__(self, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the root directory.
        self.RootDir = os.getcwd()
        # Save the components
        self.Components = opts.get_DataBookComponents()
        # Save the folder
        self.Dir = opts.get_DataBookDir()
        # Save the trajectory.
        self.x = x.Copy()
        # Save the options.
        self.opts = opts
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                opts.mkdir(fdir)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in self.Components:
            # Initialize the data book.
            self[comp] = DBComp(comp, x, opts, fdir)
        # Initialize targets.
        self.Targets = []
        # Read the targets.
        for targ in opts.get_DataBookTargets():
            # Read the file.
            self.ReadTarget(targ)
        # Initialize line loads
        self.LineLoads = []
        
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions;
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DataBook "
        # Add the number of components.
        lbl += "nComp=%i, " % len(self.Components)
        # Add the number of conditions.
        lbl += "nCase=%i>" % self[self.Components[0]].n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
        
        
    # Function to read targets if necessary
    def ReadTarget(self, targ):
        """Read a data book target if it is not already present
        
        :Call:
            >>> DB.ReadTarget(targ)
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the pyCart data book class
            *targ*: :class:`str`
                Target name
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Try to access the target.
        try:
            self.Targets[targ]
        except Exception:
            # Read the file.
            self.Targets.append(DBTarget(targ, self.x, self.opts))
            
    
        
        
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
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check the min and max data
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            ymin = min(ymin, min(P.vertices[:,1]))
            ymax = max(ymax, max(P.vertices[:,1]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount,.
        ymax += pad*ymax
        ymin -= pad*ymin
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv
    
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
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount,.
        xmax += pad*xmax
        xmin -= pad*xmin
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv
# DataBook Plot functions


# Data book for an individual component
class DBComp(dict):
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
    # Initialization method
    def __init__(self, comp, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the list of columns for that coefficient.
        cols = opts.get_DataBookCols(comp)
        # Get the directory.
        fdir = opts.get_DataBookDir()
        
        # Construct the file name.
        fcomp = 'aero_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        
        # Save relevant information
        self.x = x
        self.opts = opts
        self.comp = comp
        self.cols = cols
        # Save the target translations.
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts.
        self.DataCols = opts.get_DataBookDataCols(comp)
        # Save the file name.
        self.fname = fname
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
            
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBComp %s, " % self.comp
        # Add the number of conditions.
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
    
    # Function to read data book files
    def Read(self, fname=None):
        """Read a single data book file or initialize empty arrays
        
        :Call:
            >>> DBc.Read()
            >>> DBc.Read(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.csv' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Try to read the file.
        try:
            # DataBook delimiter
            delim = self.opts.get_Delimiter()
            # Initialize column number
            nCol = 0
            # Loop through trajectory keys.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Read the column
                self[k] = np.loadtxt(fname, 
                    delimiter=delim, dtype=str(t), usecols=[nCol])
                # Increase the column number
                nCol += 1
            # Loop through the data book columns.
            for c in self.cols:
                # Add the column.
                self[c] = np.loadtxt(fname, delimiter=delim, usecols=[nCol])
                # Increase column number.
                nCol += 1
            # Column conversion
            # Number of orders of magnitude or residual drop.
            self['nOrders'] = np.loadtxt(fname, 
                delimiter=delim, dtype=float, usecols=[nCol])
            # Last iteration number
            self['nIter'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol+1])
            # Number of iterations used for averaging.
            self['nStats'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol+2])
        except Exception:
            # Initialize empty trajectory arrays.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Initialize an empty array.
                self[k] = np.array([], dtype=str(t))
            # Initialize the data columns.
            for c in self.cols:
                self[c] = np.array([])
            # Number of orders of magnitude of residual drop
            self['nOrders'] = np.array([], dtype=float)
            # Last iteration number
            self['nIter'] = np.array([], dtype=int)
            # Number of iterations used for averaging.
            self['nStats'] = np.array([], dtype=int)
        # Set the number of points.
        self.n = len(self[c])
        
    # Function to write data book files
    def Write(self, fname=None):
        """Write a single data book file
        
        :Call:
            >>> DBc.Write()
            >>> DBc.Write(fname)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.csv' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Check for a previous old file.
        if os.path.isfile(fname+'.old'):
            # Remove it.
            os.remove(fname+'.old')
        # Check for an existing data file.
        if os.path.isfile(fname):
            # Move it to ".old"
            os.rename(fname, fname+'.old')
        # DataBook delimiter
        delim = self.opts.get_Delimiter()
        # Open the file.
        f = open(fname, 'w')
        # Write the header.
        f.write("# aero data for '%s' extracted on %s\n" %
            (self.comp, datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')))
        # Empty line.
        f.write('#\n')
        # Reference quantities
        f.write('# Reference Area = %.6E\n' %
            self.opts.get_RefArea(self.comp))
        f.write('# Reference Length = %.6E\n' %
            self.opts.get_RefLength(self.comp))
        # Get the nominal MRP.
        xMRP = self.opts.get_RefPoint(self.comp)
        # Write it.
        f.write('# Nominal moment reference point:\n')
        f.write('# XMRP = %.6E\n' % xMRP[0])
        f.write('# YMRP = %.6E\n' % xMRP[1])
        # Check for 3D.
        if len(xMRP) > 2:
            f.write('# ZMRP = %.6E\n' % xMRP[2])
        # Empty line and start of variable list.
        f.write('#\n# ')
        # Loop through trajectory keys.
        for k in self.x.keys:
            # Just write the name.
            f.write(k + delim)
        # Loop through coefficients.
        for c in self.cols:
            # Write the name. (represents the means)
            f.write(c + delim)
        # Write the number of iterations and num used for stats.
        f.write('nOrders%snIter%snStats\n' % (delim, delim))
        # Loop through the database entries.
        for i in np.arange(self.n):
            # Write the trajectory points.
            for k in self.x.keys:
                f.write('%s%s' % (self[k][i], delim))
            # Write values.
            for c in self.cols:
                f.write('%.8E%s' % (self[c][i], delim))
            # Write the residual
            f.write('%.4f%s' % (self['nOrders'][i], delim))
            # Write number of iterations.
            f.write('%i%s%i\n' % (self['nIter'][i], delim, self['nStats'][i]))
        # Close the file.
        f.close()
        
    # Function to get sorting indices.
    def ArgSort(self, key=None):
        """Return indices that would sort a data book by a trajectory key
        
        :Call:
            >>> I = DBc.ArgSort(key=None)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the data book component
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Process the key.
        if key is None: key = self.x.keys[0]
        # Check for multiple keys.
        if type(key).__name__ in ['list', 'ndarray', 'tuple']:
            # Init pre-array list of ordered n-lets like [(0,1,0), ..., ]
            Z = zip(*[self[k] for k in key])
            # Init list of key definitions
            dt = []
            # Loop through keys to get data types (dtype)
            for k in key:
                # Get the type.
                dtk = self.x.defns[k]['Value']
                # Convert it to numpy jargon.
                if dtk in ['float']:
                    # Numeric value
                    dt.append((str(k), 'f'))
                elif dtk in ['int', 'hex', 'oct', 'octal']:
                    # Stored as an integer
                    dt.append((str(k), 'i'))
                else:
                    # String is default.
                    dt.append((str(k), 'S32'))
            # Create the array to be used for multicolumn sort.
            A = np.array(Z, dtype=dt)
            # Get the sorting order
            I = np.argsort(A, order=[str(k) for k in key])
        else:
            # Indirect sort on a single key.
            I = np.argsort(self[key])
        # Output.
        return I
            
    # Function to sort data book
    def Sort(self, key=None, I=None):
        """Sort a data book according to either a key or an index
        
        :Call:
            >>> DBc.Sort()
            >>> DBc.Sort(key)
            >>> DBc.Sort(I=None)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the pyCart data book component
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Process inputs.
        if I is not None:
            # Index array specified; check its quality.
            if type(I).__name__ not in ["ndarray", "list"]:
                # Not a suitable list.
                raise TypeError("Index list is unusable type.")
            elif len(I) != self.n:
                # Incompatible length.
                raise IndexError(("Index list length (%i) " % len(I)) +
                    ("is not equal to data book size (%i)." % self.n))
        else:
            # Default key if necessary
            if key is None: key = self.x.keys[0]
            # Use ArgSort to get indices that sort on that key.
            I = self.ArgSort(key)
        # Sort all fields.
        for k in self:
            # Sort it.
            self[k] = self[k][I]
            
    # Find the index of the point in the trajectory.
    def GetTrajectoryIndex(self, j):
        """Find an entry in the run matrix (trajectory)
        
        :Call:
            >>> i = DBc.GetTrajectoryIndex(self, j)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the pyCart data book component
            *j*: :class:`int`
                Index of the case from the databook to try match
        :Outputs:
            *i*: :class:`int`
                Trajectory index or ``None``
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Initialize indices (assume all trajectory points match to start).
        i = np.arange(self.x.nCase)
        # Loop through keys requested for matches.
        for k in self.x.keys:
            # Get the target value from the data book.
            v = self[k][j]
            # Search for matches.
            try:
                # Filter test criterion.
                ik = np.where(getattr(self.x,k) == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == getattr(self.x,k)[-1]):
                    # Add the last element.
                    ik = np.union1d(ik, [self.x.nCase-1])
                # Restrict to rows that match above.
                i = np.intersect1d(i, ik)
            except Exception:
                return None
        # Output
        try:
            # There should be one match.
            return i[0]
        except Exception:
            # No matches.
            return None
        
    # Find an entry by trajectory variables.
    def FindMatch(self, i):
        """Find an entry by run matrix (trajectory) variables
        
        It is assumed that exact matches can be found.
        
        :Call:
            >>> j = DBc.FindMatch(i)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the CAPE data book component
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of index that matches the trajectory case or ``NaN``
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Loop through keys requested for matches.
        for k in self.x.keys:
            # Get the target value (from the trajectory)
            v = getattr(self.x,k)[i]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(self[k] == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == self[k][-1]):
                    # Add the last element.
                    jk = np.union1d(jk, [len(self[k])-1])
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                # No match found.
                return np.nan
        # Output
        try:
            # There should be exactly one match.
            return j[0]
        except Exception:
            # Return no match.
            return np.nan
# class DBComp


# Data book target instance
class DBTarget(dict):
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
        *DBT*: :class:`cape.dataBook.DBTarget`
            Instance of the CAPE data book target class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, targ, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
            * 2015-06-03 ``@ddalle``: Added trajectory, split into methods
        """
        # Save the target options
        self.opts = opts
        self.topts = targ
        # Save the trajectory.
        self.x = x.Copy()
        
        # Read the data
        self.ReadData()
        # Process the columns.
        self.ProcessColumns()
        # Make the trajectory data match the available list of points.
        self.UpdateTrajectory()
    
    # Read the data
    def ReadData(self):
        """Read data file according to stored options
        
        :Call:
            >>> DBT.ReadData()
        :Inputs:
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
        """
        # Source file
        fname = self.topts.get_TargetFile()
        # Name of this target.
        tname = self.topts.get_TargetName()
        # Check for the file.
        if not os.path.isfile(fname):
            raise IOError(
                "Target source file '%s' could not be found." % fname)
        # Save the name.
        self.Name = tname
        # Delimiter
        delim = self.topts.get_Delimiter()
        # Comment character
        comchar = self.topts.get_CommentChar()
        # Open the file again.
        f = open(fname)
        # Loop until finding a line that doesn't begin with comment char.
        line = comchar
        nskip = -1
        while line.strip().startswith(comchar):
            # Save the old line.
            headers = line
            # Read the next line
            line = f.readline()
            nskip += 1
        # Close the file.
        f.close()
        # Translate into headers
        self.headers = headers.lstrip('#').strip().split(delim)
        # Save number of points.
        self.n = len(self.headers)

        # Read it.
        try:
            # Read the target all at once.
            self.ReadAllData(fname, delimiter=delim, skiprows=nskip)
        except Exception:
            # Read the data by columns.
            self.ReadDataByColumn(fname, delimiter=delim, skiprows=nskip)

    # Read the data file all at once.
    def ReadAllData(self, fname, delimiter=",", skiprows=0):
        """Read target data file all at once

        :Call:
            >>> DBT.ReadAllData(fname, delimiter=",", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: First version
        """
        # Read the data.
        self.data = np.loadtxt(fname, delimiter=delimiter,
            skiprows=skiprows, dtype=float).transpose()
        # Save the number of cases.
        self.nCase = len(self.data[0])

    # Read data one column at a time
    def ReadDataByColumn(self, fname, delimiter=",", skiprows=0):
        """Read target data one column at a time
        
        :Call:
            >>> DBT.ReadDataByColumn(fname, delimiter=",", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: First version
        """
        # Initialize data.
        self.data = []
        # Loop through columns.
        for i in range(self.n):
            # Try reading as a float second.
            try:
                self.data.append(np.loadtxt(fname, delimiter=delimiter,
                    skiprows=skiprows, dtype=float, usecols=(i,)))
                continue
            except Exception:
                pass
            # Try reading as a string last.
            self.data.append(np.loadtxt(fname, delimiter=delimiter,
                skiprows=skiprows, dtype=str, usecols=(i,)))
        # Number of cases
        self.nCase = len(self.data[0])

    
    # Read the columns and split into useful dict.
    def ProcessColumns(self):
        """Process data columns and split into dictionary keys
        
        :Call:
            >>> DBT.ProcessColumns()
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
        """
        # Initialize data fields.
        cols = []
        # Names of columns corresponding to trajectory keys.
        tkeys = self.topts.get_Trajectory()
        # Loop through trajectory fields.
        for k in self.x.keys:
            # Get field name.
            col = tkeys.get(k, k)
            # Check for manually turned-off trajectory.
            if col is None:
                # Manually turned off.
                continue
            elif col not in self.headers:
                # Not present in the file.
                continue
            # Append the key.
            cols.append(col)
        # Initialize translations for force/moment coefficients
        ckeys = {}
        # List of potential components.
        tcomps = self.topts.get_TargetComponents()
        # Check for default.
        if tcomps is None:
            # Use all components.
            tcomps = self.opts.get_DataBookComponents()
        # Process the required fields.
        for comp in tcomps:
            # Initialize translations for this component.
            ckeys[comp] = {}
            # Get targets for this component.
            ctargs = self.opts.get_CompTargets(comp)
            # Loop through the possible force/moment coefficients.
            for c in ['CA','CY','CN','CLL','CLM','CLN']:
                # Get the translated name
                ctarg = ctargs.get(c, c)
                # Get the target source for this entry.
                if '/' not in ctarg:
                    # Only one target source; assume it's this one.
                    ti = self.Name
                    fi = ctarg
                else:
                    # Read the target name.
                    ti = ctarg.split('/')[0]
                    # Name of the column
                    fi = ctarg.split('/')[1]
                # Check if the target is from this target source.
                if ti != self.Name: continue
                # Check if the column is present in the headers.
                if fi not in self.headers:
                    # Check for default.
                    if ctarg in ctargs:
                        # Manually specified and not recognized: error
                        raise KeyError("There is no field '%s' in file '%s'."
                            % (fi, self.topts.get_TargetFile()))
                    else:
                        # Autoselected name but not in the file.
                        continue
                # Add the field if necessary.
                if fi in cols:
                    raise IOError(
                        "Column '%s' of file '%s' used more than once."
                        % (fi, self.topts.get_TargetFile()))
                # Add the column.
                cols.append(fi)
                # Add to the translation dictionary.
                ckeys[comp][c] = fi
        # Extract the data into a dict with a key for each relevant column.
        for col in cols:
            # Find it and save it as a key.
            self[col] = self.data[self.headers.index(col)]
        # Save the data keys translations.
        self.ckeys = ckeys
        
    # Match the databook copy of the trajectory
    def UpdateTrajectory(self):
        """Match the trajectory to the cases in the data book
        
        :Call:
            >>> DBT.UpdateTrajectory()
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Get trajectory key specifications.
        tkeys = self.topts.get_Trajectory()
        # Loop through the trajectory keys.
        for k in self.x.keys:
            # Get the column name in the target.
            tk = tkeys.get(k, k)
            # Set the value if it's a default.
            tkeys.setdefault(k, tk)
            # Check for ``None``
            if (tk is None) or (tk not in self):
                # Use NaN as the value.
                setattr(self.x,k, np.nan*np.ones(self.n))
                # Set the value.
                tkeys[k] = None
                continue
            # Update the trajectory values to match those of the trajectory.
            setattr(self.x,k, self[tk])
            # Set the text.
            self.x.text[k] = [str(xk) for xk in self[tk]]
        # Save the key translations.
        self.xkeys = tkeys
        # Set the number of cases in the "trajectory."
        self.x.nCase = self.nCase
        
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, comp, coeff, I, **kw):
        """Plot a sweep of one coefficient over several cases
        
        :Call:
            >>> h = DBT.PlotCoeff(comp, coeff, I, **kw)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: [ {None} | :class:`str` ]
                Trajectory key for *x* axis (or plot against index if ``None``)
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Legend*: [ {True} | False ]
                Whether or not to use a legend
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *MinMax*: [ {False} | True ]
                Whether to plot minimum and maximum over iterative history
            *LineOptionss*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *FigWidth*: :class:`float`
                Width of figure in inches
            *FigHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Get horizontal key.
        xk = kw.get('x')
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
        # Iterative uncertainty options
        qmmx = kw.get('MinMax', 0)
        ksig = kw.get('StDev')
        # Initialize output
        h = {}
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        else:
            # Check if the value is present.
            if xk not in self.xkeys: return
            # Extract the values.
            xv = self[self.xkeys[xk]][I]
        # Check if the coefficient is in the target data.
        if (comp not in self.ckeys) or (coeff not in self.ckeys[comp]):
            # No data.
            return
        # Extract the mean values.
        yv = self[self.ckeys[comp][coeff]][I]
        # Initialize label.
        lbl = kw.get('Label', '%s/%s' % (self.Name, comp))
        # -----------------------
        # Standard Deviation Plot
        # -----------------------
        # Initialize plot options for standard deviation.
        kw_s = odict(color='c', lw=0.0,
            facecolor='c', alpha=0.35, zorder=1)
        # Show iterative standard deviation.
        if ksig:
            # Add standard deviation to label.
            lbl = u'%s (\u00B1%s\u03C3)' % (lbl, ksig)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("StDevOptions")):
                # Option.
                o_k = kw["StDevOptions"][k]
                # Override the default option.
                if o_k is not None: kw_s[k] = o_k
            # Get the standard deviation value.
            sv = DBc[coeff+"_std"][I]
            # Plot it.
            h['std'] = plt.fill_between(xv, yv-ksig*sv, yv+ksig*sv, **kw_s)
        # ------------
        # Min/Max Plot
        # ------------
        # Initialize plot options for min/max
        kw_m = odict(color='m', lw=0.0,
            facecolor='m', alpha=0.35, zorder=2)
        # Show min/max options
        if qmmx:
            # Add min/max to label.
            lbl = u'%s (min/max)' % (lbl)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("MinMaxOptions")):
                # Option
                o_k = kw["MinMaxOptions"][k]
                # Override the default option.
                if o_k is not None: kw_m[k] = o_k
            # Get the min and max values.
            ymin = DBc[coeff+"_min"][I]
            ymax = DBc[coeff+"_max"][I]
            # Plot it.
            h['max'] = plt.fill_between(xv, ymin, ymax, **kw_m)
        # ------------
        # Primary Plot
        # ------------
        # Initialize plot options for primary plot
        kw_p = odict(color='r', marker='^', zorder=7, ls='-')
        # Plot options
        for k in util.denone(kw.get("LineOptions")):
            # Option
            o_k = kw["LineOptions"][k]
            # Override the default option.
            if o_k is not None: kw_p[k] = o_k
        # Label
        kw_p.setdefault('label', lbl)
        # Plot it.
        h['line'] = plt.plot(xv, yv, **kw_p)
        # ----------
        # Formatting
        # ----------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to requested ylabel
        if ly and ly != coeff:
            # Combine labels.
            ly = ly + '/' + coeff
        else:
            # Use the coefficient.
            ly = coeff
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(ly)
        # Get limits to include all data.
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Make sure data is included.
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
        # Legend.
        if kw.get('Legend', True):
            # Add extra room for the legend.
            h['ax'].set_ylim((ymin, 1.2*ymax-0.2*ymin))
            # Font size checks.
            if len(h['ax'].get_lines()) > 5:
                # Very small
                fsize = 7
            else:
                # Just small
                fsize = 9
            # Activate the legend.
            try:
                # Use a font that has the proper symbols.
                h['legend'] = h['ax'].legend(loc='upper center',
                    prop=dict(size=fsize, family="DejaVu Sans"),
                    bbox_to_anchor=(0.5,1.05), labelspacing=0.5)
            except Exception:
                # Default font.
                h['legend'] = h['ax'].legend(loc='upper center',
                    prop=dict(size=fsize),
                    bbox_to_anchor=(0.5,1.05), labelspacing=0.5)
        # Figure dimensions.
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # Output
        return h
        
    # Find an entry by trajectory variables.
    def FindMatch(self, x, i):
        """Find an entry by run matrix (trajectory) variables
        
        Cases will be considered matches by comparing variables specified in 
        the *DataBook* section of :file:`pyCart.json` as cases to compare
        against.  Suppose that the control file contains the following.
        
        .. code-block:: python
        
            "DataBook": {
                "Targets": {
                    "Name": "Experiment",
                    "File": "WT.dat",
                    "Trajectory": {"alpha": "ALPHA", "Mach": "MACH"}
                    "Tolerances": {
                        "alpha": 0.05,
                        "Mach": 0.01
                    }
                }
            }
        
        Then any entry in the data book target that matches the Mach number
        within 0.01 (using a column labeled *MACH*) and alpha to within 0.05 is
        considered a match.  If there are more trajectory variables, they are
        not used for this filtering of matches.
        
        :Call:
            >>> j = DBT.FindMatch(x, i)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target data carrier
            *x*: :class:`pyCart.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices that match the trajectory within tolerances
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Get the trajectory key translations.   This determines which keys to
        # filter and what those keys are called in the source file.
        tkeys = self.topts.get_Trajectory()
        # Loop through keys requested for matches.
        for k in tkeys:
            # Get the tolerance.
            tol = self.topts.get_Tol(k)
            # Get the target value (from the trajectory)
            v = getattr(x,k)[i]
            # Get the name of the column according to the source file.
            c = tkeys[k]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(np.abs(self[c] - v) <= tol)[0]
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                pass
        # Output
        return j
# class DBTarget



