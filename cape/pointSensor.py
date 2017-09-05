"""
Point sensors module: :mod:`cape.pointSensor`
===============================================

This module contains a class for reading and averaging point sensors.  It is not
included in the :mod:`cape.dataBook` module in order to give finer import
control when used in other modules.

Point sensors are set into groups, so the ``"DataBook"`` section of the JSON
file may have a point sensor group called ``"P1"`` that includes points
``"p1"``, ``"p2"``, and ``"p3"``.

If a data book is read in as *DB*, the point sensor group *DBP* for group
``"P1"`` and the point sensor *p1* are obtained using the commands below.

    .. code-block:: python
    
        // Point sensor group
        DBP = DB.PointSensors["P1"]
        // Individual point sensor
        p1 = DBP["p1"]
"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from .options   import odict
# Utilities and advanced statistics
from . import util

# Basis module
from . import dataBook

# Placeholder variables for plotting functions.
plt = 0

# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    """Import :mod:`matplotlib.pyplot` if not loaded
    
    :Call:
        >>> cape.dataBook.ImportPyPlot()
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


# Data book for group of point sensors
class DBPointSensorGroup(dict):
    """
    Point sensor group data book
    
    :Call:
        >>> DBPG = DBPointSensorGroup(x, opts, name)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *name*: :class:`str` | ``None``
            Name of data book item (defaults to *pt*)
        *pts*: :class:`list` (:class:`str`) | ``None``
            List of points to read, by default all points in the group
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2015-12-04 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
    # Initialization method
    def __init__(self, x, opts, name, **kw):
        """Initialization method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Save the interface.
        self.x = x
        self.opts = opts
        # Save the name
        self.name = name
        # Get the list of points.
        self.pts = kw.get('pts', opts.get_DBGroupPoints(name))
        # Loop through the points.
        for pt in self.pts:
            self[pt] = DBPointSensor(x, opts, pt, name)
            
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBPointSensorGroup %s, " % self.name
        # Number of cases in book
        lbl += "nPoint=%i>" % len(self.pts)
        # Output
        return lbl
    __str__ = __repr__
  # >
  
  # =========
  # I/O
  # =========
  # <
    # Output method
    def Write(self):
        """Write to file each point sensor data book in a group
        
        :Call:
            >>> DBPG.Write()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            # Sort it.
            self[pt].Sort()
            # Write it
            self[pt].Write()
  # >
  
  # ============
  # Update
  # ============
  # <
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBPG.UpdateCase(i)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Reference point
        pt = self.pts[0]
        DBP = self[pt]
        # Check update status.
        q, P = DBP._UpdateCase(i)
        # Exit if no return necessary
        if not q: return
        # Try to find a match existing in the data book
        j = self[pt].FindMatch(i)
        # Determine ninimum number of iterations required
        nStats = self.opts.get_nStats(self.name)
        nLast  = self.opts.get_nLastStats(self.name)
        # Get list of iterations
        iIter = P.i
        
        # Loop through points.
        for pt in self.pts:
            # Find the point.
            kpt = P.GetPointSensorIndex(pt)
            # Calculate statistics
            s = P.GetStats(kpt, nStats=nStats, nLast=nLast)
            
            # Save the data.
            if np.isnan(j):
                # Add the the number of cases.
                self[pt].n += 1
                # Append trajectory values.
                for k in self[pt].xCols:
                    # I hate the way NumPy does appending.
                    self[pt][k] = np.hstack((self[pt][k], 
                        [getattr(self.x,k)[i]]))
                # Append values.
                for c in self[pt].fCols:
                    self[pt][c] = np.hstack((self[pt][c], [s[c]]))
                # Append iteration counts.
                self[pt]['nIter']  = np.hstack((self[pt]['nIter'], iIter[-1:]))
                self[pt]['nStats'] = np.hstack((self[pt]['nStats'], [nStats]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in self[pt].fCols:
                    self[pt][c][j] = s[c]
                # Update the other statistics.
                self[pt]['nIter'][j]   = iIter[-1]
                self[pt]['nStats'][j]  = nStats
  # >
  
  # ============
  # Organization
  # ============
  # <
    # Sorting method
    def Sort(self):
        """Sort point sensor group
        
        :Call:
            >>> DBPG.Sort()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2016-03-08 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            self[pt].Sort()
            
    # Match the databook copy of the trajectory
    def UpdateTrajectory(self):
        """Match the trajectory to the cases in the data book
        
        :Call:
            >>> DBPG.UpdateTrajectory()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2015-05-22 ``@ddalle``: First version
        """
        # Get the first component.
        DBc = self[self.pts[0]]
        # Loop through the fields.
        for k in self.x.keys:
            # Copy the data.
            setattr(self.x, k, DBc[k])
            # Set the text.
            self.x.text[k] = [str(xk) for xk in DBc[k]]
        # Set the number of cases.
        self.x.nCase = DBc.n
  # >
# class DBPointSensorGroup


# Data book of point sensors
class DBPointSensor(dataBook.DBBase):
    """Point sensor data book
    
    Plotting methods are inherited from :class:`cape.dataBook.DBBase`, including
    :func:`cape.dataBook.DBBase.PlotHist` for plotting historgrams of point
    sensor results in particular.
    
    :Call:
        >>> DBP = DBPointSensor(x, opts, pt, name=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *pt*: :class:`str`
            Name of point
        *name*: :class:`str` | ``None``
            Name of data book item (defaults to *pt*)
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2015-12-04 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, x, opts, pt, name=None, **kw):
        """Initialization method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.pt = pt
        # Save data book title
        if name is None:
            # Default name
            self.comp = pt
        else:
            # Specified name
            self.comp = name
        
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Folder containing the data book
        fdir = opts.get_DataBookDir()
        # Folder name for compatibility
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        
        # File name
        fpt = 'pt_%s.csv' % pt
        # Absolute path to point sensors
        fname = os.path.join(fdir, fpt)
        # Save the file name
        self.fname = fname
        
        # Process columns
        self.ProcessColumns()
        print(self.cols)
        print(self.nCol)
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBPointSensor %s, " % self.pt
        # Number of cases in book
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    __str__ = __repr__
    
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBP.UpdateCase(i)
        :Inputs:
            *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
                An individual point sensor data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        pass
            
# class DBPointSensor


