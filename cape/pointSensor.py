"""
:mod:`cape.pointSensor`: Point sensor databook
================================================

This module contains a class for reading and averaging point sensors or
extracting point sensor data from a CFD solution file. It is not included in
the :mod:`cape.dataBook` module in order to give finer import control when used
in other modules.

Point sensors are often defined in two regions of the main Cape JSON file read
by :class:`cape.options.Options` or :class:`cape.cntl.Cntl`.  Usually the
coordinates of the points are defined in the ``"Config"`` section while the
groups and other databook attributes are defined in the ``"DataBook"`` section.

The database components are split into groups, so the ``"DataBook"`` section of
the JSON file may have a point sensor group called ``"P1"`` that includes
points ``"p1"``, ``"p2"``, and ``"p3"``.  To explain this example further, the
following JSON snippets could be used to define these three points in one
group.

    .. code-block:: javascript
    
        {
            "Config": {
                "Points": {
                    "p1": [2.5000, 1.00, 0.00], 
                    "p2": [2.5000, 0.00, 1.00],
                    "p3": [3.5000, 0.00, 1.00]
                }
            },
            "DataBook": {
                "P1": {
                    "Type": "TriqPoint",
                    "Points": ["p1", "p2", "p3"]
                }
            }
        }

If a data book is read in as *DB*, the point sensor group *DBP* for group
``"P1"`` and the point sensor *p1* are obtained using the commands below.

    .. code-block:: python
    
        // Point sensor group
        DBP = DB.PointSensors["P1"]
        // Individual point sensor
        p1 = DBP["p1"]
        
The same snippet could also be interpreted as a Python :class:`dict` and used
as raw inputs without using :class:`cape.options.Options`.  Note that each
point sensor group can be one of two point sensor types:

    * ``"Point"``: Point sensor data explicitly provided by CFD solver
    * ``"TriqPoint"``: Surface point sensor extracted from CFD solution
    
In many cases, the ``"Point"`` type is not fully implemented.  It is a very
sensitive method since it requires the user to specify the points before
running the CFD case (whereas ``"TriqPoint"`` just requires a surface solution
output), but it is the only way to extract point iterative histories.

"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from .options   import odict
# Utilities and advanced statistics
from . import util
from . import case
from . import dataBook

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
# def ImportPyPlot


# Data book for group of point sensors
class DBPointSensorGroup(dataBook.DBBase):
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
        # Save the columns
        self.cols = opts.get_DataBookCoeffs(name)
        # Divide columns into parts
        self.DataCols = opts.get_DataBookDataCols(name)
        # Loop through the points.
        for pt in self.pts:
            self.ReadPointSensor(pt)
            
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
    
    # Read a point sensor
    def ReadPointSensor(self, pt):
        """Read a point sensor
        
        This function needs to be customized for each derived class so that the
        correct class is used for each of the member data books
        
        :Call:
            >>> DBPG.ReadPointSensor(pt)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
            *pt*: :class:`str`
                Name of the point to read
        :Versions:
            * 2017-10-11 ``@ddalle``: First version
        """
        # Read the local class
        self[pt] = DBPointSensor(self.x, self.opts, pt, self.name)
  # >
  
  # ======
  # I/O
  # ======
  # <
    # Output method
    def Write(self, merge=False, unlock=True):
        """Write to file each point sensor data book in a group
        
        :Call:
            >>> DBPG.Write()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
            *merge*: ``True`` | {``False``}
                Whether or not to attempt a merger before writing
            *unlock*: {``True``} | ``False``
                Whether or not to delete any lock files
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            # Sort it.
            self[pt].Sort()
            # Write it
            self[pt].Write(merge=merge, unlock=unlock)
  # >
  
  # ==========
  # Case I/O
  # ==========
  # <
    # Read case point data
    def ReadCasePoint(self, pt, i):
        """Read point data from current run folder
        
        :Call:
            >>> P = DBPG.ReadCasePoint(pt, i)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointGroup`
                Point sensor group data book
            *pt*: :class:`str`
                Name of point to read
            *i*: :class:`int`
                Case index
        :Outputs:
            *P*: :class:`dict`
                Dictionary of state variables as requested from the point
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        # Read data from a custom file
        pass
  # >
  
  # =============
  # Organization
  # =============
  # <
    # Sort data book
    def Sort(self):
        """Sort each point sensor data book in a group
        
        :Call:
            >>> DBPG.Sort()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2017-10-11 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            self[pt].Sort()
  # >
  
  # ============
  # Updaters
  # ============
  # <
   # -------
   # Config
   # -------
   # [
    # Process list of components
    def ProcessComps(self, pt=None, **kw):
        """Process list of points
        
        This performs several conversions:
        
            =============  ===================
            *comp*         Output
            =============  ===================
            ``None``       ``DBPG.pts``
            :class:`str`   ``pt.split(',')``
            :class:`list`  ``pt``
            =============  ===================
        
        :Call:
            >>> DBPG.ProcessComps(pt=None)
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Point sensor group data book
            *pt*: {``None``} | :class:`list` (:class:`str`) | :class:`str`
                Point name or list of point names
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        # Get type
        t = type(pt).__name__
        # Default list of components
        if pt is None:
            # Default: all components
            return self.pts
        elif t in ['str', 'unicode']:
            # Split by comma (also ensures list)
            return pt.split(',')
        elif t in ['list', 'ndarray']:
            # Already a list?
            return pt
        else:
            # Unknown
            raise TypeError("Cannot process point list with type '%s'" % t)
   # ]
   
   # -----------
   # Update/Add
   # -----------
   # [
    # Update data book
    def Update(self, I=None, pt=None):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DBPG.Update(I=None, pt=None)
        :Inputs:
            *DBPG*: :class:`cape.dataBook.DBPointGroup`
                Point sensor group data book
            *I*: :class:`list` (:class:`int`) | ``None``
                List of trajectory indices or update all cases in trajectory
            *pt*: {``None``} | :class:`list` (:class:`str`) | :class:`str`
                Point name or list of point names
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        # Default indices (all)
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Process list of components
        pts = self.ProcessComps(pt=pt)
        # Loop through points
        for pt in pts:
            # Check type
            if pt not in self.pts: continue
            # Status update
            print("Point '%s' ..." % pt)
            # Save location
            fpwd = os.getcwd()
            # Go to root dir
            os.chdir(self.RootDir)
            # Start counter
            n = 0
            # Loop through indices
            for i in I:
                try:
                    # See if it can be updated
                    n += self.UpdateCasePoint(i, pt)
                except Excaption as e:
                    # Print error message and move on
                    print("update failed: %s" % e.message)
            # Return to original location
            os.chdir(fpwd)
            # Move on to next component if no updates
            if n == 0:
                # Unlock
                self[pt].Unlock()
                continue
            # Status update
            print("Writing %i new or updated entries" % n)
            # Sort the point 
            self[pt].Sort()
            # Write it
            self[pt].Write(merge=True, unlock=True)
            
    # Update a case (alternate grouping)
    def UpdateCase(self, i, pt=None):
        """Update all points for one case
        
        :Call:
            >>> n = DBPG.UpdateCase(i, pt=None)
        :Inputs:
            *DBPG*: :class:`cape.dataBook.DBPointGroup`
                Point sensor group data book
            *i*: :class:`int`
                Case index
            *pt*: {``None``} | :class:`list` (:class:`str`) | :class:`str`
                Point name or list of point names
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2017-10-11 ``@ddalle``: First version
        """
        # Process list of components
        pts = self.ProcessComps(pt=pt)
        # Save location
        fpwd = os.getcwd()
        # Initialize counter
        n = 0
        # Status update
        print(self.x.GetFullFolderNames(i))
        # Loop through points
        for pt in pts:
            # Check type
            if pt not in self.pts: continue
            # Go to root dir
            os.chdir(self.RootDir)
            # Update the point
            n += self.UpdateCaseComp(i, pt)
        # Output
        return n
    
    # Update or add an entry for one component
    def UpdateCaseComp(self, i, pt):
        """Update or add a case to a point data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> n = DBPG.UpdateCaseComp(i, pt)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                Point sensor group data book
            *i*: :class:`int`
                Trajectory index
            *pt*: :class:`str`
                Name of point
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
            * 2017-04-12 ``@ddalle``: Modified to work one component
            * 2017-04-23 ``@ddalle``: Added output
            * 2017-10-10 ``@ddalle``: From :class:`cape.dataBook.DataBook`
        """
        # Check if it's present
        if pt not in self:
            raise KeyError("No point sensor '%s'" % pt)
        # Print point name
        print("  %s" % pt)
        # Get the first data book component.
        DBc = self[pt]
        # Try to find a match existing in the data book.
        j = DBc.FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Go home.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            # Nothing to do.
            return 0
        # Go to the folder.
        os.chdir(frun)
        # Get the current iteration number.
        nIter = self.GetCurrentIter()
        # Get the number of iterations used for stats.
        nStats = self.opts.get_nStats()
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_nMin()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations (or zero iterations)
            print("    Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry.
            print("    Adding new databook entry at iteration %i." % nIter)
            q = True
        elif DBc['nIter'][j] < nIter:
            # Update
            print("    Updating from iteration %i to %i."
                % (DBc['nIter'][j], nIter))
            q = True
        else:
            # Up-to-date
            print("    Databook up to date.")
            q = False
        # Check for an update
        if (not q): return 0
        # Maximum number of iterations allowed.
        nMax = min(nIter-nMin, self.opts.get_nMaxStats())
        # Read data
        P = self.ReadCasePoint(pt, i)
        
        # Save the data.
        if np.isnan(j):
            # Add to the number of cases.
            DBc.n += 1
            # Append trajectory values.
            for k in self.x.keys:
                # Append to array
                DBc[k] = np.append(DBc[k], getattr(self.x,k)[i])
            # Append values.
            for c in DBc.DataCols:
                # Append
                DBc[c] = np.append(DBc[c], P[c])
            # Append iteration counts.
            if 'nIter' in DBc:
                DBc['nIter']  = np.append(DBc['nIter'], nIter)
        else:
            # Save updated trajectory values
            for k in DBc.xCols:
                # Append to that column
                DBc[k][j] = getattr(self.x,k)[i]
            # Update data values.
            for c in DBc.DataCols:
                DBc[c][j] = P[c]
            # Update the other statistics.
            if 'nIter' in DBc:
                DBc['nIter'][j]   = nIter
        # Go back.
        os.chdir(self.RootDir)
        # Output
        return 1
   # ]
            
   # -------
   # Delete
   # -------
   # [
    # Function to delete entries by index
    def DeleteCases(self, I, pt=None):
        """Delete list of cases from point sensor data book
        
        :Call:
            >>> DBPG.Delete(I)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                Point sensor group data book
            *I*: :class:`list` (:class:`int`)
                List of trajectory indices
            *pt*: {``None``} | :class:`list` (:class:`str`) | :class:`str`
                Point name or list of point names
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        # Default.
        if I is None: return
        # Process list of components
        pts = self.ProcessComps(pt=pt)
        # Loop through components
        for pt in pts:
            # Check if present
            if pt not in self.pts: continue
            # Perform deletions
            nj = self.DeleteCasesComp(I, comp)
            # Write the component
            if nj > 0:
                # Write cleaned-up data book
                self[comp].Write(unlock=True)
            else:
                # Unlock
                self[comp].Unlock()
        
    # Function to delete entries by index
    def DeleteCasesComp(self, I, pt):
        """Delete list of cases from data book
        
        :Call:
            >>> n = DBPG.Delete(I, pt)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
                Point sensor group data book
            *I*: :class:`list` (:class:`int`)
                List of trajectory indices or update all cases in trajectory
            *pt*: :class:`str`
                Name of point sensor
        :Outputs:
            *n*: :class:`int`
                Number of deleted entries
        :Versions:
            * 2015-03-13 ``@ddalle``: First version
            * 2017-04-13 ``@ddalle``: Split by component
            * 2017-10-10 ``@ddalle``: From :class:`cape.dataBook.DataBook`
        """
        # Check if it's present
        if pt not in self:
            print("WARNING: No point sensor '%s'" % pt)
        # Get the first data book component.
        DBc = self[pt]
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
        # Number of deletions
        nj = len(J)
        # Exit if no deletions
        if nj == 0:
            return nj
        # Report status
        print("  Removing %s entries from point '%s'" % (nj, pt))
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Extract data book component.
        DBc = self[comp]
        # Loop through data book columns.
        for c in DBc.keys():
            # Apply the mask
            DBc[c] = DBc[c][mask]
        # Update the number of entries.
        DBc.n = len(DBc[DBc.keys()[0]])
        # Output
        return nj
    # ]
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


# Class for surface pressures pulled from triq
class DBTriqPointGroup(DBPointSensorGroup):
    """Post-processed point sensor group data book
    
    :Call:
        >>> DBPG = DBTriqPointGroup(x, opts, name, pts=None, RootDir=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *name*: :class:`str` | ``None``
            Name of data book group
        *pts*: {``None``} | :class:`list` (:class:`str`)
            List of points to read; defaults to all points in thegroup
        *RootDir*: {``None``} | :class:`str`
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`cape.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2017-10-10 ``@ddalle``: First version
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
        # Save the columns
        self.cols = opts.get_DataBookCoeffs(name)
        # Divide columns into parts
        self.DataCols = opts.get_DataBookDataCols(name)
        # Loop through the points.
        for pt in self.pts:
            self.ReadPointSensor(pt)
            
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBTriqPointGroup %s, " % self.name
        # Number of cases in book
        lbl += "nPoint=%i>" % len(self.pts)
        # Output
        return lbl
    __str__ = __repr__
    
    # Read a point sensor
    def ReadPointSensor(self, pt):
        """Read a point sensor
        
        This function needs to be customized for each derived class so that the
        correct class is used for each of the member data books
        
        :Call:
            >>> DBPG.ReadPointSensor(pt)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBTriqPointGroup`
                A point sensor group data book
            *pt*: :class:`str`
                Name of the point to read
        :Versions:
            * 2017-10-11 ``@ddalle``: First version
        """
        # Read the local class
        self[pt] = DBTriqPoint(self.x, self.opts, pt, self.name)
  # >
  
  # ==========
  # Case I/O
  # ==========
  # <
    # Read case point data
    def ReadCasePoint(self, pt):
        """Read point data from current run folder
        
        :Call:
            >>> P = DBPG.ReadCasePoint(pt)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBTriqPointGroup`
                Point sensor group data book
            *pt*: :class:`str`
                Name of point to read
        :Outputs:
            *P*: :class:`dict`
                Dictionary of state variables as requested from the point
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        # Read data from a custom file
        pass
    

    # Read Triq file from this folder
    def ReadCaseTriq(self):
        """Read the the most recent Triq file from this folder
        
        :Call:
            >>> triq, VarList = DBPG.ReadCaseTriq()
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBTriqPointGroup`
                Point sensor group data book
        :Outputs:
            *triq*: :class:`cape.tri.Triq`
                Annotated triangulation interface
            *VarList*: :class:`list` (:class:`str`)
                List of variable names
        :Versions:
            * 2017-10-10 ``@ddalle``: First version
        """
        pass
  # >
# class DBTriqPointGroup



# Data book of point sensors
class DBPointSensor(dataBook.DBBase):
    """Point sensor data book
    
    Plotting methods are inherited from :class:`cape.dataBook.DBBase`,
    including :func:`cape.dataBook.DBBase.PlotHist` for plotting historgrams of
    point sensor results in particular.
    
    :Call:
        >>> DBP = DBPointSensor(x, opts, pt, name=None, check=False, lock=False)
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
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, create a LOCK file
    :Outputs:
        *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2015-12-04 ``@ddalle``: Started
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, x, opts, pt, name=None, check=False, lock=False, **kw):
        """Initialization method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.pt = pt
        self.name = pt
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
        # Data columns
        self.DataCols = opts.get_DataBookCoeffs(name)
        
        # Read the file or initialize empty arrays.
        self.Read(fname, check=check, lock=lock)
        
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
        
    # Read a copy
    def ReadCopy(self, check=False, lock=False):
        """Read a copied database object
        
        :Call:
            >>> DBP1 = DBP.ReadCopy(check=False, lock=False)
        :Inputs:
            *DBP*: :class:`cape.pointSensor.DBPointSensor`
                Data book base object
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Outputs:
            *DBP1*: :class:`cape.pointSensor.DBPointSensor`
                Copy of data book object
        :Versions:
            * 2017-06-26 ``@ddalle``: First version
            * 2017-10-11 ``@ddalle``: From :class:`cape.dataBook.DBBase`
        """
        # Call the object
        DBP = DBPointSensor(self.x, self.opts, self.pt, self.comp)
        # Output
        return DBP
  # >
  
  # =========
  # Updaters
  # =========
  # <
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBP.UpdateCase(i)
        :Inputs:
            *DBP*: :class:`cape.pointSensor.DBPointSensor`
                An individual point sensor data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        pass
  # >       
# class DBPointSensor



# Data book of TriQ point sensors
class DBTriqPoint(DBPointSensor):
    """TriQ point sensor data book
    
    Plotting methods are inherited from :class:`cape.dataBook.DBBase`,
    including :func:`cape.dataBook.DBBase.PlotHist` for plotting historgrams of
    point sensor results in particular.
    
    :Call:
        >>> DBP = DBTriqPoint(x, opts, pt, name=None)
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
        *DBP*: :class:`cape.pointSensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2015-12-04 ``@ddalle``: Started
    """
  # ========
  # Config
  # ========
  # <
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBTriqPoint %s, " % self.pt
        # Number of cases in book
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    __str__ = __repr__
        
    # Read a copy
    def ReadCopy(self, check=False, lock=False):
        """Read a copied database object
        
        :Call:
            >>> DBP1 = DBP.ReadCopy(check=False, lock=False)
        :Inputs:
            *DBP*: :class:`cape.pointSensor.DBTriqPoint`
                Data book base object
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Outputs:
            *DBP1*: :class:`cape.pointSensor.DBTriqPoint`
                Copy of data book object
        :Versions:
            * 2017-06-26 ``@ddalle``: First version
            * 2017-10-11 ``@ddalle``: From :class:`cape.dataBook.DBBase`
        """
        # Call the object
        DBP = DBTriqPoint(self.x, self.opts, self.pt, self.comp,
            check=check, lock=lock)
        # Output
        return DBP
  # >
  
  # =========
  # Updaters
  # =========
  # <
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBP.UpdateCase(i)
        :Inputs:
            *DBP*: :class:`cape.pointSensor.DBTriqPoint`
                An individual point sensor data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        pass
   # >       
# class DBPointSensor


