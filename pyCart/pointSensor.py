"""
Point Sensors Module: :mod:`pyCart.pointSensor`
===============================================

This module contains a class for reading and averaging point sensors.  It is not
included in the :mod:`pyCart.dataBook` module in order to give finer import
control when used in other modules

:Versions:
    * 2015-11-30 ``@ddalle``: First version
"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime
# Local function
from .util      import readline, GetTotalHistIter, GetWorkingFolder
from .bin       import tail
from .inputCntl import InputCntl

# Basis module
import cape.dataBook

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
        plt
    except AttributeError:
        # Load the modules.
        import matplotlib.pyplot as plt
        import matplotlib.transforms as tform
        from matplotlib.text import Text


# Read best input.cntl file.
def get_InputCntl():
    """Read the best ``input.cntl`` or ``input.??.cntl`` file
    
    :Call:
        >>> IC = get_InputCntl()
    :Outputs:
        *IC*: :class:`pyCart.inputCntl.InputCntl`
            File interface to Cart3D input file ``input.cntl``
    :Versions:
        * 2015-12-04 ``@ddalle``: First version
    """
    # Look for numbered input files
    fglob = glob.glob("input.[0-9][0-9]*.cntl")
    # Safety catch.
    try:
        # No phases?
        if len(fglob) == 0 and os.path.isfile('input.cntl'):
            # Read the unmarked file
            return InputCntl('input.cntl')
        else:
            # Get phase numbers
            iglob = [int(f.split('.')[1]) for f in fglob]
            # Maximum phase
            return InputCntl('input.%02i.cntl' % max(iglob))
    except Exception:
        # No handle.
        return None
    
# Check iteration number
def get_iter(fname):
    """Get iteration number from a point sensor single-iteration file
    
    :Call:
        >>> i = get_iter(fname)
    :Inputs:
        *fname*: :class:`str`
            Point sensor file name
    :Outputs:
        *i*: :class:`float`
            Iteration number or time
    :Versions:
        * 2015-11-30 ``@ddalle``: First version
    """
    # Check for file.
    if not os.path.isfile(fname): return 0
    # Safely check the last line of the file.
    try:
        # Get the last line.
        line = tail(fname, n=1)
        # Read the time step/iteration
        return float(line.split()[-1])
    except Exception:
        # No iterations
        return 0
        
# Get Mach number from function
def get_mach(IC=None):
    """Get Mach number from most appropriate :file:`input.??.cntl` file
    
    :Call:
        >>> M = get_mach(IC=None)
    :Inputs:
        *IC*: :class:`pyCart.inputCntl.InputCntl`
            File interface to Cart3D input file ``input.cntl``
    :Outputs:
        *M*: :class:`float`
            Mach number as determined from Cart3D input file
    :Versions:
        * 2015-12-01 ``@ddalle``: First version
    """
    # Look for numbered input files
    fglob = glob.glob("input.[0-9][0-9]*.cntl")
    # Safety catch.
    try:
        # Read ``input.cntl`` if necessary.
        if IC is None: IC = get_InputCntl()
        # Get the Mach number
        return IC.GetMach()
    except Exception:
        # Nothing, give 0.0
        return 0.0
        
# Get point sensor history iterations
def get_nStatsPS():
    """Return info about iterations at which point sensors have been recorded
    
    :Call:
        >>> nStats = get_nStatsPS()
    :Outputs:
        *nIter*: :class:`int`
            Last available iteration for which a point sensor is recorded
        *nStats*: :class:`int`
            Number of iterations at which point sensors are recorded
    :Versions:
        * 2015-12-04 ``@ddalle``: First version
    """
    # Check for the file.
    if not os.path.isfile('pointSensors.hist.dat'):
        return 0
    # Open the file and get the first line.
    f = open('pointSensors.hist.dat', 'r')
    line = readline(f)
    # Output
    return nStats
# end functions

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
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`pyCart.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2015-12-04 ``@ddalle``: First version
    """
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
        self.pts = opts.get_DBGroupPoints(name)
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
            
    # Output method
    def Write(self):
        """Write to file each point sensor data book in a group
        
        :Call:
            >>> DBPG.Write()
        :Inputs:
            *DBPG*: :class:`pyCart.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            self[pt].Write()
    
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBPG.UpdateCase(i)
        :Inputs:
            *DBPG*: :class:`pyCart.pointSensor.DBPointSensorGroup`
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
# class DBPointSensorGroup


# Data book of point sensors
class DBPointSensor(cape.dataBook.DBBase):
    """
    Point sensor data book
    
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
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Folder containing the data book
        fdir = opts.get_DataBookDir()
        # Folder name for compatibility
        fdir = fdir.replace("/", os.sep)
        
        # File name
        fpt = 'pt_%s.csv' % pt
        # Absolute path to point sensors
        fname = os.path.join(fdir, fpt)
        
        # Save data book title
        if name is None:
            # Default name
            self.name = pt
        else:
            # Specified name
            self.name = name
        # Save point name
        self.pt = pt
        # Save the CNTL
        self.x = x
        self.opts = opts
        # Save the file name
        self.fname = fname
        # Column types
        self.xCols = self.x.keys
        self.fCols = [
            'X', 'Y', 'Z', 'Cp', 'dp', 'U', 'V', 'W', 'P',
            'Cp_std', 'Cp_min', 'Cp_max', 'dp_std', 'dp_min', 'dp_max',
            'rho_std', 'rho_min', 'rho_max', 'U_std', 'U_min', 'U_max',
            'V_std', 'V_min', 'V_max', 'W_std', 'W_min', 'W_max',
            'P_std', 'P_min', 'P_max', 'RefLev'
        ]
        self.iCols = ['nIter', 'nStats']
        # Counts
        self.nxCol = len(self.xCols)
        self.nfCol = len(self.fCols)
        self.niCol = len(self.iCols)
        
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
        # Check update status.
        q, P = self._UpdateCase(i)
        # Exit if no return necessary
        if not q: return
        # Try to find a match existing in the data book
        j = self.FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Go to the case folder (it exists due to above tests).
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        os.chdir(frun)
        # Determine ninimum number of iterations required
        nStats = self.opts.get_nStats(self.name)
        nLast  = self.opts.get_nLastStats(self.name)
        # Get list of iterations
        iIter = P.i
        # Find the point.
        kpt = P.GetPointSensorIndex(self.pt)
        # Calculate statistics
        s = P.GetStats(kpt, nStats=nStats, nLast=nLast)
        
        # Save the data.
        if np.isnan(j):
            # Add the the number of cases.
            self.n += 1
            # Append trajectory values.
            for k in self.xCols:
                # I hate the way NumPy does appending.
                self[k] = np.hstack((self[k], [getattr(self.x,k)[i]]))
            # Append values.
            for c in self.fCols:
                self[c] = np.hstack((self[c], [s[c]]))
            # Append iteration counts.
            self['nIter']  = np.hstack((self['nIter'], iIter[-1:]))
            self['nStats'] = np.hstack((self['nStats'], [nStats]))
        else:
            # No need to update trajectory values.
            # Update data values.
            for c in self.fCols:
                self[c][j] = s[c]
            # Update the other statistics.
            self['nIter'][j]   = iIter[-1]
            self['nStats'][j]  = nStats
        # Go back.
        os.chdir(fpwd)
    
    # Process a case
    def _UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> q, P = DBP._UpdateCase(i)
        :Inputs:
            *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
                An individual point sensor data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *q*: :class:`bool`
                Whether or not to compute update
            *P*: :class:`pyCart.pointSensor.PointSensor`
                Individual case point sensor history
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Try to find a match existing in the data book
        j = self.FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Status update
        print(frun)
        # Go home
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return False, None
        # Go to the case folder.
        os.chdir(frun)
        # Determine ninimum number of iterations required
        nStats = self.opts.get_nStats(self.name)
        nMin   = self.opts.get_nMin(self.name)
        nLast  = self.opts.get_nLastStats(self.name)
        # Get last potential iteration
        nIter = int(GetTotalHistIter()) 
        # Decide whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations
            print("  Not enough iterations (%s) for analysis." % nIter)
            os.chdir(fpwd); return False, None
        elif np.isnan(j):
            # No current history
            print("  Adding new databook entry.")
        elif self['nStats'][j] != nStats:
            # New history
            print("  Recomputing statistics using %i samples." % nStats)
        elif self['nIter'][j] == nIter:
            # Up-to-date
            print("  Databook up-to-date.")
            os.chdir(fpwd); return False, None
        elif self['nIter'][j] == get_iter('pointSensors.hist.dat'):
            # Up-to-date
            print("  Databook up-to-date.")
            os.chdir(fpwd); return False, None
        # Read the point sensor history.
        P = CasePointSensor()
        # Get minimum iteration that would be included if we compute stats now
        if P.nIter < nStats:
            # Not enough samples
            print("  Not enough point samples (%s) for analysis." % PS.nIter)
            os.chdir(fpwd); return False, None
        elif P.nPoint < 1:
            # No points?
            print("  Point sensor history contains no points.")
            os.chdir(fpwd); return False, None
        # Get list of iterations
        iIter = P.i
        # Downselect if *nLast* in use
        if nLast > 0: iIter = iIter[iIter<=nLast]
        # Minimum iteration that will be included in stats
        if nStats == 0:
            # No averaging; just use last iteration
            iStats0 = iIter[-1]
        else:
            # Read backwards *nStats* samples from the end
            iStats0 = iIter[-nStats]
        # Check that.
        if iStats0 < nMin:
            # Too early
            print("  Not enough samples after min iteration %i." % nMin)
            os.chdir(fpwd); return False, None
        else:
            os.chdir(fpwd); return True, P
            
# class DBPointSensor


# Individual point sensor
class CasePointSensor(cape.dataBook.CaseData):
    """Individual case point sensor history
    
    :Call:
        >>> P = CasePointSensor()
    :Outputs:
        *P*: :class:`pyCart.pointSensor.CasePointSensor`
            Case point sensor
        *P.mach*: :class:`float`
            Mach number for this case; for calculating pressure coefficient
        *P.nPoint*: :class:`int`
            Number of point sensors
        *P.nIter*: :class:`int`
            Number of iterations recorded in point sensor history
        *P.nd*: ``2`` | ``3``
            Number of dimensions
        *P.iSteady*: :class:`int`
            Maximum steady-state iteration number
        *P.data*: :class:`numpy.ndarray` (*nPoint*, *nIter*, 10 | 12)
            Data array
    :Versions:
        * 2015-12-01 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self):
        """Initialization method"""
        # Check for history file
        if os.path.isfile('pointSensors.hist.dat'):
            # Read the file
            self.ReadHist()
        else:
            # Initialize empty data
            self.nPoint = None
            self.nIter = 0
            self.nd = None
            self.iSteady = 0
            self.data = np.zeros((0,0,12))
            self.i = np.array([])
        # Read iterations if necessary.
        self.UpdateIterations()
        # Input file
        self.InputCntl = get_InputCntl()
        # Save the Mach number
        self.mach = get_mach(self.InputCntl)
        
    
    # Read the steady-state output file
    def UpdateIterations(self):
        """Read any Cart3D point sensor output files and save them
        
        :Call:
            >>> P.UpdateIterations()
        :Inputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative point sensor history
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Get latest iteration.
        if self.nPoint > 0:
            imax = self.data[0,-1,-1]
        else:
            imax = 0
        # Check for steady-state outputs.
        fglob = glob.glob('adapt??/pointSensors.dat')
        fglob += glob.glob('pointSensors.dat')
        fglob.sort()
        # Loop through steady-state iterations
        for f in fglob:
            # Check if it's up-to-date
            if get_iter(f) <= imax: continue
            # Read the file.
            PS = PointSensor(f)
            # Save the iterations
            self.AppendIteration(PS)
            # Update the steady-state iteration count
            if self.nPoint > 0:
                self.iSteady = PS.data[0,-1]
                imax = self.iSteady
        # Check for time-accurate iterations.
        fglob = glob.glob('pointSensors.[0-9][0-9]*.dat')
        iglob = np.array([int(f.split('.')[1]) for f in fglob])
        iglob.sort()
        # Time-accurate results only; filter on *imax*
        iglob = iglob[iglob > imax-self.iSteady]
        # Read the time-accurate iterations
        for i in iglob:
            # File name
            fi = "pointSensors.%06i.dat" % i
            # Read the file.
            PS = PointSensor(fi)
            # Increase time-accurate iteration number
            PS.i += self.iSteady
            # Save the data.
            self.AppendIteration(PS)
        
        
    # Read history file
    def ReadHist(self, fname='pointSensors.hist.dat'):
        """Read point sensor iterative history file
        
        :Call:
            >>> P.ReadHist(fname='pointSensors.hist.dat')
        :Inputs:
            *fname*: :class:`str`
                Name of point sensor history file
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname):
            raise SystemError("File '%s' does not exist." % fname)
        # Open the file.
        f = open(fname, 'r')
        # Read the first line, which contains identifiers.
        line = readline(f)
        # Get the values
        nPoint, nIter, nd, iSteady = [int(v) for v in line.split()]
        # Save
        self.nPoint  = nPoint
        self.nIter   = nIter
        self.nd      = nd
        self.iSteady = iSteady
        # Number of data columns
        if nd == 2:
            # Two-dimensional data
            nCol = 10
        else:
            # Three-dimensional data
            nCol = 12
        # Read data lines
        A = np.fromfile(f, dtype=float, count=nPoint*nIter*nCol, sep=" ")
        # Reshape
        self.data = A.reshape((nPoint, nIter, nCol))
        # Save the iterations at which samples are recoreded
        self.i = self.data[0,:,-1]
        
    # Write history file
    def WriteHist(self, fname='pointSensors.hist.dat'):
        """Write point sensor iterative history file
        
        :Call:
            >>> P.WriteHist(fname='pointSensors.hist.dat')
        :Inputs:
            *fname*: :class:`str`
                Name of point sensor history file
        :Versions:
            * 2015-12-01 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'w')
        # Write column names
        f.write('# nPoint, nIter, nd, iSteady\n')
        # Write variable names
        if self.nd == 2:
            # Two-dimensional data
            f.write("# VARIABLES = X Y (P-Pinf)/Pinf RHO U V P ")
            f.write("RefLev mgCycle/Time\n")
        else:
            # Three-dimensional data
            f.write("# VARIABLES = X Y Z (P-Pinf)/Pinf RHO U V W P ")
            f.write("RefLev mgCycle/Time\n")
        # Write header.
        f.write('%i %i %i %i\n' %
            (self.nPoint, self.nIter, self.nd, self.iSteady))
        # Write flag
        if self.nd == 2:
            # Point, 2 coordinates, 5 states, refinements, iteration
            fflag = '%4i' + (' %15.8e'*7) + ' %2i %9.3f\n'
        else:
            # Point, 3 coordinates, 6 states, refinements, iteration
            fflag = '%4i' + (' %15.8e'*9) + ' %2i %9.3f\n'
        # Loop through points
        for k in range(self.nPoint):
            # Loop through iterations
            for i in range(self.nIter):
                # Write the info.
                f.write(fflag % tuple(self.data[k,i,:]))
        # Close the file.
        f.close()
        
    # Add another point sensor
    def AppendIteration(self, PS):
        """Add a single-iteration of point sensor data to the history
        
        :Call:
            >>> P.AppendIteration(PS)
        :Inputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative point sensor history
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check compatibility
        if self.nPoint is None:
            # Use the point count from the individual file.
            self.nPoint = PS.nPoint
            self.nd = PS.nd
            self.nIter = 0
            # Initialize
            if self.nd == 2:
                self.data = np.zeros((self.nPoint, 0, 10))
            else:
                self.data = np.zeros((self.nPoint, 0, 12))
        elif self.nPoint != PS.nPoint:
            # Wrong number of points
            raise IndexError(
                "History has %i points; point sensor has %i points."
                % (self.nPoint, PS.nPoint))
        elif self.nd != PS.nd:
            # Wrong number of dimensions
            raise IndexError(
                "History is %-D; point sensor is %i-D." % (self.nd, PS.nd))
        # Get data from point sensor and add point number
        A = np.hstack((np.array([range(self.nPoint)]).transpose(), PS.data))
        # Number of columns
        nCol = A.shape[1]
        # Append to history.
        self.data = np.hstack(
            (self.data, A.reshape((self.nPoint,1,nCol))))
        # Increase iteration count.
        self.nIter += 1
        
    
    # Get point sensor by name
    def GetPointSensorIndex(self, name):
        """Get the index of a point sensor by its name in ``input.cntl``
        
        :Call:
            >>> k = P.GetPointSensorIndex(name)
        :Inputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative point sensor history
            *name*: :class:`str`
                Name of the sensor
        :Outputs:
            *k*: :class;`int`
                Index of nearest point in the sampled data
        :Versions:
            * 2105-12-04 ``@ddalle``: First version
        """
        # Get coordinates from :file:`input.cntl`
        r = self.InputCntl.GetPointSensor(name)
        # Get the coordinates for all point sensors from data
        R = self.data[:,0,1:self.nd+1]
        # Calculate distances
        if self.nd == 2:
            L2 = (r[0]-R[:,0])**2 + (r[1]-R[:,1])**2
        else:
            L2 = (r[0]-R[:,0])**2 + (r[1]-R[:,1])**2 + (r[2]-R[:,2])**2
        # Find the minimum distance.
        return np.argmin(L2)
        
    # Compute statistics
    def GetStats(self, k, nStats=1, nLast=None):
        """Compute min, max, mean, and standard deviation of each quantity
        
        This includes computing pressure coefficient.  NaNs are reported as the
        standard deviation if *nStats* is 1 or 0.  If the point sensor is
        two-dimensional, i.e. *P.nd* is 2, the *W* velocity is reported as 0.0.
        
        :Call:
            >>> s = P.GetStats(k, nStats=1, nLast=None)
        :Inputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative point sensor history
            *nStats*: :class:`int`
                Number of samples to use for computing statistics
            *nLast*: :class:`int` | ``None``
                If specified, maximum iteration to use
        :Outputs:
            *s*: :class:`dict` (:class:`float`)
                Dictionary of mean, min, max, std for each variable
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Last iteration to use.
        if nLast:
            # Attempt to use requested iter.
            if nLast < self.i[0]:
                # No earlier iterative histories
                I = np.array([])
            else:
                # Apply filter.
                I = np.where(self.i <= nLast)[0]
            # Check for sufficient samples
            if I.size < nStats:
                raise RuntimeError("Less than %i samples before iteration %i"
                    % (nStats, nLast))
            # Filter last *nStats* samples.
            I = I[-nStats:]
        else:
            # Initialize to all samples included
            I = np.arange(self.nIter)
            # Check for number of iterations
            if I.size < nStats:
                raise RuntimeError("Less than %i samples before iteration %i"
                    % (nStats, nLast))
            # Filter last *nStats* samples.
            I = I[-nStats:]
        # Initialize output
        s = {}
        # Extract data for this point.
        A = self.data[k, :, 1:]
        # Insert Z, Cp, and W as appropriate
        if self.nd == 2:
            # Calculate Cp
            Cp = A[:,2:3] / (0.7*self.mach**2)
            # Insert zeros for Z and W
            Z = np.zeros_like(A[:,:1])
            B = np.hstack((A[:,:2], Z, Cp, A[:,2:5], Z, A[:,5:]))
        else:
            # Calculate Cp
            Cp = A[:,3:4] / (0.7*self.mach**2)
            # Insert *Cp*
            B = np.hstack((A[:,:3], Cp, A[:,3:]))
        # Save coordinates
        s['X'] = B[0,0]
        s['Y'] = B[0,1]
        s['Z'] = B[0,2]
        # Mean number of refinement levels
        s['RefLev'] = np.mean(B[I,-2])
        # List of states
        fcols = ['Cp', 'dp', 'rho', 'U', 'V', 'W', 'P']
        # Loop through columns
        for i in range(len(fcols)):
            # State name and values
            c = fcols[i]
            V = B[I,i+3]
            # Save mean value.
            s[c] = np.mean(V)
            # Save statistics
            s[c+'_min'] = np.min(V)
            s[c+'_max'] = np.max(V)
            s[c+'_std'] = np.std(V)
        # Output,
        return s
        
    # Extract value
    def ExtractValue(self, c, k=None):
        """Extract the iterative history for one coefficient/state
        
        The following states are available
        
            * ``"X"``: x-coordinate of point
            * ``"Y"``: y-coordinate of point
            * ``"Z"``: z-coordinate of point
            * ``"dp"``: static pressure, :math:`p/p_\infty-1`
            * ``"rho"``: static density, :math:`\rho/\rho_\infty`
            * ``"U"``: x-component of velocity, :math:`u/a_\infty`
            * ``"V"``: y-component of velocity, :math:`v/a_\infty`
            * ``"W"``: z-component of velocity, :math:`w/a_\infty`
            * ``"P"``: static pressure, :math:`p/(\gamma p_\infty)`
            * ``"Cp"``: pressure coefficient, :math:`dp/(0.7*M_\infty^2`
            * ``"p"``: static pressure, :math:`p/p_\infty`
            * ``"T"``: Static temperature, :math:`T/T_\infty`
            * ``"M"``: Mach number
        
        :Call:
            >>> C = PS.ExtractValue(c, k=None)
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor
                Point sensor
            *c*: :class:`str` 
                Name of state
            *k*: :class:`str` | :class:`int` | ``None``
                Point sensor name or index
        :Outputs:
            *C*: :class:`np.ndarray`
                Array of values for *c* at each sample
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Process point sensor specification
        if k is None:
            # Extract all data
            D = self.data.reshape((self.nPoint*self.nIter,12))
        elif type(k).__name__.startswith('int'):
            # Select the appropriate point.
            D = self.data[k,:,:]
        else:
            # Get the point index
            k = self.GetPointSensorIndex(k)
            # Select.
            D = self.data[k,:,:]
        # Extract valid states
        dp = D[:,4]
        rho = D[:,5]
        U = D[:,6]
        V = D[:,7]
        W = D[:,8]
        P = D[:,9]
        # Check input state
        if c.lower() in ['m', 'mach']:
            # Calculate velocity.
            Vt = np.sqrt(U**2 + V**2 + W**2)
            # Divide by soundspeed ratio
            C = Vt / np.sqrt(1.4*P/rho)
        elif c.lower() in ['cp', 'c_p']:
            # Get freestream Mach number
            Minf = self.mach
            # Calculate pressure coefficient
            C = dp / (0.7*Minf*Minf)
        elif c == 'T':
            # Get static temperature
            C = 1.4*P/srho
        elif c == 'p':
            # Get static pressure
            C = 1.4*P
        elif c.lower() == 'rho':
            # Static density
            C = rho
        elif c.lower() == 'dp':
            # Scaled static pressure
            C = dp
        elif c == 'P':
            # Weird static pressure
            C = P
        elif c == 'U':
            C = U
        elif c == 'V':
            C = V
        elif c == 'W':
            C = W
        else:
            raise AttributeError(
                "State '%s' is unknown for %iD data." % (c, self.nd))
        # Output
        if k is None:
            # Resize
            return C.reshape((self.nIter,self.nPoint))
        else:
            # Direct output
            return C
        
    
    # Plot history of a coefficient
    def PlotState(self, c, k, **kw):
        """Plot the iterative history of a state
        
        :Call:
            >>> h = P.PlotState(c, k, n=None, nAvg=100, **kw)
        :Inputs:
            *P*: :class:`pyCart.dataBook.PointSensor`
                Case component history class
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'Cp'``
            *k*: :class:`str` | :class:`int`
                Name or index of point sensor
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d*: :class:`float`
                Delta in the coefficient to show expected range
            *k*: :class:`float`
                Multiple of iterative standard deviation to plot
            *u*: :class:`float`
                Multiple of sampling error standard deviation to plot
            *eps*: :class:`float`
                Fixed sampling error, default uses :func:`SigmaMean`
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *nFirst*: :class:`int`
                First iteration to plot
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *LineOptions*: :class:`dict`
                Dictionary of additional options for line plot
            *StDevOptions*: :class:`dict`
                Options passed to :func:`plt.fill_between` for stdev plot
            *ErrPltOptions*: :class:`dict`
                Options passed to :func:`plt.fill_between` for uncertainty plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowEpsilon*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Process slightly improved default y-axis labels
        if "YLabel" not in kw:
            # Check the state.
            if c.lower() == 'dp':
                kw["YLabel"] = r"$(p-p_\infty)/p_\infty$"
            elif c == 'P':
                kw["YLabel"] = r"$p/\gamma p_\infty$"
            elif c.lower() == 'rho':
                kw["YLabel"] = r"Static Density, $\rho/\rho_\infty$"
            elif c == 'p':
                kw["YLabel"] = r"Static Pressure, $p/p_\infty$"
            elif c == 'T':
                kw["YLabel"] = r"Static Temperature, $T/T_\infty$"
            elif c.lower() in ['cp', 'c_p']:
                kw["YLabel"] = "Pressure Coefficient"
            elif c.lower() in ['m', 'mach']:
                kw["YLabel"] = "Mach Number"
        # Refer to parent plotting class
        return self.PlotValue(c, col=k, **kw)
        
# class CasePointSensor


# Individual file point sensor
class PointSensor(object):
    """Class for individual point sensor
    
    :Call:
        >>> PS = PointSensor(fname="pointSensors.dat", data=None)
    :Inputs:
        *fname*: :class:`str`
            Name of Cart3D output point sensors file
        *data*: :class:`np.ndarray` (:class:`float`)
            Data array with either 9 (2-D) or 11 (3-D) columns
    :Outputs:
        *PS*: :class:`pyCart.pointSensor.PointSensor`
            Point sensor
        *PS.data*: :class:`np.ndarray` (:class:`float`)
            Data array with either 9 (2-D) or 11 (3-D) columns
        *PS.nd*: ``2`` | ``3``
            Number of dimensions of the data
        *PS.nPoint*: :class:`int`
            Number of points in the file
        *PS.nIter*: :class:`int`
            Number of iterations used to calculate the average
    :Versions:
        * 2015-11-30 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, fname="pointSensors.dat", data=None):
        """Initialization method"""
        # Check for data
        if data is None:
            # Read the file.
            data = np.loadtxt(fname, comments='#')
        # Check the dimensionality.
        if data.shape[1] == 9:
            # Sort
            i = np.lexsort((data[:,1], data[:,0]))
            self.data = data[i,:]
            # Two-dimensional data
            self.nd = 2
            self.X = self.data[:,0]
            self.Y = self.data[:,1]
            self.dp  = self.data[:,2]
            self.rho = self.data[:,3]
            self.U   = self.data[:,4]
            self.V   = self.data[:,5]
            self.P   = self.data[:,6]
            self.RefLev = self.data[:,7]
            self.i      = self.data[:,8]
        else:
            # Sort
            i = np.lexsort((data[:,2], data[:,1], data[:,0]))
            self.data = data[i,:]
            # Three-dimensional data
            self.nd = 3
            self.X = self.data[:,0]
            self.Y = self.data[:,1]
            self.Z = self.data[:,2]
            self.dp  = self.data[:,3]
            self.rho = self.data[:,4]
            self.U   = self.data[:,5]
            self.V   = self.data[:,6]
            self.W   = self.data[:,7]
            self.P   = self.data[:,8]
            self.RefLev = self.data[:,9]
            self.i      = self.data[:,10]
        # Sort
        # Save number of points
        self.nPoint = self.data.shape[0]
        # Number of averaged iterations
        self.nIter = 1
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check dimensionality
        return "<PointSensor(nd=%i, nPoint=%i)>" % (self.nd, self.nPoint)
        
    # Copy a point sensor
    def copy(self):
        """Copy a point sensor
        
        :Call:
            >>> P2 = PS.copy()
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
        :Outputs:
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor copied
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        return PointSensor(data=self.data)
        
    # Write to file
    def Write(self, fname):
        """Write single-iteration point sensor file
        
        :Call:
            >>> PS.Write(fname):
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *fname*: :class:`str`
                Name of Cart3D output point sensors file
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Open the file for writing.
        f = open(fname, 'w')
        # Write header
        if self.nd == 2:
            # Two-dimensional data
            f.write("# VARIABLES = X Y (P-Pinf)/Pinf RHO U V P ")
            f.write("RefLev mgCycle/Time\n")
            # Format string
            fpr = (7*' %15.8e' + ' %i %7.3f\n')
        else:
            # Three-dimensional data
            f.write("# VARIABLES = X Y Z (P-Pinf)/Pinf RHO U V W P ")
            f.write("RefLev mgCycle/Time\n")
            # Format string
            fpr = (9*' %15.8e' + ' %i %7.3f\n')
        # Write the points
        for i in range(self.nPoint):
            f.write(fpr % tuple(self.data[i,:]))
        # Close the file.
        f.close()
        
    
    # Extract value
    def ExtractValue(self, c):
        """Extract the iterative history for one coefficient/state
        
        The following states are available
        
            * ``"X"``: x-coordinate of point
            * ``"Y"``: y-coordinate of point
            * ``"Z"``: z-coordinate of point
            * ``"dp"``: static pressure, :math:`p/p_\infty-1`
            * ``"rho"``: static density, :math:`\rho/\rho_\infty`
            * ``"U"``: x-component of velocity, :math:`u/a_\infty`
            * ``"V"``: y-component of velocity, :math:`v/a_\infty`
            * ``"W"``: z-component of velocity, :math:`w/a_\infty`
            * ``"P"``: static pressure, :math:`p/(\gamma p_\infty)`
            * ``"Cp"``: pressure coefficient, :math:`dp/(0.7*M_\infty^2`
            * ``"p"``: static pressure, :math:`p/p_\infty`
            * ``"T"``: Static temperature, :math:`T/T_\infty`
            * ``"M"``: Mach number
            
        The values *Z* and *W* are not available for 2D data
        
        :Call:
            >>> C = PS.ExtractValue(c)
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor
                Point sensor
            *c*: :class:`str` 
                Name of state
        :Outputs:
            *C*: :class:`np.ndarray`
                Array of values for *c* at each sample
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Check input state
        if c.lower() in ['m', 'mach']:
            # Calculate velocity.
            if self.nd == 2:
                Vt = np.sqrt(self.U**2 + self.V**2)
            else:
                Vt = np.sqrt(self.U**2 + self.V**2 + self.W**2)
            # Divide by soundspeed ratio
            return Vt / np.sqrt(1.4*self.P/self.rho)
        elif c.lower() in ['cp', 'c_p']:
            # Get freestream Mach number
            Minf = get_mach()
            # Calculate pressure coefficient
            return self.dp / (0.7*Minf*Minf)
        elif c == 'T':
            # Get static temperature
            return 1.4*self.P/self.rho
        elif c == 'p':
            # Get static pressure
            return 1.4*self.P
        else:
            # Direct reference
            try:
                # Version of "PS.(c)*
                return getattr(self,c)
            except AttributeError:
                raise AttributeError(
                    "State '%s' is unknown for %iD data." % (c, self.nd))
        
        
    # Plot history
    def PlotPoint(self, c, n=None, nAvg=100, **kw):
        """Plot a single point sensor iterative history
        
        :Call:
            >>> h = PS.PlotPoint(c, n=None, nAvg=100, **kw)
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *c*: {'dp'} | 'rho' | 'U' | 'V' | 'W' | 'P' 
                Name of state quantity to plot
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d*: :class:`float`
                Delta in the coefficient to show expected range
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *nFirst*: :class:`int`
                First iteration to plot
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Extract the data.
        C = self.ExtractValue(c)
    
        
    # Multiplication
    def __mul__(self, c):
        """Multiplication method
        
        :Call:
            >>> P2 = PS.__mul__(c)
            >>> P2 = PS * c
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *c*: :class:`int` | :class:`float`
                Number by which to multiply
        :Outputs:
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor copied
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check the input
        t = type(c).__name__
        if not (tc.startswith('int') or tc.startswith('float')):
            return TypeError("Point sensors can only be multiplied by scalars.")
        # Create a copy
        P2 = self.copy()
        # Multiply
        if self.nd == 2:
            # Two-dimensional data
            P2.data[:,2:7] *= c
        else:
            # Two-dimensional data
            P2.data[:,3:9] *= c
        # If integer, multiply number of iiterations included
        if type(c).startswith('int'): P2.nIter*=c
        # Output
        return P2
    
    # Multiplication, other side
    __rmul__ = __mul__
    __rmul__.__doc__ = """Right-hand multiplication method
    
        :Call:
            >>> P2 = PS.__rmul__(c)
            >>> P2 = c * PS
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *c*: :class:`int` | :class:`float`
                Number by which to multiply
        :Outputs:
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor copied
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
    """
    
    # Multiplication
    def __div__(self, c):
        """Multiplication method
        
        :Call:
            >>> P2 = PS.__div__(c)
            >>> P2 = PS / c
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *c*: :class:`int` | :class:`float`
                Number by which to divide
        :Outputs:
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor copied
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check the input
        t = type(c).__name__
        if not (tc.startswith('int') or tc.startswith('float')):
            return TypeError("Point sensors can only be multiplied by scalars.")
        # Create a copy
        P2 = self.copy()
        # Multiply
        if self.nd == 2:
            # Two-dimensional data
            P2.data[:,2:7] /= c
        else:
            # Two-dimensional data
            P2.data[:,3:9] /= c
        # Output
        return P2
    
    # Addition method
    def __add__(self, P1):
        """Addition method
        
        :Call:
            >>> P2 = PS.__add__(P1)
        :Inputs:
            *PS*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensor to add
        :Outputs:
            *P2*: :class:`pyCart.pointSensor.PointSensor`
                Point sensors added
        :Versions:
            * 2015-11-30 ``@ddalle``: First version
        """
        # Check compatibility
        if type(P1).__name__ != 'PointSensor':
            # One addend is not a point sensor
            return TypeError(
                "Only point sensors can be added to point sensors.")
        elif self.nd != P1.nd:
            # Incompatible dimension
            return IndexError("Cannot add 2D and 3D point sensors together.")
        elif self.nPoint != P1.nPoint:
            # Mismatching number of points
            return IndexError(
                "Sensor 1 has %i points, and sensor 2 has %i points." 
                % (self.nPoint, P1.nPoint))
        # Create a copy.
        P2 = self.copy()
        # Add
        if self.nd == 2:
            # Two-dimensional data
            P2.data[:,2:7] = self.data[:,2:7] + P1.data[:,2:7]
        else:
            # Two-dimensional data
            P2.data[:,3:9] = self.data[:,3:9] + P1.data[:,3:9]
        # Number of iterations
        P2.nIter = self.nIter + P1.nIter
        # Output
        return P2
# class PointSensor

