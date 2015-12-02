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
# Local function
from .util      import readline
from .bin       import tail
from .inputCntl import InputCntl


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
def get_mach():
    """Get Mach number from most appropriate :file:`input.??.cntl` file
    
    :Call:
        >>> M = get_mach()
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
        # No phases?
        if len(fglob) == 0 and os.path.isfile('input.cntl'):
            # Read the unmarked file
            ICntl = InputCntl('input.cntl')
        else:
            # Get phase numbers
            iglob = [int(f.split('.')[1]) for f in fglob]
            # Maximum phase
            ICntl = InputCntl('input.%02i.cntl' % max(iglob))
        # Get the Mach number
        return ICntl.GetMach()
    except Exception:
        # Nothing, give 0.0
        return 0.0
        
# end functions


# Data book of point sensors
class DBPointSensor(object):
    
    pass





# Individual point sensor
class CasePointSensor(object):
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
        # Read iterations if necessary.
        self.UpdateIterations()
        # Save the Mach number
        self.mach = get_mach()
        
    
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
        # Check for steady-state iteration.
        if get_iter('pointSensors.dat') > imax:
            # Read the file.
            PS = PointSensor('pointSensors.dat')
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
        
        
    # Get the pressure coefficient
    def GetCp(self, k=None, imin=None, imax=None):
        """Get pressure coefficients at points *k* for one or more iterations
        
        :Call:
            >>> CP = P.GetCp(k=None, imin=None, imax=None)
        :Inputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative point sensor history
            *k*: :class:`int` | :class:`list` (:class:`int`) | ``None``
                Point index or list of points (all points if ``None``)
            *imin*: :class:`int` | ``None``
                Minimum iteration number to include
            *imax*: :class:`int` | ``None``
                Maximum iteration number to include
        :Versions:
            * 2015-12-01 ``@ddalle``: First version
        """
        # Default point indices.
        if k is None: k = np.arange(self.nPoint)
        # List of iterations.
        iIter = self.data[0,:,-1]
        # Indices
        i = np.arange(self.nIter) > -1
        # Filter indices
        if imin is not None: i[iIter<imin] = False
        if imax is not None: i[iIter>imax] = False
        # Select the data
        return self.data[k,i,self.nd] / (0.7*self.mach**2)
        
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
            self.p   = self.data[:,2]
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
            self.p   = self.data[:,3]
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

