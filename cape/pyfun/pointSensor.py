"""
:mod:`cape.pyfun.pointSensor`: FUN3D point sensors module
========================================================

This module contains several classes for extracting point sensor data from
FUN3D solutions. The database classes, :class:`DBTriqPointGroup` and
:class:`DBTriqPoint`, are based on versions from the generic point sensor
module :mod:`cape.pointSensor`. These classes extract surface solution data
from a FUN3D boundary output file (usually with a name of
``pyfun_tec_boundary_timestep1000.plt`` or similar) using :class:`pyFun.plt`
and :class:`cape.tri` by interpolating the surface solution to the point on the
discretized surface nearest the requested point.

At present, there is no support for reading point sensor values directly from
FUN3D output that can be requested from ``fun3d.nml``.

:See also:
    * :mod:`cape.pointSensor`
    * :mod:`cape.pyfun.dataBook`
    * :mod:`cape.pyfun.cntl`
    * :mod:`cape.pyfun.plt`
    * :mod:`cape.dataBook`
    * :mod:`cape.tri`
"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime
# Local modules
from . import util
from . import case
from . import mapbc
import cape.pyfun.plt

# Basis module
import cape.dataBook
import cape.pointSensor

# Placeholder variables for plotting functions.
plt = 0

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
# def ImportPyPlot

# Data book for triq point sensors
class DBTriqPointGroup(cape.pointSensor.DBTriqPointGroup):
    """Post-processed point sensor group data book
    
    :Call:
        >>> DBPG = DBTriqPointGroup(x, opts, name, pts=None, RootDir=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *name*: :class:`str` | ``None``
            Name of data book group
        *pts*: {``None``} | :class:`list` (:class:`str`)
            List of points to read; defaults to all points in thegroup
        *RootDir*: {``None``} | :class:`str`
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`pyFun.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2017-10-10 ``@ddalle``: First version
    """
  # ==========
  # Config
  # ==========
  # <
    # Read a point sensor
    def ReadPointSensor(self, pt):
        """Read a point sensor
        
        This function needs to be customized for each derived class so that the
        correct class is used for each of the member data books
        
        :Call:
            >>> DBPG.ReadPointSensor(pt)
        :Inputs:
            *DBPG*: :class:`pyFun.pointSensor.DBTriqPointGroup`
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
    # Current iteration status
    def GetCurrentIter(self):
        """Determine iteration number of current folder
        
        :Call:
            >>> n = DB.GetCurrentIter()
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
                Instance of data book class
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        try:
            return case.GetCurrentIter()
        except Exception:
            return None
    
    # Read case point data
    def ReadCasePoint(self, pt, i, **kw):
        """Read point data from current run folder
        
        :Call:
            >>> P = DBPG.ReadCasePoint(pt, i)
        :Inputs:
            *DBPG*: :class:`cape.pointSensor.DBTriqPointGroup`
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
        # Try to set the Mach number for *Cp* conversion
        try:
            # Get conditions
            mach = self.x.GetMach(i)
            # Set it
            kw["mach"] = mach
        except Exception:
            pass
        # Read data from a custom file
        triq, VarList = self.ReadCaseTriq(**kw)
        # Get the coordinates of point *pt*
        x = self.opts.get_Point(pt)
        # Project to surface and interpolate
        x0, q = triq.InterpSurfPoint(x)
        # Initialize output
        P = {}
        # Get data columns
        for col in self.cols:
            # Check for a point
            if col == "x":
                # x-coordinate
                P["x"] = x0[0]
            elif col == "y":
                # y-coordinate
                P["y"] = x0[1]
            elif col == "z":
                # z-coordinate
                P["z"] = x0[2]
            else:
                # Make a key name for the _avg parameter
                kavg = col + "_tavg"
                # Find the index
                if kavg in VarList:
                    # Use the time-averaged parameter
                    j = VarList.index(kavg)
                elif col in VarList:
                    # Use the regular parameter
                    j = VarList.index(col)
                else:
                    # Not found
                    raise KeyError("No state named '%s' found in PLT file"%col)
                # Save the parameter
                P[col] = q[j]
        # Output
        return P
            
        

    # Read Triq file from this folder
    def ReadCaseTriq(self, **kw):
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
        # Get the PLT file
        fplt, n, i0, i1 = case.GetPltFile()
        # Read PLT file
        pplt = pyFun.plt.Plt(fplt)
        # Check for mapbc file
        fglob = glob.glob("*.mapbc")
        # Check for more than one
        if len(fglob) > 0:
            # Make a crude attempt at sorting
            fglob.sort()
            # Import the alphabetically last one (should be the same anyway)
            kw["mapbc"] = mapbc.MapBC(fglob[0])
        # Attempt to get *cp_tavg* state
        if "mach" in kw:
            pplt.GetCpTAvg(kw["mach"])
        # Set options
        kw.setdefault("avg", False)
        kw.setdefault("triload", False)
        # Convert to Triq
        triq = pplt.CreateTriq(**kw)
        # Get variable list
        VarList = [k for k in pplt.Vars if k not in ['x','y','z']]
        # Output
        return triq, VarList
        
  # >
# class DBTriqPointGroup

# Data book of point sensor data
class DBTriqPoint(cape.pointSensor.DBTriqPoint):
    """TriQ point sensor data book
    
    Plotting methods are inherited from :class:`cape.dataBook.DBBase`,
    including :func:`cape.dataBook.DBBase.PlotHist` for plotting historgrams of
    point sensor results in particular.
    
    :Call:
        >>> DBP = DBTriqPoint(x, opts, pt, name=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *pt*: :class:`str`
            Name of point
        *name*: :class:`str` | ``None``
            Name of data book item (defaults to *pt*)
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBP*: :class:`pyFun.pointSensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2015-12-04 ``@ddalle``: Started
    """
    
    pass

# class DBTriqPoint
