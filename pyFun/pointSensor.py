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
# Local modules
from . import util
from . import case
from . import plt
from . import mapbc

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
        *DBPG*: :class:`pyFun.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2017-10-10 ``@ddalle``: First version
    """
  
  # ==========
  # Case I/O
  # ==========
  # <
    # Read case point data
    def ReadCasePoint(self, pt, **kw):
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
        triq, VarList = self.ReadCaseTriq(**kw)
        # Get the coordinates of point *pt*
        x = self.opts.get_Point(pt)
        # Project to surface and interpolate
        x0, q = triq.InterpSurfPoint(x)
        # Initialize output
        P = {}
        # Get data columns
        for col in self.cols:
            # Check if it's an *x* column
            if col in self.xCols: continue
            # Check for a point
            if col == "x":
                P["x"] = x0[0]
            # Find the index
            
        
    

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
        pplt = plt.Plt(fplt)
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
        triq = plt.CreateTriq(**kw)
        # Get variable list
        VarList = [k for k in plt.Vars if k not in ['x','y','z']]
        # Output
        return triq, VarList
        
  # >
# class DBTriqPointGroup

