r"""
:mod:`cape.pyfun.pointsensor`: FUN3D point sensors module
===========================================================

This module contains several classes for extracting point sensor data
from FUN3D solutions. The database classes, :class:`DBTriqPointGroup` 
and :class:`DBTriqPoint`, are based on versions from the generic point
sensor module :mod:`cape.cfdx.pointsensor`. These classes extract 
surface solution data from a FUN3D boundary output file (usually with a
name of ``pyfun_tec_boundary_timestep1000.plt`` or similar)
using :class:`cape.pyfun.plt` and :class:`cape.tri` by interpolating the
surface  solution to the point on the discretized surface nearest the
requested  point.

At present, there is no support for reading point sensor values directly
from FUN3D output that can be requested from ``fun3d.nml``.

:See also:
    * :mod:`cape.cfdx.pointsensor`
    * :mod:`cape.pyfun.dataBook`
    * :mod:`cape.pyfun.cntl`
    * :mod:`cape.pyfun.plt`
    * :mod:`cape.cfdx.dataBook`
    * :mod:`cape.tri`
"""

# Standard library
import os
import glob

# Third party

# Local modules
from . import casecntl
from . import mapbc
from . import pltfile
from ..cfdx import pointsensor as cptsensor
from ..trifile import Triq


# Placeholder variables for plotting functions.
plt = 0


# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    r"""Import :mod:`matplotlib.pyplot` if not loaded

    :Call:
        >>> pyCart.databook.ImportPyPlot()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Make global variables
    global plt
    global tform
    global Text
    # Check for PyPlot.
    try:
        pltfile.gcf
    except AttributeError:
        # Load the modules.
        import matplotlib.pyplot as plt
        import matplotlib.transforms as tform
        from matplotlib.text import Text
# def ImportPyPlot


# Data book for triq point sensors
class DBTriqPointGroup(cptsensor.DBTriqPointGroup):
    r"""Post-processed point sensor group data book

    :Call:
        >>> DBPG = DBTriqPointGroup(cntl, opts, name, **kw)
    :Inputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *name*: :class:`str` | ``None``
            Name of data book group
        *pts*: {``None``} | :class:`list`\ [:class:`str`]
            List of points to read; defaults to all points in the group
        *RootDir*: {``None``} | :class:`str`
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`pyFun.pointsensor.DBPointSensorGroup`
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
        r"""Read a point sensor

        This function needs to be customized for each derived class so 
        that the correct class is used for each of the member data 
        books

        :Call:
            >>> DBPG.ReadPointSensor(pt)
        :Inputs:
            *DBPG*: :class:`pyFun.pointsensor.DBTriqPointGroup`
                A point sensor group data book
            *pt*: :class:`str`
                Name of the point to read
        :Versions:
            * 2017-10-11 ``@ddalle``: First version
        """
        # Read the local class
        self[pt] = DBTriqPoint(self.cntl, self.opts, pt, self.name)
  # >

  # ==========
  # Case I/O
  # ==========
  # <
    # Current iteration status
    def GetCurrentIter(self):
        r"""Determine iteration number of current folder

        :Call:
            >>> n = DB.GetCurrentIter()
        :Inputs:
            *DB*: :class:`pyFun.databook.DataBook`
                Instance of data book class
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        try:
            return casecntl.GetCurrentIter()
        except Exception:
            return None

    # Read case point data
    def ReadCasePoint(self, pt, i, **kw):
        r"""Read point data from current run folder

        :Call:
            >>> P = DBPG.ReadCasePoint(pt, i)
        :Inputs:
            *DBPG*: :class:`cape.cfdx.pointsensor.DBTriqPointGroup`
                Point sensor group data book
            *pt*: :class:`str`
                Name of point to read
            *i*: :class:`int`
                Case index
        :Outputs:
            *P*: :class:`dict`
                Dictionary of state variables as requested from the 
                point
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
        r"""Read the the most recent Triq file from this folder

        :Call:
            >>> triq, VarList = DBPG.ReadCaseTriq()
        :Inputs:
            *DBPG*: :class:`cape.cfdx.pointsensor.DBTriqPointGroup`
                Point sensor group data book
        :Outputs:
            *triq*: :class:`cape.trifile.Triq`
                Annotated triangulation interface
            *VarList*: :class:`list`\ [:class:`str`]
                List of variable names
        :Versions:
            * 2017-10-10 ``@ddalle``: v1.0
            * 2024-12-03 ``@ddalle``: v2.0; new TRIQ file method
        """
        # Get the PLT file
        ftriq, n, i0, i1 = casecntl.GetTriqFile()
        # Exit if no file
        if ftriq is None or not os.path.isfile(ftriq):
            raise FileNotFoundError("Unable to find .triq file")
        # Convert to Triq
        triq = Triq(ftriq)
        # Get variable list
        VarList = ["cp"]
        # Output
        return triq, VarList
  # >
# class DBTriqPointGroup


# Data book of point sensor data
class DBTriqPoint(cptsensor.DBTriqPoint):
    r"""TriQ point sensor data book

    Plotting methods are inherited from 
    :class:`cape.cfdx.databook.DBBase`, including
    :func:`cape.cfdx.databook.DBBase.PlotHist` for plotting historgrams
    of point sensor results in particular.

    :Call:
        >>> DBP = DBTriqPoint(cntl, opts, pt, name=None)
    :Inputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
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
        *DBP*: :class:`pyFun.pointsensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2015-12-04 ``@ddalle``: Started
    """

    pass

# class DBTriqPoint
