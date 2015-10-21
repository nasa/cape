"""
Data Book Module: :mod:`pyFun.dataBook`
=======================================

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
from .case import GetCurrentIter, GetProjectRootname
# Utilities or advanced statistics
from . import util

# Template module
import cape.dataBook

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
        >>> DB = pyFun.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyFun.trajectory.Trajectory`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`pyFun.dataBook.DataBook`
            Instance of the pyFun data book class
    :Versions:
        * 2015-10-20 ``@ddalle``: Started
    """
    pass

# class DataBook

# Force/moment history
class CaseFM(cape.dataBook.CaseFM):
    """
    This class contains methods for reading data about an the history of an
    individual component for a single case.  It reads the Tecplot file
    :file:`$proj_fm_$comp.dat` where *proj* is the lower-case root project name
    and *comp* is the name of the component.  From this file it determines
    which coefficients are recorded automatically.
    
    :Call:
        >>> FM = pyFun.dataBook.CaseFM(proj, comp)
    :Inputs:
        *proj*: :class:`str`
            Root name of the project
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *FM*: :class:`pyFun.aero.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list` (:class:`str`)
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
    def __init__(self, proj, comp):
        """Initialization method"""
        # Save component name
        self.comp = comp
        # Get the project rootname
        self.proj = proj
        # Expected name of the component history file
        self.fname = '%s_fm_%s.dat' % (proj.lower(), comp.lower())
        # Check if it exists.
        if not os.path.isfile(self.fname):
            # Make an empty CaseFM
            self.MakeEmpty()
            return
        # Process the column indices
        self.ProcessColumnNames()
        # Read the data.
        A = np.loadtxt(self.fname,
            skiprows=self._hdr, usecols=tuple(self.inds))
        # Number of columns.
        n = len(self.cols)
        # Save the values.
        for k in range(n):
            # Set the values from column *k* of *A*
            setattr(self,self.cols[k], A[:,k])
        
            
    # Function to make empty one.
    def MakeEmpty(self):
        """Create empty *CaseFM* instance
        
        :Call:
            >>> FM.MakeEmpty()
        :Inputs:
            *FM*: :class:`pyFun.dataBook.CaseFM`
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
    def ProcessColumnNames(self):
        """Determine column names
        
        :Call:
            >>> FM.ProcessColumnNames(fname)
        :Inputs:
            *FM*: :class:`pyFun.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        self._hdr = 0
        # Open the file
        f = open(self.fname)
        # Loop through lines
        while self._hdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                self._hdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'): continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2: continue
                # Split variables on as things between quotes
                vals = re.findall('"\w+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                self._hdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall('"\w+"', l)
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            else:
                # Check if it starts with an integer
                try:
                    # If it's an integer, stop reading lines.
                    float(l.split()[0])
                    break
                except Exception:
                    # Line starts with something else; continue
                    self._hdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        self.inds = []
        self.cols = []
        self.coeffs = []
        # Check for iteration column.
        if "Iteration" in keys:
            self.inds.append(keys.index("Iteration"))
            self.cols.append('i')
        # Check for CA (axial force)
        if "C_x" in keys:
            self.inds.append(keys.index("C_x"))
            self.cols.append('CA')
            self.coeffs.append('CA')
        # Check for CY (body side force)
        if "C_y" in keys:
            self.inds.append(keys.index("C_y"))
            self.cols.append('CY')
            self.coeffs.append('CY')
        # Check for CN (normal force)
        if "C_z" in keys:
            self.inds.append(keys.index("C_z"))
            self.cols.append('CN')
            self.coeffs.append('CN')
        # Check for CLL (rolling moment)
        if "C_M_x" in keys:
            self.inds.append(keys.index("C_M_x"))
            self.cols.append('CLL')
            self.coeffs.append('CLL')
        # Check for CLM (pitching moment)
        if "C_M_y" in keys:
            self.inds.append(keys.index("C_M_y"))
            self.cols.append('CLM')
            self.coeffs.append('CLM')
        # Check for CLN (yawing moment)
        if "C_M_z" in keys:
            self.inds.append(keys.index("C_M_z"))
            self.cols.append('CLN')
            self.coeffs.append('CLN')
        
        
# class CaseFM


# Class to keep track of residuals
def CaseResid(cape.dataBook.CaseResid):
    """FUN3D iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given case
    
    :Call:
        >>> hist = pyFun.dataBook.CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *hist*: :class:`pyFun.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2015-10-21 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, proj):
        """Initialization method
        
        :Versions:
            * 2015-10-21 ``@ddalle``: First version
        """
        # Save the project root name
        self.proj = proj
        # Expected name of the history file
        self.fname = "%s_hist.dat" % proj.lower()
        # Check if it esists.
        if not os.path.isfile(self.fname):
            # Make an empty history
            self.MakeEmpty()
            return
        # Process the column indices
        self.ProcessColumnNames()
        # Read the data.
        A = np.loadtxt(self.fname,
            skiprows=self._hdr, usecols=tuple(self.inds))
        # Number of columns.
        n = len(self.cols)
        # Save the values.
        for k in range(n):
            # Set the values from column *k* of *A*
            setattr(self,self.cols[k], A[:,k])
        
        
    # Function to make empty one.
    def MakeEmpty(self):
        """Create empty *CaseResid* instance
        
        :Call:
            >>> hist.MakeEmpty()
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Case residual history
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Make all entries empty.
        self.i = np.array([])
        self.R_1 = np.array([])
        self.R_2 = np.array([])
        self.R_3 = np.array([])
        self.R_4 = np.array([])
        self.R_5 = np.array([])
        self.R_6 = np.array([])
        # Save a default list of columns
        self.cols = ['i', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6']
        
    # Process the column names
    def ProcessColumnNames(self):
        """Determine column names
        
        :Call:
            >>> hist.ProcessColumnNames()
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Case force/moment history
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        self._hdr = 0
        # Open the file
        f = open(self.fname)
        # Loop through lines
        while self._hdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                self._hdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'): continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2: continue
                # Split variables on as things between quotes
                vals = re.findall('"\w+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                self._hdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall('"\w+"', l)
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            else:
                # Check if it starts with an integer
                try:
                    # If it's an integer, stop reading lines.
                    float(l.split()[0])
                    break
                except Exception:
                    # Line starts with something else; continue
                    self._hdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        self.inds = []
        self.cols = []
        # Check for iteration column.
        if "Iteration" in keys:
            self.inds.append(keys.index("Iteration"))
            self.cols.append('i')
        # Check for CA (axial force)
        if "R_1" in keys:
            self.inds.append(keys.index("R_1"))
            self.cols.append('R_1')
        # Check for CA (axial force)
        if "R_2" in keys:
            self.inds.append(keys.index("R_2"))
            self.cols.append('R_2')
        # Check for CA (axial force)
        if "R_3" in keys:
            self.inds.append(keys.index("R_3"))
            self.cols.append('R_3')
        # Check for CA (axial force)
        if "R_4" in keys:
            self.inds.append(keys.index("R_4"))
            self.cols.append('R_4')
        # Check for CA (axial force)
        if "R_5" in keys:
            self.inds.append(keys.index("R_5"))
            self.cols.append('R_5')
        # Check for CA (axial force)
        if "R_6" in keys:
            self.inds.append(keys.index("R_6"))
            self.cols.append('R_6')

# class CaseResid

