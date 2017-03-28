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
import os, glob
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
from . import case
# Special data books
from . import lineLoad
# Special class
import pyFun.plt

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
        # Check compatibility of the environment
        if os.environ.get('DISPLAY') is None:
            # Use a special MPL backend to avoid need for DISPLAY
            import matplotlib
            matplotlib.use('Agg')
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
    # Initialize a DBComp object
    def InitDBComp(self, comp, x, opts, targ=None):
        """Initialize data book for one component
        
        :Call:
            >>> DB.InitDBComp(comp, x, opts)
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *x*: :class:`pyFun.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *opts*: :class:`pyFun.options.Options`
                Global pyCart options instance
            *targ*: {``None``} | :class:`str`
                If used, read a duplicate data book as a target named *targ*
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
            * 2016-06-27 ``@ddalle``: Added *targ* keyword
        """
        self[comp] = DBComp(comp, x, opts, targ=targ)
        
    # Read line load
    def ReadLineLoad(self, comp, conf=None, targ=None):
        """Read a line load data book if it is not already present
        
        :Call:
            >>> DB.ReadLineLoad(comp)
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
                Instance of the pyFun data book class
            *comp*: :class:`str`
                Line load component group
            *conf*: {``"None"``} | :class:`cape.config.Config`
                Surface configuration interface
            *targ*: {``"None"``} | :class:`str`
                Sets alternate directory to read from, defaults to *DB.targ*
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
            * 2016-06-27 ``@ddalle``: Added *targ*
        """
        # Initialize if necessary
        try:
            self.LineLoads
        except Exception:
            self.LineLoads = {}
        # Try to access the line load
        try:
            if targ is None:
                # Check for the line load data book as is
                self.LineLoads[comp]
            else:
                # Check for the target
                self.ReadTarget(targ)
                # Check for the target line load
                self.Targets[targ].LineLoads[comp]
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Default target name
            if targ is None:
                # Read the file.
                self.LineLoads[comp] = lineLoad.DBLineLoad(self.x, self.opts,
                    comp, conf=conf, RootDir=self.RootDir, targ=self.targ)
            else:
                # Read as a specified target.
                ttl = '%s\\%s' % (targ, comp)
                # Read the file.
                self.LineLoads[ttl] = lineLoad.DBLineLoad(self.x, self.opts,
                    comp, conf=conf, RootDir=self.RootDir, targ=targ)
            # Return to starting location
            os.chdir(fpwd)
    
    # Read TrqiFM components
    def ReadTriqFM(self, comp):
        """Read a TriqFM data book if not already present
        
        :Call:
            >>> DB.ReadTriqFM(comp)
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
                Instance of pyFun data book class
            *comp*: :class:`str`
                Name of TriqFM component
        :Versions:
            * 2017-03-28 ``@ddalle``: First version
        """
        # Initialize if necessary
        try:
            self.TriqFM
        except Exception:
            self.TriqFM = {}
        # Try to access the TriqFM database
        try:
            self.TriqFM[comp]
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read data book
            self.TriqFM[comp] = DBTriqFM(self.x, self.opts, comp,
                RootDir=self.RootDir)
            # Return to starting position
            os.chdir(fpwd)
        
    # Update data book
    def UpdateDataBook(self, I=None):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DB.UpdateDataBook()
            >>> DB.UpdateDataBook(I)
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
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

    # Update or add an entry.
    def UpdateCase(self, i):
        """Update or add a case to a data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> DB.UpdateCase(i)
        :Inputs:
            *DB*: :class:`pyFun.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Trajectory index
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the first data book component.
        DBc = self.GetRefComponent()
        # Try to find a match existing in the data book.
        j = DBc.FindMatch(i)
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
        nIter = GetCurrentIter()
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
        elif DBc['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i."
                % (self[c0]['nIter'][j], nIter))
            q = True
        elif DBc['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Maximum number of iterations allowed.
        nMax = min(nIter-nMin, self.opts.get_nMaxStats())
        # Read residual
        H = CaseResid(self.proj)
        # Loop through components.
        for comp in self.Components:
            # Ensure proper type
            tcomp = self.opts.get_DataBookType(comp)
            if tcomp not in ['Force', 'Moment', 'FM']: continue
            # Get component (note this automatically defaults to *comp*)
            compID = self.opts.get_DataBookCompID(comp)
            # Check for multiple components
            if type(compID).__name__ in ['list', 'ndarray']:
                # Read the first component
                FM = CaseFM(proj, compID[0])
                # Loop through remaining components
                for compi in compID[1:]:
                    # Check for minus sign
                    if compi.startswith('-1'):
                        # Subtract the component
                        FM -= CaseFM(proj, compi.lstrip('-'))
                    else:
                        # Add in the component
                        FM += CaseFM(proj, compi)
            else:
                # Read the iterative history for single component
                FM = CaseFM(proj, compID)
            # Extract the component databook.
            DBc = self[comp]
            # List of transformations
            tcomp = self.opts.get_DataBookTransformations(comp)
            # Special transformation to reverse *CLL* and *CLN*
            tflight = {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}
            # Check for ScaleCoeffs
            if tflight not in tcomp:
                # Append a transformation to reverse *CLL* and *CLN*
                tcomp.append(tflight)
            # Loop through the transformations.
            for topts in tcomp:
                # Apply the transformation.
                FM.TransformFM(topts, self.x, i)
                
            # Process the statistics.
            s = FM.GetStats(nStats, nMax)
            # Get the corresponding residual drop
            nOrders = H.GetNOrders(s['nStats'])
            
            # Save the data.
            if np.isnan(j):
                # Add to the number of cases.
                DBc.n += 1
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DBc[k] = np.append(DBc[k], getattr(self.x,k)[i])
                # Append values.
                for c in DBc.DataCols:
                    DBc[c] = np.hstack((DBc[c], [s[c]]))
                # Append residual drop.
                DBc['nOrders'] = np.hstack((DBc['nOrders'], [nOrders]))
                # Append iteration counts.
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DBc.DataCols:
                    DBc[c][j] = s[c]
                # Update the other statistics.
                DBc['nOrders'][j] = nOrders
                DBc['nIter'][j]   = nIter
                DBc['nStats'][j]  = s['nStats']
        # Go back.
        os.chdir(self.RootDir)
# class DataBook

# Component data book
class DBComp(cape.dataBook.DBComp):
    """Individual component data book
    
    This class is derived from :class:`cape.dataBook.DBBase`. 
    
    :Call:
        >>> DBc = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`cape.options.Options`
            Global pyCart options instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
    :Outputs:
        *DBc*: :class:`pyOver.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2016-09-15 ``@ddalle``: First version
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
        >>> DBT = DBTarget(targ, x, opts)
    :Inputs:
        *targ*: :class:`pyFun.options.DataBook.DBTarget`
            Instance of a target source options interface
        *x*: :class:`pyFun.trajectory.Trajectory`
            Run matrix interface
        *opts*: :class:`pyFun.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyFun.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    pass
# class DBTarget

# TriqFM data book
class DBTriqFM(cape.dataBook.DBTriqFM):
    """Force and moment component extracted from surface triangulation
    
    :Call:
        >>> DBF = DBTriqFM(x, opts, comp, RootDir=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of TriqFM component
        *RootDir*: {``None``} | :class:`st`
            Root directory for the configuration
    :Outputs:
        *DBF*: :class:`pyFun.dataBook.DBTriqFM`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: First version
    """
    # Get file
    def GetTriqFile(self):
        """Get most recent ``triq`` file and its associated iterations
        
        :Call:
            >>> qtriq, ftriq, n, i0, i1 = DBF.GetTriqFile()
        :Inputs:
            *DBF*: :class:`pyFun.dataBook.DBTriqFM`
                Instance of TriqFM data book
        :Outputs:
            *qtriq*: {``False``}
                Whether or not to convert file from other format
            *ftriq*: :class:`str`
                Name of ``triq`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2016-12-19 ``@ddalle``: Added to the module
        """
        # Get properties of triq file
        fplt, n, i0, i1 = case.GetPltFile()
        # Get the corresponding .triq file name
        ftriq = fplt.rstrip('.plt') + '.triq'
        # Check if the TRIQ file exists
        if os.path.isfile(ftriq):
            # No conversion needed
            qtriq = False
        else:
            # Need to convert PLT file to TRIQ
            qtriq = True
        # Output
        return qtriq, ftriq, n, i0, i1
    
    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, ftriq, **kw):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> DBL.PreprocessTriq(ftriq, i=None)
        :Inputs:
            *DBF*: :class:`pyFun.dataBook.DBTriqFM`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of triq file
            *i*: {``None``} | :class:`int`
                Case index (else read from :file:`conditions.json`)
        :Versions:
            * 2017-03-28 ``@ddalle``: First version
        """
        # Get name of plt file
        fplt = ftriq.rstrip('triq') + 'plt'
        # Get case index
        i = kw.get('i')
        # Read Mach number
        if i is None:
            # Read from :file:`conditions.json`
            mach = case.ReadConditions('mach')
        else:
            # Get from trajectory
            mach = self.x.GetMach(i)
        # Read the plt information
        pyFun.plt.Plt2Triq(fplt, ftriq, mach=mach)
# class DBTriqFM
    

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
        * 2016-05-05 ``@ddalle``: Handles adaptive; ``pyfun00,pyfun01,...``
        * 2016-10-28 ``@ddalle``: Catching iteration resets
    """
    # Initialization method
    def __init__(self, proj, comp):
        """Initialization method"""
        # Save component name
        self.comp = comp
        # Get the project rootname
        self.proj = proj
        # File names use lower case here
        projl = proj.lower()
        compl = comp.lower()
        # Check for ``Flow`` folder
        if os.path.isdir('Flow'):
            # Dual setup
            qdual = True
            os.chdir('Flow')
        else:
            # Single folder
            qdual = False
        # Expected name of the component history file(s)
        self.fname = '%s_fm_%s.dat' % (projl, compl)
        # Full list
        if os.path.isfile(self.fname):
            # Single project; check for history resets
            fglob1 = glob.glob('%s_fm_%s.[0-9][0-9].dat' % (projl, compl))
            fglob1.sort()
            # Add in main file name
            self.fglob = fglob1 + [self.fname]
        else:
            # Multiple projects
            fglob2 = glob.glob('%s[0-9][0-9]_fm_%s.dat' % (projl, compl))
            fglob3 = glob.glob('%s[0-9][0-9]_fm_%s.[0-9][0-9].dat' \
                % (projl, compl))
            # Combine them
            self.fglob = fglob2 + fglob3
            self.fglob.sort()
        # Check for available files.
        if len(self.fglob) > 0:
            # Read the first file
            self.ReadFileInit(self.fglob[0])
            # Loop through other files
            for fname in self.fglob[1:]:
                # Append the data
                self.ReadFileAppend(fname)
        else:
            # Make an empty CaseFM
            self.MakeEmpty()
        # Return if necessary
        if qdual:
            os.chdir('..')
            
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
        
    # Read data from an initial file
    def ReadFileInit(self, fname=None):
        """Read data from a file and initialize columns
        
        :Call:
            >>> FM.ReadFileInit(fname=None)
        :Inputs:
            *FM*: :class:`pyFun.dataBook.CaseFM`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                Name of file to process (defaults to *FM.fname*)
        :Versions:
            * 2016-05-05 ``@ddalle``: First version
        """
        # Default file name
        if fname is None: fname = self.fname
        # Process the column names
        nhdr, cols, coeffs, inds = self.ProcessColumnNames(fname)
        # Save entries
        self._hdr   = nhdr
        self.cols   = cols
        self.coeffs = coeffs
        self.inds   = inds
        # Read the data.
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Number of columns.
        n = len(self.cols)
        # Save the values.
        for k in range(n):
            # Set the values from column *k* of *A*
            setattr(self,cols[k], A[:,k])
            
    # Read data from a second or later file
    def ReadFileAppend(self, fname):
        """Read data from a file and append it to current history
        
        :Call:
            >>> FM.ReadFileAppend(fname)
        :Inputs:
            *FM*: :class:`pyFun.dataBook.CaseFM`
                Case force/moment history
            *fname*: :class:`str`   
                Name of file to read
        :Versions:
            * 2016-05-05 ``@ddalle``: First version
            * 2016-10-28 ``@ddalle``: Catching iteration resets
        """
        # Process the column names
        nhdr, cols, coeffs, inds = self.ProcessColumnNames(fname)
        # Check entries
        for col in cols:
            # Check for existing column
            if col in self.cols: continue
            # Initialize the column
            setattr(self,col, np.zeros_like(self.i, dtype=float))
            # Append to the end of the list
            self.cols.append(col)
        # Read the data.
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Number of columns.
        n = len(cols)
        # Append the values.
        for k in range(n):
            # Column name
            col = cols[k]
            # Value to use
            V = A[:,k]
            # Check for iteration number reset
            if col == 'i' and V[0] < self.i[-1]:
                # Keep counting iterations from the end of the previous one.
                V += (self.i[-1] - V[0] + 1)
            # Append
            setattr(self,col, np.hstack((getattr(self,col), V)))
                
        
    # Process the column names
    def ProcessColumnNames(self, fname=None):
        """Determine column names
        
        :Call:
            >>> nhdr, cols, coeffs, inds = FM.ProcessColumnNames(fname=None)
        :Inputs:
            *FM*: :class:`pyFun.dataBook.CaseFM`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                Name of file to process, defaults to *FM.fname*
        :Outputs:
            *nhdr*: :class:`int`
                Number of header rows to skip
            *cols*: :class:`list` (:class:`str`)
                List of column names
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficient names
            *inds*: :class:`list` (:class:`int`)
                List of column indices for each entry of *cols*
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
            * 2016-05-05 ``@ddalle``: Using outputs instead of saving to *FM*
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Default file name
        if fname is None: fname = self.fname
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                nhdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'): continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2: continue
                # Split variables on as things between quotes
                vals = re.findall('"[\w ]+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                nhdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall('"[\w ]+"', l)
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
                    nhdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        coeffs = []
        # Check for iteration column.
        if "Iteration" in keys:
            inds.append(keys.index("Iteration"))
            cols.append('i')
        # Check for CA (axial force)
        if "C_x" in keys:
            inds.append(keys.index("C_x"))
            cols.append('CA')
            coeffs.append('CA')
        # Check for CY (body side force)
        if "C_y" in keys:
            inds.append(keys.index("C_y"))
            cols.append('CY')
            coeffs.append('CY')
        # Check for CN (normal force)
        if "C_z" in keys:
            inds.append(keys.index("C_z"))
            cols.append('CN')
            coeffs.append('CN')
        # Check for CLL (rolling moment)
        if "C_M_x" in keys:
            inds.append(keys.index("C_M_x"))
            cols.append('CLL')
            coeffs.append('CLL')
        # Check for CLM (pitching moment)
        if "C_M_y" in keys:
            inds.append(keys.index("C_M_y"))
            cols.append('CLM')
            coeffs.append('CLM')
        # Check for CLN (yawing moment)
        if "C_M_z" in keys:
            inds.append(keys.index("C_M_z"))
            cols.append('CLN')
            coeffs.append('CLN')
        # Check for CL
        if "C_L" in keys:
            inds.append(keys.index("C_L"))
            cols.append('CL')
            coeffs.append('CL')
        # Check for CD
        if "C_D" in keys:
            inds.append(keys.index("C_D"))
            cols.append('CD')
            coeffs.append('CD')
        # Check for CA (axial force)
        if "C_xp" in keys:
            inds.append(keys.index("C_xp"))
            cols.append('CAp')
            coeffs.append('CAp')
        # Check for CY (body side force)
        if "C_yp" in keys:
            inds.append(keys.index("C_yp"))
            cols.append('CYp')
            coeffs.append('CYp')
        # Check for CN (normal force)
        if "C_zp" in keys:
            inds.append(keys.index("C_zp"))
            cols.append('CNp')
            coeffs.append('CNp')
        # Check for CLL (rolling moment)
        if "C_M_xp" in keys:
            inds.append(keys.index("C_M_xp"))
            cols.append('CLLp')
            coeffs.append('CLLp')
        # Check for CLM (pitching moment)
        if "C_M_yp" in keys:
            inds.append(keys.index("C_M_yp"))
            cols.append('CLMp')
            coeffs.append('CLMp')
        # Check for CLN (yawing moment)
        if "C_M_zp" in keys:
            inds.append(keys.index("C_M_zp"))
            cols.append('CLNp')
            coeffs.append('CLNp')
        # Check for CL
        if "C_Lp" in keys:
            inds.append(keys.index("C_Lp"))
            cols.append('CLp')
            coeffs.append('CLp')
        # Check for CD
        if "C_Dp" in keys:
            inds.append(keys.index("C_Dp"))
            cols.append('CDp')
            coeffs.append('CDp')
        # Check for CA (axial force)
        if "C_xv" in keys:
            inds.append(keys.index("C_xv"))
            cols.append('CAv')
            coeffs.append('CAv')
        # Check for CY (body side force)
        if "C_yv" in keys:
            inds.append(keys.index("C_yv"))
            cols.append('CYv')
            coeffs.append('CYv')
        # Check for CN (normal force)
        if "C_zv" in keys:
            inds.append(keys.index("C_zv"))
            cols.append('CNv')
            coeffs.append('CNv')
        # Check for CLL (rolling moment)
        if "C_M_xv" in keys:
            inds.append(keys.index("C_M_xv"))
            cols.append('CLLv')
            coeffs.append('CLLv')
        # Check for CLM (pitching moment)
        if "C_M_yv" in keys:
            inds.append(keys.index("C_M_yv"))
            cols.append('CLMv')
            coeffs.append('CLMv')
        # Check for CLN (yawing moment)
        if "C_M_zv" in keys:
            inds.append(keys.index("C_M_zv"))
            cols.append('CLNv')
            coeffs.append('CLNv')
        # Check for CL
        if "C_Lv" in keys:
            inds.append(keys.index("C_Lv"))
            cols.append('CLv')
            coeffs.append('CLv')
        # Check for CD
        if "C_Dv" in keys:
            inds.append(keys.index("C_Dv"))
            cols.append('CDv')
            coeffs.append('CDv')
        # Output
        return nhdr, cols, coeffs, inds
        
# class CaseFM


# Class to keep track of residuals
class CaseResid(cape.dataBook.CaseResid):
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
        * 2016-10-28 ``@ddalle``: Catching iteration resets
    """
    
    # Initialization method
    def __init__(self, proj):
        """Initialization method
        
        :Versions:
            * 2015-10-21 ``@ddalle``: First version
        """
        # Save the project root name
        self.proj = proj
        # Use lower case for Fortran code
        projl = proj.lower()
        # Check for ``Flow`` folder
        if os.path.isdir('Flow'):
            # Dual setup
            qdual = True
            os.chdir('Flow')
        else:
            # Single folder
            qdual = False
        # Expected name of the history file
        self.fname = "%s_hist.dat" % proj.lower()
        # Full list
        if os.path.isfile(self.fname):
            # Single project; check for history resets
            fglob1 = glob.glob('%s_hist.[0-9][0-9].dat' % projl)
            fglob1.sort()
            # Add in main file name
            self.fglob = fglob1 + [self.fname]
        else:
            # Multiple adaptations
            fglob2 = glob.glob('%s[0-9][0-9]_hist.dat' % projl)
            fglob2.sort()
            # Check for history resets
            fglob1 = glob.glob('%s[0-9][0-9]_hist.[0-9][0-9].dat' % projl)
            fglob1.sort()
            # Combine history resets
            if len(fglob2) > 0:
                # Combine history and **current** project
                self.fglob = fglob1 + fglob2[-1:]
            else:
                # Only have historical iterations right now
                self.fglob = fglob1
        # Check for which file(s) to use
        if len(self.fglob) > 0:
            # Read the first file
            self.ReadFileInit(self.fglob[0])
            # Loop through other files
            for fname in self.fglob[1:]:
                # Append the data
                self.ReadFileAppend(fname)
        else:
            # Make an empty history
            self.MakeEmpty()
        # Save number of iterations
        self.nIter = len(self.i)
        # Initialize residuals
        L2 = np.zeros(self.nIter)
        L0 = np.zeros(self.nIter)
        # Check residuals
        if 'R_1' in self.cols: L2 += (self.R_1**2)
        if 'R_2' in self.cols: L2 += (self.R_2**2)
        if 'R_3' in self.cols: L2 += (self.R_3**2)
        if 'R_4' in self.cols: L2 += (self.R_4**2)
        if 'R_5' in self.cols: L2 += (self.R_5**2)
        # Check initial subiteration residuals
        if 'R_10' in self.cols: L0 += (self.R_10**2)
        if 'R_20' in self.cols: L0 += (self.R_20**2)
        if 'R_30' in self.cols: L0 += (self.R_30**2)
        if 'R_40' in self.cols: L0 += (self.R_40**2)
        if 'R_50' in self.cols: L0 += (self.R_50**2)
        # Save residuals
        self.L2Resid = np.sqrt(L2)
        self.L2Resid0 = np.sqrt(L0)
        # Return if appropriate
        if qdual: os.chdir('..')
        
    # Plot R_1
    def PlotR1(self, **kw):
        """Plot the density
        
        :Call:
            >>> h = hist.PlotR1(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-10-21 ``@ddalle``: First version
        """
        # Plot "R_1"
        return self.PlotResid('R_1', YLabel='Density Residual', **kw)
        
    # Plot turbulence residual
    def PlotTurbResid(self, **kw):
        """Plot the turbulence residual
        
        :Call:
            >>> h = hist.PlotTurbResid(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-10-21 ``@ddalle``: First version
        """
        # Plot "R_6"
        return self.PlotResid('R_6', YLabel='Turbulence Residual', **kw)
        
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
        self.t = np.array([])
        self.R_1 = np.array([])
        self.R_2 = np.array([])
        self.R_3 = np.array([])
        self.R_4 = np.array([])
        self.R_5 = np.array([])
        self.R_6 = np.array([])
        # Residuals
        self.L2Resid = np.array([])
        self.L2Resid0 = np.array([])
        # Number of iterations
        self.nIter = 0
        # Save a default list of columns
        self.cols = ['i', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6']
        
    # Process the column names
    def ProcessColumnNames(self, fname=None):
        """Determine column names
        
        :Call:
            >>> nhdr, cols, inds = hist.ProcessColumnNames(fname=None)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                File name to process, defaults to *FM.fname*
        :Outputs:
            *nhdr* :class:`int`
                Number of header rows
            *cols*: :class:`list` (:class:`str`)
                List of columns
            *inds*: :class:`list` (:class:`int`)
                List of indices in columns
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
            * 2016-05-05 ``@ddalle``: Use output instead of saving to *FM*
        """
        # Default file name
        if fname is None: fname = self.fname
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                nhdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'): continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2: continue
                # Split variables on as things between quotes
                vals = re.findall('"[\w ]+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                nhdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall('"[\w ]+"', l)
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
                    nhdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        # Check for iteration column.
        if "Iteration" in keys:
            inds.append(keys.index("Iteration"))
            cols.append('i')
        if "Fractional_Time_Step" in keys:
            inds.append(keys.index("Fractional_Time_Step"))
            cols.append('j')
        if "Wall Time" in keys:
            inds.append(keys.index("Wall Time"))
            cols.append('CPUtime')
        # Check for CA (axial force)
        if "R_1" in keys:
            inds.append(keys.index("R_1"))
            cols.append('R_1')
        # Check for CA (axial force)
        if "R_2" in keys:
            inds.append(keys.index("R_2"))
            cols.append('R_2')
        # Check for CA (axial force)
        if "R_3" in keys:
            inds.append(keys.index("R_3"))
            cols.append('R_3')
        # Check for CA (axial force)
        if "R_4" in keys:
            inds.append(keys.index("R_4"))
            cols.append('R_4')
        # Check for CA (axial force)
        if "R_5" in keys:
            inds.append(keys.index("R_5"))
            cols.append('R_5')
        # Check for CA (axial force)
        if "R_6" in keys:
            inds.append(keys.index("R_6"))
            cols.append('R_6')
        # Output
        return nhdr, cols, inds
    
    # Read initial data
    def ReadFileInit(self, fname=None):
        """Initialize history by reading a file
        
        :Call:
            >>> hist.ReadFileInit(fname=None)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                File name to process, defaults to *FM.fname*
        :Outputs:
            *nhdr* :class:`int`
                Number of header rows
            *cols*: :class:`list` (:class:`str`)
                List of columns
            *inds*: :class:`list` (:class:`int`)
                List of indices in columns
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
            * 2016-05-05 ``@ddalle``: Now an output
        """
        # Default file name
        if fname is None: fname = self.fname
        # Process the column names
        nhdr, cols, inds = self.ProcessColumnNames(fname)
        # Save entries
        self._hdr = nhdr
        self.cols = cols
        self.inds = inds
        # Read the data.
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Number of columns.
        n = len(self.cols)
        # Save the values.
        for k in range(n):
            # Set the values from column *k* of *A*
            setattr(self,cols[k], A[:,k])
        # Check for subiteration history
        Vsub = fname.split('.')
        fsub = Vsub[0][:-40 ]+ "subhist." + (".".join(Vsub[1:]))
        # Check for the file
        if os.path.isfile(fsub):
            # Process subiteration
            self.ReadSubhist(fsub)
            return
        # Initialize residuals
        for k in range(n):
            # get column name
            col = cols[k]
            c0 = col + '0'
            # Check for special commands
            if not col.startswith('R'): continue
            # Copy the shape of the residual
            setattr(self,c0, np.nan*np.ones_like(getattr(self,col)))
            # Append column list
            self.cols.append(c0)
    
    # Read data from a second or later file
    def ReadFileAppend(self, fname):
        """Read data from a file and append it to current history
        
        :Call:
            >>> hist.ReadFileAppend(fname)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Case force/moment history
            *fname*: :class:`str`   
                Name of file to read
        :Versions:
            * 2016-05-05 ``@ddalle``: First version
            * 2016-10-28 ``@ddalle``: Catching iteration resets
        """
        # Process the column names
        nhdr, cols, inds = self.ProcessColumnNames(fname)
        # Check entries
        for col in cols:
            # Check for existing column
            if col in self.cols: continue
            # Initialize the column
            setattr(self,col, np.zeros_like(self.i, dtype=float))
            # Append to the end of the list
            self.cols.append(col)
        # Read the data.
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Number of columns.
        n = len(cols)
        # Save current last iteration
        i1 = self.i[-1]
        # Append the values.
        for k in range(n):
            # Column name
            col = cols[k]
            # Value to use
            V = A[:,k]
            # Check for iteration number reset
            if col == 'i' and V[0] < self.i[-1]:
                # Keep counting iterations from the end of the previous one.
                V += (i1 - V[0] + 1)
            # Append
            setattr(self,col, np.hstack((getattr(self,col), V)))
        # Check for subiteration history
        Vsub = fname.split('.')
        fsub = Vsub[0][:-4] + "subhist." + (".".join(Vsub[1:]))
        I = np.hstack((self.i[:-1]>self.i[1:], [True]))
        # Check for the file
        if os.path.isfile(fsub):
            # Read the subiteration history
            self.ReadSubhist(fsub, iend=i1)
            return
        # Initialize residuals
        for k in range(n):
            # Get column name
            col = cols[k]
            c0 = col + '0'
            # Check for special commands
            if not col.startswith('R'): continue
            # Copy the shape of the residual
            setattr(self,c0, np.hstack(
                (getattr(self,c0), np.nan*np.ones_like(V))))
            
    # Read subiteration history
    def ReadSubhist(self, fname, iend=0):
        """Read subiteration history
        
        :Call:
            >>> hist.ReadSubhist(fname)
        :Inputs:
            *hist*: :class:`pyFun.dataBook.CaseResid`
                Fun3D residual history interface
            *fname*: :class:`str`
                Name of subiteration history file
            *iend*: {``0``} | positive :class:`int`
                Last iteration number before reading this file
        :Versions:
            * 2016-10-29 ``@ddalle``: First version
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Count line
            nhdr += 1
            # Check for "variables"
            if not l.lower().startswith('variables'): continue
            # Split on '=' sign.
            L = l.split('=')
            # Check for first variable.
            if len(L) < 2: break
            # Split variables on as things between quotes
            vals = re.findall('"[\w ]+"', L[1])
            # Append to the list.
            keys += [v.strip('"') for v in vals]
            break
        # Number of keys
        nkey = len(keys)
        # Read the data
        B = np.fromfile(f, sep=' ')
        # Get number of complete records
        nA = int(len(B) / nkey)
        # Reshape
        A = np.reshape(B[:nA*nkey], (nA, nkey))
        # Close the file
        f.close()
        # Initialize the output
        d = {}
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        # Check for iteration column.
        if "Fractional_Time_Step" in keys:
            inds.append(keys.index("Fractional_Time_Step"))
            cols.append('i')
        # Check for residual of state 1
        if "R_1" in keys:
            inds.append(keys.index("R_1"))
            cols.append('R_1')
        # Check for residual of state 2
        if "R_2" in keys:
            inds.append(keys.index("R_2"))
            cols.append('R_2')
        # Check for residual of state 3
        if "R_3" in keys:
            inds.append(keys.index("R_3"))
            cols.append('R_3')
        # Check for residual of state 4
        if "R_4" in keys:
            inds.append(keys.index("R_4"))
            cols.append('R_4')
        # Check for residual of state 5
        if "R_5" in keys:
            inds.append(keys.index("R_5"))
            cols.append('R_5')
        # Check for turbulent residual
        if "R_6" in keys:
            inds.append(keys.index("R_6"))
            cols.append('R_6')
        # Loop through columns
        n = len(cols)
        for k in range(n):
            # Column name
            col = cols[k]
            # Save it
            d[col] = A[:,inds[k]]
        # Check for integers
        if 'i' not in d:
            return
        # Indices of matching integers
        I = d['i'] == np.array(d['i'], dtype='int')
        # Don't read past the last write of '*_hist.dat'
        I = np.logical_and(I, d['i']+iend<=self.i[-1])
        # Loop through the columns again to save them
        for k in range(n):
            # Column name
            col = cols[k]
            c0  = col + '0'
            # Get the values
            v = d[col][I]
            # Check integers
            if col == 'i':
                # Get expected iteration numbers
                ni = len(v)
                ip = self.i[-ni:]
                # Offset current iteration numbers by reset iter
                iv = v + iend
                # Compare to existing iteration numbers
                if np.any(ip != iv):
                    raise ValueError("Mismatch between nominal history " +
                        ("(%i-%i) and subiteration history (%i-%i)" %
                        (ip[0], ip[-1], iv[0], iv[-1])))
            # Check to append
            try:
                # Check if the attribute is present
                v0 = getattr(self,c0)
                # Save it if that command succeeded
                setattr(self,c0, np.hstack((v0, v)))
            except AttributeError:
                # Save the value as a new one
                setattr(self,c0, v)
                # Save column
                self.cols.append(c0)
        
    # Number of orders of magintude of residual drop
    def GetNOrders(self, nStats=1):
        """Get the number of orders of magnitude of residual drop
        
        :Call:
            >>> nOrders = hist.GetNOrders(nStats=1)
            
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
                Instance of the DataBook residual history
            *nStats*: :class:`int`
                Number of iterations to use for averaging the final residual
        :Outputs:
            *nOrders*: :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-10-21 ``@ddalle``: First versoin
        """
        
        # Process the number of usable iterations available.
        i = max(self.nIter-nStats, 0)
        # Get the maximum residual.
        L1Max = np.log10(np.max(self.R_1))
        # Get the average terminal residual.
        L1End = np.log10(np.mean(self.R_1[i:]))
        # Return the drop
        return L1Max - L1End
        
    
# class CaseResid

