"""
:mod:`cape.cfdx.lineLoad`: Sectional loads databook
========================================================

This module contains functions for reading and processing sectional loads.
This module is developed from :mod:`cape.cfdx.dataBook`, which is the overall
databook interface.  It provides the primary class :class:`DBLineLoad`, which
is a subclass of :class:`cape.cfdx.dataBook.DBBase`.  This class is an interface to
all line load data for a specific surface component.

Overall, this module provides three classes:

    * :class:`DBLineLoad`: Line load database for one component
    * :class:`CaseLL`: Line load data for one component of one CFD solution
    * :class:`CaseSeam`: Interface to "seam curves" to plot outline of surface
    
In addition to a database interface, this module also creates line loads.
Specific modifications to the generic template provided here are needed for
each individual CFD solver:

    * :mod:`cape.pycart.lineLoad`
    * :mod:`cape.pyfun.lineLoad`
    * :mod:`cape.pyover.lineLoad`
    
To calculate line loads, this module utilizes the Chimera Grid Tools executable
called ``triloadCmd``.  This works by taking a Cart3D annotated surface
triangulation (``triq`` file), slicing the surface component into slices, and
computing the loads on each slice.  In order to create this surface
triangulation, some solvers require steps to process the native CFD output.
Those steps are performed by the solver-specific :mod:`lineLoad` modules.

"""

# Standard library
import os
import glob

# Standard library: direct imports
from datetime import datetime

# Third-party modules
import numpy as np

# Local modules
from .. import util
from .. import tar
from . import dataBook
from . import case
from . import queue

# CAPE module: direct imports
from .options import odict
from cape.util import RangeString

# Placeholder variables for plotting functions.
plt = 0

# Radian -> degree conversion
deg = np.pi / 180.0

# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    """Import :mod:`matplotlib.pyplot` if not already loaded
    
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
        if case.os.environ.get('DISPLAY') is None:
            # Use a special MPL backend to avoid need for DISPLAY
            import matplotlib
            matplotlib.use('Agg')
        # Load the modules.
        import matplotlib.pyplot as plt
        # Other modules
        import matplotlib.transforms as tform
        from matplotlib.text import Text
# def ImportPyPlot

# Data book of line loads
class DBLineLoad(dataBook.DBBase):
    """Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(cntl, comp, conf=None, RootDir=None, targ=None)
    :Inputs:
        *cntl*: :class:`cape.cntl.Cntl`
            CAPE run matrix control instance
        *comp*: :class:`str`
            Name of line load component
        *conf*: {``None``} | :class:`cape.config.Config`
            Surface configuration interface
        *RootDir*: {``None``} | :class:`str`
            Root directory for the configuration
        *targ*: {``None``} | :class:`str`
            If used, read target data book's folder
    :Outputs:
        *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
            Instance of line load data book
        *DBL.nCut*: :class:`int`
            Number of *x*-cuts to make, from *opts*
        *DBL.RefL*: :class:`float`
            Reference length
        *DBL.MRP*: :class:`numpy.ndarray` shape=(3,)
            Moment reference center
        *DBL.x*: :class:`numpy.ndarray` shape=(*nCut*,)
            Locations of *x*-cuts
        *DBL.CA*: :class:`numpy.ndarray` shape=(*nCut*,)
            Axial force sectional load, d(CA)/d(x/RefL))
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
        * 2016-05-11 ``@ddalle``: Moved to :mod:`cape`
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, comp, cntl, conf=None, RootDir=None, **kw):
        """Initialization method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
            * 2016-06-07 ``@ddalle``: Updated slightly
        """
        # Save attributes
        self.cntl = cntl
        # Unpack key parts of cntl
        x = cntl.x
        opts = cntl.opts
        # Targ and keys
        targ = kw.get('targ')
        keys = kw.get("keys")
        # Save root directory
        if RootDir is None:
            # Use the current folder
            self.RootDir = os.getcwd()
        else:
            self.RootDir = RootDir
        
        # Get the data book directory.
        if targ is None:
            # Read from base directory
            fdir = opts.get_DataBookDir()
        else:
            # Read from target directory
            fdir = opts.get_DataBookTargetDir(targ)
        # Compatibility
        fdir = fdir.replace("/", os.sep)
        # Save folder
        self.fdir = fdir
        
        # Construct the file name.
        fcomp = 'll_%s.csv' % comp
        # Full file name
        fname = os.path.join(fdir, fcomp)
        
        # Safely change to root directory
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Create directories if necessary
        if not os.path.isdir(fdir):
            # Create data book folder (should not occur)
            self.mkdir(fdir)
        # Check for lineload folder
        if not os.path.isdir(os.path.join(fdir, 'lineload')):
            # Create line load folder
            os.mkdir(os.path.join(fdir, 'lineload'))
        # Return to original location
        os.chdir(fpwd)
        
        # Save the CFD run info
        self.x = x.Copy()
        self.opts = opts
        self.conf = conf
        # Specific options for this component
        self.copts = opts['DataBook'][comp]
        # Save component name
        self.proj = self.opts.get_DataBookPrefix(comp)
        self.comp = comp
        self.sec  = self.opts.get_DataBookSectionType(comp)
        # Defaults
        if self.proj is None: self.proj = 'LineLoad'
        if self.sec  is None: self.sec  = 'dlds'
        # Save the file name.
        self.fname = fname
        
        # Figure out reference component and list of CompIDs
        self.GetCompID()
        # Number of cuts
        self.nCut = self.opts.get_DataBook_nCut(self.comp)
        # Reference areas
        self.RefA = opts.get_RefArea(self.RefComp)
        self.RefL = opts.get_RefLength(self.RefComp)
        # Moment reference point
        self.MRP = np.array(opts.get_RefPoint(self.RefComp))
        # Read the file or initialize empty arrays.
        self.Read(fname, keys=keys)
        # Try to read the seams
        self.ReadSeamCurves()
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBLineLoad %s, " % self.comp
        # Number of cases in book
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    __str__ = __repr__
    
    # Get component ID numbers
    def GetCompID(self):
        """Create list of component IDs
        
        :Call:
            >>> DBL.GetCompID()
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Instance of line load data book
        :Versions:
            * 2016-12-22 ``@ddalle``: First version, extracted from __init__
        """
        # Figure out reference component
        self.CompID = self.opts.get_DataBookCompID(self.comp)
        # Make sure it's not a list
        if type(self.CompID).__name__ == 'list':
            # Take the first component
            self.RefComp = self.CompID[0]
        else:
            # One component listed; use it
            self.RefComp = self.CompID
        # Try to get all components
        try:
            # Use the configuration interface
            self.CompID = self.conf.GetCompID(self.CompID)
        except Exception:
            pass
            
  # >
    
  # ====
  # I/O
  # ====
  # <
   # --------
   # Main I/O
   # --------
   # [
    # function to read line load data book summary
    def Read(self, fname=None, keys=None):
        """Read a data book summary file for a single line load group
        
        :Call:
            >>> DBL.Read()
            >>> DBL.Read(fname)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Instance of line load data book
            *fname*: :class:`str`
                Name of summary file
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Default list of keys
        if keys is None:
            keys = self.x.cols
        # Save column names
        self.cols = keys + ['XMRP','YMRP','ZMRP','nIter','nStats']
        # Try to read the file.
        try:
            # Data book delimiter
            delim = self.opts.get_Delimiter()
            # Initialize column number.
            nCol = 0
            # Loop through the trajectory keys.
            for k in keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Read the column
                self[k] = np.loadtxt(fname,
                    delimiter=delim, dtype=str(t), usecols=[nCol])
                # Increase the column number.
                nCol += 1
            # MRP
            self['XMRP'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol])
            self['YMRP'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol+1])
            self['ZMRP'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol+2])
            # Iteration number
            nCol += 3
            self['nIter'] = np.loadtxt(fname,
                delimiter=delim, dtype=int, usecols=[nCol])
            # Stats
            nCol += 1
            self['nStats'] = np.loadtxt(fname,
                delimiter=delim, dtype=int, usecols=[nCol])
            # Number of cases
            self.n = self[k].size
            # Check for singletons
            if self[k].ndim == 0:
                # Loop through all keys
                for k in self.cols:
                    # Convert to array
                    self[k] = np.array([self[k]])
        except Exception as e:
            # Initialize empty trajectory arrays
            for k in self.x.cols:
                # get the type.
                t = self.x.defns[k].get('Value', 'float')
                # convert type
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Initialize an empty array.
                self[k] = np.array([], dtype=str(t))
            # Initialize Other parameters.
            self['XMRP'] = np.array([], dtype=float)
            self['YMRP'] = np.array([], dtype=float)
            self['ZMRP'] = np.array([], dtype=float)
            self['nIter'] = np.array([], dtype=int)
            self['nStats'] = np.array([], dtype=int)
            # No cases
            self.n = 0
    
    # Function to write line load data book summary file
    def Write(self, fname=None):
        """Write a single line load data book summary file
        
        :Call:
            >>> DBL.Write()
            >>> DBL.Write(fname)
        :Inputs:
            *DBL*: :class:`pycart.lineLoad.DBLineLoad`
                Instance of line load data book
            *fname*: :class:`str`
                Name of summary file
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # check for a previous old file.
        if os.path.isfile(fname+".old"):
            # Remove it.
            os.remove(fname+".old")
        # Check for an existing data file.
        if os.path.isfile(fname):
            # Move it to ".old"
            os.rename(fname, fname+".old")
        # DataBook delimiter
        delim = self.opts.get_Delimiter()
        # Open the file.
        f = open(fname, 'w')
        # Write the header
        f.write("# Line load summary for '%s' extracted on %s\n" %
            (self.comp, datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')))
        # Empty line.
        f.write('#\n')
        # Reference quantities
        f.write('# Reference Area = %.6E\n' % self.RefA)
        f.write('# Reference Length = %.6E\n' % self.RefL)
        # Moment reference point
        f.write('# Nominal moment reference point:\n')
        f.write('# XMRP = %.6E\n' % self.MRP[0])
        f.write('# YMRP = %.6E\n' % self.MRP[1])
        f.write('# ZMRP = %.6E\n' % self.MRP[2])
        # Empty line and start of variable list
        f.write('#\n# ')
        # Write the name of each trajectory key.
        for k in self.x.cols:
            f.write(k + delim)
        # Write the extra column titles.
        f.write('XMRP%sYMRP%sZMRP%snIter%snStats\n' %
            tuple([delim]*4))
        # Loop through database entries.
        for i in np.arange(self.n):
            # Write the trajectory values.
            for k in self.x.cols:
                f.write('%s%s' % (self[k][i], delim))
            # Write data values
            f.write('%s%s' % (self['XMRP'][i], delim))
            f.write('%s%s' % (self['YMRP'][i], delim))
            f.write('%s%s' % (self['ZMRP'][i], delim))
            # Iteration counts
            f.write('%i%s' % (self['nIter'][i], delim))
            f.write('%i\n' % (self['nStats'][i]))
        # Close the file.
        f.close()
        # Try to write the seam curves
        self.WriteSeamCurves()
   # ]
    
   # --------
   # Seam I/O
   # --------
   # [
    # Read the seam curves
    def ReadSeamCurves(self):
        """Read seam curves from a data book directory
        
        :Call:
            >>> DBL.ReadSeamCurves()
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
        :Versions:
            * 2015-09-17 ``@ddalle``: First version (:class:`CaseLL`)
            * 2016-06-09 ``@ddalle``: Adapted for :class:`DBLineLoad`
        """
        # Expected folder
        fll = os.path.join(self.RootDir, self.fdir, 'lineload')
        # Seam file name prefix
        fpre = os.path.join(fll, '%s_%s' % (self.proj, self.comp))
        # Name of output files.
        fsmx = '%s.smx' % fpre
        fsmy = '%s.smy' % fpre
        fsmz = '%s.smz' % fpre
        # Read the seam curves.
        self.smx = CaseSeam(fsmx)
        self.smy = CaseSeam(fsmy)
        self.smz = CaseSeam(fsmz)
    
    # Write (i.e. save) seam curves
    def WriteSeamCurves(self):
        """Write seam curves to a common data book directory
        
        :Call:
            >>> DBL.WriteSeamCurves()
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        # Expected folder
        fll = os.path.join(self.RootDir, self.fdir, 'lineload')
        # Check for folder
        if not os.path.isdir(fll): self.mkdir(fll)
        # Seam file name prefix
        fpre = os.path.join(fll, '%s_%s' % (self.proj, self.comp))
        # Name of seam files
        fsmx = '%s.smx' % fpre
        fsmy = '%s.smy' % fpre
        fsmz = '%s.smz' % fpre
        # Write the x-cuts if necessary and possible
        if not os.path.isfile(fsmx):
            try:
                self.smx.Write(fsmx)
            except Exception:
                pass
        # Write the x-cuts if necessary and possible
        if not os.path.isfile(fsmy):
            try:
                self.smy.Write(fsmy)
            except Exception:
                pass
        # Write the x-cuts if necessary and possible
        if not os.path.isfile(fsmz):
            try:
                self.smz.Write(fsmz)
            except Exception:
                pass
   # ]
    
   # ---------
   # Case I/O
   # ---------
   # [
    # Read a case from the data book
    def ReadCase(self, i):
        """Read data from a case from the data book archive
        
        :Call:
            >>> DBL.ReadCase(i=None, j=None)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number from run matrix
            *j*: :class:`int`
                Case number from data book
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
            * 2017-04-18 ``@ddalle``: Alternate index inputs
        """
        # Check if already up to date
        if i in self:
            return
        # Path to lineload folder
        fll = os.path.join(self.RootDir, self.fdir, 'lineload')
        # Get name of case
        frun = os.path.join(fll, self.x.GetFullFolderNames(i))
        # Check if the case is present
        if not os.path.isdir(frun):
            return
        # File name
        fname = os.path.join(frun, '%s_%s.csv' % (self.proj, self.comp))
        # Check for the file
        if not os.path.isfile(fname):
            return
        # Read the file
        self[i] = CaseLL(self.comp, 
            proj=self.proj, typ=self.sec, ext='csv', fdir=frun)
        # Copy the seam curves
        self[i].smx = self.smx
        self[i].smy = self.smy
        self[i].smz = self.smz
   # ]
   
  # >
    
  # ============
  # Organization
  # ============
  # <
    # Match the databook copy of the trajectory
    def UpdateRunMatrix(self):
        """Match the trajectory to the cases in the data book
        
        :Call:
            >>> DBL.UpdateRunMatrix()
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
        :Versions:
            * 2015-05-22 ``@ddalle``: First version
            * 2016-08-12 ``@ddalle``: Copied from data book
        """
        # Loop through the fields.
        for k in self.x.cols:
            # Check if the key is present
            if k in self:
                # Copy the data.
                self.x[k] = self[k]
                # Set the text.
                self.x.text[k] = [str(xk) for xk in self[k]]
            else:
                # Set faulty data
                self.x[k] = np.nan*np.ones(self.n)
                self.x.text[k] = ['' for xk in range(self.n)]
        # Set the number of cases.
        self.x.nCase = self.n
  # >
    
  # ===========
  # Calculation
  # ===========
  # <
    # Update a case
    def UpdateCase(self, i, qpbs=False, seam=False):
        """Update one line load entry if necessary
        
        :Call:
            >>> n = DBL.UpdateLineLoadCase(i, qpbs=False, seam=False)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number
            *qpbs*: ``True`` | {``False``}
                Whether or not to submit as a script
            *seam*: ``True`` | {``False``}
                Option to always read local seam curves
        :Outputs:
            *n*: ``0`` | ``1``
                Number of cases updated or added
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
            * 2016-12-19 ``@ddalle``: Modified for generic module
            * 2016-12-21 ``@ddalle``: Added PBS
            * 2017-04-24 ``@ddalle``: Removed PBS and added output
            * 2021-12-01 ``@ddalle``: Added *deam*
        """
        # Try to find a match in the data book
        j = self.FindMatch(i)
        # Get the name of the folder
        frun = self.x.GetFullFolderNames(i)
        # Go to root directory safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check if the folder exits
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return 0
        # Go to the folder.
        os.chdir(frun)
        # Determine minimum number of iterations required
        nAvg = self.opts.get_nStats(self.comp)
        nMin = self.opts.get_nMin(self.comp)
        # Get the number of iterations
        qtriq, ftriq, nStats, n0, nIter = self.GetTriqFile()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nAvg):
            # Not enough iterations (or zero)
            print("    %s" % frun)
            print("      Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry, but may have *lds files in run folder
            q = True
        elif self['nIter'][j] < nIter:
            # Update
            print("    %s" % frun)
            print("      Updating from iteration %i to %i." %
                (self['nIter'][j], nIter))
            q = True
        elif self['nStats'][j] < nStats:
            # Change statistics
            print("    %s" % frun)
            print("      Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            #print("  Databook '%s' up to date." % self.comp)
            q = False
        # Check for update
        if not q:
            os.chdir(fpwd)
            return 0
        # Create lineload folder if necessary
        if not os.path.isdir('lineload'):
            self.mkdir('lineload')
        # Enter lineload folder
        os.chdir('lineload')
        # Append to triq file
        ftriq = os.path.join('..', ftriq)
        # Name of loads file
        flds = '%s_%s.%s' % (self.proj, self.comp, self.sec)
        # Check whether or not to compute
        if not os.path.isfile(flds):
            # No loads yet
            q = True
        elif not os.path.isfile(ftriq):
            # TRIQ file needs preprocessing
            # This does imply out-of-date loads
            q = True
        elif os.path.getmtime(flds) < os.path.getmtime(ftriq):
            # Loads files are older than surface file
            q = True
        else:
            # Loads up to date
            q = False
        # Run triload if necessary
        if q:
            # Status update
            print("    " + frun)
            print("      Adding new databook entry at iteration %i." % nIter)
            # Write triloadCmd input file
            self.WriteTriloadInput(ftriq, i)
            # Run the command
            self.RunTriload(qtriq, ftriq, i=i)
        else:
            # Status update
            print("    " + frun)
            print("      Reading from %s/lineload/ folder" % frun)
        # Check number of seams
        try:
            # Get seam counts
            nsmx = self.smx.n
            nsmy = self.smy.n
            nsmz = self.smz.n
            # Check if at least some seam segments
            nsm = max(nsmx, nsmy, nsmz)
        except:
            # No seams yet
            nsm = 0
        # Read the loads file
        self[i] = CaseLL(self.comp, self.proj, self.sec, fdir=None, seam=seam)
        # Get the raw option from the data book
        db_transforms = self.opts.get_DataBookTransformations(self.comp)
        # Loop through transformations
        for topts in db_transforms:
            # Get type
            ttype = topts.get("Type")
            # Only apply ScaleCoeffs
            if ttype != "ScaleCoeffs":
                continue
            # Get multipliers and apply them
            self[i].CA *= topts.get("CA", 1.0)
            self[i].CY *= topts.get("CY", 1.0)
            self[i].CN *= topts.get("CN", 1.0)
            self[i].CLL *= topts.get("CLL", 1.0)
            self[i].CLM *= topts.get("CLM", 1.0)
            self[i].CLN *= topts.get("CLN", 1.0)
        # Check for null loads
        if self[i].x.size == 0:
            return 0
        # Check whether or not to read seams
        if nsm == 0:
            # Read the seam curves from this output
            if not seam:
                self[i].ReadSeamCurves()
            # Copy the seams
            self.smx = self[i].smx
            self.smy = self[i].smy
            self.smz = self[i].smz
        # CSV folder names
        fll  = os.path.join(self.RootDir, self.fdir, 'lineload')
        fgrp = os.path.join(fll, frun.split(os.sep)[0])
        fcas = os.path.join(fll, frun)
        # Create folders as necessary
        if not os.path.isdir(fll):  self.opts.mkdir(fll)
        if not os.path.isdir(fgrp): self.mkdir(fgrp)
        if not os.path.isdir(fcas): self.mkdir(fcas)
        # CSV file name
        fcsv = os.path.join(fcas, '%s_%s.csv' % (self.proj, self.comp))
        # Write the CSV file
        self[i].WriteCSV(fcsv)
        # Save the stats
        if np.isnan(j):
            # Add to the number of cases
            self.n += 1
            # Append trajectory values.
            for k in self.x.cols:
                # Append to numpy array
                self[k] = np.hstack((self[k], [self.x[k][i]]))
            # Append relevant values
            self['XMRP'] = np.hstack((self['XMRP'], [self.MRP[0]]))
            self['YMRP'] = np.hstack((self['YMRP'], [self.MRP[1]]))
            self['ZMRP'] = np.hstack((self['ZMRP'], [self.MRP[2]]))
            self['nIter']  = np.hstack((self['nIter'],  [nIter]))
            self['nStats'] = np.hstack((self['nStats'], [nStats]))
        else:
            # Update the relevant values
            self['XMRP'][j] = self.MRP[0]
            self['YMRP'][j] = self.MRP[1]
            self['ZMRP'][j] = self.MRP[2]
            self['nIter'][j] = nIter
            self['nStats'][j] = nStats
        # Return to original directory
        os.chdir(fpwd)
        # Output
        return 1
    
    # Get file
    def GetTriqFile(self):
        """Get most recent ``triq`` file and its associated iterations
        
        :Call:
            >>> qtriq, ftriq, n, i0, i1 = DBL.GetTriqFile()
        :Inputs:
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
                Instance of line load data book
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
        ftriq, n, i0, i1 = case.GetTriqFile()
        # Output
        return False, ftriq, n, i0, i1
        
    # Write triload.i input file
    def WriteTriloadInput(self, ftriq, i, **kw):
        """Write ``triload.i`` input file to ``triloadCmd``
        
        :Call:
            >>> DBL.WriteTriloadInput(ftriq, i, **kw)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of the ``triq`` file to analyze
            *i*: :class:`int`
                Case number
        :Keyword arguments:
            *mach*: :class:`float`
                Override Mach number
            *Re*: :class:`float`
                Override Reynolds number input
            *gamma*: :class:`float`
                Override ratio of specific heats
            *MRP*: :class:`float`
                Override the moment reference point from the JSON input file
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
            * 2017-01-11 ``@ddalle``: Moved code to WriteTriloadInputBase
        """
        self.WriteTriloadInputBase(ftriq, i, **kw)
        
    # Write triload.i input file
    def WriteTriloadInputBase(self, ftriq, i, **kw):
        """Write ``triload.i`` input file to ``triloadCmd``
        
        :Call:
            >>> DBL.WriteTriloadInput(ftriq, i, **kw)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of the ``triq`` file to analyze
            *i*: :class:`int`
                Case number
        :Keyword arguments:
            *mach*: :class:`float`
                Override Mach number
            *Re*: :class:`float`
                Override Reynolds number input
            *gamma*: :class:`float`
                Override ratio of specific heats
            *MRP*: :class:`float`
                Override the moment reference point from the JSON input file
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Setting for output triq file
        trimOut = self.opts.get_DataBookTrim(self.comp)
        # Momentum setting
        qm = self.opts.get_DataBookMomentum(self.comp)
        # Number of cuts
        nCut = self.opts.get_DataBook_nCut(self.comp)
        self.nCut = nCut
        # Get components and type of the input
        compID = self.CompID
        COMPID = self.opts.get_DataBookCompID(self.comp)
        tcomp  = type(compID).__name__
        # File name
        fcmd = 'triload.%s.i' % self.comp
        # Open the file anew
        f = open(fcmd, 'w')
        # Write the triq file name
        f.write(ftriq + '\n')
        # Write the prefix na me
        f.write(self.proj + '\n')
        # Get Mach number, Reynolds number, and ratio of specific heats
        mach = kw.get('mach',  self.x.GetMach(i))
        Re   = kw.get('Re',    self.x.GetReynoldsNumber(i))
        gam  = kw.get('gamma', self.x.GetGamma(i))
        # Check for NaNs
        if mach is None: mach = 1.0
        if Re   is None: Re   = 1.0
        if gam  is None: gam  = 1.4
        # Let's save these parameters
        self.mach = mach
        self.Re   = Re
        self.gam  = gam
        # Moment reference point
        MRP = kw.get('MRP', self.MRP)
        # Write the Mach number, reference Reynolds number, and ratio of heats
        f.write('%s %s %s\n' % (mach, Re, gam))
        # Moment center
        f.write('%s %s %s\n' % (self.MRP[0], self.MRP[1], self.MRP[2]))
        # Setting for gauge pressure and non-dimensional output
        f.write('0 0\n')
        # Reference length and area
        f.write('%s %s\n' % (self.RefL, self.RefA))
        # Whether or not to include momentum
        if qm:
            # Include momentum
            f.write('y\n')
        else:
            # Do not include momentum
            f.write('n\n')
        # Group name
        f.write(self.comp + ' ')
        # Write components if any (otherwise, triLoad will use all tris)
        if type(compID).__name__ in ['list', 'ndarray']:
            # Write list of component IDs as a convenient range string
            # i.e. "3-10,12-15,17,19,21-24"
            f.write(RangeString(compID))
            f.write('\n')
        # Number of cuts
        if trimOut:
            # Only write tris included in at least one component
            f.write('%s 1\n' % nCut)
        else:
            # Write all tris trimmed
            f.write('%s 0\n' % nCut)
        # Write min and max; use '1 1' to make it automatically detect
        f.write('1 1\n')
        # Write the cut type
        f.write('const x\n')
        # Write coordinate transform
        self.WriteTriloadTransformations(i, f)
        # Close the input file
        f.close()
        
    # Get triload transformations
    def WriteTriloadTransformations(self, i, f):
        r"""Write transformations to a ``triload.i`` input file
        
        Usually this just writes an ``n`` for "no", but it can also
        write a 3x3 transformation matrix if ``"Transformations"`` are
        defined for *DBL.comp*.
        
        :Call:
            >>> DBL.WriteTriloadTransformations(i, f)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number
            *f*: :class:`file`
                Open file handle from :func:`WriteTriloadInputBase`
        :Versions:
            * 2017-04-14 ``@ddalle``: First version
        """
        # Get the raw option from the data book
        db_transforms = self.opts.get_DataBookTransformations(self.comp)
        # Initialize transformations
        R = None
        # Loop through transformations
        for topts in db_transforms:
            # Check for rotation matrix
            Ri = self.CalculateTriloadTransformation(i, topts)
            # Multiply
            if Ri is not None:
                if R is None:
                    # First transformation
                    R = Ri
                else:
                    # Compound
                    R = np.dot(R, Ri)
        # Check if no transformations
        if R is None:
            f.write('n\n')
            return
        # Yes, we are doing transformations
        f.write('y\n')
        # Write the transformation
        for row in R:
            f.write("%9.6f %9.6f %9.6f\n" % tuple(row))
    
    # Calculate transformations
    def CalculateTriloadTransformation(self, i, topts):
        """Write transformations to a ``triload.i`` input file
        
        Usually this just writes an ``n`` for "no", but it can also write a 3x3
        transformation matrix if ``"Transformations"`` are defined for
        *DBL.comp*.
        
        :Call:
            >>> R = DBL.CalculateTriloadTransformation(i, topts)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number
            *topts*: :class:`dict`
                Dictionary of transformation options
        :Outputs:
            *R*: :class:`np.ndarray` shape=(3,3)
                Rotation matrix
        :Versions:
            * 2017-04-14 ``@ddalle``: First version
        """
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Check it.
        if ttype in ["Euler321", "Euler123"]:
            # Get the angle variable names.
            # Use same as default in case it's obvious what they should be.
            kph = topts.get('phi', 0.0)
            kth = topts.get('theta', 0.0)
            kps = topts.get('psi', 0.0)
            # Extract roll
            if type(kph).__name__ not in ['str', 'unicode']:
                # Fixed value
                phi = kph*deg
            elif kph.startswith('-'):
                # Negative roll angle.
                phi = -self.x[kph[1:]][i]*deg
            else:
                # Positive roll
                phi = self.x[kph][i]*deg
            # Extract pitch
            if type(kth).__name__ not in ['str', 'unicode']:
                # Fixed value
                theta = kth*deg
            elif kth.startswith('-'):
                # Negative pitch
                theta = -self.x[kth[1:]][i]*deg
            else:
                # Positive pitch
                theta = self.x[kth][i]*deg
            # Extract yaw
            if type(kps).__name__ not in ['str', 'unicode']:
                # Fixed value
                psi = kps*deg
            elif kps.startswith('-'):
                # Negative yaw
                psi = -self.x[kps[1:]][i]*deg
            else:
                # Positive pitch
                psi = self.x[kps][i]*deg
            # Sines and cosines
            cph = np.cos(phi); cth = np.cos(theta); cps = np.cos(psi)
            sph = np.sin(phi); sth = np.sin(theta); sps = np.sin(psi)
            # Make the matrices.
            # Roll matrix
            R1 = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
            # Pitch matrix
            R2 = np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]])
            # Yaw matrix
            R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
            # Combined transformation matrix.
            # Remember, these are applied backwards in order to undo the
            # original Euler transformation that got the component here.
            if ttype == "Euler321":
                return np.dot(R1, np.dot(R2, R3))
            elif ttype == "Euler123":
                return np.dot(R3, np.dot(R2, R1))
        elif ttype in ["ScaleCoeffs"]:
            # Handled elsewhere
            return
        else:
            raise ValueError(
                "Transformation type '%s' is not recognized." % ttype)

    # Run triload
    def RunTriload(self, qtriq=False, ftriq=None, qpbs=False, i=None):
        r"""Run ``triload`` for a case
        
        :Call:
            >>> DBL.RunTriload(**kw)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *qtriq*: ``True`` | {``False``}
                Whether or not preprocessing is needed to create TRIQ file
            *ftriq*: {``None``} | :class:`str`
                Name of TRIQ file (if needed)
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
            * 2016-12-21 ``@ddalle``: PBS added
        """
        # Convert
        if qtriq:
            self.PreprocessTriq(ftriq, qpbs=qpbs, i=i)
        # Triload command
        cmd = 'triloadCmd < triload.%s.i > triload.%s.o'%(self.comp,self.comp)
        # Status update
        print("    %s" % cmd)
        # Run triload
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            return SystemError("Failure while running ``triloadCmd``")
    
    # Convert
    def PreprocessTriq(self, ftriq, **kw):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBL.PreprocessTriq(ftriq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *ftriq*: :class:`str`
                Name of triq file
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
            *i*: {``None``} | :class:`int`
                Case index
        :Versions:
            * 2016-12-19 ``@ddalle``: First version
            * 2016-12-21 ``@ddalle``: Added PBS
        """
        pass
  # >
    
  # ========
  # Database
  # ========
  # <
    # Get a proper orthogonal decomposition
    def GetCoeffPOD(self, coeff, n=2, f=None, **kw):
        """Create a Proper Orthogonal Decomposition of lineloads for one coeff
        
        :Call:
            >>> u, s = DBL.GetPOD(coeff, n=None, f=None, **kw)
        :Inputs:
            *DBL*: :class:`cape.cfdx.lineLoad.DBLineLoad`
                Line load data book
            *n*: {``2``} | positive :class:`int`
                Number of modes to keep
            *f*: {``None``} | 0 < :class:`float` <= 1
                Keep enough modes so that this fraction of energy is kept
            *cons*: {``None``} | :class:`list` (:class:`str`)
                Constraints for which cases to include in basis
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List of cases to include in basis
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Get list of cases to include
        I = self.x.GetIndices(**kw)
        # Make sure all cases are present
        for i in I:
            self.ReadCase(i)
        # Initialize basis
        C = np.zeros((self.nCut+1,len(I)))
        j = 0
        # Loop through cases
        for i in I:
            # Check if this case is present in the database
            if i not in self: continue
            # Append to the database
            C[:,j] = getattr(self[i], coeff)
            # Increase the count
            j += 1
        # Downsize *C* as appropriate
        C = C[:,:j]
        # Check for null database
        if j == 0:
            return np.zeros((self.nCut,0)), np.zeros(0)
        # Calculate singular value decomposition
        u, s, v = np.linalg.svd(C, full_matrices=False)
        # Normalize singular values
        s = np.cumsum(s) / np.sum(s)
        # Figure out how many components needed to meet *f*
        if f is None:
            # Default: keep nothing for *f*
            nf = 0
        elif f > 1:
            # Bad value
            raise ValueError(("Received POD fraction of %s; " % f) +
                "cannot keep more than 100% of modes")
        else:
            # Calculate the fraction using cumulative sum of singular values
            nf = np.where(s >= f)[0][0] + 1
        # Keep the maximum of *n* and *nf* modes
        ns = np.max(n, nf)
        # Output
        return u[:,:ns], s[:ns]
        
  # >

# class DBLineLoad
    
# Line load from one case
class CaseLL(object):
    """Interface to individual sectional load
    
    :Call:
        >>> LL = CaseLL(comp, proj='LineLoad', sec='slds')
    :Inputs:
        *comp*: :class:`str`
            Name of component
        *proj*: :class:`str`
            Prefix for sectional load output files
        *sec*: ``"clds"`` | {``"dlds"``} | ``"slds"``
            Cut type, cumulative, derivative, or sectional
        *ext*: ``"clds"`` | {``"dlds"``} | ``"slds"`` | ``"csv"``
            File extension 
        *fdir* {``None``} | :class:`str`
            Name of sub folder to use
    :Outputs:
        *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
            Individual line load for one component from one case
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
        * 2016-06-07 ``@ddalle``: Second version, universal
    """
  # =============
  # Configuration
  # =============
  # <
    # Initialization method
    def __init__(self, comp, proj='LineLoad', sec='dlds', **kw):
        """Initialization method
        
        :Versions:
            * 2016-06-07 ``@ddalle``: First universal version
        """
        # Save input options
        self.comp = comp
        self.proj = proj
        self.sec  = sec
        # Default extension
        if self.sec == 'slds':
            # Sectional
            ext0 = 'slds'
        elif self.sec == 'clds':
            # Cumulative
            ext0 = 'clds'
        else:
            # Derivative
            ext0 = 'dlds'
        # Keyword inputs
        self.ext  = kw.get('ext', ext0)
        self.fdir = kw.get('fdir', None)
        self.seam = kw.get('seam', True)
        # File name
        self.fname = '%s_%s.%s' % (proj, comp, self.ext)
        if self.fdir is not None:
            # Prepend folder name
            self.fname = os.path.join(self.fdir, self.fname)
        # Read the file
        try:
            # Check if we are reading triload output file or data book file
            if self.ext.lower() == "csv":
                # Read csv file
                self.ReadCSV(self.fname)
            else:
                # Read triload output file
                self.ReadLDS(self.fname)
        except Exception:
            # Create empty line loads
            self.x   = np.zeros(0)
            self.CA  = np.zeros(0)
            self.CY  = np.zeros(0)
            self.CN  = np.zeros(0)
            self.CLL = np.zeros(0)
            self.CLM = np.zeros(0)
            self.CLN = np.zeros(0)
        # Read the seams
        if self.seam:
            self.ReadSeamCurves()
    
    # Function to display contents
    def __repr__(self):
        """Representation method
        
        Returns the following format:
        
            * ``<CaseLL comp='CORE' (csv)>``
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        return "<CaseLL comp='%s' (%s)>" % (self.comp, self.ext)
        
    # Copy
    def Copy(self):
        """Create a copy of a case line load object
        
        :Call:
            >>> LL2 = LL.Copy()
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Single-case, single-component line load interface
        :Outputs:
            *LL2*: :class:`cape.cfdx.lineLoad.CaseLL`
                Copy of the line load interface
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Initialize (empty) object
        LL = CaseLL(self.comp, proj=self.proj, sec=self.sec,
            ext=self.ext, fdir=self.fdir, seam=self.seam)
        # Copy the data
        LL.x   = self.x.copy()
        LL.CA  = self.CA.copy()
        LL.CY  = self.CY.copy()
        LL.CN  = self.CN.copy()
        LL.CLL = self.CLL.copy()
        LL.CLM = self.CLM.copy()
        LL.CLN = self.CLN.copy()
        # Output
        return LL
    
  # >
    
  # ====
  # I/O
  # ====
  # <
    # Function to read a file
    def ReadLDS(self, fname=None):
        """Read a sectional loads ``*.?lds`` file from `triloadCmd`
        
        :Call:
            >>> LL.ReadLDS(fname)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Single-case, single component, line load interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Default file name
        if fname is None: fname = self.fname
        # Open the file
        with open(fname, 'r') as fp:
            # Read lines until it is not a comment.
            line = '#'
            while (line.lstrip().startswith('#')) and (len(line)>0):
                # Read the next line.
                line = fp.readline()
            # Exit if empty.
            if len(line) == 0:
                raise ValueError("Empty triload file '%s'" % fname)
            # Number of columns
            nCol = len(line.split())
            # Go backwards one line from current position.
            fp.seek(fp.tell() - len(line))
            # Read the rest of the file.
            D = np.fromfile(fp, count=-1, sep=' ')
            # Reshape to a matrix
            D = D.reshape((D.size//nCol, nCol))
            # Save the keys.
            self.x = D[:,0]
            self.CA = D[:,1]
            self.CY = D[:,2]
            self.CN = D[:,3]
            self.CLL = D[:,4]
            self.CLM = D[:,5]
            self.CLN = D[:,6]

    # Function to read a databook file
    def ReadCSV(self, fname=None, delim=','):
        """Read a sectional loads ``csv`` file from the data book
        
        :Call:
            >>> LL.ReadCSV(fname, delim=',')
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Single-case, single component, line load interface
            *fname*: :class:`str`
                Name of file to read
            *delim*: {``','``} | ``' '`` | :class:`str`
                Text delimiter
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Default file name
        if fname is None:
            # Replace base extension with csv
            fname = self.fname.rstrip(self.ext) + 'csv'
        # Read the rest of the file.
        D = np.loadtxt(fname, delimiter=delim)
        # Ensure array
        if D.ndim == 0:
            D = np.array([D])
        # Save the keys.
        self.x = D[:,0]
        self.CA = D[:,1]
        self.CY = D[:,2]
        self.CN = D[:,3]
        self.CLL = D[:,4]
        self.CLM = D[:,5]
        self.CLN = D[:,6]
        
    # Write CSV file
    def WriteCSV(self, fname=None, delim=','):
        """Write a sectional loads ``csv`` file
        
        :Call:
            >>> LL.WriteCSV(fname, delim=',')
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Single-case, single component, line load interface
            *fname*: :class:`str`
                Name of file to write
            *delim*: {``','``} | ``' '`` | :class:`str`
                Text delimiter
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Default file name
        if fname is None:
            # Replace base extension with csv
            fname = self.fname.rstrip(self.ext) + 'csv'
        # Open the file to write
        f = open(fname, 'w')
        # Write the header line
        f.write('# ')
        f.write(delim.join(['x', 'CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']))
        f.write('\n')
        # Generate write flag
        ffmt = delim.join(['%13.6E'] * 7) + '\n'
        # Loop through the values
        for i in range(len(self.x)):
            # Write data
            f.write(ffmt % (self.x[i], self.CA[i], self.CY[i], self.CN[i],
                self.CLL[i], self.CLM[i], self.CLN[i]))
        # Close the file
        f.close()
        
    # Read the seam curves
    def ReadSeamCurves(self):
        """Read seam curves from a data book directory
        
        :Call:
            >>> LL.ReadSeamCurves()
        :Inputs:
            *LL*: :class:`pyCart.lineLoad.CaseLL`
                Instance of data book line load interface
        :Versions:
            * 2015-09-17 ``@ddalle``: First version
        """
        # Seam file names
        if self.fdir is None:
            # Folder
            fpre = '%s_%s' % (self.proj, self.comp)
        else:
            # Include subfolder
            fpre = os.path.join(self.fdir, '%s_%s' % (self.proj, self.comp))
        # Name of output files.
        fsmx = '%s.smx' % fpre
        fsmy = '%s.smy' % fpre
        fsmz = '%s.smz' % fpre
        # Read the seam curves.
        self.smx = CaseSeam(fsmx)
        self.smy = CaseSeam(fsmy)
        self.smz = CaseSeam(fsmz)
  # >
    
  # =========
  # Plotting
  # =========
  # <
    # Plot a line load
    def Plot(self, coeff, **kw):
        """Plot a single line load
        
        :Call:
            >>> LL.Plot(coeff, **kw)
        :Inputs:
            *LL*: :class:`pyCart.lineLoad.CaseLL`
                Instance of data book line load interface
            *coeff*: :class:`str`
                Name of coefficient to plot
            *x*: {``"x"``} | ``"y"`` | ``"z"``
                Axis to use for independent axis
            *Seams*: {``[]``} | :class:`list` (:class:`str` | :class:`CaseSeam`)
                List of seams to plot
            *SeamLocation*: {``"bottom"``} | ``"left"`` | ``"right"`` | ``"top"``
                Location on which to plot seams
            *Orientation*: {``"vertical"``} | ``"horizontal"``
                If not 'vertical', flip *x* and *y* axes
            *LineOptions*: {``{}``} | :class:`dict`
                Dictionary of plot options
            *SeamOptions*: {``{}``} | :class:`dict`
                Dictionary of plot options
            *Label*: {*LL.comp*} | :class:`str`
                Plot label, ``LineOptions['label']`` supersedes this variable
            *XLabel*: {``"x/Lref"``} | :class:`str`
                Label for x-axis
            *YLabel*: {*coeff*} | :class:`str`
                Label for y-axis
            *Legend*: [ {True} | False ]
                Whether or not to use a legend
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *SubplotMargin*: {``0.015``} | :class:`float`
                Margin between subplots
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
       # -------
       # Options
       # -------
        # Ensure plot modules are necessary
        ImportPyPlot()
        # Get x-axis values
        kx = kw.get('x', 'x')
        # Get values
        x = getattr(self, kx)
        y = getattr(self, coeff)
        # Axis flip setting
        q_vert = kw.get('Orientation', 'vertical') == 'vertical'
        # Other plot options
        fw = kw.get('FigWidth')
        fh = kw.get('FigHeight')
        # Process seams
        sms = kw.get('Seams', [])
        # Default values -> None
        if sms in ['', None, False]:
            sms = []
        # Seam location
        sm_loc = kw.get('SeamLocation')
        # Check for single seam
        if type(sms).__name__ not in ['list', 'ndarray']: sms = [sms]
        # Number of seams
        nsm = len(sms)
        # Ensure seam location is also a list
        if type(sm_loc).__name__ not in ['list', 'ndarray']:
            sm_loc = [sm_loc] * nsm
        # Convert seams and seam locations to proper formats
        for i in range(nsm):
            # Get the location
            loc_i = sm_loc[i]
            # Check default
            if loc_i is None and q_vert:
                # Default is to plot seams below
                sm_loc[i] = 'bottom'
            elif loc_i is None:
                # Default is to plot on the left
                sm_loc[i] = 'left'
            # Get the seam
            sm = sms[i]
            # Seam type
            tsm = type(sm).__name__
            # Check for labeled axes
            if tsm not in ['str', 'unicode']:
                # Already a seam
                continue
            elif sm.lower() in ['x', 'smx']:
                # Get the seam handle
                sms[i] = self.smx
            elif sm.lower() in ['y', 'smy']:
                # Get the y-cut seam handle
                sms[i] = self.smy
            else:
                # Get the z-cut seam handle
                sms[i] = self.smz
       # ------------------
       # Initialize Figures
       # ------------------
        # Initialize handles
        h = {}
        # Check for seam plots
        if nsm > 0:
            # Call subplot command first to avoid deleting plots
            if q_vert:
                # Number of seams above
                sfigll = 1 + sm_loc.count('top')
                # Plot seams above and below
                plt.subplot(nsm+1, 1, sfigll)
            else:
                # Number of seams to the left
                sfigll = 1 + sm_loc.count('left')
                # Plot seams to the left or right
                plt.subplot(1, nsm+1, sfigll)
       # ------------
       # Primary plot
       # ------------
        # Initialize primary plot options
        kw_p = odict(color=kw.get("color","k"), ls="-", lw=1.5, zorder=7)
        # Extract plot optiosn from kwargs
        for k in util.denone(kw.get("LineOptions", {})):
            # Override the default option
            if kw["LineOptions"][k] is not None:
                kw_p[k] = kw["LineOptions"][k]
        # Apply label
        kw_p.setdefault('label', kw.get('Label', self.comp))
        # Plot
        if q_vert:
            # Regular orientation
            h[coeff] = plt.plot(x, y, **kw_p)
        else:
            # Flip axes
            h[coeff] = plt.plot(y, x, **kw_p)
       # -----------------
       # Margin adjustment
       # -----------------
        # Get the figure and axes handles
        h['fig'] = plt.gcf()
        h['ax']  = plt.gca()
        # Check for existing label
        if q_vert:
            ly = h['ax'].get_ylabel()
        else:
            ly = h['ax'].get_xlabel()
        # Default labels
        if self.sec == 'slds':
            # Sectional loads
            ly0 = coeff
            lx0 = '%s/Lref' % kx
        elif self.sec == 'clds':
            # Cumulative loads
            ly0 = coeff
            lx0 = '%s/Lref' % kx
        else:
            # Derivative label
            ly0 = 'd%s/d(%s/Lref)' % (coeff, kx)
            lx0 = '%s/Lref' % kx
        # Compare to the requested ylabel
        if not ly: ly = ly0
        # Check orientation
        if q_vert:
            # Get label inputs
            xlbl = kw.get('XLabel', lx0)
            ylbl = kw.get('YLabel', ly)
        else:
            # Get label inputs with flipped defaults
            xlbl = kw.get('XLabel', ly)
            ylbl = kw.get('YLabel', kx0)
        # Label handles
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Get actual limits
        xmin, xmax = util.get_xlim(h['ax'], **kw)
        ymin, ymax = util.get_ylim(h['ax'], **kw)
        # Set the axis limits
        h['ax'].set_xlim((xmin, xmax))
        h['ax'].set_ylim((ymin, ymax))
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Margins
        adj_l = kw.get('AdjustLeft')
        adj_r = kw.get('AdjustRight')
        adj_t = kw.get('AdjustTop')
        adj_b = kw.get('AdjustBottom')
        # Subplot margin
        w_sfig = kw.get('SubplotMargin', 0.015)
        # Make adjustments
        if adj_l: plt.subplots_adjust(left=adj_l)
        if adj_r: plt.subplots_adjust(right=adj_r)
        if adj_t: plt.subplots_adjust(top=adj_t)
        if adj_b: plt.subplots_adjust(bottom=adj_b)
        # Report the actual limits
        h['xmin'] = xmin
        h['xmax'] = xmax
        h['ymin'] = ymin
        h['ymax'] = ymax
       # ------
       # Legend
       # ------
        # Check for legend setting
        if kw.get('Legend', False):
            # Get current limits.
            ymin, ymax = util.get_ylim(h['ax'], pad=0.05)
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
       # ----------
       # Seam plots
       # ----------
        # Exit if no seams
        if nsm < 1: return h
        # Initialize seam handles
        H = [None for i in range(nsm+1)]
        # Save main plot axis limits
        xlim = h['ax'].get_xlim()
        ylim = h['ax'].get_ylim()
        # Save the main plot handles
        H[sfigll-1] = h
        # Initialize aspect ratios
        AR = np.zeros(nsm+1)
        # Relevant position info from primary plot
        pax = h['ax'].get_position().get_points()
        # Save all the positions; some will be overwritten
        xax_min, yax_min = pax[0]
        xax_max, yax_max = pax[1]
        # Seam plot options
        kw_S = kw.copy()
        # Transfer options
        kw_S['LineOptions'] = kw.get('SeamOptions')
        # Default to last line options
        if kw_S['LineOptions'] in [{}, None]:
            kw_S['LineOptions'] = kw_p
        # Dictionary of subplot indices for each seam plot
        sfigs = {}
        # Loop through seams
        for i in range(nsm):
            # Get subfig number
            if q_vert:
                # Check numbers of top/bottom seam plots
                if sm_loc[i] == 'top':
                    # Count previous top figures
                    sfigi = 1 + sm_loc[:i].count('top')
                else:
                    # Count previous seam figures and all other figs above
                    sfigi = i + 2 + sm_loc[i:].count('top')
                # Select the plot
                plt.subplot(nsm+1, 1, sfigi)
            else:
                # Check numbers of left/right seam plots
                if sm_loc[i] == 'left':
                    # Count previous left figures
                    sfigi = 1 + sm_loc[:i].count('left')
                else:
                    # Count previous seam figures and all other left figs
                    sfigi = i + 2 + sm_loc[i:].count('left')
                # Select the plot
                plt.subplot(1, nsm+1, sfigi)
            # Save subfigs
            sfigs[i] = sfigi
            # Plot the seam
            hi = sms[i].Plot(**kw_S)
            # Save the handles
            H[sfigi-1] = hi
            # Copy axes handle
            axi = hi['ax']
            # Save aspect ratio
            AR[sfigi-1] = (hi['ymax'] - hi['ymin']) / (hi['xmax'] - hi['xmin'])
            # Plot offsets
            dxi = kw.get('XSeamOffset', 0.0)
            dyi = kw.get('YSeamOffset', 0.0)
            # Middle coordinates
            xi = 0.5*(hi['xmin'] + hi['xmax']) + dxi
            yi = 0.5*(hi['ymin'] + hi['ymax']) + dyi
            # Scaled axis widths
            wxi = (ylim[1] - ylim[0]) / AR[sfigi-1]
            wyi = (xlim[1] - xlim[0]) * AR[sfigi-1]
            # Scaled axis limits after copying from main plot
            xlimi = [xi - 0.5*wxi, xi + 0.5*wxi]
            ylimi = [yi - 0.5*wyi, yi + 0.5*wyi]
            # Get axes position
            pi = axi.get_position().get_points()
            # Follow up for each type
            if q_vert:
                # Copy xlims from line load plot
                axi.set_xlim(xlim)
                axi.set_ylim(ylimi)
                plt.draw()
                # Check for top/bottom plot for absolute limits
                if sfigi == nsm+1:
                    # Bottom figure
                    yax_min = pi[0,1]
                elif sfigi == 1:
                    # Top figure
                    yax_max = pi[1,1]
            else:
                # Copy ylims from line load plot
                axi.set_ylim(ylim)
                axi.set_xlim(xlimi)
                # Cehck for left/right plot for absolute limits
                if sfigi == 1:
                    # Left figure
                    xax_min = pi[0,0]
                elif sfigi == nsm+1:
                    # Right figure
                    xax_max = pi[1,0]
        # Nominal width/height of axes position
        wax = xax_max - xax_min
        hax = yax_max - yax_min
        # Loop through subplots to set aspect ratios, etc.
        for i in range(nsm+1):
            # Get axis handle
            ax = H[i]['ax']
            # Only use ticks on first subplot
            if q_vert and i < nsm:
                # Turn off xticks
                ax.set_xticklabels([])
                # Turn off x label
                ax.set_xlabel('')
            elif not q_vert and i > 0:
                # Turn off yticks
                ax.set_yticklabels([])
                # Turn off y label
                ax.set_ylabel('')
            # Handle main plot last
            if i+1 == sfigll:
                continue
            # Get seam number
            sfigi = sfigs[i - (i>=sfigll)]
            # Get current axis limits
            xlimj = axi.get_xlim()
            ylimj = axi.get_ylim()
            # Aspect ratio of the existing figure
            ARi = (ylimj[1]-ylimj[0]) / (xlimj[1]-xlimj[0])
            # Set margins
            if q_vert:
                # Automatic axis height based on aspect ratio
                haxi = AR[i] * wax
                # Select subplot
                plt.subplot(1+nsm, 1, sfigi)
                # Modify top/bottom margins
                if i+1 < sfigll:
                    # Work from the top
                    ax.set_position([xax_min, yax_max-haxi, wax, haxi])
                    # Update top position
                    yax_max = yax_max - haxi - w_sfig
                else:
                    # Work from the bottom
                    ax.set_position([xax_min, yax_min, wax, haxi])
                    # Update the bottom position
                    yax_min = yax_min + haxi + w_sfig
                # Target *x* limits
                wxi = xlim[1] - xlim[0]
                # Set the *y* limits appropriately
                ylimi = [yi-0.5*ARi*wxi, yi+0.5*ARi*wxi]
                # Copy the limits again
                ax.set_xlim(xlim)
                ax.set_ylim(ylimi)
                plt.draw()
                #plt.axis([xlim[0], xlim[1], ylimi[0], ylimi[1]])
                # Minimal ticks on y-axis
                try: plt.locator_params(axis='y', nbins=4)
                except Exception: pass
            else:
                # Automatic axis width based on aspect ratio
                waxi = hax / AR[i]
                # Select subplot
                plt.subplot(1+nsm, sfigi, 1)
                # Modify left/right margins
                if i > sfigll:
                    # Work from the right
                    ax.set_position([xax_max-waxi, yax_min, waxi, hax])
                    # Update right position
                    xax_max = xax_max - waxi - w_sfig
                else:
                    # Work from the left
                    ax.set_position([wax_min, yax_min, waxi, hax])
                    # Update left position
                    xax_min = xax_min + waxi + w_sfig
                # Target *y* limits
                wyi = ylim[1] - ylim[0]
                # Set the *x* limits appropriately
                xlimi = [xi-0.5*wyi/ARi, xi+0.5*wyi/ARi]
                # Reset axis limits
                axi.set_ylim(ylim)
                axi.set_xlim(xlimi)
                # Minimal ticks on y-axis
                try: plt.locator_params(axis='x', nbins=3)
                except Exception: pass
        # Make sure to give handle back to primary plot
        if q_vert:
            # Seams are above and below
            plt.subplot(nsm+1, 1, sfigll)
        else:
            # Plot seams to the left or right
            plt.subplot(1, nsm+1, sfigll)
        # Finally, set the position for the position for the main figure
        h['ax'].set_position([
                xax_min, yax_min, xax_max-xax_min, yax_max-yax_min
        ])
        # Reset limits
        h['ax'].set_xlim(xlim)
        h['ax'].set_ylim(ylim)
        # Modify output
        h['sm'] = H[:sfigll-1] + H[sfigll:]
        # Output
        return h
        
    # Plot a seam
    def PlotSeam(self, s='z', **kw):
        """Plot a set of seam curves
        
        :Call:
            >>> h = LL.PlotSeam(s='z', **kw)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of data book line load interface
            *s*: ``"x"`` | ``"y"`` | {``"z"``}
                Type of slice to plot
            *x*: {``"x"``} | ``"y"`` | ``"z"``
                Axis to plot on x-axis
            *y*: ``"x"`` | {``"y"``} | ``"z"``
                Axis to plot on y-axis
            *LineOptions*: {``{}``} | :class:`dict`
                Dictionary of plot options
            *Label*: {*LL.comp*} | :class:`str`
                Plot label, ``LineOptions['label']`` supersedes this variable
            *XLabel*: {``"x/Lref"``} | :class:`str`
                Label for x-axis
            *XLabel*: {*coeff*} | :class:`str`
                Label for y-axis
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        # Get name of plot
        ksm = 'sm' + s
        # Get handle
        sm = getattr(self, ksm)
        # Plot
        h = sm.Plot(**kw)
        # Output
        return h
  # >
    
  # ===========
  # Corrections
  # ===========
  # <
    # Correct line loads using linear basis functions
    def CorrectLinear(self, CN, CLM, CY, CLN, xMRP=0.0):
        """Correct line loads to match target integrated values using lines
        
        :Call:
            >>> LL2 = LL.CorrectLinear(CN, CLM, xMRP=0.0)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CN*: :class:`float`
                Target integrated value of *CN*
            *CLM*: :class:`float`
                Target integrated value of *CLM*
            *CY*: :class:`float`
                Target integrated value of *CY*
            *CLN*: :class:`float`
                Target integrated value of *CLN*
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of MRP divided by reference length
        :Outputs:
            *LL2*: :class:`cape.cfdx.lineLoad.CaseLL`
                Line loads with integrated loads matching *CN* and *CLM*
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Create basis functions
        CN1 = np.ones_like(self.x)
        CN2 = np.linspace(0, 1, len(self.x))
        # Create a copy
        LL = self.Copy()
        # Correct *CN* and *CLM*
        LL.CorrectCN2(CN, CLM, CN1, CN2, xMRP=xMRP)
        LL.CorrectCY2(CY, CLN, CN1, CN2, xMRP=xMRP)
        # Output
        return LL
        
    # Correct *CN* and *CLM* using *n* functions
    def CorrectCN(self, CN, CLM, UCN, sig=None, xMRP=0.0):
        """Correct *CN* and *CLM* given *n* unnormalized functions
        
        This function takes an *m* by *n* matrix where *m* is the size of
        *LL.CN*. It then calculates an increment to *LL.CN* that is a linear
        combination of the columns of that matrix *UCN* such that the
        integrated normal force coefficient (*CN*) and pitching moment
        coefficient (*CLM*) match target values provided by the user.  The
        increment is
        
            .. math::
                
                \Delta C_N = \sum_{i=1}^n k_i\\phi_i
                
        where :math:`\\phi_i`` is the *i*th column of *UCN* scaled so that it
        has an L2 norm of 1.
        
        The weights of the linear coefficient are chosen in order to minimize
        the sum of an objective function subject to the integration constraints
        mentioned above.  This objective function is
        
            .. math::
            
                \\sum_{i=1}^n a_i k_i^2 / \\sigma_i
                
        where :math:`a_i` is the maximum absolute value of column *i* of *UCN*
        and :math:`\\sigma_i` is the associated singular value.
        
        :Call:
            >>> LL.CorrectCN(CN, CLM, UCN, sig=None, xMRP=0.0)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CN*: :class:`float`
                Target integrated value of *CN*
            *CLM*: :class:`float`
                Target integrated value of *CLM*
            *UCN*: :class:`np.ndarray` (*LL.x.size*,*n*)
                Matrix of *CN* adjustment basis functions
            *sig*: {``None``} | :class:`np.ndarray` (*n*,)
                Array of singular values
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of MRP divided by reference length
        :Versions:
            * 2017-02-02 ``@ddalle``: First version
        """
        # Get the current loads
        CN0  = np.trapz(self.CN,  self.x)
        CLM0 = np.trapz(self.CLM, self.x)
        # Dimensionality of line load vector
        nx = len(self.x)
        # Check dimension count of the dispersions matrix
        if UCN.ndims != 2:
            raise ValueError(
                "Adjustment basis function *UCN* must be 2D array")
        # Dimensions of the dispersion matrix
        m, n = UCN.shape
        # Check dims
        if m != nx:
            raise ValueError(
                ("Adjustment basis functions have %i entries " % m) +
                ("but line load has %i" % nx))
        # L2-norm of each basis vector
        L = np.sqrt(np.sum(UCN**2, axis=0))
        # Initialize normalized basis functions
        VCN = nnp.zeros(m, n)
        # Loop through basis vectors
        for i in range(n):
            # Normalize
            VCN[:,i] = UCN[:,i]/L[i]
        # Max values
        mxCN = np.max(np.abs(VCN), axis=0)
        # Default singular values
        if sig is None:
            sig = np.ones(n)
        # Weights
        w = mxCN / sig
        # Calculate the increment from each mode
        dCN = np.zeros(n)
        dCLM = np.zeros(n)
        # Loop through modes
        for i in range(n):
            dCN[i]  = np.trapz(UCN[:,i], self.x)
            dCLM[i] = np.trapz(UCN[:,i]*(xMRP-self.x), self.x)
        # Form matrix for linear system
        dC = np.array([dCN, dCLM])
        # First two equations: equality constraints on *CN* and *CLM*
        A1 = np.hstack((dC, np.zeros((2,2))))
        # Last *n* equations: derivatives of the Lagrangian
        A2 = np.hstack((np.diag(2*w), -np.transpose(dC)))
        # Assemble matrices
        A = np.vstack((A1, A2))
        # Right-hand sides of equations
        b = np.hstack(([CN-CN0, CLM-CLM0], np.zeros(n)))
        # Solve the linear system (and optimization problem)
        x = np.linalg.solve(A, b)
        # Calculate linear combination (discard Lagrange multipliers)
        phi = np.dot(UCN, x[:n])
        # Apply increment
        self.CN  = self.CN + phi
        self.CLM = self.CLM + (self.x-xMRP)*phi
        
    
        
    # Correct *CY* and *CLN* using *n* functions
    def CorrectCY(self, CY, CLN, UCY, sig=None, xMRP=0.0):
        """Correct *CY* and *CLN* given *n* unnormalized functions
        
        This function takes an *m* by *n* matrix where *m* is the size of
        *LL.CY*. It then calculates an increment to *LL.CY* that is a linear
        combination of the columns of that matrix *UCY* such that the
        integrated normal force coefficient (*CY*) and pitching moment
        coefficient (*CLN*) match target values provided by the user.  The
        increment is
        
            .. math::
                
                \Delta C_Y = \sum_{i=1}^n k_i\\phi_i
                
        where :math:`\\phi_i`` is the *i*th column of *UCY* scaled so that it
        has an L2 norm of 1.
        
        The weights of the linear coefficient are chosen in order to minimize
        the sum of an objective function subject to the integration constraints
        mentioned above.  This objective function is
        
            .. math::
            
                \\sum_{i=1}^n a_i k_i^2 / \\sigma_i
                
        where :math:`a_i` is the maximum absolute value of column *i* of *UCY*
        and :math:`\\sigma_i` is the associated singular value.
        
        :Call:
            >>> LL.CorrectCY(CY, CLN, UCY, sig=None, xMRP=0.0)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CY*: :class:`float`
                Target integrated value of *CY*
            *CLN*: :class:`float`
                Target integrated value of *CLN*
            *UCY*: :class:`np.ndarray` (*LL.x.size*,*n*)
                Matrix of *CY* adjustment basis functions
            *sig*: {``None``} | :class:`np.ndarray` (*n*,)
                Array of singular values
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of MRP divided by reference length
        :Versions:
            * 2017-02-02 ``@ddalle``: First version
        """
        # Get the current loads
        CY0  = np.trapz(self.CY,  self.x)
        CLN0 = np.trapz(-self.CLN, self.x)
        # Dimensionality of line load vector
        nx = len(self.x)
        # Check dimension count of the dispersions matrix
        if UCY.ndims != 2:
            raise ValueError(
                "Adjustment basis function *UCN* must be 2D array")
        # Dimensions of the dispersion matrix
        m, n = UCY.shape
        # Check dims
        if m != nx:
            raise ValueError(
                ("Adjustment basis functions have %i entries " % m) +
                ("but line load has %i" % nx))
        # L2-norm of each basis vector
        L = np.sqrt(np.sum(UCY**2, axis=0))
        # Initialize normalized basis functions
        VCY = nnp.zeros(m, n)
        # Loop through basis vectors
        for i in range(n):
            # Normalize
            VCY[:,i] = UCY[:,i]/L[i]
        # Max values
        mxCY = np.max(np.abs(VCY), axis=0)
        # Default singular values
        if sig is None:
            sig = np.ones(n)
        # Weights
        w = mxCY / sig
        # Calculate the increment from each mode
        dCY = np.zeros(n)
        dCLN = np.zeros(n)
        # Loop through modes
        for i in range(n):
            dCY[i]  = np.trapz(UCY[:,i], self.x)
            dCLN[i] = np.trapz(UCY[:,i]*(self.x-xMRP), self.x)
        # Form matrix for linear system
        dC = np.array([dCY, dCLN])
        # First two equations: equality constraints on *CN* and *CLM*
        A1 = np.hstack((dC, np.zeros((2,2))))
        # Last *n* equations: derivatives of the Lagrangian
        A2 = np.hstack((np.diag(2*w), -np.transpose(dC)))
        # Assemble matrices
        A = np.vstack((A1, A2))
        # Right-hand sides of equations
        b = np.hstack(([CY-CY0, CLN-CLN0], np.zeros(n)))
        # Solve the linear system (and optimization problem)
        x = np.linalg.solve(A, b)
        # Calculate linear combination (discard Lagrange multipliers)
        phi = np.dot(UCY, x[:n])
        # Apply increment
        self.CY  = self.CY + phi
        self.CLN = self.CLN - (self.x-xMRP)*phi
        
    
    # Correct *CN* and *CLM* given two functions
    def CorrectCN2(self, CN, CLM, CN1, CN2, xMRP=0.0):
        """Correct *CN* and *CLM* given two unnormalized functions
        
        This function takes two functions with the same dimensions as *LL.CN*
        and adds a linear combination of them so that the integrated normal
        force coefficient (*CN*) and pitching moment coefficient (*CLM*) match
        target integrated values given by the user.
        
        The user must specify two basis functions for correcting the *CN*
        sectional loads, and they must be linearly independent.  The
        corrections to *CLM* will be selected automatically to ensure
        consistency between the *CN* and *CLM*.
        
        :Call:
            >>> LL.CorrectCN2(CN, CLM, CN1, CN2)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CN*: :class:`float`
                Target integrated value of *CN*
            *CLM*: :class:`float`
                Target integrated value of *CLM*
            *CN1*: :class:`np.ndarray` (*LL.x.size*)
                First *CN* sectional load correction basis function
            *CN2*: :class:`np.ndarray` (*LL.x.size*)
                Second *CN* sectional load correction basis function
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of MRP divided by reference length
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Get the current loads
        CN0  = np.trapz(self.CN,  self.x)
        CLM0 = np.trapz(self.CLM, self.x)
        # Correction values
        dCN  = CN - CN0
        dCLM = CLM - CLM0
        # Exit if close
        if np.abs(dCN) <= 1e-4 and np.abs(dCLM) <= 1e-4: return
        # Integrated values from the input functions
        dCN1 = np.trapz(CN1, self.x)
        dCN2 = np.trapz(CN2, self.x)
        # Normalize so that dCN == 1.0 unless this would cause an issue
        if np.abs(dCN1)>1e-4:
            CN1 = CN1/dCN1
            dCN1 = 1.0
        if np.abs(dCN2)>1e-4:
            CN2 = CN2/dCN2
            dCN2 = 1.0
        # Get moment correction functions
        CLM1 = (xMRP - self.x) * CN1
        CLM2 = (xMRP - self.x) * CN2
        # Integrated values of $\Delta C_{LM}$
        dCLM1 = np.trapz(CLM1, self.x)
        dCLM2 = np.trapz(CLM2, self.x)
        # Form matrix
        A = np.array([[dCN1, dCN2], [dCLM1, dCLM2]])
        # Check for error
        if abs(np.linalg.det(A)) < 1e-8:
            # Not linearly independent
            print("  WARNING: Two functions are not linearly independent; " +
                "Cannot correct both *CN* and *CLM* (%s)" % np.linalg.det(A))
            return
        # Solve for the weights
        x = np.linalg.solve(A, [dCN, dCLM])
        # Modify the loads
        self.CN  = self.CN  + x[0]*CN1  + x[1]*CN2
        self.CLM = self.CLM + x[0]*CLM1 + x[1]*CLM2
    
    # Correct *CY* and *CLN* given two functions
    def CorrectCY2(self, CY, CLN, CY1, CY2, xMRP=0.0):
        """Correct *CY* and *CLN* given two unnormalized functions
        
        This function takes two functions with the same dimensions as *LL.CY*
        and adds a linear combination of them so that the integrated side
        force coefficient (*CY*) and yawing moment coefficient (*CLN*) match
        target integrated values given by the user.
        
        The user must specify two basis functions for correcting the *CY*
        sectional loads, and they must be linearly independent.  The
        corrections to *CLN* will be selected automatically to ensure
        consistency between the *CY* and *CLN*.
        
        :Call:
            >>> LL.CorrectCY2(CY, CLN, CY1, CY2)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CY*: :class:`float`
                Target integrated value of *CY*
            *CLN*: :class:`float`
                Target integrated value of *CLN*
            *CY1*: :class:`np.ndarray` (*LL.x.size*)
                First *CY* sectional load correction basis function
            *CY2*: :class:`np.ndarray` (*LL.x.size*)
                Second *CY* sectional load correction basis function
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of MRP divided by reference length
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Get the current loads
        CY0  = np.trapz(self.CY,  self.x)
        CLN0 = np.trapz(-self.CLN, self.x)
        # Correction values
        dCY  = CY - CY0
        dCLN = CLN - CLN0
        # Exit if close
        if np.abs(dCY) <= 1e-4 and np.abs(dCLN) <= 1e-4: return
        # Integrated values from the input functions
        dCY1 = np.trapz(CY1, self.x)
        dCY2 = np.trapz(CY2, self.x)
        # Normalize so that dCN == 1.0 unless this would cause an issue
        if np.abs(dCY1)>1e-4:
            CY1 = CY1/dCY1
            dCY1 = 1.0
        if np.abs(dCY2)>1e-4:
            CY2 = CY2/dCY2
            dCY2 = 1.0
        # Get moment correction functions
        CLN1 = (xMRP - self.x) * CY1
        CLN2 = (xMRP - self.x) * CY2
        # Integrated values of $\Delta C_{LM}$
        dCLN1 = np.trapz(-CLN1, self.x)
        dCLN2 = np.trapz(-CLN2, self.x)
        # Form matrix
        A = np.array([[dCY1, dCY2], [dCLN1, dCLN2]])
        # Check for error
        if abs(np.linalg.det(A)) < 1e-8:
            # Not linearly independent
            print("  WARNING: Two functions are not linearly independent; " +
                ("Cannot correct both *CY* and *CLN* (%s)" % np.linalg.det(A)))
            return
        # Solve for the weights
        x = np.linalg.solve(A, [dCY, dCLN])
        # Modify the loads
        self.CY  = self.CY  + x[0]*CY1  + x[1]*CY2
        self.CLN = self.CLN + x[0]*CLN1 + x[1]*CLN2
    
    
    # Correct *CA* using a correction function
    def CorrectCA(self, CA, CA1):
        """Correct *CA* using an unnormalized function
        
        This function takes a function with the same dimensions as *LL.CA* and
        adds a multiple of it so that the integrated axial force coefficient
        (*CA*) matches a target integrated value given by the user.
        
        :Call:
            >>> LL.CorrectCA(CA, CA1)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CA*: :class:`float`
                Target integrated value of *CA*
            *CA1*: :class:`np.ndarray` (*LL.x.size*)
                Basis function to correct *CA*
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Get the current loads
        CA0  = np.trapz(self.CA,  self.x)
        # Correction values
        dCA = CA - CA0
        # Exit if close
        if np.abs(dCA) <= 1e-4: return
        # Integrated values from the input functions
        dCA1 = np.trapz(CA1, self.x)
        # Normalize
        if np.abs(dCA1)>1e-4: CA1 /= dCA1
        # Check for error
        if np.abs(dCA1) < 1e-8:
            # Not linearly independent
            print("WARNING: Basis function does not change *CA*")
            return
        # Solve for the weights
        x = dCA / dCA1
        # Modify the loads
        self.CA = self.CA + x*CA1
    
    
    # Correct *CLL* using a correction function
    def CorrectCLL(self, CLL, CLL1):
        """Correct *CLL* using an unnormalized function
        
        This function takes a function with the same dimensions as *LL.CLL* and
        adds a multiple of it so that the integrated rolling moment coefficient
        (*CLL*) matches a target integrated value given by the user.
        
        :Call:
            >>> LL.CorrectCLL(CLL, CLL1)
        :Inputs:
            *LL*: :class:`cape.cfdx.lineLoad.CaseLL`
                Instance of single-case line load interface
            *CLL*: :class:`float`
                Target integrated value of *CLL*
            *CLL1*: :class:`np.ndarray` (*LL.x.size*)
                Basis function to correct *CLL*
        :Versions:
            * 2016-12-27 ``@ddalle``: First version
        """
        # Get the current loads
        CLL0 = np.trapz(self.CLL, self.x)
        # Correction values
        dCLL = CLL - CLL0
        # Exit if close
        if np.abs(dCLL) <= 1e-4: return
        # Integrated values from the input functions
        dCLL1 = np.trapz(CLL1, self.x)
        # Normalize
        if np.abs(dCLL1)>1e-4: CLL1 /= dCLL1
        # Check for error
        if np.abs(dCLL1) < 1e-8:
            # Not linearly independent
            print("WARNING: Basis function does not change *CLL*")
            return
        # Solve for the weights
        x = dCLL / dCLL1
        # Modify the loads
        self.CLL= self.CLL + x*CLL1
        
  # >
# class CaseLL

# Class for seam curves
class CaseSeam(object):
    """Seam curve interface
    
    :Call:
        >>> S = CaseSeam(fname, comp='entire', proj='LineLoad')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *comp*: :class:`str`
            Name of the component
    :Outputs:
        *S* :class:`cape.cfdx.lineLoad.CaseSeam`
            Seam curve interface
        *S.ax*: ``"x"`` | ``"y"`` | ``"z"``
            Name of coordinate being held constant
        *S.x*: :class:`float` | {:class:`list` (:class:`np.ndarray`)}
            x-coordinate or list of seam x-coordinate vectors
        *S.y*: :class:`float` | {:class:`list` (:class:`np.ndarray`)}
            y-coordinate or list of seam y-coordinate vectors
        *S.z*: {:class:`float`} | :class:`list` (:class:`np.ndarray`)
            z-coordinate or list of seam z-coordinate vectors
    :Versions:
        * 2016-06-09 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname, comp='entire', proj='LineLoad'):
        """Initialization method
        
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        # Save file
        self.fname = fname
        # Save prefix and component name
        self.proj = proj
        self.comp = comp
        # Read file
        self.Read()
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        return "<CaseSeam '%s', n=%s>" % (
            os.path.split(self.fname)[-1], self.n)
        
    # Function to read a seam file
    def Read(self, fname=None):
        """Read a seam  ``*.sm[yz]`` file
        
        :Call:
            >>> S.Read(fname=None)
        :Inputs:
            *S* :class:`cape.cfdx.lineLoad.CaseSeam`
                Seam curve interface
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *S.n*: :class:`int`
                Number of points in vector entries
            *S.x*: :class:`list` (:class:`numpy.ndarray`)
                List of *x* coordinates of seam curves
            *S.y*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
                Fixed *y* coordinate or list of seam curve *y* coordinates
            *S.z*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
                Fixed *z* coordinate or list of seam curve *z* coordinates
        :Versions:
            * 2015-09-17 ``@ddalle``: First version
            * 2016-06-09 ``@ddalle``: Added possibility of x-cuts
        """
        # Default file name
        if fname is None: fname = self.fname
        # Initialize seam count
        self.n = 0
        # Initialize seams
        self.x = []
        self.y = []
        self.z = []
        self.ax = 'y'
        # Check for the file
        if not os.path.isfile(fname): return
        # Open the file.
        f = open(fname, 'r')
        # Read first line.
        line = f.readline()
        # Get the axis and value
        txt = line.split()[-2]
        ax  = txt.split('=')[0]
        val = float(txt.split('=')[1])
        # Name of cut axis
        self.ax = ax
        # Save the value
        setattr(self, ax, val)
        # Read two lines.
        f.readline()
        f.readline()
        # Loop through curves.
        while line != '':
            # Get data
            D = np.fromfile(f, count=-1, sep=" ")
            # Check size.
            m = int(np.floor(D.size/2) * 2)
            # Save the data.
            if ax == 'x':
                # x-cut
                self.y.append(D[0:m:2])
                self.z.append(D[1:m:2])
            elif ax == 'y':
                # y-cut
                self.x.append(D[0:m:2])
                self.z.append(D[1:m:2])
            else:
                # z-cut
                self.x.append(D[0:m:2])
                self.y.append(D[1:m:2])
            # Segment count
            self.n += 1
            # Read two lines.
            line = f.readline()
            line = f.readline()
        # Cleanup
        f.close()
            
    # Function to write a seam file
    def Write(self, fname=None):
        """Write a seam curve file
        
        :Call:
            >>> S.Write(fname)
        :Inputs:
            *S* :class:`cape.cfdx.lineLoad.CaseSeam`
                Seam curve interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-17 ``@ddalle``: First version
            * 2016-06-09 ``@ddalle``: Added possibility of x-cuts
            * 2016-06-09 ``@ddalle``: Moved to seam class
        """
        # Default file name
        if fname is None:
            fname = '%s_%s.sm%s' % (self.proj, self.comp, self.ax)
        # Check if there's anything to write.
        if self.n < 1: return
        # Check axis
        if self.ax == 'x':
            # x-cuts
            x1 = 'y'
            x2 = 'z'
        elif self.ax == 'z':
            # z-cuts
            x1 = 'x'
            x2 = 'y'
        else:
            # y-cuts
            x1 = 'x'
            x2 = 'z'
        # Save axis
        ax = self.ax
        # Open the file.
        f = open(fname, 'w')
        # Write the header line.
        f.write(' #Seam curves for %s=%s plane\n'
            % (self.ax, getattr(self,ax)))
        # Loop through seems
        for i in range(self.n):
            # Header
            f.write(' #Seam curve %11i\n' % i)
            # Extract coordinates
            x = getattr(self,x1)[i]
            y = getattr(self,x2)[i]
            # Write contents
            for j in np.arange(len(x)):
                f.write(" %11.6f %11.6f\n" % (x[j], y[j]))
        # Cleanup
        f.close()
        
    # Function to plot a set of seam curves
    def Plot(self, **kw):
        """Plot a set of seam curves
        
        :Call:
            >>> h = S.Plot(**kw)
        :Inputs:
            *S* :class:`cape.cfdx.lineLoad.CaseSeam`
                Seam curve interface
            *x*: {``"x"``} | ``"y"`` | ``"z"``
                Axis to plot on x-axis
            *y*: ``"x"`` | {``"y"``} | ``"z"``
                Axis to plot on y-axis
            *LineOptions*: {``{}``} | :class:`dict`
                Dictionary of plot options
            *Label*: :class:`str`
                Plot label, ``LineOptions['label']`` supersedes this variable
            *XLabel*: {``"x/Lref"``} | :class:`str`
                Label for x-axis
            *XLabel*: {*coeff*} | :class:`str`
                Label for y-axis
            *xpad*: {``0.03``} | :class:`float`
                Relative margin to pad x-axis limits
            *ypad*: {``0.03``} | :class:`float`
                Relative margin to pad y-axis limits
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        # Ensure plotting modules
        ImportPyPlot()
        # Other plot options
        fw = kw.get('FigWidth')
        fh = kw.get('FigHeight')
        # Get default axes
        if self.ax == 'x':
            # X-cuts
            x0 = 'y'
            y0 = 'z'
        elif self.ax == 'y':
            # Y-cuts
            x0 = 'x'
            y0 = 'z'
        else:
            # Z-cuts
            x0 = 'x'
            y0 = 'y'
        # Get axes
        kx = kw.get('x', x0)
        ky = kw.get('y', y0)
        # Name for plot handles
        ksm = 'sm' + self.ax
        # ------------
        # Primary plot
        # ------------
        # Initialize primary plot options
        kw_p = odict(color=kw.get("color","k"), ls="-", lw=1.5, zorder=7)
        # Extract plot optiosn from kwargs
        for k in util.denone(kw.get("LineOptions", {})):
            # Override the default option
            if kw["LineOptions"][k] is not None:
                kw_p[k] = kw["LineOptions"][k]
        # Apply label
        kw_p.setdefault('label', kw.get('Label', self.comp))
        # Initialize handles
        h = {ksm: []}
        # Loop through curves
        for i in range(self.n):
            # Turn off labels after first plot
            if i == 1: del kw_p['label']
            # Get coordinates
            x = getattr(self, kx)[i]
            y = getattr(self, ky)[i]
            # Plot
            h[ksm].append(plt.plot(x, y, **kw_p))
        # --------------
        # Figure margins
        # --------------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Process axis labels
        xlbl = kw.get('XLabel', kx + '/Lref')
        ylbl = kw.get('YLabel', ky + '/Lref')
        # Label handles
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Get actual limits
        xmin, xmax = util.get_xlim_ax(h['ax'], **kw)
        ymin, ymax = util.get_ylim_ax(h['ax'], **kw)
        # DO NOT Ensure proper aspect ratio; leave commented
        # This comment is here to remind you not to do it!
        # plt.axis('equal')
        # Set the axis limits
        h['ax'].set_xlim((xmin, xmax))
        h['ax'].set_ylim((ymin, ymax))
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # Margins
        adj_l = kw.get('AdjustLeft')
        adj_r = kw.get('AdjustRight')
        adj_t = kw.get('AdjustTop')
        adj_b = kw.get('AdjustBottom')
        # Make adjustments
        if adj_l: plt.subplots_adjust(left=adj_l)
        if adj_r: plt.subplots_adjust(right=adj_r)
        if adj_t: plt.subplots_adjust(top=adj_t)
        if adj_b: plt.subplots_adjust(bottom=adj_b)
        # Report the actual limits
        h['xmin'] = xmin
        h['xmax'] = xmax
        h['ymin'] = ymin
        h['ymax'] = ymax
        # Output
        return h
        
# class CaseSeam

