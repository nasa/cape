"""
Sectional Loads Module: :mod:`cape.lineLoad`
============================================

This module contains functions for reading and processing sectional loads.  It
is a submodule of :mod:`pyCart.dataBook`.

:Versions:
    * 2015-09-15 ``@ddalle``: Started
"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime

# Utilities or advanced statistics
from . import util
from . import case
from . import dataBook
from cape import tar

# Finer control of dicts
from .options import odict

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
        # Load the modules.
        import matplotlib.pyplot as plt
        import matplotlib.transforms as tform
        from matplotlib.text import Text


# Data book of line loads
class DBLineLoad(dataBook.DBBase):
    """Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(x, opts, comp, conf=None, RootDir=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of line load component
        *conf*: {``"None"``} | :class:`cape.config.Config`
            Surface configuration interface
        *RootDir*: {``"None"``} | :class:`str`
            Root directory for the configuration
    :Outputs:
        *DBL*: :class:`cape.lineLoad.DBLineLoad`
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
    def __init__(self, x, opts, comp, proj='LineLoad', conf=None, RootDir=None):
        """Initialization method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
            * 2016-06-07 ``@ddalle``: Updated slightly
        """
        # Save root directory
        if RootDir is None:
            # Use the current folder
            self.RootDir = os.getcwd()
        else:
            self.RootDir = RootDir
        
        # Get the data book directory.
        fdir = opts.get_DataBookDir()
        # Compatibility
        fdir = fdir.replace("/", os.sep)
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
            opts.mkdir(fdir)
        # Check for lineload folder
        if not os.path.isdir(os.path.join(fdir, 'lineload')):
            # Create line load folder
            os.mkdir(os.path.join(fdir, 'lineload'))
        # Return to original location
        os.chdir(fpwd)
        
        # Save the CFD run info
        self.x = x
        self.opts = opts
        self.conf = conf
        # Specific options for this component
        self.copts = opts['DataBook'][comp]
        # Save component name
        self.proj = self.opts.get_DataBookPrefix(comp)
        self.comp = comp
        self.sec  = self.opts.get_DataBookSectionType(comp)
        # Save the file name.
        self.fname = fname
        
        # Figure out reference component
        self.CompID = opts.get_DataBookCompID(comp)
        # Make sure it's not a list
        if type(self.CompID).__name__ == 'list':
            # Take the first component
            self.RefComp = self.RefComp[0]
        else:
            # One component listed; use it
            self.RefComp = self.CompID
        # Try to get all components
        try:
            # Use the configuration interface
            self.CompID = self.conf.GetCompID(self.CompID)
        except Exception:
            pass
        # Reference areas
        self.RefA = opts.get_RefArea(self.RefComp)
        self.RefL = opts.get_RefLength(self.RefComp)
        # Moment reference point
        self.MRP = np.array(opts.get_RefPoint(self.RefComp))
        # Read the file or initialize empty arrays.
        self.Read(fname)
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
        
    # function to read line load data book summary
    def Read(self, fname=None):
        """Read a data book summary file for a single line load group
        
        :Call:
            >>> DBL.Read()
            >>> DBL.Read(fname)
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Instance of line load data book
            *fname*: :class:`str`
                Name of summary file
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Save column names
        self.cols = self.x.keys + ['XMRP','YMRP','ZMRP','nIter','nStats']
        # Try to read the file.
        try:
            # Data book delimiter
            delim = self.opts.get_Delimiter()
            # Initialize column number.
            nCol = 0
            # Loop through the trajectory keys.
            for k in self.x.keys:
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
            k = 'ZMRP'
            # Iteration number
            nCol += 3
            self['nIter'] = np.loadtxt(fname,
                delimiter=delim, dtype=int, usecols=[nCol])
            k = 'nIter'
            # Stats
            nCol += 1
            k = 'nStats'
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
        except Exception:
            # Initialize empty trajectory arrays
            for k in self.x.keys:
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
        for k in self.x.keys:
            f.write(k + delim)
        # Write the extra column titles.
        f.write('Mach%sRe%sXMRP%sYMRP%sZMRP%snIter%snStats\n' %
            tuple([delim]*6))
        # Loop through database entries.
        for i in np.arange(self.n):
            # Write the trajectory values.
            for k in self.x.keys:
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
    
    # Read the seam curves
    def ReadSeamCurves(self):
        """Read seam curves from a data book directory
        
        :Call:
            >>> DBL.ReadSeamCurves()
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
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
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Line load data book
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
        # Expected folder
        fll = os.path.join(self.RootDir, self.fdir, 'lineload')
        # Check for folder
        if not os.path.isdir(fll): self.opts.mkdir(fll)
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
                self.smy.Write(fsmx)
            except Exception:
                pass
        # Write the x-cuts if necessary and possible
        if not os.path.isfile(fsmz):
            try:
                self.smz.Write(fsmx)
            except Exception:
                pass
        
    # Read a case from the data book
    def ReadCase(self, i):
        """Read data from a case from the data book archive
        
        :Call:
            >>> DBL.ReadCase(i)
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Search for match of this case in the data book
        j = self.FindMatch(i)
        # Check if current case is in the data book
        if np.isnan(j): return
        # Check if already up to date
        if i in self: return
        # Path to lineload folder
        fll = os.path.join(self.RootDir, self.fdir, 'lineload')
        # Get name of case
        frun = os.path.join(fll, self.x.GetFullFolderNames(i))
        # Check if the case is present
        if not os.path.isdir(frun): return
        # File name
        fname = os.path.join(frun, '%s_%s.csv' % (self.proj, self.comp))
        # Check for the file
        if not os.path.isfile(fname): return
        # Read the file
        self[i] = CaseLL(self.comp, 
            proj=self.proj, typ=self.sec, ext='csv', fdir=frun)
        
    # Write triload.i input file
    def WriteTriloadInput(self, ftriq, i, **kw):
        """Write ``triload.i`` input file to ``triloadCmd``
        
        :Call:
            >>> DBL.WriteTriloadInput(ftriq, i, **kw)
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
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
        # Get components and type of the input
        compID = self.CompID
        tcomp  = type(compID).__name__
        # Convert to string if appropriate
        if tcomp in ['list', 'ndarray']:
            compID = [str(comp) for comp in compID]
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
            # Write list of component IDs
            f.write(','.join(compID) + '\n')
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
        f.write('n\n')
        # Close the input file
        f.close()
        
    # Run triload
    def RunTriload(self):
        """Run ``triload`` for a case
        
        :Call:
            >>> DBL.RunTriload()
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Line load data book
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Run triload
        cmd = 'triloadCmd < triload.%s.i > triload.o' % self.comp
        # Status update
        print("    %s" % cmd)
        # Run triload
        ierr = os.system(cmd)
        # Check for errors
        if ierr:
            return SystemError("Failure while running ``triloadCmd``")
    
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
        *LL*: :class:`cape.lineLoad.CaseLL`
            Individual line load for one component from one case
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
        * 2016-06-07 ``@ddalle``: Second version, universal
    """
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
        
            * ``<CaseLL nCut=200>``
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        return "<CaseLL comp='%s' (%s)>" % (self.comp, self.ext)
    
    # Function to read a file
    def ReadLDS(self, fname=None):
        """Read a sectional loads ``*.?lds`` file from `triloadCmd`
        
        :Call:
            >>> LL.ReadLDS(fname)
        :Inputs:
            *LL*: :class:`cape.lineLoad.CaseLL`
                Single-case, single component, line load interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Default file name
        if fname is None: fname = self.fname
        # Open the file.
        f = open(fname, 'r')
        # Read lines until it is not a comment.
        line = '#'
        while (line.lstrip().startswith('#')) and (len(line)>0):
            # Read the next line.
            line = f.readline()
        # Exit if empty.
        if len(line) == 0:
            raise ValueError("Empty triload file '%s'" % fname)
        # Number of columns
        nCol = len(line.split())
        # Go backwards one line from current position.
        f.seek(-len(line), 1)
        # Read the rest of the file.
        D = np.fromfile(f, count=-1, sep=' ')
        # Reshape to a matrix
        D = D.reshape((D.size/nCol, nCol))
        # Save the keys.
        self.x   = D[:,0]
        self.CA  = D[:,1]
        self.CY  = D[:,2]
        self.CN  = D[:,3]
        self.CLL = D[:,4]
        self.CLM = D[:,5]
        self.CLN = D[:,6]
        # Cloe the file
        f.close()
        
    # Function to read a databook file
    def ReadCSV(self, fname=None, delim=','):
        """Read a sectional loads ``csv`` file from the data book
        
        :Call:
            >>> LL.ReadCSV(fname, delim=',')
        :Inputs:
            *LL*: :class:`cape.lineLoad.CaseLL`
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
        # Open the file.
        f = open(fname, 'r')
        # Read lines until it is not a comment.
        line = '#'
        while (not line.lstrip().startswith('#')) and (len(line)>0):
            # Read the next line.
            line = f.readline()
        # Exit if empty.
        if len(line) == 0:
            return {}
        # Number of columns
        nCol = len(line.split())
        # Go backwards one line from current position.
        f.seek(-len(line), 1)
        # Read the rest of the file.
        D = np.fromfile(f, count=-1, sep=delim)
        # Reshape to a matrix
        D = D.reshape((D.size/nCol, nCol))
        # Save the keys.
        self.x   = D[:,0]
        self.CA  = D[:,1]
        self.CY  = D[:,2]
        self.CN  = D[:,3]
        self.CLL = D[:,4]
        self.CLM = D[:,5]
        self.CLN = D[:,6]
        
    # Write CSV file
    def WriteCSV(self, fname=None, delim=','):
        """Write a sectional loads ``csv`` file
        
        :Call:
            >>> LL.WriteCSV(fname, delim=',')
        :Inputs:
            *LL*: :class:`cape.lineLoad.CaseLL`
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
            *Label*: {*LL.comp*} | :class:`str`
                Plot label, ``LineOptions['label']`` supersedes this variable
            *XLabel*: {``"x/Lref"``} | :class:`str`
                Label for x-axis
            *YLabel*: {*coeff*} | :class:`str`
                Label for y-axis
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
            *SubplotMargin*: {``0.015``} | :class:`float`
                Margin between subplots
        :Versions:
            * 2016-06-09 ``@ddalle``: First version
        """
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
        ly = h['ax'].get_ylabel()
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
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
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
            # Plot the seam
            hi = sms[i].Plot(**kw)
            # Save the handles
            H[sfigi-1] = hi
            # Copy axes handle
            axi = hi['ax']
            # Save aspect ratio
            AR[sfigi-1] = (hi['ymax']-hi['ymin']) / (hi['xmax']-hi['xmin'])
            # Get axes position
            pi = axi.get_position().get_points()
            # Follow up for each type
            if q_vert:
                # Copy xlims from line load plot
                axi.set_xlim(xlim)
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
            # Set margins
            if q_vert:
                # Automatic axis height based on aspect ratio
                haxi = AR[i] * wax
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
                # Copy the limits again
                ax.set_xlim(xlim)
                print("position: [%s, %s, %s, %s]" %
                    (xax_min, yax_min, xax_max, yax_max))
            else:
                # Automatic axis width based on aspect ratio
                waxi = hax / AR[i]
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
                # Reset axis limits
                ax.set_ylim(ylim)
        # Finally, set the position for the position for the main figure
        h['ax'].set_position([xax_min,yax_min,xax_max-xax_min,yax_max-yax_min])
        # REset limits
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
            *LL*: :class:`pyCart.lineLoad.CaseLL`
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
        *S* :class:`cape.lineLoad.CaseSeam`
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
        return "<CaseSeam '%s', n=%s>" % (self.fname, self.n)
        
    # Function to read a seam file
    def Read(self, fname=None):
        """Read a seam  ``*.sm[yz]`` file
        
        :Call:
            >>> S.Read(fname=None)
        :Inputs:
            *S* :class:`cape.lineLoad.CaseSeam`
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
            m = np.floor(D.size/2) * 2
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
            *S* :class:`cape.lineLoad.CaseSeam`
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
        # Axis types
        vx = type(self.x).__name__ in ['list', 'ndarray']
        vy = type(self.y).__name__ in ['list', 'ndarray']
        vz = type(self.z).__name__ in ['list', 'ndarray']
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
        # Open the file.
        f = open(fname)
        # Write the header line.
        f.write(' #Seam curves for %s=%s plane\n' % (self.ax, getattr(self,ax)))
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
            *S* :class:`cape.lineLoad.CaseSeam`
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
        # Ensure proper aspect ratio
        plt.axis('equal')
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

# Function to determine newest triangulation file
def GetTriqFile(proj='Components'):
    """Get most recent ``triq`` file and its associated iterations
    
    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile(proj='Components')
    :Inputs:
        *proj*: {``"Components"``} | :class:`str`
            File root name
    :Outputs:
        *ftriq*: :class:`str`
            Name of ``triq`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
    """
    # Get the working directory.
    fwrk = case.GetWorkingFolder()
    # Go there.
    fpwd = os.getcwd()
    os.chdir(fwrk)
    # Get the glob of numbered files.
    fglob3 = glob.glob('%s.*.*.*.triq'  % proj)
    fglob2 = glob.glob('%s.*.*.triq'    % proj)
    fglob1 = glob.glob('%s.[0-9]*.triq' % proj)
    # Check it.
    if len(fglob3) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[3]) for f in fglob3]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob3[j].split('.')[2])
        # Count
        n = int(fglob3[j].split('.')[1])
        # File name
        ftriq = fglob3[j]
    if len(fglob2) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[2]) for f in fglob2]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = int(fglob2[j].split('.')[1])
        # File name
        ftriq = fglob2[j]
    # Check it.
    elif len(fglob1) > 0:
        # Get last iterations
        I0 = [int(f.split('.')[1]) for f in fglob1]
        # Index of best iteration
        j = np.argmax(I0)
        # Iterations there.
        i1 = I0[j]
        i0 = I0[j]
        # Count
        n = i1 - i0 + 1
        # File name
        ftriq = fglob1[j]
    # Plain file
    elif os.path.isfile('%s.i.triq' % proj):
        # Iteration counts: assume it's most recent iteration
        i1 = self.cart3d.CheckCase(self.i)
        i0 = i1
        # Count
        n = 1
        # file name
        ftriq = '%s.i.triq' % proj
    else:
        # No iterations
        i1 = None
        i0 = None
        n = None
        ftriq = None
    # Output
    os.chdir(fpwd)
    return ftriq, n, i0, i1
            
