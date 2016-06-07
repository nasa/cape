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
        fname = os.apth.join(fdir, fcomp)
        
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
        self.proj = self.copts.get('Prefix', 'LineLoad')
        self.comp = comp
        self.ext  = self.copts.get("Extension", "dlds")
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
            nCol += 2
            self['XMRP'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol])
            self['YRMP'] = np.loadtxt(fname,
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
            self['YRMP'] = np.array([], dtype=float)
            self['ZMRP'] = np.array([], dtype=float)
            self['nIter'] = np.array([], dtype=int)
            self['nStats'] = np.array([], dtype=int)
        # Save the number of cases analyzed.
        self.n = len(self[k])
    
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
            f.write('%i\n' % (self['nStats'][i], delim))
        # Close the file.
        f.close()
        
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
        self[i] = CaseLL(self.comp, proj=self.proj, ext='csv', fdir=frun)
        
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
        # Get components
        compID = self.CompID
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
        self.write(self.comp + ' ')
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
        ierr = os.system('triloadCmd < triload.%s.i > triload.out')
        # Check for errors
        if ierr:
            return SystemError("Failure while running ``triloadCmd``")
    
# class DBLineLoad
    
# Line load from one case
class CaseLL(object):
    """Interface to individual sectional load
    
    :Call:
        >>> LL = CaseLL(comp, proj='LineLoad', ext='slds')
    :Inputs:
        *comp*: :class:`str`
            Name of component
        *proj*: :class:`str`
            Prefix for sectional load output files
        *ext*: ``"slds"`` | ``"clds"`` | {``"dlds"``}
            File extension for section, cumulative, or derivative loads
    :Outputs:
        *LL*: :class:`cape.lineLoad.CaseLL`
            Individual line load for one component from one case
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
        * 2016-06-07 ``@ddalle``: Second version, universal
    """
    # Initialization method
    def __init__(self, comp, proj='LineLoad', ext='dlds', fdir='lineload'):
        """Initialization method
        
        :Versions:
            * 2016-06-07 ``@ddalle``: First universal version
        """
        # Save input options
        self.comp = comp
        self.proj = proj
        self.ext  = ext
        self.fdir = fdir
        # File name
        if fdir is None:
            # Use the working folder
            self.fname = '%s_%s.%s' % (proj, comp, ext)
        else:
            # Corral line load files in separate folder
            self.fname = os.path.join(fdir, '%s_%s.%s' % (proj, comp, ext))
        # Read the file
        try:
            # Check if we are reading triload output file or data book file
            if ext.lower() == "csv":
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
        """Read seam curves from a line load directory
        
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
        fsmy = '%s.smy' % fpre
        fsmz = '%s.smz' % fpre
        # Read the seam curves.
        self.smy = ReadSeam(fsmy)
        self.smz = ReadSeam(fsmz)
    
# class CaseLL
        
# Function to read a seam file
def ReadSeam(fname):
    """Read a seam  ``*.sm[yz]`` file
    
    :Call:
        >>> s = ReadSeam(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *s*: :class:`dict`
            Dictionary of seem curves
        *s['x']*: :class:`list` (:class:`numpy.ndarray`)
            List of *x* coordinates of seam curves
        *s['y']*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
            Fixed *y* coordinate or list of seam curve *y* coordinates
        *s['z']*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
            Fixed *z* coordinate or list of seam curve *z* coordinates
    :Versions:
        * 2015-09-17 ``@ddalle``: First version
    """
    # Initialize data.
    s = {'x':[], 'y':[], 'z':[]}
    # Check for the file
    if not os.path.isfile(fname): return s
    # Open the file.
    f = open(fname, 'r')
    # Read first line.
    line = f.readline()
    # Get the axis and value
    txt = line.split()[-2]
    ax  = txt.split('=')[0]
    val = float(txt.split('=')[1])
    # Save it.
    s[ax] = val
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
        if ax == 'y':
            # y-cut
            s['x'].append(D[0:m:2])
            s['z'].append(D[1:m:2])
        else:
            # z-cut
            s['x'].append(D[0:m:2])
            s['y'].append(D[1:m:2])
        # Read two lines.
        f.readline()
        f.readline()
    # Cleanup
    f.close()
    # Output
    return s
        
# Function to write a seam file
def WriteSeam(fname, s):
    """Write a seam curve file
    
    :Call:
        >>> WriteSeam(fname, s)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *s*: :class:`dict`
            Dictionary of seem curves
        *s['x']*: :class:`list` (:class:`numpy.ndarray`)
            List of *x* coordinates of seam curves
        *s['y']*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
            Fixed *y* coordinate or list of seam curve *y* coordinates
        *s['z']*: :class:`float` or :class:`list` (:class:`numpy.ndarray`)
            Fixed *z* coordinate or list of seam curve *z* coordinates
    :Versions:
        * 2015-09-17 ``@ddalle``: First version
    """
    # Check axis
    if type(s['y']).__name__ in ['list', 'ndarray']:
        # z-cuts
        ax = 'z'
        ct = 'y'
    else:
        # y-cuts
        ax = 'y'
        ct = 'z'
    # Open the file.
    f = open(fname)
    # Write the header line.
    f.write(' #Seam curves for %s=%s plane\n' % (ax, s[ax]))
    # Loop through seems
    for i in range(len(s['x'])):
        # Header
        f.write(' #Seam curve %11i\n' % i)
        # Extract coordinates
        x = s['x'][i]
        y = s[ct][i]
        # Write contents
        for j in np.arange(len(x)):
            f.write(" %11.6f %11.6f\n" % (x[j], y[j]))
    # Cleanup
    f.close()

# Function to determine newest triangulation file
def GetTriqFile():
    """Get most recent ``triq`` file and its associated iterations
    
    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile()
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
    fglob3 = glob.glob('Components.*.*.*.triq')
    fglob2 = glob.glob('Components.*.*.triq')
    fglob1 = glob.glob('Components.[0-9]*.triq')
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
    elif os.path.isfile('Components.i.triq'):
        # Iteration counts: assume it's most recent iteration
        i1 = self.cart3d.CheckCase(self.i)
        i0 = i1
        # Count
        n = 1
        # file name
        ftriq = 'Components.i.triq'
    else:
        # No iterations
        i1 = None
        i0 = None
        n = None
        ftriq = None
    # Output
    os.chdir(fpwd)
    return ftriq, n, i0, i1
            
