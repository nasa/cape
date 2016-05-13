"""
Sectional Loads Module: :mod:`pyCart.lineLoad`
==============================================

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
from cape import tar


# Data book of line loads
class DBLineLoad(object):
    """Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(cart3d, comp)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master pyCart interface
        *i*: :class:`int`
            Case index
        *comp*: :class:`str`
            Name of line load group
    :Outputs:
        *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
            Instance of line load data book
        *DBL.nCut*: :class:`int`
            Number of *x*-cuts to make, based on options in *cart3d*
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
    """
    def __init__(self, cart3d, comp):
        """Initialization method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Get the data book directory.
        fdir = opts.get_DataBookDir()
        # Compatibility
        fdir = fdir.repalce("/", os.sep)
        
        # Construct the file name.
        fcomp = 'll_%s.csv' % comp
        # Full file name
        fname = os.apth.join(fdir, fcomp)
        
        # Save the Cart3D run info
        self.cart3d = cart3d
        # Save the file name.
        self.fname = fname
        
        # Reference areas
        self.RefA = cart3d.opts.get_RefArea(self.RefComp)
        self.RefL = cart3d.opts.get_RefLength(self.RefComp)
        # Moment reference point
        self.MRP = np.array(cart3d.opts.get_RefPoint(self.RefComp))
        
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
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
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
            for k in self.cart3d.x.keys:
                # Get the type.
                t = self.cart3d.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Read the column
                self[k] = np.loadtxt(fname,
                    delimiter=delim, dtype=str(t), usecols=[nCol])
                # Increase the column number.
                nCol += 1
            # Mach number and Reynolds number
            self['Mach'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol])
            self['Re'] = np.loadtxt(fname,
                delimiter=delim, dtype=float, usecols=[nCol+1])
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
            for k in self.cart3d.x.keys:
                # get the type.
                t = self.cart3d.x.defns[k].get('Value', 'float')
                # convert type
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Initialize an empty array.
                self[k] = np.array([], dtype=str(t))
            # Initialize Other parameters.
            self['Mach'] = np.array([], dtype=float)
            self['Re']   = np.array([], dtype=float)
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
        delim = self.cart3d.opts.get_Delimiter()
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
        for k in self.cart3d.x.keys:
            f.write(k + delim)
        # Write the extra column titles.
        f.write('Mach%sRe%sXMRP%sYMRP%sZMRP%snIter%snStats\n' %
            tuple([delim]*6))
        # Loop through database entries.
        for i in np.arange(self.n):
            # Write the trajectory values.
            for k in self.cart3d.x.keys:
                f.write('%s%s' % (self[k][i], delim))
            # Write data values
            f.write('%s%s' % (self['Mach'][i], delim))
            f.write('%s%s' % (self['Re'][i], delim))
            f.write('%s%s' % (self['XMRP'][i], delim))
            f.write('%s%s' % (self['YMRP'][i], delim))
            f.write('%s%s' % (self['ZMRP'][i], delim))
            # Iteration counts
            f.write('%i%s' % (self['nIter'][i], delim))
            f.write('%i\n' % (self['nStats'][i], delim))
        # Close the file.
        f.close()
        
    # Function to sort the data book
    def ArgSort(self, key=None):
        """Return indices that would sort a a data book by a trajectory key
        
        :Call:
            >>> I = DBL.ArgSort(key=None)
        :Inputs:
            *DBL*: :class;`pyCart.lineLoad.DBLineLoad`
                Instance of line load group data book
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Process the key.
        if key is None: key = self.x.keys[0]
        # Check for multiple keys.
        if type(key).__name__ in ['list', 'ndarray', 'tuple']:
            # Init pre-array list of ordered n-lets like [(0,1,0), ..., ]
            Z = zip(*[self[k] for k in key])
            # Init list of key definitions
            dt = []
            # Loop through keys to get data types (dtype)
            for k in key:
                # Get the type.
                dtk = self.cart3d.x.defns[k]['Value']
                # Convert it to numpy jargon.
                if dtk in ['float']:
                    # Numeric value
                    dt.append((str(k), 'f'))
                elif dtk in ['int', 'hex', 'oct', 'octal']:
                    # Stored as an integer
                    dt.append((str(k), 'i'))
                else:
                    # String is default.
                    dt.append((str(k), 'S32'))
            # Create the array to be used for multicolumn sort.
            A = np.array(Z, dtype=dt)
            # Get the sorting order
            I = np.argsort(A, order=[str(k) for k in key])
        else:
            # Indirect sort on a single key.
            I = np.argsort(self[key])
        # Output.
        return I
        
    # Function to sort data book
    def Sort(self, key=None, I=None):
        """Sort a data book according to either a key or an index
        
        :Call:
            >>> DBL.Sort()
            >>> DBL.Sort(key)
            >>> DBL.Sort(I=None)
        :Inputs:
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
                Instance of the pyCart data book line load group
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Process inputs.
        if I is not None:
            # Index array specified; check its quality.
            if type(I).__name__ not in ["ndarray", "list"]:
                # Not a suitable list.
                raise TypeError("Index list is unusable type.")
            elif len(I) != self.n:
                # Incompatible length.
                raise IndexError(("Index list length (%i) " % len(I)) +
                    ("is not equal to data book size (%i)." % self.n))
        else:
            # Default key if necessary
            if key is None: key = self.x.keys[0]
            # Use ArgSort to get indices that sort on that key.
            I = self.ArgSort(key)
        # Sort all fields.
        for k in self:
            # Sort it.
            self[k] = self[k][I]
        
    # Find an entry by trajectory variables.
    def FindMatch(self, i):
        """Find an entry by run matrix (trajectory) variables
        
        It is assumed that exact matches can be found.
        
        :Call:
            >>> j = DBL.FindMatch(i)
        :Inputs:
            *DBL*: :class:`pyCart.lineLoad.DBLineLoad`
                Instance of the pyCart line load data book
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of index that matches the trajectory case or ``NaN``
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
            * 2015-09-16 ``@ddalle``: Copied from :class:`dataBook.DBComp`
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Loop through keys requested for matches.
        for k in self.cart3d.x.keys:
            # Get the target value (from the trajectory)
            v = getattr(self.x,k)[i]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(self[k] == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == self[k][-1]):
                    # Add the last element.
                    jk = np.union1d(jk, [len(self[k])-1])
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                # No match found.
                return np.nan
        # Output
        try:
            # There should be exactly one match.
            return j[0]
        except Exception:
            # Return no match.
            return np.nan
# class DBLineLoad
    

# Line loads
class CaseLL(object):
    """Individual class line load class
    
    :Call:
        >>> LL = CaseLL(cart3d, i, comp)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master pyCart interface
        *i*: :class:`int`
            Case index
        *comp*: :class:`str`
            Name of line load group
    :Outputs:
        *LL*: :class:`pyCart.lineLoad.CaseLL`
            Instance of individual case line load interface
        *LL.nCut*: :class:`int`
            Number of *x*-cuts to make, based on options in *cart3d*
        *LL.nIter*: :class:`int`
            Last iteration in line load file
        *LL.nStats*: :class:`int`
            Number of iterations in line load file
        *LL.RefL*: :class:`float`
            Reference length
        *LL.MRP*: :class:`numpy.ndarray` shape=(3,)
            Moment reference center
        *LL.x*: :class:`numpy.ndarray` shape=(*nCut*,)
            Locations of *x*-cuts
        *LL.CA*: :class:`numpy.ndarray` shape=(*nCut*,)
            Axial force sectional load, d(CA)/d(x/RefL))
    :Versions:
        * 2015-09-16 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, proj='ll', comp=None):
        """Initialization method"""
        # Save options
        self.proj = proj
        self.comp = comp
        # File prefix
        self.pre = '%s_%s' % (proj, comp)
        # Loads file
        self.fdlds = '%s.dlds' % self.pre
        # Read files
        self.ReadLDS(self.fdlds)
    
    # Function to display contents
    def __repr__(self):
        """Representation method
        
        Returns the following format:
        
            * ``<CaseLL nCut=200>``
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        return "<CaseLL '%s' nCut=%i>" % (self,pre, self.nCut)
        
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
        """# Change to root directory.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        # Name of triload folder
        ftri = 'fomo-lineload'
        # Name of output files.
        fsmy = 'LineLoad_%s.smy' % self.comp
        fsmz = 'LineLoad_%s.smz' % self.comp
        # Get working directory
        fdir = self.cart3d.x.GetFullFolderNames(i)
        # Enter
        os.chdir(frun)
        # Enter the lineload folder and untar if necessary.
        tar.chdir_in(ftri)
        # Read the seam curves.
        self.smy = ReadSeam(fsmy)
        self.smz = ReadSeam(fsmz)
        # Clean up.
        tar.chdir_up()
        os.chdir(fpwd)
    
    # Function to read a file
    def ReadLDS(self, fname):
        """Read a sectional loads ``*.?lds`` from `triloadCmd`
        
        :Call:
            >>> LL.ReadLDS(fname)
        :Inputs:
            *LL*: :class:`pyCart.lineLoad.CaseLL`
                Single-case line load interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Open the file.
        f = open(fname, 'r')
        # Read lines until it is not a comment.
        line = '#'
        while (line.lstrip().startswith('#')) and (len(line)>0):
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
        
    # Plot a line load
    def PlotLDS(self, coeff):
        pass
    
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
            
