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
# Line load template
import cape.lineLoad


# Data book of line loads
class DBLineLoad(cape.lineLoad.DBLineLoad):
    """Line load (sectional load) data book for one group
    
    :Call:
        >>> DBL = DBLineLoad(x, opts. comp, conf=None, RootDir=None)
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
        
    # Update a case
    def UpdateCase(self, i):
        """Update one line load entry if necessary
        
        :Call:
            >>> DBL.UpdateLineLoadCase(i)
        :Inputs:
            *DBL*: :class:`cape.lineLoad.DBLineLoad`
                Line load data book
            *i*: :class:`int`
                Case number
        :Versions:
            * 2016-06-07 ``@ddalle``: First version
        """
        # Try to find a match in the data book
        j = self.FindMatch(i)
        # Get the name of the folder
        frun = self.x.GetFullFolderNames(i)
        # Status update
        print(frun)
        # Go to root directory safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check if the folder exits
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Determine minimum number of iterations required
        nAvg = self.opts.get_nStats(self.comp)
        nMin = self.opts.get_nMin(self.comp)
        # Get the number of iterations
        ftriq, nStats, n0, nIter = GetTriqFile()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations (or zero)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif self['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i." %
                (self['nIter'][j], nIter))
            q = True
        elif self['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for update
        if not q:
            os.chdir(fpwd)
            return
        # Create lineload folder if necessary
        if not os.path.isdir('lineload'): self.opts.mkdir('lineload')
        # Enter lineload folder
        os.chdir('lineload')
        # Append to triq file
        ftriq = os.path.join('..', ftriq)
        # Name of loads file
        flds = '%s_%s.%s' % (self.proj, self.comp, self.ext)
        # Name of triload input file
        fcmd = 'triload.%s.i' % self.comp
        # Process existing input file
        if os.path.isfile(fcmd):
            # Open input file
            f = open(fcmd)
            # Read first line
            otriq = f.readline().strip()
            # Close file
            f.close()
        else:
            # No input file
            otriq = ''
        # Check whether or not to compute
        print("otriq: '%s'" % otriq)
        print("ftriq: '%s'" % ftriq)
        print("flds:  '%s'" % flds)
        print("?flds: %s" % os.path.isfile(flds))
        print("PWD=%s"  % os.getcwd())
        if not otriq != ftriq:
            # Not using the most recent triq file
            q = True
        elif not os.path.isfile(flds):
            # No loads yet
            q = True
        elif os.path.getmtime(flds) < os.path.getmtime(ftriq):
            # Loads files are older than surface file
            q = True
        else:
            # Loads up to date
            q = False
        # Run triload if necessary
        if q:
            # Write triloadCmd input file
            self.WriteTriloadInput(ftriq, i)
            # Run the command
            self.RunTriload()
        # Read the loads file
        print(os.listdir())
        self[i] = CaseLL(self.comp, self.proj, self.ext, fdir=None)
        # CSV folder names
        fll  = os.path.join(self.RootDir, self.fdir, 'lineload')
        fgrp = os.path.join(fll, frun.split(os.sep)[0])
        fcas = os.path.join(fll, frun)
        # Create folders as necessary
        if not os.path.isdir(fll):  self.opts.mkdir(fll)
        if not os.path.isdir(fgrp): self.opts.mkdir(fgrp)
        if not os.path.isdir(fcas): self.opts.mkdir(fcas)
        # CSV file name
        fcsv = os.path.join(fcas, '%s_%s.csv' % (self.proj, self.comp))
        # Write the CSV file
        self[i].WriteCSV(fcsv)
        # Save the stats
        if np.isnan(j):
            # Add to the number of cases
            self.n += 1
            # Append trajectory values.
            for k in self.x.keys:
                # Append to numpy array
                self[k] = np.hstack((self[k], [getattr(self.x,k)[i]]))
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
# class DBLineLoad
    

# Line loads
class CaseLL(cape.lineLoad.CaseLL):
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
        * 2016-06-07 ``@ddalle``: Subclassed
    """
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
            
