"""
Aerodynamic Iterative History: :mod:`pyCart.aero`
=================================================

This module contains functions for reading and processing forces, moments, and
other statistics from a run directory.

:Versions:
    * 2014-11-12 ``@ddalle``: Starter version
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re
# Import history interface
from history import History
# Plotting
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import PdfPages


# Aerodynamic history class
class Aero(dict):
    """
    Aerodynamic history class
    =========================
    
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> aero = pyCart.aero.Aero(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class
        *aero*: :class:`dict` (:class:`numpy.ndarray`)
            Dictionary of force and/or moment histories
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self, comps=[]):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Process the best data folder.
        if os.path.isfile('history.dat'):
            # Use current directory.
            fdir = '.'
        elif os.path.islink('BEST'):
            # There's a BEST/ folder; use it as most recent adaptation cycle.
            fdir = 'BEST'
        elif os.path.isdir('adapt00'):
            # It's an adaptive run, but it hasn't gotten far yet.
            fdir = 'adapt00'
        else:
            # This is not an adaptive cycle; use root folder.
            fdir = '.'
        # Read the loadsCC.dat file to see what components are requested.
        self.ReadLoadsCC()
        # Default component list.
        if (not comps):
            # Extract keys from dictionary.
            comps = self.Components.keys()
        elif (type(comps).__name__ == "str"):
            # Make a singleton list.
            comps = [comps]
        # Loop through components.
        for comp in comps:
            # Expected name of the history file.
            fname = os.path.join(fdir, comp+'.dat')
            # Check if it exists.
            if not os.path.isfile(fname):
                # Warn and got to the next component.
                print("Warning: Component '%s' was not found." % comp)
                continue
            # Otherwise, read the file.
            L = open(fname).readlines()
            # Filter comments
            L = [l for l in L if not l.startswith('#')]
            # Columns to use: 0 and {-6,-3}:
            n = len(self.Components[comp]['C'])
            # Select the columns
            L = [l.split()[0:1] + l.split()[-n:] for l in L]
            # Create an array.
            A = np.array([[float(v) for v in l] for l in L])
            # Extract info from components for readability
            d = self.Components[comp]
            # Make the component.
            self[comp] = FM(d['C'], MRP=d['MRP'], A=A)
            
            
    # Function to read 'loadsCC.dat'
    def ReadLoadsCC(self):
        """Read forces and moments from a :file:`loadsCC.dat` file if possible
        
        :Call:
            >> aero.ReadLoadsCC()
        :Inputs:
            *aero*: :class:`pyCart.aero.Aero`
                Instance of the aero history class
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize list of components.
        self.Components = {}
        # Check for the file.
        if os.path.isfile(os.path.join('BEST', 'loadsCC.dat')):
            # Use the most recent file.
            fCC = os.path.join('BEST', 'loadsCC.dat')
        elif os.path.isfile(os.path.join('adapt00', 'loadsCC.dat')):
            # Most recent adaptation currently running; no file in BEST
            fCC = os.path.join('adapt00', 'loadsCC.dat')
        elif os.path.isfile('loadsCC.dat'):
            # Non-adaptive run
            fCC = 'loadsCC.dat'
        else:
            # Nothing to do.
            return None
        # Read the file.
        linesCC = open(fCC).readlines()
        # Loop through the lines.
        for line in linesCC:
            # Strip line.
            line = line.strip()
            # Check for empty line or comment.
            if (not line) or line.startswith('#'): continue
            # Get name of component.
            comp = line.split()[0]
            # Add line to dictionary if necessary.
            if comp not in self.Components:
                self.Components[comp] = {'C':[], 'MRP':None}
            # Try to get the coefficient name.
            try:
                # Find text like '(C_A)' and return 'C_A'.
                c = re.search('\(([A-Za-z_]+)\)', line).group(1)
            except Exception:
                # Failed to find expected text.
                print("Warning: no coefficient name in line:\n  '%s'" % line) 
                continue
            # Filter the coefficient.
            if c == 'C_A':
                # Axial force
                self.Components[comp]['C'].append('CA')
                continue
            elif c == 'C_Y': 
                # Lateral force
                self.Components[comp]['C'].append('CY')
                continue
            elif c == 'C_N':
                # Normal force
                self.Components[comp]['C'].append('CN')
                continue
            elif c == 'C_M_x':
                # Rolling moment
                self.Components[comp]['C'].append('CLL')
            elif c == 'C_M_y':
                # Pitching moment
                self.Components[comp]['C'].append('CLM')
            elif c == 'C_M_z':
                # Yaw moment
                self.Components[comp]['C'].append('CLN')
            else:
                # Extra coefficient such as lift, drag, etc.
                continue
            # Only process reference point once.
            if self.Components[comp]['MRP'] is not None: continue
            # Try to find reference point.
            try:
                # Search for text like '(17.0, 0, 0)'.
                txt = re.search('\(([0-9., -]+)\)', line).group(1)
                # Split into coordinates.
                MRP = np.array([float(v) for v in txt.split(',')])
                # Save it.
                self.Components[comp]['MRP'] = MRP
            except Exception:
                # Failed to find expected text.
                print("Warning: no reference point in line:\n  '%s'" % line)
                
        
# Individual component force and moment
class FM(object):
    """
    Individual force and moment class
    =================================
    
    This class contains methods for reading data about an individual component. 
    The list of available components comes from a :file:`loadsCC.dat` file if
    one exists.
    
    :Call:
        >>> FM = pyCart.aero.FM(C, MRP=None, A=None)
    :Inputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *FM*: :class:`pyCart.aero.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list` (:class:`str`)
            List of coefficients
        *FM.MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
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
    """
    # Initialization method
    def __init__(self, C, MRP=None, A=None):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Save component list.
        self.C = C
        # Initialize iteration list.
        self.i = np.array([])
        # Loop through components.
        for c in C:
            setattr(self, c, np.array([]))
        # Save the MRP.
        self.MRP = np.array(MRP)
        # Check for data.
        if A is not None:
            # Use method to parse.
            self.AddData(A)
            
    # Function to display contents
    def __repr__(self):
        """Representation method
        
        Returns one of the following:
        
            * ``'<aero.FM Force, i=100>'``
            * ``'<aero.FM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<aero.FM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize the string.
        txt = '<aero.FM '
        # Check for a moment.
        if ('CA' in self.C) and ('CLL' in self.C):
            # Force and moment.
            txt += 'FM'
        elif ('CA' in self.C):
            # Force only
            txt += 'Force'
        elif ('CLL' in self.C):
            # Moment only
            txt += 'Moment'
        # Add number of iterations.
        txt += (', i=%i' % self.i.size)
        # Add MRP if possible.
        if (self.MRP.size == 3):
            txt += (', MRP=(%.2f, %.2f, %.2f)' % tuple(self.MRP))
        # Finish the string and return it.
        return txt + '>'
        
    # String method
    def __str__(self):
        """String method
        
        Returns one of the following:
        
            * ``'<aero.FM Force, i=100>'``
            * ``'<aero.FM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<aero.FM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        return self.__repr__()
        
            
    # Method to add data to instance
    def AddData(self, A):
        """Add iterative force and/or moment history for a component
        
        :Call:
            >>> FM.AddData(A)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
                Matrix of forces and/or moments at *N* iterations
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Get size of A.
        n, m = A.shape
        # Save the iterations.
        self.i = np.arange(n) + 1
        # Check size.
        if m == 7:
            # Save all fields.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            self.CLL = A[:,4]
            self.CLM = A[:,5]
            self.CLN = A[:,6]
        elif (self.MRP.size==3) and (m == 4):
            # Save only moments.
            self.CLL = A[:,1]
            self.CLM = A[:,2]
            self.CLN = A[:,3]
        elif (m == 4):
            # Save only forces.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            
    # Function to plot force coeffs and residual (resid is only a blank)
    def Plot4(self, **kw):
        """Initialize a plot for three force coefficients and L1 residual
        
        :Call:
            >>> h = FM.Plot4(**kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`pyCart.aero.Aero.plot`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Plot the forces and residual.
        h = self.Plot(2, 2, ['CA', 'CY', 'CN', 'L1'], **kw)
        # Output
        return h
        
    # Function to plot force coefficients only
    def PlotForce(self, **kw):
        """Initialize a plot for three force coefficients
        
        :Call:
            >>> h = FM.PlotForce(**kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`pyCart.aero.Aero.plot`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Plot the forces, but leave a spot for residual.
        h = self.Plot(3, 1, ['CA', 'CY', 'CN'], **kw)
        # Output
        return h
        
    # Function to plot longitudinal forces and moments.
    def PlotLongitudinal(self, **kw):
        """Initialize a plot for longitudinal forces and moments
        
        :Call:
            >>> h = FM.PlotLongitudinal(**kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`pyCart.aero.Aero.plot`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Plot the forces, but leave a spot for residual.
        h = self.Plot(3, 1, ['CA', 'CN', 'CLM'], **kw)
        # Output
        return h
        
    # Function to plot longitudinal forces and moments.
    def Plot4Long(self, **kw):
        """Initialize a plot for longitudinal forces and moments and L1 resid
        
        :Call:
            >>> h = FM.Plot4Long(**kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`pyCart.aero.Aero.plot`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Plot the forces and residual.
        h = self.Plot(2, 2, ['CA', 'CN', 'CLM', 'L1'], **kw)
        # Output
        return h
        
    # Function to plot lateral forces and moments.
    def PlotLateral(self, **kw):
        """Initialize a plot for longitudinal forces and moments and L1 resid
        
        :Call:
            >>> h = FM.PlotLateral(**kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`pyCart.aero.Aero.plot`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Plot the forces, but leave a spot for residual.
        h = self.Plot(2, 2, ['CY', 'CN', 'CLM', 'CLN'], **kw)
        # Output
        return h
            
    # Function to plot several coefficients.
    def Plot(self, nRow, nCol, C, n=1000, nAvg=100, d={}, d0=0.01, **kw):
        """Plot one or several component histories
        
        :Call:
            >>> h = FM.Plot(nRow, nCol, C, n=1000, nAvg=100, d0=0.01, **kw)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *nRow*: :class:`int`
                Number of rows of subplots to make
            *nCol*: :class:`int`
                Number of columns of subplots to make
            *C*: :class:`list` (:class:`str`)
                List of coefficients or ``'L1'`` to plot
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d0*: :class:`float`
                Default delta to use
            *d*: :class:`dict`
                Dictionary of deltas for each component
            *tag*: :class:`str` 
                Tag to put in upper corner, for instance case number and name
            *restriction*: :class:`str`
                Type of data, e.g. ``"SBU - ITAR"`` or ``"U/FOUO"``
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Check for single input.
        if type(C).__name__ == "str": C = [C]
        # Number of components
        nC = len(C)
        # Check inputs.
        if nC > nRow*nCol:
            raise IOError("Too many components")
        # Initialize handles.
        h = {}
        # Loop through components.
        for i in range(nC):
            # Get coefficient.
            c = C[i]
            # Pull up the subplot.
            plt.subplot(nRow, nCol, i+1)
            # Check if residual was requested.
            if c == 'L1':
                # Read the history.
                hist = Hist()
                # Plot it.
                h[c] = hist.PlotL1(n=n)
            else:
                # Get the delta
                di = d.get(c, d0)
                # Plot
                h[c] = self.PlotCoeff(c, n=n, nAvg=nAvg, d=di)
            # Turn off overlapping xlabels for condensed plots.
            if (nCol==1 or nRow>2) and (i+nCol<nC):
                # Kill the xlabel and xticklabels.
                h[c]['ax'].set_xticklabels(())
                h[c]['ax'].set_xlabel('')
        # Max of number 
        n0 = max(nCol, nRow)
        # Determine target font size.
        if n0 == 1:
            # Font size (default)
            fsize = 12
        elif n0 == 2:
            # Smaller
            fsize = 9
        else:
            # Really small
            fsize = 8
        # Loop through the text labels.
        for h_t in plt.gcf().findobj(Text):
            # Apply the target font size.
            h_t.set_fontsize(fsize)
        # Add tag.
        tag = kw.get('tag', '')
        h['tag'] = plt.figtext(0.015, 0.985, tag, verticalalignment='top')
        # Add restriction.
        txt = kw.get('restriction', '')
        h['restriction'] = plt.figtext(0.5, 0.01, txt,
            horizontalalignment='center')
        # Attempt to use the tight_layout() utility.
        try:
            # Add room for labels with *rect*, and tighten up other margins.
            plt.gcf().tight_layout(pad=0.2, w_pad=0.5, h_pad=0.7,
                rect=(0,0.015,1,0.91))
        except Exception:
            pass
        # Save the figure.
        h['fig'] = plt.gcf()
        # Output
        return h
        
    # Function to plot a single coefficient.
    def PlotCoeff(self, c, n=1000, nAvg=100, d=0.01):
        """Plot a single coefficient history
        
        :Call:
            >>> h = FM.PlotCoeff(c, n=1000, nAvg=100, d=0.01)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d*: :class:`float`
                Delta in the coefficient to show expected range
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Extract the data.
        C = getattr(self, c)
        # Number of iterations present.
        nIter = self.i.size
        # Process min indices for plotting and averaging.
        i0 = max(0, nIter-n)
        i0Avg = max(0, nIter-nAvg)
        # Calculate mean.
        cAvg = np.mean(C[i0Avg:])
        # Initialize dictionary of handles.
        h = {}
        # Calculate range of interest.
        if d:
            # Limits
            cMin = cAvg-d
            cMax = cAvg+d
            # Plot the target window boundaries.
            h['min'] = (
                plt.plot([i0,i0Avg], [cMin,cMin], 'r:', lw=0.8) +
                plt.plot([i0Avg,nIter], [cMin,cMin], 'r-', lw=0.8))
            h['max'] = (
                plt.plot([i0,i0Avg], [cMax,cMax], 'r:', lw=0.8) +
                plt.plot([i0Avg,nIter], [cMax,cMax], 'r-', lw=0.8))
        # Plot the mean.
        h['mean'] = (
            plt.plot([i0,i0Avg], [cAvg, cAvg], 'r--', lw=1.0) + 
            plt.plot([i0Avg,nIter], [cAvg, cAvg], 'r-', lw=1.0))
        # Plot the coefficient.
        h[c] = plt.plot(self.i[i0:], C[i0:], 'k-', lw=1.5)
        # Labels.
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel(c)
        # Get the axes.
        h['ax'] = plt.gca()
        # Set the xlimits.
        h['ax'].set_xlim((i0, nIter+25))
        # Make a label for the mean value.
        lbl = u'%s = %.4f' % (c, cAvg)
        h['val'] = plt.text(0.81, 1.06, lbl, horizontalalignment='right',
            verticalalignment='top', transform=h['ax'].transAxes)
        # Make a label for the deviation.
        lbl = u'\u00B1 %.4f' % d
        h['d'] = plt.text(1.0, 1.06, lbl, color='r',
            horizontalalignment='right', verticalalignment='top',
            transform=h['ax'].transAxes)
        # Output.
        return h
            
    

# Aerodynamic history class
class Hist(History):
    """
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = pyCart.aero.Hist()
    :Outputs:
        *hist*: :class:`pyCart.history.History`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Plot function
    def PlotL1(self, n=None):
        """Plot the L1 residual
        
        :Call:
            >>> h = hist.PlotL1()
            >>> h = hist.PlotL1(n)
        :Inputs:
            *hist*: :class:`pyCart.aero.Hist`
                Instance of the run history class
            *n*: :class:`int`
                Only show the last *n* iterations
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize dictionary.
        h = {}
        # Get iteration numbers.
        if n is None:
            # Use all iterations
            i0 = 0
        else:
            # Use only last *n* iterations
            i0 = max(0, self.nIter - n)
        # Plot the residual.
        h['L1'] = plt.semilogy(self.i[i0:], self.L1Resid[i0:], 'k-', lw=1.5)
        # Labels
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel('L1 Residual')
        # Get the axes.
        h['ax'] = plt.gca()
        # Set the xlimits.
        h['ax'].set_xlim((i0, self.nIter+25))
        # Output.
        return h
        
