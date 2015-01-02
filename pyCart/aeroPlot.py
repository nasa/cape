"""
Plotting tools: :mod:`pyCart.Plot`
==================================


"""


# File interface
import os
# Numerics
import numpy as np
# Plotting
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import PdfPages

# Modules
from .dataBook import Aero
# System interface
import os

        
      
# Module for plotting components
class AeroPlot(Aero):
    """
    Aerodynamic history class with plotting
    =======================================
    
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> AP = pyCart.aeroPlot.AeroPlot(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *AP*: :class:`pyCart.aeroplot.AeroPlot`
            Instance of the aero history class, similar to dictionary of force
            and/or moment histories
    :Versions:
        * 2014-12-10 ``@ddalle``: First version
    """
    
        
    # Function to plot a single coefficient.
    def PlotCoeff(self, comp, c, n=1000, nAvg=100, d=0.01):
        """Plot a single coefficient history
        
        :Call:
            >>> h = AP.PlotCoeff(comp, c, n=1000, nAvg=100, d=0.01)
        :Inputs:
            *AP*: :class:`pyCart.aeroPlot.AeroPlot`
                Instance of the force history plotting class
            *comp*: :class:`str`
                Name of component to plot
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
            * 2014-12-09 ``@ddalle``: Transferred to :class:`AeroPlot`
        """
        # Extract the component.
        FM = self[comp]
        # Extract the data.
        C = getattr(FM, c)
        # Number of iterations present.
        nIter = FM.i.size
        # Process min indices for plotting and averaging.
        i0 = max(0, nIter-n)
        i0Avg = max(0, nIter-nAvg)
        # Calculate mean.
        cAvg = np.mean(C[i0Avg:])
        # Get the actual iteration numbers to use for averaging.
        iA = FM.i[i0Avg]
        iB = FM.i[-1]
        # Initialize dictionary of handles.
        h = {}
        # Calculate range of interest.
        if d:
            # Limits
            cMin = cAvg-d
            cMax = cAvg+d
            # Plot the target window boundaries.
            h['min'] = (
                plt.plot([i0,iA], [cMin,cMin], 'r:', lw=0.8) +
                plt.plot([iA,iB], [cMin,cMin], 'r-', lw=0.8))
            h['max'] = (
                plt.plot([i0,iA], [cMax,cMax], 'r:', lw=0.8) +
                plt.plot([iA,iB], [cMax,cMax], 'r-', lw=0.8))
        # Plot the mean.
        h['mean'] = (
            plt.plot([i0,iA], [cAvg, cAvg], 'r--', lw=1.0) + 
            plt.plot([iA,iB], [cAvg, cAvg], 'r-', lw=1.0))
        # Plot the coefficient.
        h[c] = plt.plot(FM.i[i0:], C[i0:], 'k-', lw=1.5)
        # Labels.
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel(c)
        # Get the axes.
        h['ax'] = plt.gca()
        # Set the xlimits.
        h['ax'].set_xlim((i0, iB+25))
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
    
    # Plot function
    def PlotL1(self, n=None):
        """Plot the L1 residual
        
        :Call:
            >>> h = AP.PlotL1()
            >>> h = AP.PlotL1(n)
        :Inputs:
            *AP*: :class:`pyCart.aero.AeroPlot`
                Instance of the force history plotting class
            *n*: :class:`int`
                Only show the last *n* iterations
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
        """
        # Initialize dictionary.
        h = {}
        # Get iteration numbers.
        if n is None:
            # Use all iterations
            i0 = 0
        else:
            # Use only last *n* iterations
            i0 = max(0, self.Residual.nIter - n)
        # Extract iteration numbers and residuals.
        i  = self.Residual.i[i0:]
        L1 = self.Residual.L1Resid[i0:]
        L0 = self.Residual.L1Resid0[i0:]
        # Plot the initial residual if there are any unsteady iterations.
        if L0[-1] > L1[-1]:
            h['L0'] = plt.semilogy(i, L0, 'b-', lw=1.2)
        # Plot the residual.
        h['L1'] = plt.semilogy(i, L1, 'k-', lw=1.5)
        # Labels
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel('L1 Residual')
        # Get the axes.
        h['ax'] = plt.gca()
        # Set the xlimits.
        h['ax'].set_xlim((i0, self.Residual.i[-1]+25))
        # Output.
        return h
            
    # Function to plot several coefficients.
    def Plot(self, comp, C, d={}, **kw):
        """Plot one or several component histories
        
        :Call:
            >>> h = AP.Plot(comp, C, d={}, n=1000, nAvg=100, **kw)
        :Inputs:
            *AP*: :class:`pyCart.aero.AeroPlot`
                Instance of the force history plotting class
            *comp*: :class:`str`
                Name of component to plot
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
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
        """
        # Read inputs
        nRow = kw.get('nRow', 2)
        nCol = kw.get('nCol', 2)
        n    = kw.get('n', 1000)
        nAvg = kw.get('nAvg', 100)
        d0   = kw.get('d0', 0.01)
        # Check for single input.
        if type(C).__name__ == "str": C = [C]
        # Number of components
        nC = len(C)
        # Check inputs.
        if nC > nRow*nCol:
            raise IOError("Too many components for %i rows and %i columns" 
                % (nRow, nCol))
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
                # Plot it.
                h[c] = self.PlotL1(n=n)
            else:
                # Get the delta
                di = d.get(c, d0)
                # Plot
                h[c] = self.PlotCoeff(comp, c, n=n, nAvg=nAvg, d=di)
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
        # Add PASS label (empty but handle is useful)
        h['pass'] = plt.figtext(0.99, 0.97, "", color="#00E500",
            horizontalalignment='right')
        # Add iteration label
        h['iter'] = plt.figtext(0.99, 0.94, "%i/" % self[comp].i[-1],
            horizontalalignment='right', size=9)
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
            
    # Function to plot force coeffs and residual (resid is only a blank)
    def Plot4(self, comp, **kw):
        """Initialize a plot for three force coefficients and L1 residual
        
        :Call:
            >>> h = AP.Plot4(comp, **kw)
        :Inputs:
            *AP*: :class:`pyCart.aero.AeroPlot`
                Instance of the force history plotting class
            *comp*: :class:`str`
                Name of component to plot
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
        h = self.Plot(comp, ['CA', 'CY', 'CN', 'L1'], nRow=2, nCol=2, **kw)
        # Output
        return h
        
    # Function to plot force coefficients only
    def PlotForce(self, comp, **kw):
        """Initialize a plot for three force coefficients
        
        :Call:
            >>> h = AP.PlotForce(comp, **kw)
        :Inputs:
            *AP*: :class:`pyCart.aero.AeroPlot`
                Instance of the force history plotting class
            *comp*: :class:`str`
                Name of component to plot
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
        h = self.Plot(comp, ['CA', 'CY', 'CN'], nRow=3, nCol=1, **kw)
        # Output
        return h
    
    
    
    
