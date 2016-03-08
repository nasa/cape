#!/usr/bin/env python
"""
OVERFLOW Point Sensor Module
============================

"""

# File interface
import os, glob
# Basic numerics
import numpy as np
# Date processing
from datetime import datetime
# Local function
from .options   import odict
# Utilities and advanced statistics
from . import util

# Basis module
import cape.dataBook


# Data book for group of point sensors
class DBPointSensorGroup(dict):
    """
    Point sensor group data book
    
    :Call:
        >>> DBPG = DBPointSensorGroup(x, opts, name)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *name*: :class:`str` | ``None``
            Name of data book item (defaults to *pt*)
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBPG*: :class:`pyOver.pointSensor.DBPointSensorGroup`
            A point sensor group data book
    :Versions:
        * 2015-12-04 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, x, opts, name, **kw):
        """Initialization method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Save the interface.
        self.x = x
        self.opts = opts
        # Save the name
        self.name = name
        # Get the list of points.
        self.pts = opts.get_DBGroupPoints(name)
        # Loop through the points.
        for pt in self.pts:
            self[pt] = DBPointSensor(x, opts, pt, name)
            
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBPointSensorGroup %s, " % self.name
        # Number of cases in book
        lbl += "nPoint=%i>" % len(self.pts)
        # Output
        return lbl
    __str__ = __repr__
    
    # Sorting method
    def Sort(self):
        """Sort point sensor group
        
        :Call:
            >>> DBPG.Sort()
        :Inputs:
            *DBPG*: :class:`pyCart.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2016-03-08 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            self[pt].Sort()
            
            
    # Output method
    def Write(self):
        """Write to file each point sensor data book in a group
        
        :Call:
            >>> DBPG.Write()
        :Inputs:
            *DBPG*: :class:`pyCart.pointSensor.DBPointSensorGroup`
                A point sensor group data book
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Loop through points
        for pt in self.pts:
            # Sort it.
            self[pt].Sort()
            # Write it
            self[pt].Write()
    
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBPG.UpdateCase(i)
        :Inputs:
            *DBPG*: :class:`pyOver.pointSensor.DBPointSensorGroup`
                A point sensor group data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-02-17 ``@ddalle``: Placeholder
        """
        pass
# class DBPointSensorGroup

# Data book of point sensors
class DBPointSensor(cape.dataBook.DBBase):
    """
    Point sensor data book
    
    :Call:
        >>> DBP = DBPointSensor(x, opts, pt, name=None)
    :Inputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Trajectory/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *pt*: :class:`str`
            Name of point
        *name*: :class:`str` | ``None``
            Name of data book item (defaults to *pt*)
        *RootDir*: :class:`str` | ``None``
            Project root directory absolute path, default is *PWD*
    :Outputs:
        *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
            An individual point sensor data book
    :Versions:
        * 2016-02-17 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, x, opts, pt, name=None, **kw):
        """Initialization method
        
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Folder containing the data book
        fdir = opts.get_DataBookDir()
        # Folder name for compatibility
        fdir = fdir.replace("/", os.sep)
        
        # File name
        fpt = 'pt_%s.csv' % pt
        # Absolute path to point sensors
        fname = os.path.join(fdir, fpt)
        
        # Save data book title
        if name is None:
            # Default name
            self.name = pt
        else:
            # Specified name
            self.name = name
        # Save point name
        self.pt = pt
        # Save the CNTL
        self.x = x
        self.opts = opts
        # Save the file name
        self.fname = fname
        # Column types
        self.xCols = self.x.keys
        self.fCols = [
            'Cp', 'Cp_std', 'Cp_min', 'Cp_max'
        ]
        self.iCols = ['nIter', 'nStats']
        # Counts
        self.nxCol = len(self.xCols)
        self.nfCol = len(self.fCols)
        self.niCol = len(self.iCols)
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBPointSensor %s, " % self.pt
        # Number of cases in book
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    __str__ = __repr__
    
    # Process a case
    def UpdateCase(self, i):
        """Prepare to update one point sensor case if necessary
        
        :Call:
            >>> DBP.UpdateCase(i)
        :Inputs:
            *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
                An individual point sensor data book
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-02-17 ``@ddalle``: Placeholder
        """
        pass
            
    # Plot a sweep of one or more coefficients
    def PlotValueHist(self, coeff, I, **kw):
        """Plot a histogram of one coefficient over several cases
        
        :Call:
            >>> h = DBi.PlotHist(coeff, I, **kw)
        :Inputs:
            *DB*: :class:`cape.dataBook.DBBase`
                Instance of the data book component class
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *TargetValue*: :class:`float` | :class:`list` (:class:`float`)
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowEpsilon*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
            * 2015-12-14 ``@ddalle``: Added error bars
        """
        # -----------
        # Preparation
        # -----------
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
        # Extract the values
        V = self[coeff][I]
        # Calculate basic statistics
        vmu = np.mean(V)
        vstd = np.std(V)
        # Check for outliers ...
        ostd = kw.get('OutlierSigma', 7.0)
        # Apply outlier tolerance
        if ostd:
            # Find indices of cases that are within outlier range
            J = np.abs(V-vmu)/vstd <= ostd
            # Downselect
            V = V[J]
            # Recompute statistics
            vmu = np.mean(V)
            vstd = np.std(V)
        # Uncertainty options
        ksig = kw.get('StDev')
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in [None, False]:
            vtarg = []
        elif type(vtarg).__name__ not in ['list', 'tuple', 'ndarray']:
            vtarg = [vtarg]
        # Create appropriate target list for 
        if type(ltarg).__name__ not in ['list', 'tuple', 'ndarray']:
            ltarg = [ltarg]
        # --------
        # Plotting
        # --------
        # Initialize dictionary of handles.
        h = {}
        # --------------
        # Histogram Plot
        # --------------
        # Initialize plot options for histogram.
        kw_h = odict(facecolor='c', zorder=2, bins=20)
        # Extract options from kwargs
        for k in util.denone(kw.get("HistOptions", {})):
            # Override the default option.
            if kw["HistOptions"][k] is not None:
                kw_h[k] = kw["HistOptions"][k]
        # Check for range based on standard deviation
        if kw.get("Range"):
            # Use this number of pair of numbers as multiples of *vstd*
            r = kw["Range"]
            # Check for single number or list
            if type(r).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - r[0]*vstd
                vmax = vmu + r[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - r*vstd
                vmax = vmu + r*vstd
            # Overwrite any range option in *kw_h*
            kw_h['range'] = (vmin, vmax)
        # Plot the historgram.
        h['hist'] = plt.hist(V, **kw_h)
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Get current axis limits
        pmin, pmax = h['ax'].get_ylim()
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", True)
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # ---------
        # Mean Plot
        # ---------
        # Option whether or not to plot mean as vertical line.
        if kw.get("PlotMean", True):
            # Initialize options for mean plot
            kw_m = odict(color='k', lw=2, zorder=6)
            kw_m["label"] = "Mean value"
            # Extract options from kwargs
            for k in util.denone(kw.get("MeanOptions", {})):
                # Override the default option.
                if kw["MeanOptions"][k] is not None:
                    kw_m[k] = kw["MeanOptions"][k]
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot([vmu,vmu], [pmin,pmax], **kw_m)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot([pmin,pmax], [vmu,vmu], **kw_m)
        # -----------
        # Target Plot
        # -----------
        # Option whether or not to plot targets
        if vtarg is not None and len(vtarg)>0:
            # Initialize options for target plot
            kw_t = odict(color='k', lw=2, ls='--', zorder=8)
            # Set label
            if ltarg is not None:
                # User-specified list of labels
                kw_t["label"] = ltarg
            else:
                # Default label
                kw_t["label"] = "Target"
            # Extract options for target plot
            for k in util.denone(kw.get("TargetOptions", {})):
                # Override the default option.
                if kw["TargetOptions"][k] is not None:
                    kw_t[k] = kw["TargetOptions"][k]
            # Loop through target values
            for i in range(len(vtarg)):
                # Select the value
                vt = vtarg[i]
                # Check for NaN or None
                if np.isnan(vt) or vt in [None, False]: continue
                # Downselect options
                kw_ti = {}
                for k in kw_t:
                    kw_ti[k] = kw_t.get_key(k, i)
                # Initialize handles
                h['target'] = []
                # Check orientation
                if q_vert:
                    # Plot a vertical line for the target.
                    h['target'].append(
                        plt.plot([vt,vt], [pmin,pmax], **kw_ti))
                else:
                    # Plot a horizontal line for the target.
                    h['target'].append(
                        plt.plot([pmin,pmax], [vt,vt], **kw_ti))
        # -----------------------
        # Standard Deviation Plot
        # -----------------------
        # Check whether or not to plot it
        if ksig and len(I)>2:
            # Check for single number or list
            if type(ksig).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - ksig[0]*vstd
                vmax = vmu + ksig[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - ksig*vstd
                vmax = vmu + ksig*vstd
            # Initialize options for std plot
            kw_s = odict(color='b', lw=2, zorder=5)
            # Extract options from kwargs
            for k in util.denone(kw.get("StDevOptions", {})):
                # Override the default option.
                if kw["StDevOptions"][k] is not None:
                    kw_s[k] = kw["StDevOptions"][k]
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = (
                    plt.plot([vmin,vmin], [pmin,pmax], **kw_s) +
                    plt.plot([vmax,vmax], [pmin,pmax], **kw_s))
            else:
                # Plot a horizontal line for the min and max
                h['std'] = (
                    plt.plot([pmin,pmax], [vmin,vmin], **kw_s) +
                    plt.plot([pmin,pmax], [vmax,vmax], **kw_s))
        # ----------
        # Delta Plot
        # ----------
        # Check whether or not to plot it
        if dc:
            # Initialize options for delta plot
            kw_d = odict(color="r", ls="--", lw=1.0, zorder=3)
            # Extract options from kwargs
            for k in util.denone(kw.get("DeltaOptions", {})):
                # Override the default option.
                if kw["DeltaOptions"][k] is not None:
                    kw_d[k] = kw["DeltaOptions"][k]
                # Check for single number or list
            if type(dc).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                cmin = vmu - dc[0]
                cmax = vmu + dc[1]
            else:
                # Use as a single number
                cmin = vmu - dc
                cmax = vmu + dc
            # Check orientation
            if q_vert:
                # Plot vertical lines for the reference length
                h['delta'] = (
                    plt.plot([cmin,cmin], [pmin,pmax], **kw_d) +
                    plt.plot([cmax,cmax], [pmin,pmax], **kw_d))
            else:
                # Plot horizontal lines for reference length
                h['delta'] = (
                    plt.plot([pmin,pmax], [cmin,cmin], **kw_d) +
                    plt.plot([pmin,pmax], [cmax,cmax], **kw_d))
        # ----------
        # Formatting
        # ----------
        # Default value-axis label
        lx = coeff
        # Default probability-axis label
        if q_normed:
            # Size of bars is probability
            ly = "Probability Density"
        else:
            # Size of bars is count
            ly = "Count"
        # Process axis labels
        xlbl = kw.get('XLabel')
        ylbl = kw.get('YLabel')
        # Apply defaults
        if xlbl is None: xlbl = lx
        if ylbl is None: ylbl = ly
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # ------
        # Labels
        # ------
        # y-coordinates of the current axes w.r.t. figure scale
        ya = h['ax'].get_position().get_points()
        ha = ya[1,1] - ya[0,1]
        # y-coordinates above and below the box
        yf = 2.5 / ha / h['fig'].get_figheight()
        yu = 1.0 + 0.065*yf
        yl = 1.0 - 0.04*yf
        # Make a label for the mean value.
        if kw.get("ShowMu", True):
            # printf-style format flag
            flbl = kw.get("MuFormat", "%.4f")
            # Form: CA = 0.0204
            lbl = (u'%s = %s' % (coeff, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(0.99, yu, lbl, color=kw_m['color'],
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['mu'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the deviation.
        if dc and kw.get("ShowDelta", True):
            # printf-style flag
            flbl = kw.get("DeltaFormat", "%.4f")
            # Form: \DeltaCA = 0.0050
            lbl = (u'\u0394%s = %s' % (coeff, flbl)) % dc
            # Create the handle.
            h['d'] = plt.text(0.01, yl, lbl, color=kw_d.get_key('color',1),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['d'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the standard deviation.
        if len(I)>2 and ((ksig and kw.get("ShowSigma", True)) 
                or kw.get("ShowSigma", False)):
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Form \sigma(CA) = 0.0032
            lbl = (u'\u03C3(%s) = %s' % (coeff, flbl)) % vstd
            # Create the handle.
            h['sig'] = plt.text(0.01, yu, lbl, color=kw_s.get_key('color',1),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the iterative uncertainty.
        if len(vtarg)>0 and kw.get("ShowTarget", True):
            # printf-style format flag
            flbl = kw.get("TargetFormat", "%.4f")
            # Form Target = 0.0032
            lbl = (u'%s = %s' % (ltarg[0], flbl)) % vtarg[0]
            # Create the handle.
            h['t'] = plt.text(0.99, yl, lbl, color=kw_t.get_key('color',0),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['t'].set_family("DejaVu Sans")
            except Exception: pass
        # Output.
        return h
        
# class DBPointSensor

