"""
Data Book Module: :mod:`pyCart.dataBook`
========================================

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

# Finer control of dicts
from .options import odict
# Utilities or advanced statistics
from . import util

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# ---------------------------------
#-->

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
class DataBook(dict):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyCart.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyCart.trajectory.Trajectory`
            The current pyCart trajectory (i.e. run matrix)
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
    :Outputs:
        *DB*: :class:`pyCart.dataBook.DataBook`
            Instance of the pyCart data book class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the root directory.
        self.RootDir = os.getcwd()
        # Save the components
        self.Components = opts.get_DataBookComponents()
        # Save the folder
        self.Dir = opts.get_DataBookDir()
        # Save the trajectory.
        self.x = x.Copy()
        # Save the options.
        self.opts = opts
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                opts.mkdir(fdir)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in self.Components:
            # Initialize the data book.
            self.InitDBComp(comp, x, opts)
        # Initialize targets.
        self.Targets = []
        # Read the targets.
        for targ in opts.get_DataBookTargets():
            # Read the file.
            self.ReadTarget(targ)
        # Initialize line loads
        self.LineLoads = []
        
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions;
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DataBook "
        # Add the number of components.
        lbl += "nComp=%i, " % len(self.Components)
        # Add the number of conditions.
        lbl += "nCase=%i>" % self[self.Components[0]].n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
        
    # Initialize a DBComp object
    def InitDBComp(self, comp, x, opts):
        """Initialize data book for one component
        
        :Call:
            >>> DB.InitDBComp(comp, x, opts)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *x*: :class:`pyCart.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *opts*: :class:`pyCart.options.Options`
                Global pyCart options instance
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
        """
        self[comp] = DBComp(comp, x, opts)
        
    # Function to read targets if necessary
    def ReadTarget(self, targ):
        """Read a data book target if it is not already present
        
        :Call:
            >>> DB.ReadTarget(targ)
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the CAPE data book class
            *targ*: :class:`str`
                Target name
        :Versions:
            * 2015-09-16 ``@ddalle``: First version
        """
        # Try to access the target.
        try:
            self.Targets[targ]
        except Exception:
            # Read the file.
            self.Targets.append(DBTarget(targ, self.x, self.opts))
            
    # Match the databook copy of the trajectory
    def UpdateTrajectory(self):
        """Match the trajectory to the cases in the data book
        
        :Call:
            >>> DB.UpdateTrajectory()
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the CAPE data book class
        :Versions:
            * 2015-05-22 ``@ddalle``: First version
        """
        # Get the first component.
        DBc = self[self.Components[0]]
        # Loop through the fields.
        for k in self.x.keys:
            # Copy the data.
            setattr(self.x, k, DBc[k])
            # Set the text.
            self.x.text[k] = [str(xk) for xk in DBc[k]]
        # Set the number of cases.
        self.x.nCase = DBc.n
                    
    # Get target to use based on target name
    def GetTargetByName(self, targ):
        """Get a target handle by name of the target
        
        :Call:
            >>> DBT = DB.GetTargetByName(targ)
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the pyCart data book class
            *targ*: :class:`str`
                Name of target to find
        :Outputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the pyCart data book target class
        :Versions:
            * 2015-06-04 ``@ddalle``: First version
        """
        # List of target names.
        targs = [DBT.Name for DBT in self.Targets]
        # Check for the target.
        if targ not in targs:
            # Target not found.
            raise ValueError("Target named '%s' not in data book." % targ)
        # Return the target handle.
        return self.Targets[targs.index(targ)]
        
    # Restrict the data book object to points in the trajectory.
    def MatchTrajectory(self):
        """Restrict the data book object to points in the trajectory
        
        :Call:
            >>> DB.MatchTrajectory()
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the CAPE data book class
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Get the first component.
        DBc = self[self.Components[0]]
        # Initialize indices of points to keep.
        I = []
        J = []
        # Loop through trajectory points.
        for i in range(self.x.nCase):
            # Look for a match
            j = DBc.FindMatch(i)
            # Check for no matches.
            if np.isnan(j): continue
            # Match: append to both lists.
            I.append(i)
            J.append(j)
        # Loop through the trajectory keys.
        for k in self.x.keys:
            # Restrict to trajectory points that were found.
            setattr(self.x,k, getattr(self.x,k)[I])
        # Loop through the databook components.
        for comp in self.Components:
            # Loop through fields.
            for k in DBc.keys():
                # Restrict to matched cases.
                self[comp][k] = self[comp][k][J]
            
    # Write the data book
    def Write(self):
        """Write the current data book in Python memory to file
        
        :Call:
            >>> DB.Write()
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the CAPE data book class
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
            * 2015-06-19 ``@ddalle``: New multi-key sort
        """
        # Start from root directory.
        os.chdir(self.RootDir)
        # Get the sort key.
        skey = self.opts.get_SortKey()
        # Sort the data book if there is a key.
        if skey is not None:
            # Sort on either a single key or multiple keys.
            self.Sort(skey)
        # Loop through the components.
        for comp in self.Components:
            # Write individual component.
            self[comp].Write()
            
    # Function to sort data book
    def Sort(self, key=None, I=None):
        """Sort a data book according to either a key or an index
        
        :Call:
            >>> DB.Sort()
            >>> DB.Sort(key)
            >>> DB.Sort(I=None)
        :Inputs:
            *DB*: :class:`cape.dataBook.DataBook`
                Instance of the CAPE data book class
            *key*: :class:`str` or :class:`list` (:class:`str`)
                Name of trajectory key or list of keys on which to sort
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
            * 2015-06-19 ``@ddalle``: New multi-key sort
        """
        # Process inputs.
        if I is None:
            # Use indirect sort on the first component.
            I = self[self.Components[0]].ArgSort(key)
        # Loop through components.
        for comp in self.Components:
            # Apply the DBComp.Sort() method.
            self[comp].Sort(I=I)
            
    
    
    # Get index of target to use based on coefficient name
    def GetTargetIndex(self, ftarg):
        """Get the index of the target to use based on a name
        
        For example, if "UPWT/CAFC" will use the target "UPWT" and the column
        named "CAFC".  If there is no "/" character in the name, the first
        available target is used.
        
        :Call:
            >>> i, c = self.GetTargetIndex(ftarg)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *ftarg*: :class:`str`
                Name of the target and column
        :Outputs:
            *i*: :class:`int`
                Index of the target to use
            *c*: :class:`str`
                Name of the column to use from that target
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Check if there's a slash
        if "/" in ftarg:
            # List of target names.
            TNames = [DBT.Name for DBT in self.Targets]
            # Split.
            ctarg = ftarg.split("/")
            # Find the name,
            i = TNames.index(ctarg[0])
            # Column name
            c = ctarg[1]
        else:
            # Use the first target.
            i = 0
            c = ftarg
        # Output
        return i, c
        
    # Get lists of indices of matches
    def GetTargetMatches(self, ftarg, tol=0.0, tols={}):
        """Get vectors of indices matching targets
        
        :Call:
            >>> I, J = DB.GetTargetMatches(ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys (``0.0`` enforces equality)
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *I*: :class:`numpy.ndarray`
                Array of data book indices with matches
            *J*: :class:`numpy.ndarray`
                Array of target indices for each data book index
        :Versions:
            * 2015-08-30 ``@ddalle``: First version
        """
        # First component.
        DBC = self[self.Components[0]]
        # Initialize indices of targets *J*
        I = []
        J = []
        # Loop through cases.
        for i in np.arange(DBC.n):
            # Get the match.
            j = self.GetTargetMatch(i, ftarg, tol=tol, tols=tols)
            # Check it.
            if np.isnan(j): continue
            # Append it.
            I.append(i)
            J.append(j)
        # Convert to array.
        I = np.array(I)
        J = np.array(J)
        # Output
        return I, J
    
    # Get match for a single index
    def GetTargetMatch(self, i, ftarg, tol=0.0, tols={}):
        """Get index of a target match (if any) for one data book entry
        
        :Call:
            >>> j = DB.GetTargetMatch(i, ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Data book index
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys (``0.0`` enforces equality)
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *j*: :class:`int` or ``np.nan``
                Data book target index
        :Versions:
            * 2015-08-30 ``@ddalle``: First version
        """
        # Check inputs.
        if type(tols).__name__ not in ['dict']:
            raise IOError("Keyword argument *tols* to " +
                ":func:`GetTargetMatches` must be a :class:`dict`.") 
        # First component.
        DBC = self[self.Components[0]]
        # Get the target.
        DBT = self.GetTargetByName(ftarg)
        # Get trajectory keys.
        tkeys = DBT.topts.get_Trajectory()
        # Initialize constraints.
        cons = {}
        # Loop through trajectory keys
        for k in self.x.keys:
            # Get the column name.
            col = tkeys.get(k, k)
            # Continue if column not present.
            if col is None or col not in DBT: continue
            # Get the constraint
            cons[k] = tols.get(k, tol)
            # Set the key.
            tkeys.setdefault(k, col)
        # Initialize match indices
        m = np.arange(DBT.nCase)
        # Loop through tkeys
        for k in tkeys:
            # Get the trajectory key.
            tk = tkeys[k]
            # Make sure there's a key.
            if tk is None: continue
            # Check type.
            if self.x.defns[k]['Value'].startswith('float'):
                # Apply the constraint.
                m = np.intersect1d(m, np.where(
                    np.abs(DBC[k][i] - DBT[tk]) <= cons[k])[0])
            else:
                # Apply equality constraint.
                m = np.intersect1d(m, np.where(DBC[k][i]==DBT[tk])[0])
            # Check if empty; if so exit with no match.
            if len(m) == 0: return np.nan
        # Return the first match.
        return m[0]
    
    # Get match for a single index
    def GetDBMatch(self, h, ftarg, tol=0.0, tols={}):
        """Get index of a target match (if any) for one data book entry
        
        :Call:
            >>> i = DB.GetDBMatch(j, ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *j*: :class:`int` or ``np.nan``
                Data book target index
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys (``0.0`` enforces equality)
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *i*: :class:`int`
                Data book index
        :Versions:
            * 2015-08-30 ``@ddalle``: First version
        """
        # Check inputs.
        if type(tols).__name__ not in ['dict']:
            raise IOError("Keyword argument *tols* to " +
                ":func:`GetTargetMatches` must be a :class:`dict`.") 
        # First component.
        DBC = self[self.Components[0]]
        # Get the target.
        DBT = self.GetTargetByName(ftarg)
        # Get trajectory keys.
        tkeys = DBT.topts.get_Trajectory()
        # Initialize constraints.
        cons = {}
        # Loop through trajectory keys
        for k in self.x.keys:
            # Get the column name.
            col = tkeys.get(k, k)
            # Continue if column not present.
            if col is None or col not in DBT: continue
            # Get the constraint
            cons[k] = tols.get(k, tol)
            # Set the key.
            tkeys.setdefault(k, col)
        # Initialize match indices
        m = np.arange(DBC.n)
        # Loop through tkeys
        for k in tkeys:
            # Get the trajectory key.
            tk = tkeys[k]
            # Make sure there's a key.
            if tk is None: continue
            # Check type.
            if self.x.defns[k]['Value'].startswith('float'):
                # Apply the constraint.
                m = np.intersect1d(m, np.where(
                    np.abs(DBC[k] - DBT[tk][j]) <= cons[k])[0])
            else:
                # Apply equality constraint.
                m = np.intersect1d(m, np.where(DBC[k]==DBT[tk][j])[0])
            # Check if empty; if so exit with no match.
            if len(m) == 0: return np.nan
        # Return the first match.
        return m[0]
            
        
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, comp, coeff, I, **kw):
        """Plot a sweep of one coefficients over several cases
        
        :Call:
            >>> h = DB.PlotCoeff(comp, coeff, I, **kw)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: [ {None} | :class:`str` ]
                Trajectory key for *x* axis (or plot against index if ``None``)
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Legend*: [ {True} | False ]
                Whether or not to use a legend
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *MinMax*: [ {False} | True ]
                Whether to plot minimum and maximum over iterative history
            *LineOptionss*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *FigWidth*: :class:`float`
                Width of figure in inches
            *FigHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Extract the component.
        DBc = self[comp]
        # Get horizontal key.
        xk = kw.get('x')
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
        # Iterative uncertainty options
        qmmx = kw.get('MinMax', 0)
        ksig = kw.get('StDev')
        # Initialize output
        h = {}
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        else:
            # Extract the values.
            xv = DBc[xk][I]
        # Extract the mean values.
        yv = DBc[coeff][I]
        # Initialize label.
        lbl = kw.get('Label', comp)
        # -----------------------
        # Standard Deviation Plot
        # -----------------------
        # Initialize plot options for standard deviation
        kw_s = odict(color='b', lw=0.0,
            facecolor='b', alpha=0.35, zorder=1)
        # Show iterative standard deviation.
        if ksig:
            # Add standard deviation to label.
            lbl = u'%s (\u00B1%s\u03C3)' % (lbl, ksig)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("StDevOptions")):
                # Option.
                o_k = kw["StDevOptions"][k]
                # Override the default option.
                if o_k is not None: kw_s[k] = o_k
            # Get the standard deviation value.
            sv = DBc[coeff+"_std"][I]
            # Plot it.
            h['std'] = plt.fill_between(xv, yv-ksig*sv, yv+ksig*sv, **kw_s)
        # ------------
        # Min/Max Plot
        # ------------
        # Initialize plot options for min/max
        kw_m = odict(color='g', lw=0.0,
            facecolor='g', alpha=0.35, zorder=2)
        # Show min/max options
        if qmmx:
            # Add min/max to label.
            lbl = u'%s (min/max)' % (lbl)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("MinMaxOptions")):
                # Option
                o_k = kw["MinMaxOptions"][k]
                # Override the default option.
                if o_k is not None: kw_m[k] = o_k
            # Get the min and max values.
            ymin = DBc[coeff+"_min"][I]
            ymax = DBc[coeff+"_max"][I]
            # Plot it.
            h['max'] = plt.fill_between(xv, ymin, ymax, **kw_m)
        # ------------
        # Primary Plot
        # ------------
        # Initialize plot options for primary plot
        kw_p = odict(color='k', marker='^', zorder=8, ls='-')
        # Plot options
        for k in util.denone(kw.get("LineOptions")):
            # Option
            o_k = kw["LineOptions"][k]
            # Override the default option.
            if o_k is not None: kw_p[k] = o_k
        # Label
        kw_p.setdefault('label', lbl)
        # Plot it.
        h['line'] = plt.plot(xv, yv, **kw_p)
        # ----------
        # Formatting
        # ----------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to requested ylabel
        if ly and ly != coeff:
            # Combine labels.
            ly = ly + '/' + coeff
        else:
            # Use the coefficient.
            ly = coeff
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(ly)
        # Get limits that include all data (and not extra).
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Make sure data is included.
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
        # Legend.
        if kw.get('Legend', True):
            # Get current limits.
            ymin, ymax = get_ylim(h['ax'], pad=0.05)
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
        # Figure dimensions.
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # Output
        return h
        
        
# class DataBook
        
            
# Function to automatically get inclusive data limits.
def get_ylim(ha, pad=0.05):
    """Calculate appropriate *y*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.
    
    :Call:
        >>> ymin, ymax = get_ylim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check the min and max data
            ymin = min(ymin, min(h.get_ydata()))
            ymax = max(ymax, max(h.get_ydata()))
        elif t == 'PolyCollection':
            # Get the path.
            P = h.get_paths()[0]
            # Get the coordinates.
            ymin = min(ymin, min(P.vertices[:,1]))
            ymax = max(ymax, max(P.vertices[:,1]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount,.
        ymax += pad*ymax
        ymin -= pad*ymin
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv
    
# Function to automatically get inclusive data limits.
def get_xlim(ha, pad=0.05):
    """Calculate appropriate *x*-limits to include all lines in a plot
    
    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.
    
    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Check the min and max data
            xmin = min(xmin, min(h.get_xdata()))
            xmax = max(xmax, max(h.get_xdata()))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount,.
        xmax += pad*xmax
        xmin -= pad*xmin
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv
# DataBook Plot functions


# Data book for an individual component
class DBBase(dict):
    """
    Individual item data book basis class
    
    :Call:
        >>> DBi = DBBase(comp, cntl)
    :Inputs:
        *comp*: :class:`str`
            Name of the component or other item name
        *cntl*: :class:`cape.cntl.Cntl` or derivative
            Code control and settings interface
    :Outputs:
        *DBi*: :class:`cape.dataBook.DBBase`
            An individual item data book
    :Versions:
        * 2014-12-22 ``@ddalle``: First version
        * 2015-12-04 ``@ddalle``: Forked from :class:`DBComp`
    """
    # Initialization method
    def __init__(self, comp, cntl):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the directory.
        fdir = opts.get_DataBookDir()
        
        # Construct the file name.
        fcomp = 'aero_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        
        # Save relevant information
        self.cntl = cntl
        self.comp = comp
        # Save column names.
        self.xCols = cntl.x.keys
        self.fCols = []
        self.iCols = []
        # Counts
        self.nxCol = len(self.xCols)
        self.nfCol = len(self.fCols)
        self.niCol = len(self.iCols)
        # Save the file name.
        self.fname = fname
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
            
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize string
        return "<DBBase, nCase=%i>" % self.n
    # String conversion
    __str__ = __repr__
    
    # Read point sensor data
    def Read(self, fname=None):
        """Read a data book statistics file for a single point sensor
        
        :Call:
            >>> DBP.Read()
            >>> DBP.Read(fname)
        :Inputs:
            *DBP*: :class:`pyCart.pointSensor.DBPointSensor`
                An individual point sensor data book
            *fname*: :class:`str`
                Name of data file to read
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Try to read the file.
        try:
            # Data book delimiter
            delim = self.cntl.opts.get_Delimiter()
            # Initialize column number.
            nxCol = 0
            # Loop through the trajectory keys.
            for k in self.xCols:
                # Get the type.
                t = self.cntl.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Read the column
                self[k] = np.loadtxt(fname,
                    delimiter=delim, dtype=str(t), usecols=[nxCol])
                # Fix single-entry values.
                if self[k].ndim == 0: self[k] = np.array([self[k]])
                # Increase the column number.
                nxCol += 1
            # Read the float columns
            A = np.loadtxt(fname, delimiter=delim, dtype=float,
                usecols=range(nxCol,nxCol+self.nfCol))
            # Read the integer columns
            B = np.loadtxt(fname, delimiter=delim, dtype=int,
                usecols=range(nxCol+self.nfCol,nxCol+self.nfCol+self.niCol))
            # Fix single-entry values.
            if A.ndim == 0:
                A = np.array([A])
                B = np.array([B])
            # Distribute.
            for i in range(self.nfCol):
                self[self.fCols[i]] = A[:,i]
            for i in range(self.niCol):
                self[self.iCols[i]] = B[:,i]
        except Exception:
            # Initialize empty trajectory arrays
            for k in self.xCols:
                # get the type.
                t = self.cntl.x.defns[k].get('Value', 'float')
                # convert type
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Initialize an empty array.
                self[k] = np.array([], dtype=str(t))
            # Initialize float parameters
            for col in self.fCols:
                self[col] = np.array([], dtype=float)
            # Initialize integer counts
            for col in self.iCols:
                self[col] = np.array([], dtype=int)
        # Number of cases
        self.n = len(self[k])
        
    # Output
    def Write(self, fname=None):
        """Write a single point sensor data book summary file
        
        :Call:
            >>> DBi.Write()
            >>> DBi.Write(fname)
        :Inputs:
            *DBi*: :class:`cape.dataBook.DBBase`
                An individual item data book
            *fname*: :class:`str`
                Name of data file to read
        :Versions:
            * 2015-12-04 ``@ddalle``: First version
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
        f.write("# Point sensor statistics for '%s' extracted on %s\n" %
            (self.pt, datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')))
        # Empty line.
        f.write('#\n#')
        # Variable list
        f.write(delim.join(self.xCols) + ' ')
        f.write(delim.join(self.fCols) + ' ')
        f.write(delim.join(self.iCols) + '\n')
        # Loop through database entries
        for i in np.arange(self.n):
            # Write the trajectory values.
            for k in self.xCols:
                f.write('%s%s' % (self[k][i], delim))
            # Write data values
            for k in self.fCols:
                f.write('%s%s' % (self[k][i], delim))
            # Iteration counts
            for k in self.iCols[:-1]:
                f.write('%i%s' % (self[k][i], delim))
            # Last column
            f.write('%i\n' % self[self.iCols[-1])
        # Close the file.
        f.close()
        
    # Function to get sorting indices.
    def ArgSort(self, key=None):
        """Return indices that would sort a data book by a trajectory key
        
        :Call:
            >>> I = DBi.ArgSort(key=None)
        :Inputs:
            *DBi*: :class:`cape.dataBook.DBBase`
                An individual item data book
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
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
                dtk = self.cntl.x.defns[k]['Value']
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
            >>> DBi.Sort()
            >>> DBi.Sort(key)
            >>> DBi.Sort(I=None)
        :Inputs:
            *DBi*: :class:`cape.dataBook.DBBase`
                An individual item data book
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
            if key is None: key = self.cntl.x.keys[0]
            # Use ArgSort to get indices that sort on that key.
            I = self.ArgSort(key)
        # Sort all fields.
        for k in self:
            # Sort it.
            self[k] = self[k][I]
            
    # Find the index of the point in the trajectory.
    def GetTrajectoryIndex(self, j):
        """Find an entry in the run matrix (trajectory)
        
        :Call:
            >>> i = DBi.GetTrajectoryIndex(self, j)
        :Inputs:
            *DBi*: :class:`cape.dataBook.DBBase`
                An individual item data book
            *j*: :class:`int`
                Index of the case from the databook to try match
        :Outputs:
            *i*: :class:`int`
                Trajectory index or ``None``
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Initialize indices (assume all trajectory points match to start).
        i = np.arange(self.cntl.x.nCase)
        # Loop through keys requested for matches.
        for k in self.cntl.x.keys:
            # Get the target value from the data book.
            v = self[k][j]
            # Search for matches.
            try:
                # Filter test criterion.
                ik = np.where(getattr(self.cntl.x,k) == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == getattr(self.cntl.x,k)[-1]):
                    # Add the last element.
                    ik = np.union1d(ik, [self.cntl.x.nCase-1])
                # Restrict to rows that match above.
                i = np.intersect1d(i, ik)
            except Exception:
                return None
        # Output
        try:
            # There should be one match.
            return i[0]
        except Exception:
            # No matches.
            return None
        
    # Find an entry by trajectory variables.
    def FindMatch(self, i):
        """Find an entry by run matrix (trajectory) variables
        
        It is assumed that exact matches can be found.
        
        :Call:
            >>> j = DBi.FindMatch(i)
        :Inputs:
            *DBi*: :class:`cape.dataBook.DBBase`
                An individual item data book
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of index that matches the trajectory case or ``NaN``
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Loop through keys requested for matches.
        for k in self.cntl.x.keys:
            # Get the target value (from the trajectory)
            v = getattr(self.cntl.x,k)[i]
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
# class DBBase


# Data book for an individual component
class DBComp(dict):
    """
    Individual component data book
    
    :Call:
        >>> DBi = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyCart.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
    :Outputs:
        *DBi*: :class:`pyCart.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2014-12-22 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, comp, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the list of columns for that coefficient.
        cols = opts.get_DataBookCols(comp)
        # Get the directory.
        fdir = opts.get_DataBookDir()
        
        # Construct the file name.
        fcomp = 'aero_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        
        # Save relevant information
        self.x = x
        self.opts = opts
        self.comp = comp
        self.cols = cols
        # Save the target translations.
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts.
        self.DataCols = opts.get_DataBookDataCols(comp)
        # Save the file name.
        self.fname = fname
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
            
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize string
        lbl = "<DBComp %s, " % self.comp
        # Add the number of conditions.
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
    
    # Function to read data book files
    def Read(self, fname=None):
        """Read a single data book file or initialize empty arrays
        
        :Call:
            >>> DBc.Read()
            >>> DBc.Read(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.csv' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Try to read the file.
        try:
            # DataBook delimiter
            delim = self.opts.get_Delimiter()
            # Initialize column number
            nCol = 0
            # Loop through trajectory keys.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Read the column
                self[k] = np.loadtxt(fname,
                    delimiter=delim, dtype=str(t), usecols=[nCol])
                # Fix single-entry values.
                if self[k].ndim == 0: self[k] = np.array([self[k]])
                # Increase the column number
                nCol += 1
            # Loop through the data book columns.
            for c in self.cols:
                # Add the column.
                self[c] = np.loadtxt(fname, delimiter=delim, usecols=[nCol])
                # Fix single-entry values.
                if self[c].ndim == 0: self[c] = np.array([self[c]])
                # Increase column number.
                nCol += 1
            # Number of orders of magnitude or residual drop.
            self['nOrders'] = np.loadtxt(fname, 
                delimiter=delim, dtype=float, usecols=[nCol])
            # Last iteration number
            self['nIter'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol+1])
            # Number of iterations used for averaging.
            self['nStats'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol+2])
            # Fix singletons.
            for k in ['nOrders', 'nIter', 'nStats']:
                if self[k].ndim == 0: self[k] = np.array([self[k]])
        except Exception:
            # Initialize empty trajectory arrays.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Convert type.
                if t in ['hex', 'oct', 'octal', 'bin']: t = 'int'
                # Initialize an empty array.
                self[k] = np.array([], dtype=str(t))
            # Initialize the data columns.
            for c in self.cols:
                self[c] = np.array([])
            # Number of orders of magnitude of residual drop
            self['nOrders'] = np.array([], dtype=float)
            # Last iteration number
            self['nIter'] = np.array([], dtype=int)
            # Number of iterations used for averaging.
            self['nStats'] = np.array([], dtype=int)
        # Set the number of points.
        self.n = len(self[c])
        
    # Function to write data book files
    def Write(self, fname=None):
        """Write a single data book file
        
        :Call:
            >>> DBc.Write()
            >>> DBc.Write(fname)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.csv' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Check for a previous old file.
        if os.path.isfile(fname+'.old'):
            # Remove it.
            os.remove(fname+'.old')
        # Check for an existing data file.
        if os.path.isfile(fname):
            # Move it to ".old"
            os.rename(fname, fname+'.old')
        # DataBook delimiter
        delim = self.opts.get_Delimiter()
        # Open the file.
        f = open(fname, 'w')
        # Write the header.
        f.write("# aero data for '%s' extracted on %s\n" %
            (self.comp, datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')))
        # Empty line.
        f.write('#\n')
        # Reference quantities
        f.write('# Reference Area = %.6E\n' %
            self.opts.get_RefArea(self.comp))
        f.write('# Reference Length = %.6E\n' %
            self.opts.get_RefLength(self.comp))
        # Get the nominal MRP.
        xMRP = self.opts.get_RefPoint(self.comp)
        # Write it.
        f.write('# Nominal moment reference point:\n')
        f.write('# XMRP = %.6E\n' % xMRP[0])
        f.write('# YMRP = %.6E\n' % xMRP[1])
        # Check for 3D.
        if len(xMRP) > 2:
            f.write('# ZMRP = %.6E\n' % xMRP[2])
        # Empty line and start of variable list.
        f.write('#\n# ')
        # Loop through trajectory keys.
        for k in self.x.keys:
            # Just write the name.
            f.write(k + delim)
        # Loop through coefficients.
        for c in self.cols:
            # Write the name. (represents the means)
            f.write(c + delim)
        # Write the number of iterations and num used for stats.
        f.write('nOrders%snIter%snStats\n' % (delim, delim))
        # Loop through the database entries.
        for i in np.arange(self.n):
            # Write the trajectory points.
            for k in self.x.keys:
                f.write('%s%s' % (self[k][i], delim))
            # Write values.
            for c in self.cols:
                f.write('%.8E%s' % (self[c][i], delim))
            # Write the residual
            f.write('%.4f%s' % (self['nOrders'][i], delim))
            # Write number of iterations.
            f.write('%i%s%i\n' % (self['nIter'][i], delim, self['nStats'][i]))
        # Close the file.
        f.close()
        
    # Function to get sorting indices.
    def ArgSort(self, key=None):
        """Return indices that would sort a data book by a trajectory key
        
        :Call:
            >>> I = DBc.ArgSort(key=None)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the data book component
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
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
                dtk = self.x.defns[k]['Value']
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
            >>> DBc.Sort()
            >>> DBc.Sort(key)
            >>> DBc.Sort(I=None)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the pyCart data book component
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
            
    # Find the index of the point in the trajectory.
    def GetTrajectoryIndex(self, j):
        """Find an entry in the run matrix (trajectory)
        
        :Call:
            >>> i = DBc.GetTrajectoryIndex(self, j)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the pyCart data book component
            *j*: :class:`int`
                Index of the case from the databook to try match
        :Outputs:
            *i*: :class:`int`
                Trajectory index or ``None``
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Initialize indices (assume all trajectory points match to start).
        i = np.arange(self.x.nCase)
        # Loop through keys requested for matches.
        for k in self.x.keys:
            # Get the target value from the data book.
            v = self[k][j]
            # Search for matches.
            try:
                # Filter test criterion.
                ik = np.where(getattr(self.x,k) == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == getattr(self.x,k)[-1]):
                    # Add the last element.
                    ik = np.union1d(ik, [self.x.nCase-1])
                # Restrict to rows that match above.
                i = np.intersect1d(i, ik)
            except Exception:
                return None
        # Output
        try:
            # There should be one match.
            return i[0]
        except Exception:
            # No matches.
            return None
        
    # Find an entry by trajectory variables.
    def FindMatch(self, i):
        """Find an entry by run matrix (trajectory) variables
        
        It is assumed that exact matches can be found.
        
        :Call:
            >>> j = DBc.FindMatch(i)
        :Inputs:
            *DBc*: :class:`cape.dataBook.DBComp`
                Instance of the CAPE data book component
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of index that matches the trajectory case or ``NaN``
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Loop through keys requested for matches.
        for k in self.x.keys:
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
# class DBComp


# Data book target instance
class DBTarget(dict):
    """
    Class to handle data from data book target files.  There are more
    constraints on target files than the files that data book creates, and raw
    data books created by pyCart are not valid target files.
    
    :Call:
        >>> DBT = pyCart.dataBook.DBTarget(targ, x, opts)
    :Inputs:
        *targ*: :class:`pyCart.options.DataBook.DBTarget`
            Instance of a target source options interface
        *x*: :class:`pyCart.trajectory.Trajectory`
            Run matrix interface
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`cape.dataBook.DBTarget`
            Instance of the CAPE data book target class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, targ, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
            * 2015-06-03 ``@ddalle``: Added trajectory, split into methods
        """
        # Save the target options
        self.opts = opts
        self.topts = targ
        # Save the trajectory.
        self.x = x.Copy()
        
        # Read the data
        self.ReadData()
        # Process the columns.
        self.ProcessColumns()
        # Make the trajectory data match the available list of points.
        self.UpdateTrajectory()
    
    # Read the data
    def ReadData(self):
        """Read data file according to stored options
        
        :Call:
            >>> DBT.ReadData()
        :Inputs:
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
        """
        # Source file
        fname = self.topts.get_TargetFile()
        # Name of this target.
        tname = self.topts.get_TargetName()
        # Check for the file.
        if not os.path.isfile(fname):
            raise IOError(
                "Target source file '%s' could not be found." % fname)
        # Save the name.
        self.Name = tname
        # Delimiter
        delim = self.topts.get_Delimiter()
        # Comment character
        comchar = self.topts.get_CommentChar()
        # Open the file again.
        f = open(fname)
        # Loop until finding a line that doesn't begin with comment char.
        line = comchar
        nskip = -1
        while line.strip().startswith(comchar):
            # Save the old line.
            headers = line
            # Read the next line
            line = f.readline()
            nskip += 1
        # Close the file.
        f.close()
        # Translate into headers
        self.headers = headers.lstrip('#').strip().split(delim)
        # Save number of points.
        self.n = len(self.headers)

        # Read it.
        try:
            # Read the target all at once.
            self.ReadAllData(fname, delimiter=delim, skiprows=nskip)
        except Exception:
            # Read the data by columns.
            self.ReadDataByColumn(fname, delimiter=delim, skiprows=nskip)

    # Read the data file all at once.
    def ReadAllData(self, fname, delimiter=",", skiprows=0):
        """Read target data file all at once

        :Call:
            >>> DBT.ReadAllData(fname, delimiter=",", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: First version
        """
        # Read the data.
        self.data = np.loadtxt(fname, delimiter=delimiter,
            skiprows=skiprows, dtype=float).transpose()
        # Save the number of cases.
        self.nCase = len(self.data[0])

    # Read data one column at a time
    def ReadDataByColumn(self, fname, delimiter=",", skiprows=0):
        """Read target data one column at a time
        
        :Call:
            >>> DBT.ReadDataByColumn(fname, delimiter=",", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: First version
        """
        # Initialize data.
        self.data = []
        # Loop through columns.
        for i in range(self.n):
            # Try reading as a float second.
            try:
                self.data.append(np.loadtxt(fname, delimiter=delimiter,
                    skiprows=skiprows, dtype=float, usecols=(i,)))
                continue
            except Exception:
                pass
            # Try reading as a string last.
            self.data.append(np.loadtxt(fname, delimiter=delimiter,
                skiprows=skiprows, dtype=str, usecols=(i,)))
        # Number of cases
        self.nCase = len(self.data[0])

    
    # Read the columns and split into useful dict.
    def ProcessColumns(self):
        """Process data columns and split into dictionary keys
        
        :Call:
            >>> DBT.ProcessColumns()
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
        """
        # Initialize data fields.
        cols = []
        # Names of columns corresponding to trajectory keys.
        tkeys = self.topts.get_Trajectory()
        # Loop through trajectory fields.
        for k in self.x.keys:
            # Get field name.
            col = tkeys.get(k, k)
            # Check for manually turned-off trajectory.
            if col is None:
                # Manually turned off.
                continue
            elif col not in self.headers:
                # Not present in the file.
                continue
            # Append the key.
            cols.append(col)
        # Initialize translations for force/moment coefficients
        ckeys = {}
        # List of potential components.
        tcomps = self.topts.get_TargetComponents()
        # Check for default.
        if tcomps is None:
            # Use all components.
            tcomps = self.opts.get_DataBookComponents()
        # Process the required fields.
        for comp in tcomps:
            # Initialize translations for this component.
            ckeys[comp] = {}
            # Get targets for this component.
            ctargs = self.opts.get_CompTargets(comp)
            # Loop through the possible force/moment coefficients.
            for c in ['CA','CY','CN','CLL','CLM','CLN']:
                # Get the translated name
                ctarg = ctargs.get(c, c)
                # Get the target source for this entry.
                if '/' not in ctarg:
                    # Only one target source; assume it's this one.
                    ti = self.Name
                    fi = ctarg
                else:
                    # Read the target name.
                    ti = ctarg.split('/')[0]
                    # Name of the column
                    fi = ctarg.split('/')[1]
                # Check if the target is from this target source.
                if ti != self.Name: continue
                # Check if the column is present in the headers.
                if fi not in self.headers:
                    # Check for default.
                    if ctarg in ctargs:
                        # Manually specified and not recognized: error
                        raise KeyError("There is no field '%s' in file '%s'."
                            % (fi, self.topts.get_TargetFile()))
                    else:
                        # Autoselected name but not in the file.
                        continue
                # Add the field if necessary.
                if fi in cols:
                    raise IOError(
                        "Column '%s' of file '%s' used more than once."
                        % (fi, self.topts.get_TargetFile()))
                # Add the column.
                cols.append(fi)
                # Add to the translation dictionary.
                ckeys[comp][c] = fi
        # Extract the data into a dict with a key for each relevant column.
        for col in cols:
            # Find it and save it as a key.
            self[col] = self.data[self.headers.index(col)]
        # Save the data keys translations.
        self.ckeys = ckeys
        
    # Match the databook copy of the trajectory
    def UpdateTrajectory(self):
        """Match the trajectory to the cases in the data book
        
        :Call:
            >>> DBT.UpdateTrajectory()
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Get trajectory key specifications.
        tkeys = self.topts.get_Trajectory()
        # Loop through the trajectory keys.
        for k in self.x.keys:
            # Get the column name in the target.
            tk = tkeys.get(k, k)
            # Set the value if it's a default.
            tkeys.setdefault(k, tk)
            # Check for ``None``
            if (tk is None) or (tk not in self):
                # Use NaN as the value.
                setattr(self.x,k, np.nan*np.ones(self.n))
                # Set the value.
                tkeys[k] = None
                continue
            # Update the trajectory values to match those of the trajectory.
            setattr(self.x,k, self[tk])
            # Set the text.
            self.x.text[k] = [str(xk) for xk in self[tk]]
        # Save the key translations.
        self.xkeys = tkeys
        # Set the number of cases in the "trajectory."
        self.x.nCase = self.nCase
        
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, comp, coeff, I, **kw):
        """Plot a sweep of one coefficient over several cases
        
        :Call:
            >>> h = DBT.PlotCoeff(comp, coeff, I, **kw)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: [ {None} | :class:`str` ]
                Trajectory key for *x* axis (or plot against index if ``None``)
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Legend*: [ {True} | False ]
                Whether or not to use a legend
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *MinMax*: [ {False} | True ]
                Whether to plot minimum and maximum over iterative history
            *LineOptionss*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *FigWidth*: :class:`float`
                Width of figure in inches
            *FigHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: First version
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Get horizontal key.
        xk = kw.get('x')
        # Figure dimensions
        fw = kw.get('FigWidth', 6)
        fh = kw.get('FigHeight', 4.5)
        # Iterative uncertainty options
        qmmx = kw.get('MinMax', 0)
        ksig = kw.get('StDev')
        # Initialize output
        h = {}
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        else:
            # Check if the value is present.
            if xk not in self.xkeys: return
            # Extract the values.
            xv = self[self.xkeys[xk]][I]
        # Check if the coefficient is in the target data.
        if (comp not in self.ckeys) or (coeff not in self.ckeys[comp]):
            # No data.
            return
        # Extract the mean values.
        yv = self[self.ckeys[comp][coeff]][I]
        # Initialize label.
        lbl = kw.get('Label', '%s/%s' % (self.Name, comp))
        # -----------------------
        # Standard Deviation Plot
        # -----------------------
        # Initialize plot options for standard deviation.
        kw_s = odict(color='c', lw=0.0,
            facecolor='c', alpha=0.35, zorder=1)
        # Show iterative standard deviation.
        if ksig:
            # Add standard deviation to label.
            lbl = u'%s (\u00B1%s\u03C3)' % (lbl, ksig)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("StDevOptions")):
                # Option.
                o_k = kw["StDevOptions"][k]
                # Override the default option.
                if o_k is not None: kw_s[k] = o_k
            # Get the standard deviation value.
            sv = DBc[coeff+"_std"][I]
            # Plot it.
            h['std'] = plt.fill_between(xv, yv-ksig*sv, yv+ksig*sv, **kw_s)
        # ------------
        # Min/Max Plot
        # ------------
        # Initialize plot options for min/max
        kw_m = odict(color='m', lw=0.0,
            facecolor='m', alpha=0.35, zorder=2)
        # Show min/max options
        if qmmx:
            # Add min/max to label.
            lbl = u'%s (min/max)' % (lbl)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("MinMaxOptions")):
                # Option
                o_k = kw["MinMaxOptions"][k]
                # Override the default option.
                if o_k is not None: kw_m[k] = o_k
            # Get the min and max values.
            ymin = DBc[coeff+"_min"][I]
            ymax = DBc[coeff+"_max"][I]
            # Plot it.
            h['max'] = plt.fill_between(xv, ymin, ymax, **kw_m)
        # ------------
        # Primary Plot
        # ------------
        # Initialize plot options for primary plot
        kw_p = odict(color='r', marker='^', zorder=7, ls='-')
        # Plot options
        for k in util.denone(kw.get("LineOptions")):
            # Option
            o_k = kw["LineOptions"][k]
            # Override the default option.
            if o_k is not None: kw_p[k] = o_k
        # Label
        kw_p.setdefault('label', lbl)
        # Plot it.
        h['line'] = plt.plot(xv, yv, **kw_p)
        # ----------
        # Formatting
        # ----------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to requested ylabel
        if ly and ly != coeff:
            # Combine labels.
            ly = ly + '/' + coeff
        else:
            # Use the coefficient.
            ly = coeff
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(ly)
        # Get limits to include all data.
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Make sure data is included.
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
        # Legend.
        if kw.get('Legend', True):
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
        # Figure dimensions.
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try: plt.tight_layout()
        except Exception: pass
        # Output
        return h
        
    # Find an entry by trajectory variables.
    def FindMatch(self, x, i):
        """Find an entry by run matrix (trajectory) variables
        
        Cases will be considered matches by comparing variables specified in 
        the *DataBook* section of :file:`pyCart.json` as cases to compare
        against.  Suppose that the control file contains the following.
        
        .. code-block:: python
        
            "DataBook": {
                "Targets": {
                    "Name": "Experiment",
                    "File": "WT.dat",
                    "Trajectory": {"alpha": "ALPHA", "Mach": "MACH"}
                    "Tolerances": {
                        "alpha": 0.05,
                        "Mach": 0.01
                    }
                }
            }
        
        Then any entry in the data book target that matches the Mach number
        within 0.01 (using a column labeled *MACH*) and alpha to within 0.05 is
        considered a match.  If there are more trajectory variables, they are
        not used for this filtering of matches.
        
        :Call:
            >>> j = DBT.FindMatch(x, i)
        :Inputs:
            *DBT*: :class:`cape.dataBook.DBTarget`
                Instance of the CAPE data book target data carrier
            *x*: :class:`pyCart.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices that match the trajectory within tolerances
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Get the trajectory key translations.   This determines which keys to
        # filter and what those keys are called in the source file.
        tkeys = self.topts.get_Trajectory()
        # Loop through keys requested for matches.
        for k in tkeys:
            # Get the tolerance.
            tol = self.topts.get_Tol(k)
            # Get the target value (from the trajectory)
            v = getattr(x,k)[i]
            # Get the name of the column according to the source file.
            c = tkeys[k]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(np.abs(self[c] - v) <= tol)[0]
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                pass
        # Output
        return j
# class DBTarget


# Individual component force and moment
class CaseFM(object):
    """
    This class contains methods for reading data about an the histroy of an
    individual component for a single case.  The list of available components
    comes from a :file:`loadsCC.dat` file if one exists.
    
    :Call:
        >>> FM = cape.dataBook.CaseFM(C, MRP=None, A=None)
    :Inputs:
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *FM*: :class:`cape.aero.FM`
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
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.FM`
    """
    # Initialization method
    def __init__(self, comp):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2015-10-16 ``@ddalle``: Trivial generic version
        """
        # Save the component name.
        self.comp = comp
        # Empty iterations
        self.i = np.array([])
            
    # Function to display contents
    def __repr__(self):
        """Representation method
        
        Returns the following format, with ``'entire'`` replaced with the
        component name, *FM.comp*
        
            * ``'<dataBook.CaseFM('entire', i=100)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2015-10-16 ``@ddalle``: Generic version
        """
        return "<dataBook.CaseFM('%s', i=%i)>" % (self.comp, len(self.i))
    # String method
    __str__ = __repr__
    
    # Method to add data to instance
    def AddData(self, A):
        """Add iterative force and/or moment history for a component
        
        :Call:
            >>> FM.AddData(A)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
                Matrix of forces and/or moments at *N* iterations
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2015-10-16 ``@ddalle``: Version 2.0, complete rewrite
        """
        # Save the values.
        for k in range(len(self.cols)):
            # Set the values from column *k* of the data
            setattr(self,self.cols[k], A[:,k])
    
    # Transform force or moment reference frame
    def TransformFM(self, topts, x, i):
        """Transform a force and moment history
        
        Available transformations and their required parameters are listed
        below.
        
            * "Euler321": "psi", "theta", "phi"
            
        Trajectory variables are used to specify values to use for the
        transformation variables.  For example,
        
            .. code-block:: python
            
                topts = {"Type": "Euler321",
                    "psi": "Psi", "theta": "Theta", "phi": "Phi"}
        
        will cause this function to perform a reverse Euler 3-2-1 transformation
        using *x.Psi[i]*, *x.Theta[i]*, and *x.Phi[i]* as the angles.
        
        :Call:
            >>> FM.TransformFM(topts, x, i)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *topts*: :class:`dict`
                Dictionary of options for the transformation
            *x*: :class:`pyCart.trajectory.Trajectory`
                The run matrix used for this analysis
            *i*: :class:`int`
                The index of the case to transform in the current run matrix
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Check it.
        if ttype in ["Euler321"]:
            # Get the angle variable names.
            # Use same as default in case it's obvious what they should be.
            kph = topts.get('phi', 'phi')
            kth = topts.get('theta', 'theta')
            kps = topts.get('psi', 'psi')
            # Extract roll
            if kph.startswith('-'):
                # Negative roll angle.
                phi = -getattr(x,kph[1:])[i]*deg
            else:
                # Positive roll
                phi = getattr(x,kph)[i]*deg
            # Extract pitch
            if kth.startswith('-'):
                # Negative pitch
                theta = -getattr(x,kth[1:])[i]*deg
            else:
                # Positive pitch
                theta = getattr(x,kth)[i]*deg
            # Extract yaw
            if kps.startswith('-'):
                # Negative yaw
                psi = -getattr(x,kps[1:])[i]*deg
            else:
                # Positive pitch
                psi = getattr(x,kps)[i]*deg
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
            R = np.dot(R1, np.dot(R2, R3))
            # Force transformations
            if 'CY' in self.coeffs:
                # Assemble forces.
                Fc = np.vstack((self.CA, self.CY, self.CN))
                # Transform.
                Fb = np.dot(R, Fc)
                # Extract (is this necessary?)
                self.CA = Fb[0]
                self.CY = Fb[1]
                self.CN = Fb[2]
            elif 'CN' in self.coeffs:
                # Use zeros for side force.
                CY = np.zeros_like(self.CN)
                # Assemble forces.
                Fc = np.vstack((self.CA, CY, self.CN))
                # Transform.
                Fb = np.dot(R, Fc)
                # Extract
                self.CA = Fb[0]
                self.CN = Fb[2]
            # Moment transformations
            if 'CLN' in self.coeffs:
                # Assemble moment vector.
                Mc = np.vstack((self.CLL, self.CLM, self.CLN))
                # Transform.
                Mb = np.dot(R, Mc)
                # Extract.
                self.CLL = Mb[0]
                self.CLM = Mb[1]
                self.CLN = Mb[2]
            elif 'CLM' in self.coeffs:
                # Use zeros for roll and yaw moment.
                CLL = np.zeros_like(self.CLM)
                CLN = np.zeros_like(self.CLN)
                # Assemble moment vector.
                Mc = np.vstack((CLL, self.CLM, CLN))
                # Transform.
                Mb = np.dot(R, Mc)
                # Extract.
                self.CLM = Mb[1]
                
        elif ttype in ["ScaleCoeffs"]:
            # Loop through coefficients.
            for c in topts:
                # Check if it's an available coefficient.
                if c not in self.coeffs: continue
                # Get the value.
                k = topts[c]
                # Check if it's a number.
                if type(k).__name__ not in ["float", "int"]:
                    # Assume they meant to flip it.
                    k = -1.0
                # Scale.
                setattr(self,c, k*getattr(self,c))
            
        else:
            raise IOError(
                "Transformation type '%s' is not recognized." % ttype)
        
    # Method to shift the MRC
    def ShiftMRP(self, Lref, x, xi=None):
        """Shift the moment reference point
        
        :Call:
            >>> FM.ShiftMRP(Lref, x, xi=None)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *Lref*: :class:`float`
                Reference length
            *x*: :class:`list` (:class:`float`)
                Target moment reference point
            *xi*: :class:`list` (:class:`float`)
                Current moment reference point (default: *self.MRP*)
        :Versions:
            * 2015-03-02 ``@ddalle``: First version
        """
        # Check for moments.
        if ('CA' not in self.coeffs) or ('CLM' not in self.coeffs):
            # Not a force/moment history
            return
        # Rolling moment: side force
        if ('CLL' in self.coeffs) and ('CY' in self.coeffs):
            self.CLL -= (xi[2]-x[2])/Lref*self.CY
        # Rolling moment: normal force
        if ('CLL' in self.coeffs) and ('CN' in self.coeffs):
            self.CLL += (xi[1]-x[1])/Lref*self.CN
        # Pitching moment: normal force
        if ('CLM' in self.coeffs) and ('CN' in self.coeffs):
            self.CLM -= (xi[0]-x[0])/Lref*self.CN
        # Pitching moment: axial force
        if ('CLM' in self.coeffs) and ('CA' in self.coeffs):
            self.CLM += (xi[2]-x[2])/Lref*self.CA
        # Yawing moment: axial force
        if ('CLN' in self.coeffs) and ('CA' in self.coeffs):
            self.CLN += (x[1]-xi[1])/Lref*self.CA
        # Yawing moment: axial force
        if ('CLN' in self.coeffs) and ('CY' in self.coeffs):
            self.CLN += (x[0]-xi[0])/Lref*self.CY
    
    # Function to get index of a certain iteration number
    def GetIterationIndex(self, i):
        """Return index of a particular iteration in *FM.i*
        
        If the iteration *i* is not present in the history, the index of the
        last available iteration less than or equal to *i* is returned.
        
        :Call:
            >>> j = FM.GetIterationIndex(i)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *i*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Index of last iteration in *FM.i* less than or equal to *i*
        :Versions:
            * 2015-03-06 ``@ddalle``: First version
        """
        # Check for *i* less than first iteration.
        if i < self.i[0]: return 0
        # Find the index.
        j = np.where(self.i <= i)[0][-1]
        # Output
        return j
        
        
    # Method to get averages and standard deviations
    def GetStatsN(self, nStats=100, nLast=None):
        """Get mean, min, max, and standard deviation for all coefficients
        
        :Call:
            >>> s = FM.GetStatsN(nStats, nFirst=None, nLast=None)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *nStats*: :class:`int`
                Number of iterations in window to use for statistics
            *nLast*: :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict` (:class:`float`)
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
            * 2015-02-28 ``@ddalle``: Renamed from :func:`GetStats`
            * 2015-03-04 ``@ddalle``: Added last iteration capability
        """
        # Last iteration to use.
        if nLast:
            # Attempt to use requested iter.
            if nLast < self.i.size:
                # Using an earlier iter; make sure to use one in the hist.
                jLast = self.GetIterationIndex(nLast)
                # Find the iterations that are less than i.
                iLast = self.i[jLast]
            else:
                # Use the last iteration.
                iLast = self.i.size
        else:
            # Just use the last iteration
            iLast = self.i.size
        # Default values.
        if (nStats is None) or (nStats < 2):
            # Use last iteration
            i0 = iLast - 1
        else:
           # Process min indices for plotting and averaging.
            i0 = max(0, iLast-nStats)
        # Initialize output.
        s = {}
        # Loop through coefficients.
        for c in self.coeffs:
            # Get the values
            F = getattr(self, c)
            # Save the mean value.
            s[c] = np.mean(F[i0:])
            # Check for statistics.
            if (nStats is not None) or (nStats < 2):
                # Save the statistics.
                s[c+'_min'] = np.min(F[i0:iLast])
                s[c+'_max'] = np.max(F[i0:iLast])
                s[c+'_std'] = np.std(F[i0:iLast])
                s[c+'_err'] = util.SigmaMean(F[i0:iLast])
        # Output
        return s
            
    # Method to get averages and standard deviations
    def GetStats(self, nStats=100, nMax=None, nLast=None):
        """Get mean, min, max, and standard deviation for all coefficients
        
        :Call:
            >>> s = FM.GetStats(nStats, nMax=None, nLast=None)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the force and moment class
            *nStats*: :class:`int`
                Minimum number of iterations in window to use for statistics
            *nMax*: :class:`int`
                Maximum number of iterations to use for statistics
            *nLast*: :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict` (:class:`float`)
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2015-02-28 ``@ddalle``: First version
            * 2015-03-04 ``@ddalle``: Added last iteration capability
        """
        # Make sure the number of iterations used is an integer.
        if not nStats: nStats = 1
        # Process list of candidate numbers of iterations for statistics.
        if nMax and (nStats > 1) and (nMax >= 1.5*nStats):
            # Nontrivial list of candidates
            # Multiples of *nStats*
            N = [k*nStats for k in range(1, int(nMax/nStats)+1)]
            # Check if *nMax* should also be considered.
            if nMax >= 1.5*N[-1]:
                # Add *nMax*
                N.append(nMax)
        else:
            # Only one candidate.
            N = [nStats]
        # Initialize error as infinity.
        e = np.inf;
        # Loop through list of candidate iteration counts
        for n in N:
            # Get the statistics.
            sn = self.GetStatsN(n, nLast=nLast)
            # Save the number of iterations used.
            sn['nStats'] = n
            # If there is only one candidate, return it.
            if len(N) == 1: return sn
            # Calculate the composite error.
            en = np.sqrt(np.sum([sn[c+'_err']**2 for c in self.coeffs]))
            # Calibrate to slightly favor less iterations
            en = en * (0.75 + 0.25*np.sqrt(n)/np.sqrt(N[0]))
            # Check if this error is an improvement.
            if (n == min(N)) or (en < e):
                # Select these statistics, and update the best scaled error.
                s = sn
                e = en
        # Output.
        return s
    
    
    
    # Plot iterative force/moment history
    def PlotCoeff(self, c, n=None, nAvg=100, **kw):
        """Plot a single coefficient history
        
        :Call:
            >>> h = FM.PlotCoeff(comp, c, n=1000, nAvg=100, **kw)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the component force history class
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d*: :class:`float`
                Delta in the coefficient to show expected range
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *nFirst*: :class:`int`
                First iteration to plot
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Transferred to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Extract the data.
        C = getattr(self, c)
        # Process inputs.
        nLast = kw.get('nLast')
        nFirst = kw.get('nFirst')
        # Iterative uncertainty options
        dc = kw.get("d", 0.0)
        ksig = kw.get("k", 0.0)
        uerr = kw.get("u", 0.0)
        # Other plot options
        fw = kw.get('FigWidth')
        fh = kw.get('FigHeight')
        # Get statistics
        s = self.GetStatsN(nAvg, nLast=nLast) 
        # ---------
        # Last Iter 
        # ---------
        # Most likely last iteration
        iB = self.i[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                # Find the iterations that are less than i.
                jB = self.GetIterationIndex(nLast)
                iB = self.i[jB]
        # Get the index of *iB* in *self.i*.
        jB = self.GetIterationIndex(iB)
        # ----------
        # First Iter
        # ----------
        # Default number of iterations: all
        if n is None: n = len(self.i)
        # Get the starting iteration number to use.
        i0 = max(0, iB-n, nFirst) + 1
        # Make sure *iA* is in *self.i* and get the index.
        j0 = self.GetIterationIndex(i0)
        # Reselect *iA* in case initial value was not in *self.i*.
        i0 = self.i[j0]
        # --------------
        # Averaging Iter
        # --------------
        # Get the first iteration to use in averaging.
        iA = max(0, iB-nAvg) + 1
        # Make sure *iV* is in *self.i* and get the index.
        jA = self.GetIterationIndex(iA)
        # Reselect *iV* in case initial value was not in *self.i*.
        iA = self.i[jA]
        # -----------------------
        # Standard deviation plot
        # -----------------------
        # Initialize dictionary of handles.
        h = {}
        # Shortcut for the mean
        cAvg = s[c]
        # Initialize plot options for standard deviation
        kw_s = odict(color='b', lw=0.0,
            facecolor="b", alpha=0.35, zorder=1)
        # Show iterative n*standard deviation
        if ksig and nAvg>2:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("StDevOptions", {})):
                # Ignore linestyle and ls
                if k in ['ls', 'linestyle']: continue
                # Override the default option.
                if kw["StDevOptions"][k] is not None:
                    kw_s[k] = kw["StDevOptions"][k]
            # Limits
            cMin = cAvg - ksig*s[c+"_std"]
            cMax = cAvg + ksig*s[c+"_std"]
            # Plot the target window boundaries.
            h['std'] = plt.fill_between([iA,iB], [cMin]*2, [cMax]*2, **kw_s)
        # --------------------------
        # Iterative uncertainty plot
        # --------------------------
        kw_u = odict(color='g', ls="none",
            facecolor="g", alpha=0.4, zorder=2)
        # Show iterative n*standard deviation
        if uerr and nAvg>2:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("ErrPltOptions", {})):
                # Ignore linestyle and ls
                if k in ['ls', 'linestyle']: continue
                # Override the default option.
                if kw["ErrPltOptions"][k] is not None:
                    kw_u[k] = kw["ErrPltOptions"][k]
            # Limits
            cMin = cAvg - uerr*s[c+"_err"]
            cMax = cAvg + uerr*s[c+"_err"]
            # Plot the target window boundaries.
            h['err'] = plt.fill_between([iA,iB], [cMin]*2, [cMax]*2, **kw_u)
        # ---------
        # Mean plot
        # ---------
        # Initialize plot options for mean.
        kw_m = odict(color=kw.get("color", "0.1"),
            ls=[":", "-"], lw=1.0, zorder=8)
        # Extract plot options from kwargs
        for k in util.denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Turn into two groups.
        kw0 = {}; kw1 = {}
        for k in kw_m:
            kw0[k] = kw_m.get_key(k, 0)
            kw1[k] = kw_m.get_key(k, 1)
        # Plot the mean.
        h['mean'] = (
            plt.plot([i0,iA], [cAvg, cAvg], **kw0) + 
            plt.plot([iA,iB], [cAvg, cAvg], **kw1))
        # ----------
        # Delta plot
        # ----------
        # Initialize options for delta.
        kw_d = odict(color="r", ls="--", lw=0.8, zorder=4)
        # Calculate range of interest.
        if dc:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("DeltaOptions", {})):
                # Override the default option.
                if kw["DeltaOptions"][k] is not None:
                    kw_d[k] = kw["DeltaOptions"][k]
            # Turn into two groups.
            kw0 = {}; kw1 = {}
            for k in kw_m:
                kw0[k] = kw_d.get_key(k, 0)
                kw1[k] = kw_d.get_key(k, 1)
            # Limits
            cMin = cAvg-dc
            cMax = cAvg+dc
            # Plot the target window boundaries.
            h['min'] = (
                plt.plot([i0,iA], [cMin,cMin], **kw0) +
                plt.plot([iA,iB], [cMin,cMin], **kw1))
            h['max'] = (
                plt.plot([i0,iA], [cMax,cMax], **kw0) +
                plt.plot([iA,iB], [cMax,cMax], **kw1))
        # ------------
        # Primary plot
        # ------------
        # Initialize primary plot options.
        kw_p = odict(color=kw.get("color","k"), ls="-", lw=1.5, zorder=7)
        # Extract plot options from kwargs
        for k in util.denone(kw.get("LineOptions", {})):
            # Override the default option.
            if kw["LineOptions"][k] is not None:
                kw_p[k] = kw["LineOptions"][k]
        # Plot the coefficient.
        h[c] = plt.plot(self.i[j0:], C[j0:], **kw_p)
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to the requested ylabel
        if ly and ly != c:
            # Combine labels
            ly = ly + '/' + c
        else:
            # Use the coefficient
            ly = c
        # Labels.
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel(ly)
        # Set the xlimits.
        h['ax'].set_xlim((i0, iB+25))
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
            # Form: CA = 0.0204
            lbl = u'%s = %.4f' % (c, cAvg)
            # Create the handle.
            h['mu'] = plt.text(0.99, yu, lbl, color=kw_p['color'],
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['mu'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the deviation.
        if dc and kw.get("ShowDelta", True):
            # Form: \DeltaCA = 0.0050
            lbl = u'\u0394%s = %.4f' % (c, dc)
            # Create the handle.
            h['d'] = plt.text(0.99, yl, lbl, color=kw_d.get_key('color',1),
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['d'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the standard deviation.
        if ksig and nAvg>2 and kw.get("ShowSigma", True):
            # Form \sigma(CA) = 0.0032
            lbl = u'\u03C3(%s) = %.4f' % (c, ksig*s[c+'_std'])
            # Create the handle.
            h['sig'] = plt.text(0.01, yu, lbl, color=kw_s.get_key('color',1),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Make a label for the iterative uncertainty.
        if uerr and nAvg>2 and kw.get("ShowEpsilon", True):
            # Form \sigma(CA) = 0.0032
            lbl = u'\u0395(%s) = %.4f' % (c, ueps*s[c+'_err'])
            # Create the handle.
            h['eps'] = plt.text(0.01, yl, lbl, color=kw_u.get_key('color',1),
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font.
            try: h['sig'].set_family("DejaVu Sans")
            except Exception: pass
        # Output.
        return h
    
    # Plot coefficient histogram
    def PlotCoeffHist(self, c, nAvg=100, nBin=20, nLast=None, **kw):
        """Plot a single coefficient histogram
        
        :Call:
            >>> h = FM.PlotCoeffHist(comp, c, n=1000, nAvg=100, **kw)
        :Inputs:
            *FM*: :class:`cape.dataBook.CaseFM`
                Instance of the component force history class
            *comp*: :class:`str`
                Name of component to plot
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *nBin*: :class:`int`
                Number of bins to plot
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-02-15 ``@ddalle``: First version
            * 2015-03-06 ``@ddalle``: Added *nLast* and fixed documentation
            * 2015-03-06 ``@ddalle``: Copied to :class:`CaseFM`
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Extract the data.
        C = getattr(self, c)
        # Process other options
        fw = kw.get('FigWidth')
        fh = kw.get('FigHeight')
        # ---------
        # Last Iter 
        # ---------
        # Most likely last iteration
        iB = self.i[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                # Find the iterations that are less than i.
                jB = self.GetIterationIndex(nLast)
                iB = self.i[jB]
        # Get the index of *iB* in *FM.i*.
        jB = self.GetIterationIndex(iB)
        # --------------
        # Averaging Iter
        # --------------
        # Get the first iteration to use in averaging.
        iA = max(0, iB-nAvg) + 1
        # Make sure *iV* is in *FM.i* and get the index.
        jA = self.GetIterationIndex(iA)
        # Reselect *iV* in case initial value was not in *FM.i*.
        iA = self.i[jA]
        # --------
        # Plotting
        # --------
        # Calculate statistics.
        cAvg = np.mean(C[jA:jB+1])
        cStd = np.std(C[jA:jB+1])
        cErr = util.SigmaMean(C[jA:jB+1])
        # Calculate # of independent samples
        # Number of available samples
        nStat = jB - jA + 1
        # Initialize dictionary of handles.
        h = {}
        # Plot the histogram.
        h[c] = plt.hist(C[jA:jB+1], nBin,
            normed=1, histtype='bar', rwidth=0.85, color='#2020ff')
        # Labels.
        h['x'] = plt.xlabel(c)
        h['y'] = plt.ylabel('PDF')
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try:
            plt.tight_layout()
        except Exception:
            pass
        # Make a label for the mean value.
        lbl = u'\u03BC(%s) = %.4f' % (c, cAvg)
        h['mu'] = plt.text(1.0, 1.06, lbl, horizontalalignment='right',
            verticalalignment='top', transform=h['ax'].transAxes)
        # Make a label for the standard deviation.
        lbl = u'\u03C3(%s) = %.4f' % (c, cStd)
        h['sigma'] = plt.text(0.02, 1.06, lbl, horizontalalignment='left',
            verticalalignment='top', transform=h['ax'].transAxes)
        # Make a label for the uncertainty.
        lbl = u'\u03C3(\u03BC) = %.4f' % cErr
        h['err'] = plt.text(0.02, 0.98, lbl, horizontalalignment='left',
            verticalalignment='top', transform=h['ax'].transAxes)
        # Attempt to set font to one with Greek symbols.
        try:
            # Set the fonts.
            h['mu'].set_family("DejaVu Sans")
            h['sigma'].set_family("DejaVu Sans")
            h['err'].set_family("DejaVu Sans")
        except Exception:
            pass
        # Output.
        return h
# class CaseFM


# Aerodynamic history class
class CaseResid(object):
    """
    Iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = cape.dataBook.CaseResid()
    :Outputs:
        *hist*: :class:`cape.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
        
    # Number of orders of magnitude of residual drop
    def GetNOrders(self, nStats=1):
        """Get the number of orders of magnitude of residual drop
        
        :Call:
            >>> nOrders = hist.GetNOrders(nStats=1)
        :Inputs:
            *hist*: :class:`pyCart.dataBook.CaseResid`
                Instance of the DataBook residual history
            *nStats*: :class:`int`
                Number of iterations to use for averaging the final residual
        :Outputs:
            *nOrders*: :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: First versoin
        """
        # Process the number of usable iterations available.
        i = max(self.nIter-nStats, 0)
        # Get the maximum residual.
        L1Max = np.log10(np.max(self.L1Resid))
        # Get the average terminal residual.
        L1End = np.log10(np.mean(self.L1Resid[i:]))
        # Return the drop
        return L1Max - L1End
        
    # Number of orders of unsteady residual drop
    def GetNOrdersUnsteady(self, n=1):
        """
        Get the number of orders of magnitude of unsteady residual drop for each
        of the last *n* unsteady iteration cycles.
        
        :Call:
            >>> nOrders = hist.GetNOrders(n=1)
        :Inputs:
            *hist*: :class:`pyCart.dataBook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Number of iterations to analyze
        :Outputs:
            *nOrders*: :class:`numpy.ndarray` (:class:`float`), shape=(n,)
                Number of orders of magnitude of unsteady residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: First versoin
        """
        # Process the number of usable iterations available.
        i = max(self.nIter-n, 0)
        # Get the initial residuals
        L1Init = np.log10(self.L1Resid0[i:])
        # Get the terminal residuals.
        L1End = np.log10(self.L1Resid[i:])
        # Return the drop
        return L1Init - L1End
        
    # Plot function
    def PlotResid(self, c='L1Resid', n=None, nFirst=None, nLast=None, **kw):
        """Plot a residual by name
        
        :Call:
            >>> h = hist.PlotResid(c='L1Resid', n=None, **kw)
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
                Instance of the DataBook residual history
            *c*: :class:`str`
                Name of coefficient to plot
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
            *YLabel*: :class:`str`
                Label for *y*-axis
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: Copied from :func:`PlotL1`
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Initialize dictionary.
        h = {}
        # Get iteration numbers.
        if n is None:
            # Use all iterations
            n = self.i[-1]
        # Process other options
        fw = kw.get('FigWidth')
        fh = kw.get('FigHeight')
        # ---------
        # Last Iter 
        # ---------
        # Most likely last iteration
        iB = self.i[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                jB = self.GetIterationIndex(nLast)
                # Find the iterations that are less than i.
                iB = self.i[jB]
        # Get the index of *iB* in *FM.i*.
        jB = np.where(self.i == iB)[0][-1]
        # ----------
        # First Iter
        # ----------
        # Get the starting iteration number to use.
        i0 = max(0, iB-n, nFirst) + 1
        # Make sure *iA* is in *FM.i* and get the index.
        j0 = self.GetIterationIndex(i0)
        # Reselect *iA* in case initial value was not in *FM.i*.
        i0 = self.i[j0]
        # --------
        # Plotting
        # --------
        # Extract iteration numbers and residuals.
        i  = self.i[i0:]
        # Nominal residual
        try:
            L1 = getattr(self,c)[i0:]
        except Exception:
            L1 = np.zeros_like(i)
        # Residual before subiterations
        try:
            L0 = getattr(self,c+'0')[i0:]
        except Exception:
            L0 = np.zeros_like(i)
        # Check if L0 is too long.
        if len(L0) > len(i):
            # Trim it.
            L0 = L0[:len(i)]
        # Plot the initial residual if there are any unsteady iterations.
        if L0[-1] > L1[-1]:
            h['L0'] = plt.semilogy(i, L0, 'b-', lw=1.2)
        # Plot the residual.
        h['L1'] = plt.semilogy(i, L1, 'k-', lw=1.5)
        # Labels
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel(kw.get('YLabel', c))
        # Get the figures and axes.
        h['ax'] = plt.gca()
        h['fig'] = plt.gcf()
        # Set figure dimensions
        if fh: h['fig'].set_figheight(fh)
        if fw: h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes.
        try:
            plt.tight_layout()
        except Exception:
            pass
        # Set the xlimits.
        h['ax'].set_xlim((i0, iB+25))
        # Output.
        return h
        
    # Plot function
    def PlotL1(self, n=None, nFirst=None, nLast=None, **kw):
        """Plot the L1 residual
        
        :Call:
            >>> h = hist.PlotL1(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
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
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: Referred to :func:`PlotResid`
        """
        # Plot 'L1Resid'
        return self.PlotResid('L1Resid', 
            n=n, nFirst=nFirst, nLast=nLast, YLabel='L1 Residual')
        
    # Plot function
    def PlotL2(self, n=None, nFirst=None, nLast=None, **kw):
        """Plot the L2 residual
        
        :Call:
            >>> h = hist.PlotL2(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
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
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: Referred to :func:`PlotResid`
        """
        # Plot 'L1Resid'
        return self.PlotResid('L2Resid', 
            n=n, nFirst=nFirst, nLast=nLast, YLabel='L2 Residual')
        
        
    # Function to get index of a certain iteration number
    def GetIterationIndex(self, i):
        """Return index of a particular iteration in *hist.i*
        
        If the iteration *i* is not present in the history, the index of the
        last available iteration less than or equal to *i* is returned.
        
        :Call:
            >>> j = hist.GetIterationIndex(i)
        :Inputs:
            *hist*: :class:`cape.dataBook.CaseResid`
                Instance of the residual history class
            *i*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Index of last iteration in *FM.i* less than or equal to *i*
        :Versions:
            * 2015-03-06 ``@ddalle``: First version
        """
        # Check for *i* less than first iteration.
        if i < self.i[0]: return 0
        # Find the index.
        j = np.where(self.i <= i)[0][-1]
        # Output
        return j
# class CaseResid

