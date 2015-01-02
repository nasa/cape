"""
Data Book Module: :mod:`pyCart.dataBook`
========================================

This module contains functions for reading and processing forces, moments, and
other statistics from a trajectory

:Versions:
    * 2014-12-20 ``@ddalle``: Started
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re
# Date processing
from datetime import datetime

# Use this to only update entries with newer iterations.
from .case import GetCurrentIter

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
PdfPages = 0

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
    global PdfPages
    # Check for PyPlot.
    try:
        plt.gcf
    except AttributeError:
        # Load the modules.
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages


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
        self.x = x
        # Save the options.
        self.opts = opts
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                os.mkdir(fdir, dmask)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in self.Components:
            # Initialize the data book.
            self[comp] = DBComp(comp, x, opts, fdir)
        # Initialize targets.
        self.Targets = []
        # Read the targets.
        for targ in opts.get_DataBookTargets():
            # Read the file.
            self.Targets.append(DBTarget(targ, opts))
            
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
            
    # Write the data book
    def Write(self):
        """Write the current data book in Python memory to file
        
        :Call:
            >>> DB.Write()
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Start from root directory.
        os.chdir(self.RootDir)
        # Get the sort key.
        skey = self.opts.get_SortKey()
        # Sort the data book if there is a key.
        if skey is not None:
            # Check for a list.
            if type(skey).__name__ in ["list", "ndarray"]:
                # Loop through sort keys.
                for k in skey:
                    self.Sort(k)
            else:
                # Sort on the single key.
                self.Sort(skey)
        # Loop through the components.
        for comp in self.Components:
            self[comp].Write()
            
    # Function to sort data book
    def Sort(self, key=None, I=None):
        """Sort a data book according to either a key or an index
        
        :Call:
            >>> DB.Sort()
            >>> DB.Sort(key)
            >>> DB.Sort(I=None)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Process inputs.
        if I is None:
            # Use indirect sort on the first component.
            I = self[self.Components[0]].ArgSort(key)
        # Loop through components.
        for comp in self.Components:
            # Apply the DBComp.Sort() method.
            self[comp].Sort(I=I)
            
            
    # Update data book
    def UpdateDataBook(self, I=None):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DB.UpdateDataBook()
            >>> DB.UpdateDataBook(I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
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
        """Update or add a trajectory to a data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> DB.UpdateCase(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Trajectory index
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the first data book component.
        c0 = self.Components[0]
        # Try to find a match existing in the data book.
        j = self[c0].FindMatch(i)
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
        nIter = int(GetCurrentIter())
        # Get the number of iterations used for stats.
        nStats = self.opts.get_nStats()
        # Process whether or not to update.
        if (not nIter) or (nIter < nStats):
            # Not enough iterations (or zero iterations)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif np.isnan(j):
            # No current entry.
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif self[c0]['nIter'][j] < nIter:
            # Update
            print("  Updating from iteration %i to %i."
                % (self[c0]['nIter'][j], nIter))
            q = True
        elif self[c0]['nStats'][j] != nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Read the history.
        A = Aero(self.Components)
        # Process the residual drop
        nOrders = A.Residual.GetNOrders(nStats)
        # Loop through components.
        for comp in self.Components:
            # Extract the component history and component databook.
            FM = A[comp]
            DC = self[comp]
            # This is the part where we do transformations....
            # Loop through the transformations.ss
            for topts in self.opts.get_DataBookTransformations(comp):
                # Apply the transformation.
                FM.TransformFM(topts, self.x, i)
                
            # Process the statistics.
            s = FM.GetStats(nStats)
            # Save the data.
            if np.isnan(j):
                # Add the the number of cases.
                DC.n += 1
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DC[k] = np.hstack((DC[k], [getattr(self.x,k)[i]]))
                # Append values.
                for c in DC.DataCols:
                    DC[c] = np.hstack((DC[c], [s[c]]))
                # Append residual drop.
                DC['nOrders'] = np.hstack((DC['nOrders'], [nOrders]))
                # Append iteration counts.
                DC['nIter']  = np.hstack((DC['nIter'], [nIter]))
                DC['nStats'] = np.hstack((DC['nStats'], [nStats]))
                # Process the target.
                if len(DC.TargetCols) > 0:
                    # Select one.
                    c = DC.TargetCols[0]
                    # Determine which target is in use.
                    it, ct = self.GetTargetIndex(c)
                    # Select the target.
                    DBT = self.Targets[it]
                    # Find matches
                    jt = DBT.FindMatch(self.x, i)
                    # Check for a match.
                    if len(jt) > 0:
                        # Select the first match.
                        jt = jt[0]
                    else:
                        # No match
                        jt = np.nan
                # Append targets.
                for c in DC.targs:
                    # Determine the target to use.
                    it, ct = self.GetTargetIndex(DC.targs[c])
                    # Target name of the coefficient
                    cc = c + '_t'
                    # Store it.
                    if np.isnan(jt):
                        # No match found
                        DC[cc] = np.hstack((DC[cc], [np.nan]))
                    else:
                        # Append the match.
                        DC[cc] = np.hstack((DC[cc], [DBT[ct][jt]]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DC.DataCols:
                    DC[c][j] = s[c]
                # No reason to update targets, either.
        # Go back.
        os.chdir(self.RootDir)
                    
    # Get target to use based on target name
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
            *targ*: :class:`str`
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
            
    # Initialize a sweep plot
    def InitPlot(self, i):
        """Initialize databook plot *i*
        
        :Call:
            >>> DB.InitPlot(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Index of data book plot to initialize
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Open a figure.
        self.fig = plt.figure()
        # Save the axes.
        self.ax = plt.gca()
        # Axes labels.
        plt.xlabel(DBP["XLabel"])
        plt.ylabel(DBP["YLabel"])
        # Add the restriction text.
        txt = DBP.get('Restriction', '')
        self.restriction = plt.figtext(0.5, 0.01, txt,
            horizontalalignment='center')
        # Compress the plot slightly if there's a restriction.
        if len(txt) > 0:
            # Get the position.
            box = self.ax.get_position()
            # Move it up slightly.
            self.ax.set_position([box.x0+0.02*box.width,
                box.y0+0.03*box.height, 1.04*box.width, 0.99*box.height])
        # Initialize the tag (states variables that are constant)
        self.tag = plt.figtext(0.015, 0.985, '', verticalalignment='top')
        
    # Function to create a plot for an individual sweep
    def PlotSweep(self, I, i, istyle=0, lbl=None):
        """Create databook plot *i* for a single sweep
        
        :Call:
            >>> DB.PlotSweep(I, i, istyle=0, lbl=None
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices in sweep
            *i*: :class:`int`
                Index of data book plot to initialize
            *istyle*: :class:`bool`
                Plot style offset.  Start from plot style *istyle*; see also
                :func:`pyCart.options.DataBook.DBPlot.get_PlotOptions`
            *lbl*: :class:`str`
                Additional text to add to legend labels
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
        """
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Axis variables
        xv = DBP["XAxis"]
        yv = DBP["YAxis"]
        # Sweep specifications
        kw = DBP["Sweep"]
        # Get the components.
        comps = DBP["Components"]
        # Initial component
        DBc = self[comps[0]]
        
        # Copy of sweep keys for target search.
        tkw = kw.copy()
        # Check for carpet.
        o_carpet = DBP["Carpet"]
        # Check if it's a nonempty dict.
        if o_carpet and (type(o_carpet).__name__ == "dict"):
            # Get the carpet parameter.
            cv = o_carpet.keys()[0]
            # Add it to the target sweep criteria.
            tkw[cv] = o_carpet[cv]
        # Plot the min/max and standard deviation plots first.
        # This prevents the region plots from covering mean results.
        for j in range(len(comps)):
            # Get the component.
            DBc = self[comps[j]]
            # Plot it.
            DBc.PlotSweepMinMax(I, i, j+istyle)
            DBc.PlotSweepStDev(I, i, j+istyle)
        # Loop through the components.
        for j in range(len(comps)):
            # Get the component.
            DBc = self[comps[j]]
            # Plot it.
            line = DBc.PlotSweep(I, i, j+istyle)
            # Add min/max to label.
            if DBP.get("MinMax", False):
                # Add (min/max) to the label.
                flbl = "%s (min/max)" % line.get_label()
                line.set_label(flbl)
            # Add standard deviation to the label.
            if DBP.get("StandardDeviation", 0):
                # Add (\pm k*\sigma) to the label.
                flbl = u"%s (\u00B1%.1f\u03C3)" % (
                    line.get_label(), DBP["StandardDeviation"])
                line.set_label(flbl)
            # Append a label if so requested.
            if lbl:
                # Form the new label.
                flbl = u"%s, %s" % (line.get_label(), lbl)
                # Set it.
                line.set_label(flbl)
            # Check for a target.
            if DBc.targs.get(yv) is None: continue
            # Possible to not want targets.
            if not DBP["Targets"]: continue
            # Target indicies
            it, ct = self.GetTargetIndex(DBc.targs.get(yv))
            # Extract the target.
            DBT = self.Targets[it]
            # Get the sweep.
            It = DBT.FindSweep(DBc, I[0], key=xv, **tkw)
            # Plot the target sweep.
            line = DBT.PlotSweep(It, i, j, istyle)
            # Append a label if so requested.
            if lbl:
                # Form the new label.
                flbl = u"%s, %s" % (line.get_label(), lbl)
                # Set it.
                line.set_label(flbl)
        # Add margin to the y-axis limits
        ymin, ymax = self.ax.get_ylim()
        self.ax.set_ylim((1.05*ymin-0.05*ymax, 1.21*ymax-0.21*ymin))
        # See if font size needs to be smaller.
        if len(self.ax.get_lines()) > 5:
            # Very small
            fsize = 7
        else:
            # Just small
            fsize = 9
        # Figure tag list.
        tags = []
        # Loop through sweep parameters.
        for k in kw:
            # Check the parameter type.
            if self.x.defns[k]["Value"] == "float":
                # Short float label.
                tags.append('%s=%.2f' % (k, DBc[k][I[0]]))
            else:
                # Use literal string conversion.
                tags.append('%s=%s' % (k, DBc[k][I[0]]))
        # Activate legend.
        try:
            # Try to force the DejaVu Sans font to get "sigma".
            self.legend = self.ax.legend(loc='upper center',
                prop=dict(size=fsize, family="DejaVu Sans"),
                bbox_to_anchor=(0.5, 1.05), labelspacing=0.5)
        except Exception:
            # Legend with default font.
            self.legend = self.ax.legend(loc='upper center',
                prop=dict(size=fsize),
                bbox_to_anchor=(0.5, 1.05), labelspacing=0.5)
        # Set the figure text.
        self.tag.set_text(", ".join(tags))
        # Draw the figure.
        plt.draw()
        
        
        
    # Function to create a plot for an individual sweep
    def Plot(self, i):
        """Create data book plot *i*, a multipage PDF
        
        :Call:
            >>> DB.Plot(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Index of data book plot to initialize
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
            * 2014-12-30 ``@ddalle``: Separated out into two functions
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Check the type.
        if DBP.get("Carpet"):
            # Carpet plot.
            self.PlotCarpets(i)
        else:
            # Regular sweep plot.
            self.PlotSweeps(i)
        
        
    # Function to create a plot for an individual sweep
    def PlotSweeps(self, i):
        """Create multipage data book plot *i* if it is not a carpet plot
        
        :Call:
            >>> DB.PlotSweeps(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Index of data book plot to initialize
        :Versions:
            * 2014-12-30 ``@ddalle``: Copied from old :func:`DB.Plot`
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Axis variables
        xv = DBP["XAxis"]
        yv = DBP["YAxis"]
        # Sweep specifications
        kw = DBP["Sweep"]
        # Get the components.
        comps = DBP["Components"]
        # Get output option.
        o_out = DBP.get("Output", "")
        # Output folder
        fdir = os.path.join(self.RootDir, self.opts.get_DataBookDir())
        # Get the label for this plot.
        flbl = DBP["Label"]
        # Default label
        if flbl is None or len(flbl)==0:
            # Use list of components.
            flbl = "-".join(comps)
        # Output file name.
        fname = 'db_%s_%s_%s' % (flbl,  yv, xv)
        # File name. 
        fbase = os.path.join(fdir, fname)
        # Save plot file name. (e.g. "db_RSRB-LSRB_CY_Mach.pdf")
        self.figname = os.path.join(fdir, fname+".pdf")
        # Select the first component.
        DBc = self[comps[0]]
        # Get the sweeps.
        I = DBc.GetSweeps(xv, **kw)
        # Initialize the PDF.
        self.pdf = PdfPages(self.figname)
        # Loop through the sweeps.
        for j in range(len(I)):
            # Initialize a component plot.
            self.InitPlot(i)
            # Call the individual sweep plot function.
            self.PlotSweep(I[j], i)
            # Add the plot.
            self.pdf.savefig(self.fig)
            # Individual filename (if needed)
            f_i = fbase + ("_Sweep%03i" % j)
            # Process individual output.
            if o_out in ["pdf", "PDF"]:
                # Save as a PDF.
                self.fig.savefig(f_i+".pdf")
            elif o_out in ["svg", "SVG"]:
                # Save as an SVG.
                self.fig.savefig(f_i+".svg")
            elif o_out in ["png", "PNG"]:
                # Get resolution.
                fdpi = DBP.get("DPI", 120)
                # Save as a PNG.
                self.fig.savefig(f_i+".png", dpi=fdpi)
            # Close the figure.
            plt.close(self.fig)
        # Close the multipage PDF to create the document.
        self.pdf.close()
        
    # Function to create a set of carpet plots
    def PlotCarpets(self, i):
        """Create multipage data book plot *i* if it _is_ a carpet plot
        
        :Call:
            >>> DB.PlotCarpets(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Index of data book plot to initialize
        :Versions:
            * 2014-12-30 ``@ddalle``: Forked from old :func:`DB.Plot`
        """
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Number of components.
        nComp = len(DBP["Components"])
        # Axis variables
        xv = DBP["XAxis"]
        yv = DBP["YAxis"]
        
        # Carpet dictionary.
        o_carpet = DBP.get("Carpet")
        # Check it.  No errors; just quit.
        if type(o_carpet).__name__ != "dict": return
        # Carpet variable.
        cv = o_carpet.keys()[0]
        ctol = o_carpet[cv]
        # Check the carpet variable.
        if cv not in self.x.keys:
            raise IOError(("Carpet search variable '%s' is not a " % cv)
                + "trajectory key.")
            
        # Sweep specifications
        kw = DBP["Sweep"]
        # Get the components.
        comps = DBP["Components"]
        # Get output option.
        o_out = DBP.get("Output", "")
        # Output folder
        fdir = os.path.join(self.RootDir, self.opts.get_DataBookDir())
        # Get the label for this plot.
        flbl = DBP["Label"]
        # Default label
        if flbl is None or len(flbl)==0:
            # Use list of components.
            flbl = "-".join(comps)
        # Output file name.
        fname = 'db_%s_%s_%s-%s' % (flbl,  yv, xv, cv)
        # File name. 
        fbase = os.path.join(fdir, fname)
        # Save plot file name. (e.g. "db_RSRB_CY_Mach-alpha.pdf")
        self.figname = os.path.join(fdir, fname+".pdf")
        
        # Select the first component.
        DBc = self[comps[0]]
        # Get the sweeps.
        J = DBc.GetCarpets(xv, cv, ctol, **kw)
        # String format for additional legend label to specify *cv* value.
        if self.x.defns[cv]["Value"] == "float":
            # Short label.
            s_lgnd = "%s=%%.2f" % cv
        else:
            # Use literal
            s_lgnd = "%s=%%s" % cv
        
        # Initialize the PDF.
        self.pdf = PdfPages(self.figname)
        # Loop through the carpets.
        for ij in range(len(J)):
            # Initialize a component plot.
            self.InitPlot(i)
            # Extract sweep.
            I = J[ij]
            # Loop through the sweeps.
            for jj in range(len(I)):
                # Extract indices
                j = I[jj]
                # Form the additional label using initial value of *cv*.
                f_lgnd = s_lgnd % DBc[cv][j[0]]
                # Call the individual sweep plot function.
                self.PlotSweep(j, i, istyle=jj*nComp, lbl=f_lgnd)
            # Add the plot.
            self.pdf.savefig(self.fig)
            # Individual filename (if needed)
            f_i = fbase + ("_Carpet%03i" % ij)
            # Process individual output.
            if o_out in ["pdf", "PDF"]:
                # Save as a PDF.
                self.fig.savefig(f_i+".pdf")
            elif o_out in ["svg", "SVG"]:
                # Save as an SVG.
                self.fig.savefig(f_i+".svg")
            elif o_out in ["png", "PNG"]:
                # Get resolution.
                fdpi = DBP.get("DPI", 120)
                # Save as a PNG.
                self.fig.savefig(f_i+".png", dpi=fdpi)
        # Close the multipage PDF to create the document.
        self.pdf.close()
        # Close all the figures.
        plt.close('all')
                
    
                
# Individual component data book
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
        *fdir*: :class:`str`
            Data book folder (forward slash separators)
    :Outputs:
        *DBi*: :class:`pyCart.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, comp, x, opts, fdir="data"):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the list of columns for that coefficient.
        cols = opts.get_DataBookCols(comp)
        # Get the directory.
        fdir = opts.get_DataBookDir()
        
        # Construct the file name.
        fcomp = 'aero_%s.dat' % comp
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
        self.TargetCols = opts.get_DataBookTargetCols(comp)
        # Save the file name.
        self.fname = fname
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
            
    # Command-line representation
    def __repr__(self):
        """Representation method
        
        :Versions;
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
            >>> DBi.Read()
            >>> DBi.Read(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.dat' % self.comp``)
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
                # Read the column
                self[k] = np.loadtxt(fname, 
                    delimiter=delim, dtype=t, usecols=[nCol])
                # Increase the column number
                nCol += 1
            # Loop through the data book columns.
            for c in self.cols:
                # Add the column.
                self[c] = np.loadtxt(fname, delimiter=delim, usecols=[nCol])
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
        except Exception:
            # Initialize empty trajectory arrays.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Initialize an empty array.
                self[k] = np.array([], dtype=t)
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
        """Write a single data book file or initialize empty arrays
        
        :Call:
            >>> DBi.Write()
            >>> DBi.Write(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.dat' % self.comp``)
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
            >>> I = DBi.ArgSort(key=None)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
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
        # Indirect sort on it.
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
            *DBi*: :class:`pyCart.dataBook.DBComp`
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
                raise IOError("Index list is unusable type.")
            elif len(I) != self.n:
                # Incompatible length.
                raise IOError(("Index list length (%i) " % len(I)) +
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
            >>> j = DBi.FindMatch(i)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
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
            
    # Find entries in a sweep.
    def FindSweep(self, i, key=None, j0=None, **kw):
        """Find a the indices of values in a sweep of one variable
        
        The goal of this function is to return a list of indices of points from
        the databook that sweeps through a single variable while the other
        variables remain constant (to tolerance).
        
        The search is seeded by a point in the databook, and tolerances are
        specified for the other variables.  Only trajectory keys specified as
        keyword arguments are used as filter criteria.
        
        :Call:
            >>> j = DBi.FindSweep(i, key=None, **kw)
            >>> j = DBi.FindSweep(i, key=None, j0=None, **kw)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *i*: :class:`int`
                Index of databook entry to seed the search
            *key*: :class:`str`
                Name of variable to sort by (defaults to first trajectory key)
            *j0*: :class:`numpy.ndarray` (:class:`int`) or ``None``
                List of indices of cases to filter
            *kw*: :class:`dict` (:class:`float` or :class:`int`)
                Keyword arguments of tolerances for keys to use in filter
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                List of indices of cases meeting sweep criteria
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize indices.
        if j0 is None:
            # Use all cases.
            j = np.arange(self.n)
        elif type(j0).__name__ == "ndarray":
            # Copy the initial values.
            j = j0.copy()
        else:
            # Assume it's a list...
            j = j0
        # Get default key if necessary.
        if (key is None):
            # Use the default value.
            key = self.x.keys[0]
        # Make sure the key is usable.
        if (key not in self.x.keys):
            raise IOError("Sweep key '%s' is not a trajectory variable." % key)
        # Loop through keys.
        for k in kw:
            # Check if the kwarg is a trajectory key.
            if k not in self.x.keys: continue
            # Get the target value.
            v = self[k][i]
            # Check the criteria.
            try:
                # Check for strings.
                if self.x.defns[k]["Value"] in ["str", "unicode"]:
                    # Filter strings.
                    qj = self[k][j] == v
                    # Check last element.
                    if (not qj[-1]) and (self[k][j[-1]]==v):
                        # Set the last element to True.
                        qj[-1] = True
                else:
                    # Filter test.
                    qj = np.abs(self[k][j] - v) <= kw[k]
                    # Check last element.
                    if (not qj[-1]) and (np.abs(self[k][j[-1]]-v)<=kw[k]):
                        # Set the last element to True.
                        qj[-1] = True
                # Restrict to cases that pass this test.
                j = j[qj]
            except Exception:
                # No match or failed test.
                return np.array([])
        # Get the order that would sort *j*.
        jo = np.argsort(self[key][j])
        # Output (sorted)
        return j[jo]
        
    # Function to divide data book into sweeps
    def GetSweeps(self, key=None, **kw):
        """Divide databook entries into sweeps of one variable
        
        :Call:
            >>> I = DBi.GetSweeps(key=None, **kw)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *key*: :class:`str`
                Name of variable to sort by (defaults to first trajectory key)
        :Outputs:
            *I*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep arrays
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize data set.
        jNoMatch = np.arange(self.n)
        # Initialize output
        I = []
        # Escape criterion
        n = 0
        # Loop until all cases are in a sweep.
        while (n<1000) and (len(jNoMatch)>0):
            # Get the first index that hasn't been put into a sweep yet.
            i = jNoMatch[0]
            # Get the sweep.
            j = self.FindSweep(i, key, **kw)
            # Save the sweep.
            I.append(j)
            # Remove the cases in the current sweep.
            jNoMatch = np.setdiff1d(jNoMatch, j)
            # Increase the safety counter.
            n += 1
        # Output.
        return I
        
    # Function to divide data book into carpets
    def GetCarpets(self, key, ckey, tol, **kw):
        """Divide data book entries into carpet sweeps of two variables
        
        :Call:
            >>> J = DBi.GetCarpets(key, ckey, tol, **kw)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *key*: :class:`str`
                Primary sweep variable, *x*-axis of plots
            *ckey*: :class:`str`
                Secondary sweep variable
            *tol*: :class:`float`
                Tolerance for the secondary sweep variable
        :Outputs:
            *J*: :class:`list` (:class:`list` (:class:`numpy.ndarray`))
                List of sweep arrays
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Initialize data set.
        jNoMatch = np.arange(self.n)
        # Initialize output
        J = []
        # Escape criterion
        n = 0
        # Dictionary to use as keyword arguments for secondary searches.
        cdict = {ckey: tol}
        # Loop until all cases are in a carpet (rug?)
        while (n<1000) and (len(jNoMatch)>0):
            # Get the first index that hasn't been put into a carpet yet.
            i = jNoMatch[0]
            # Get the sweep.
            j = self.FindSweep(i, key, **kw)
            # Remove the cases in the current sweep.
            jNoMatch = np.setdiff1d(jNoMatch, j)
            # Increase the safety counter.
            n += 1
            # Safety subcounter
            ni = 0
            # Initialize individual carpet.
            I = []
            # Loop until the big sweep is divided.
            while (ni<100) and (len(j)>0):
                # Get the first index that hasn't been subdivided yet.
                ii = j[0]
                # Get the sweep. (only
                ji = self.FindSweep(ii, key, j0=j, **cdict)
                # Remove the cases from the sweep.
                j = np.setdiff1d(j, ji)
                # Increase safety counter.
                ni += 1
                # Save the subsweep.
                I.append(ji)
            # Save the carpet.
            J.append(I)
        # Output
        return J
            
        
        
    # Function to plot a single sweep min/max plot
    def PlotSweepMinMax(self, I, i, j=0):
        """Plot minimum and maximum for a fixed set of indices
        
        :Call:
            >>> lines = DBi.PlotSweepMinMax(I, i, j=0)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *I*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep arrays
            *i*: :class:`int`
                Index of data book plot options to use
            *j*: :class:`int`
                Index of plot options to use (if there are multiple curves)
        :Outputs:
            *line*: :class:`matplotlib.collections.PolyCollection`
                Handles for the sweep line that is drawn
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Ensure plot modules are loaded
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Check for min/max
        if DBP['MinMax'] and self.opts.get_nStats():
            # Determine the axes.
            xv = DBP['XAxis']
            yv = DBP['YAxis']
            # Get the min/max plot options
            o_plt = DBP.get_MinMaxOptions(j)
            # Plot it.
            line = plt.fill_between(self[xv][I],
                self[yv+'_min'][I], self[yv+'_max'][I], **o_plt)
        else:
            # No line.
            line = None
        # Output
        return line
        
    # Function to plot a single sweep min/max plot
    def PlotSweepStDev(self, I, i, j=0):
        """Plot standard deviation spread for a fixed set of indices
        
        :Call:
            >>> lines = DBi.PlotSweepStDev(I, i, j=0)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *I*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep arrays
            *i*: :class:`int`
                Index of data book plot options to use
            *j*: :class:`int`
                Index of plot options to use (if there are multiple curves)
        :Outputs:
            *line*: :class:`matplotlib.collections.PolyCollection`
                Handles for the sweep line that is drawn
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        # Ensure plot modules are loaded
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Check for standard deviation.
        if DBP['StandardDeviation'] and self.opts.get_nStats():
            # Determine the axes.
            xv = DBP['XAxis']
            yv = DBP['YAxis']
            # Get the standard deviation plot options.
            o_plt = DBP.get_StDevOptions(j)
            # Multiplier.
            ksig = DBP['StandardDeviation']
            # Extract values
            y = self[yv][I]
            s = self[yv+'_std'][I]
            # Plot it.
            line = plt.fill_between(self[xv][I], y-ksig*s, y+ksig*s, **o_plt)
        else:
            # No line.
            line = None
        # Output
        return line
        
    # Function to plot a single sweep.
    def PlotSweep(self, I, i, j=0):
        """Plot a fixed set of indices with known options
        
        :Call:
            >>> line = DBi.PlotSweep(I, i, j=0)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *I*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep arrays
            *i*: :class:`int`
                Index of data book plot options to use
            *j*: :class:`int`
                Index of plot options to use (if there are multiple curves)
        :Outputs:
            *line*: :class:`matplotlib.lines.Line2D`
                Handles for the sweep line that is drawn
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
            * 2014-12-29 ``@ddalle``: Added min/max and standard deviation
            * 2014-12-30 ``@ddalle``: Moved min/max to sep func for layering
        """
        # Ensure plot modules are loaded
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Determine the axes.
        xv = DBP['XAxis']
        yv = DBP['YAxis']
        # Get the options
        o_plt = DBP.get_PlotOptions(j)
        # Initialize the label.
        lbl = self.comp
        # Plot
        line = plt.plot(self[xv][I], self[yv][I], label=lbl, **o_plt)[0]
        # Output.
        return line
        
        
        
# Data book target instance
class DBTarget(dict):
    """
    Class to handle data from data book target files.  There are more
    constraints on target files than the files that data book creates, and raw
    data books created by pyCart are not valid target files.
    
    :Call:
        >>> DBT = pyCart.dataBook.DBTarget(targ, opts)
    :Inputs:
        *targ*: :class:`pyCart.options.DataBook.DBTarget`
            Instance of a target source options interface
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyCart.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, targ, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the target options
        self.opts = opts
        self.topts = targ
        # Source file
        fname = targ.get_TargetFile()
        # Name of this target.
        tname = targ.get_TargetName()
        # Check for the file.
        if not os.path.isfile(fname):
            raise IOError(
                "Target source file '%s' could not be found." % fname)
        # Save the name.
        self.Name = tname
        # Delimiter
        delim = targ.get_Delimiter()
        # Comment character
        comchar = targ.get_CommentChar()
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
        self.headers = headers.strip().split(delim)

        # Read it.
        self.data = np.loadtxt(fname, delimiter=delim, skiprows=nskip)
        # Initialize requested fields with the fields that correspond to
        # trajectory keys
        cols = targ.get_Trajectory().values()
        # Process the required fields.
        for comp in opts.get_DataBookComponents():
            # Loop through the targets.
            for ctarg in opts.get_CompTargets(comp).values():
                # Get the target source for this entry.
                if '/' not in ctarg:
                    # Only one target source; assume it's this one.
                    ti = tname
                    fi = ctarg
                else:
                    # Read the target name.
                    ti = ctarg.split('/')[0]
                    # Name of the column
                    fi = ctarg.split('/')[1]
                # Check if the target is from this target source.
                if ti != tname: continue
                # Add the field if necessary.
                if fi not in cols:
                    # Check if the column is present.
                    if fi not in self.headers:
                        raise IOError("There is not field '%s' in '%s'."
                            % (fi, ti))
                    # Add the column
                    cols.append(fi)
        # Process the columns.
        for col in cols:
            # Find it and save it as a key.
            self[col] = self.data[:,self.headers.index(col)]
        
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
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the pyCart data book target data carrier
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
        j = np.arange(self.data.shape[0])
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
        
    # Find a sweep by component databook variables
    def FindSweep(self, DBi, i, key=None, j0=None, **kw):
        """Find a sweep of a single variable within a databook target
        
        :Call:
            >>> j = DBT.FindSweep(DBi, i, key=None, **kw)
        :Inputs:
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the pyCart data book target data carrier
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book component
            *i*: :class:`int`
                Index of databook entry to seed the search
            *j0*: :class:`numpy.ndarray` (:class:`int`) or ``None``
                List of indices of cases to filter
            *key*: :class:`str`
                Name of variable to sort by (defaults to first trajectory key)
            *kw*: :class:`dict` (:class:`float` or :class:`int`)
                Keyword arguments of tolerances for keys to use in filter
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                List of indices of cases meeting sweep criteria
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Initialize indices.
        if j0 is None:
            # Use all cases.
            j = np.arange(self.data.shape[0])
        elif type(j0).__name__ == "ndarray":
            # Copy the initial values.
            j = j0.copy()
        else:
            # Assume it's a list...
            j = j0
        # Get the trajectory key translations.   This determines which keys to
        # filter and what those keys are called in the source file.
        tkeys = self.topts.get_Trajectory()
        # Get default key if necessary.
        if (key is None):
            # Use the default value.
            key = DBi.x.keys[0]
        # Make sure the key is usable.
        if (key not in DBi.x.keys):
            raise IOError("Sweep key '%s' is not a trajectory variable." % key)
        # Loop through keys.
        for k in kw:
            # Check if the kwarg is a trajectory key.
            if k not in DBi.x.keys: continue
            # Get the target value.
            v = DBi[k][i]
            # Get the name of the column according to the source file.
            c = tkeys[k]
            # Check the criteria.
            try:
                # Check for strings.
                if DBi.x.defns[k]["Value"] in ["str", "unicode"]:
                    # Filter strings.
                    qj = self[c][j] == v
                    # Check last element.
                    if (not qj[-1]) and (self[c][j[-1]]==v):
                        # Set the last element to True.
                        qj[-1] = True
                else:
                    # Filter test.
                    qj = np.abs(self[c][j] - v) <= kw[k]
                    # Check last element.
                    if (not qj[-1]) and (np.abs(self[c][j[-1]]-v)<=kw[k]):
                        # Set the last element to True.
                        qj[-1] = True
                # Restrict to cases that pass this test.
                j = j[qj]
            except Exception:
                # No match or failed test.
                return np.array([])
        # Get the order that would sort *j*.
        jo = np.argsort(self[tkeys[key]][j])
        # Output (sorted)
        return j[jo]
        
    # Function to plot a single sweep.
    def PlotSweep(self, I, i, j=0, istyle=0,):
        """Plot a fixed set of indices with known options
        
        :Call:
            >>> line = DBT.PlotSweep(I, i, j=0, istyle=0)
        :Inputs:
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the pyCart data book target data carrier
            *I*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep arrays
            *i*: :class:`int`
                Index of data book target plot options to use
            *j*: :class:`int`
                Index of component to use (if there are multiple components)
            *istyle*: :class:`int`
                Offset for plot options
        :Outputs:
            *line*: :class:`matplotlib.lines.Line2D`
                Handle for the sweep line that is drawn
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Ensure plot modules are loaded
        ImportPyPlot()
        # Extract the options.
        DBP = self.opts.get_DataBookPlots()[i]
        # Determine what component we're plotting.
        comp = DBP.get_Component(j)
        # Determine the axes.
        xv = DBP['XAxis']
        yv = DBP['YAxis']
        # Convert trajectory variable into target column.
        xt = self.topts.get_Trajectory().get(xv)
        # Convert y-axis variable into target column.
        yt = self.opts.get_CompTargets(comp).get(yv)
        # Make sure something was found.
        if yt is None: return
        # Get the options
        o_plt = DBP.get_TargetOptions(j+istyle)
        # Initialize the label.
        lbl = '%s, %s' % (self.Name, comp)
        # Plot
        line = plt.plot(self[xt][I], self[yt][I], label=lbl, **o_plt)[0]
        # Output.
        return line
        
        
# Aerodynamic history class
class Aero(dict):
    """
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> aero = pyCart.dataBook.Aero(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class, similar to dictionary of force
            and/or moment histories
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.Aero`
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
        # Read the residuals.
        self.Residual = CaseResid()
        # Default component list.
        if (type(comps).__name__ in ["str", "unicode", "int"]):
            # Make a singleton list.
            comps = [comps]
        elif len(comps) < 1:
            # Extract keys from dictionary.
            comps = self.Components.keys()
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
            lines = open(fname).readlines()
            # Filter comments
            lines = [l for l in lines if not l.startswith('#')]
            # Convert all the values to floats
            # Can't make this an array yet because it's not rectangular.
            V = [[float(v) for v in l.split()] for l in lines]
            # Columns to use: 0 and {-6,-3}:
            n = len(self.Components[comp]['C'])
            # Create an array with the original data.
            A = np.array([v[0:1] + v[-n:] for v in V])
            # Get the number of entries in each row.
            # This will be one larger if a time-accurate iteration.
            # It's a column of zeros, and it's the second column.
            L = np.array([len(v) for v in V])
            # Check for steady-state iterations.
            if np.any(L == n+1):
                # At least one steady-state iteration
                n0 = np.max(A[L==n+1,0])
                # Add that iteration number to the time-accurate steps.
                A[L!=n+1,0] += n0
            # Extract info from components for readability
            d = self.Components[comp]
            # Make the component.
            self[comp] = CaseFM(d['C'], MRP=d['MRP'], A=A)
            
            
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
class CaseFM(object):
    """
    This class contains methods for reading data about an the histroy of an
    individual component for a single case.  The list of available components
    comes from a :file:`loadsCC.dat` file if one exists.
    
    :Call:
        >>> FM = pyCart.dataBook.CaseFM(C, MRP=None, A=None)
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
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.FM`
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
        
            * ``'<dataBook.CaseFM Force, i=100>'``
            * ``'<dataBook.CaseFM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<dataBook.CaseFM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize the string.
        txt = '<dataBook.CaseFM '
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
        
            * ``'<dataBook.CaseFM Force, i=100>'``
            * ``'<dataBook.CaseFM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<dataBook.CaseFM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
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
            *FM*: :class:`pyCart.dataBook.CaseFM`
                Instance of the force and moment class
            *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
                Matrix of forces and/or moments at *N* iterations
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Get size of A.
        n, m = A.shape
        # Save the iterations.
        self.i = A[:,0]
        # Check size.
        if m == 7:
            # Save all fields.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            self.CLL = A[:,4]
            self.CLM = A[:,5]
            self.CLN = A[:,6]
            # Save list of coefficients.
            self.coeffs = ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']
        elif (self.MRP.size==3) and (m == 4):
            # Save only moments.
            self.CLL = A[:,1]
            self.CLM = A[:,2]
            self.CLN = A[:,3]
            # Save list of coefficients.
            self.coeffs = ['CLL', 'CLM', 'CLN']
        elif (m == 4):
            # Save only forces.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            # Save list of coefficients.
            self.coeffs = ['CA', 'CY', 'CN']
        
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
            *FM*: :class:`pyCart.aero.FM`
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
            # Extract values from the trajectory.
            phi   = getattr(x,kph)[i]*deg
            theta = getattr(x,kth)[i]*deg
            psi   = getattr(x,kps)[i]*deg
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
        
        
        
    # Method to get averages and standard deviations
    def GetStats(self, nAvg=100):
        """Get mean, min, max, and standard deviation for all coefficients
        
        :Call:
            >>> s = FM.GetStats(nAvg=100)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *nAvg*: :class:`int`
                Number of iterations in window
        :Outputs:
            *s*: :class:`dict` (:class:`float`)
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
        """
        # Default values.
        if (nAvg is None) or (nAvg < 2):
            # Use last iteration
            i0 = self.i.size - 1
        else:
           # Process min indices for plotting and averaging.
            i0 = max(0, self.i.size-nAvg)
        # Initialize output.
        s = {}
        # Loop through coefficients.
        for c in self.coeffs:
            # Get the values
            F = getattr(self, c)
            # Save the mean value.
            s[c] = np.mean(F[i0:])
            # Check for statistics.
            if (nAvg is not None) or (nAvg == 0):
                # Save the statistics.
                s[c+'_min'] = np.min(F[i0:])
                s[c+'_max'] = np.max(F[i0:])
                s[c+'_std'] = np.std(F[i0:])
        # Output
        return s
            

    

# Aerodynamic history class
class CaseResid(object):
    """
    Iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = pyCart.dataBook.CaseResid()
    :Outputs:
        *hist*: :class:`pyCart.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        
        # Process the best data folder.
        if os.path.isfile('history.dat'):
            # Subsequent non-adaptive runs
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
        # History file name.
        fhist = os.path.join(fdir, 'history.dat')
        # Read the file.
        lines = open(fhist).readlines()
        # Filter comments.
        lines = [l for l in lines if not l.startswith('#')]
        # Convert all the values to floats.
        A = np.array([[float(v) for v in l.split()] for l in lines])
        # Get the indices of steady-state iterations.
        # (Time-accurate iterations are marked with decimal step numbers.)
        i = np.array(['.' not in l.split()[0] for l in lines])
        # Check for steady-state iterations.
        if np.any(i):
            # Get the last steady-state iteration.
            n0 = np.max(A[i,0])
            # Add this to the time-accurate iteration numbers.
            A[np.logical_not(i),0] += n0
        else:
            # Not steady-state iterations.
            n0 = 0
        # Process unsteady iterations if any.
        if A[-1,0] > n0:
            # Get the integer values of the iteration indices.
            # For *ni0*, 2000.000 --> 1999; 2000.100 --> 2000
            ni0 = np.array(A[n0:,0]-1e-4, dtype=int)
            # For *ni0*, 2000.000 --> 2000; 1999.900 --> 1999
            ni1 = np.array(A[n0:,0], dtype=int)
            # Look for iterations where the index crosses an integer.
            i0 = np.insert(np.where(ni0[1:] > ni0[:-1])[0]+1, 0, 0) + n0
            i1 = np.where(ni1[1:] > ni1[:-1])[0] + 1 + n0
        else:
            # No unsteady iterations.
            i0 = np.array([], dtype=int)
            i1 = np.array([], dtype=int)
        # Prepend the steady-state iterations.
        i0 = np.hstack((np.arange(n0), i0))
        i1 = np.hstack((np.arange(n0), i1))
        # Make sure these stupid things are ints.
        i0 = np.array(i0, dtype=int)
        i1 = np.array(i1, dtype=int)
        # Save the initial residuals.
        self.L1Resid0 = A[i0, 3]
        # Rewrite the history.dat file without middle subiterations.
        if not os.path.isfile('RUNNING'):
            # Iterations to keep.
            i = np.union1d(i0, i1)
            # Write the integer iterations and the first subiterations.
            open(fhist, 'w').writelines(np.array(lines)[i])
        # Eliminate subiterations.
        A = A[np.mod(A[:,0], 1.0) == 0.0]
        # Save the number of iterations.
        self.nIter = int(A[-1,0])
        # Save the iteration numbers.
        self.i = A[:,0]
        # Save the CPU time per processor.
        self.CPUtime = A[:,1]
        # Save the maximum residual.
        self.maxResid = A[:,2]
        # Save the global residual.
        self.L1Resid = A[:,3]
        # Process the CPUtime used for steady cycles.
        if n0 > 0:
            # At least one steady-state cycle.
            t = self.CPUtime[n0-1]
        else:
            # No steady state cycles.
            t = 0.0
        # Process the unsteady cycles.
        if self.nIter > n0:
            # Add up total CPU time for unsteady cycles.
            t += np.sum(self.CPUtime[n0:])
        # Check for a 'user_time.dat' file.
        if os.path.isfile('user_time.dat'):
            # Loop through lines.
            for line in open('user_time.dat').readlines():
                # Check comment.
                if line.startswith('#'): continue
                # Add to the time everything except flowCart time.
                t += np.sum([float(v) for v in line.split()[2:]])
        # Save the time.
        self.CPUhours = t / 3600.
        
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
        
        
