"""Function for Automated Report Interface"""

# File system interface
import os, json, shutil, glob
# Numerics
import numpy as np

# Import basis module
import cape.report

# Local modules needed
from cape import tex, tar
# Data book and plotting
from .dataBook import Aero, CaseFM, CaseResid
# Folder and Tecplot management
from .case    import LinkPLT
from .tecplot import ExportLayout, Tecscript
# Configuration and surface tri
from .tri    import Tri
from .config import Config

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# Change the umask to a reasonable value.
os.umask(umask)

# ---------------------------------
#-->


# Class to interface with report generation and updating.
class Report(cape.report.Report):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyCart.report.Report(cart3d, rep)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyCart.report.Report
            Automated report interface
    :Versions:
        * 2015-03-07 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, cart3d, rep):
        """Initialization method"""
        # Save the interface
        self.cart3d = cart3d
        self.cntl = cart3d
        # Check for this report.
        if rep not in cart3d.opts.get_ReportList():
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Go to home folder.
        fpwd = os.getcwd()
        os.chdir(cart3d.RootDir)
        # Create the report folder if necessary.
        if not os.path.isdir('report'): cart3d.mkdir('report', dmask)
        # Go into the report folder.
        os.chdir('report')
        # Get the options and save them.
        self.rep = rep
        self.opts = cart3d.opts['Report'][rep]
        # Initialize a dictionary of handles to case LaTeX files
        self.sweeps = {}
        self.cases = {}
        # Read the file if applicable
        self.OpenMain()
        # Return
        os.chdir(fpwd)
        
    # Function to update report
    def UpdateReport(self, I=None, cons=[], **kw):
        """Update a report based on the list of figures
        
        :Call:
            >>> R.UpdateReport(I)
            >>> R.UpdateReport(cons=[])
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-22 ``@ddalle``: First version
        """
        # Update any sweep figures.
        self.UpdateSweeps(I, cons, **kw)
        # Update any case-by-case figures.
        if self.HasCaseFigures():
            self.UpdateCases(I, cons, **kw)
        # Write the file.
        self.tex.Write()
        # Compmile it.
        print("Compiling...")
        self.tex.Compile()
        # Need to compile twice for links
        print("Compiling...")
        self.tex.Compile()
        # Clean up
        print("Cleaning up...")
        # Clean up sweeps
        self.CleanUpSweeps(I=I, cons=cons)
        # Clean up cases
        if self.HasCaseFigures():
            self.CleanUpCases(I=I, cons=cons)
        # Get other 'report-*.*' files.
        fglob = glob.glob('%s*' % self.fname[:-3])
        # Delete most of them.
        for f in fglob:
            # Check for the two good ones.
            if f[-3:] in ['tex', 'pdf']: continue
            # Else remove it.
            os.remove(f)
            
    
    # Function to update sweeps
    def UpdateSweeps(self, I=None, cons=[], **kw):
        """Update pages of the report related to data book sweeps
        
        :Call:
            >>> R.UpdateSweeps(I=None, cons=[], **kw)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-28 ``@ddalle``: Started
        """
        # Clear out the lines.
        if 'Sweeps' in self.tex.Section:
            del self.tex.Section['Sweeps'][1:-1]
        # Get sweeps.
        fswps = self.cart3d.opts.get_ReportSweepList(self.rep)
        # Check for a list.
        if type(fswps).__name__ not in ['list', 'ndarray']: fswps = [fswps]
        # Loop through the sweep figures.
        for fswp in fswps:
            # Update the figure.
            self.UpdateSweep(fswp, I=I, cons=cons)
        # Update the text.
        self.tex._updated_sections = True
        self.tex.UpdateLines()
        # Master file location
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        
        
    # Function to update report for several cases
    def UpdateCases(self, I=None, cons=[], **kw):
        """Update several cases and add the lines to the master LaTeX file
        
        :Call:
            >>> R.UpdateCases(I)
            >>> R.UpdateCases(cons=[])
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
            * 2015-05-22 ``@ddalle``: Moved compilation portion to UpdateReport
        """
        # Check for use of constraints instead of direct list.
        I = self.cart3d.x.GetIndices(cons=cons, I=I)
        # Clear out the lines.
        del self.tex.Section['Cases'][1:-1]
        # Loop through those cases.
        for i in I:
            # Update the case
            self.UpdateCase(i)
        # Update the text.
        self.tex._updated_sections = True
        self.tex.UpdateLines()
        # Master file location
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        
    # Clean up cases
    def CleanUpCases(self, I=None, cons=[]):
        """Clean up case figures
        
        :Call:
            >>> R.CleanUpCases(I=None, cons=[])
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for use of constraints instead of direct list.
        I = self.cart3d.x.GetIndices(cons=cons, I=I)
        # Check for folder archiving
        if self.cart3d.opts.get_ReportArchive():
            # Loop through folders.
            for frun in self.cart3d.x.GetFullFolderNames(I):
                # Go to the folder.
                os.chdir(frun)
                # Go up one, archiving if necessary.
                self.cd('..')
                # Go back to root directory.
                os.chdir('..')
        
    # Clean up sweeps
    def CleanUpSweeps(self, I=None, cons=[]):
        """Clean up the folders for all sweeps
        
        :Call:
            >>> R.CleanUpSweeps(I=None, cons=[])
        :Inputs:
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for folder archiving
        if not self.cart3d.opts.get_ReportArchive(): return
        # Get sweep list
        fswps = self.opts.get('Sweeps', [])
        # Check type.
        if type(fswps).__name__ not in ['list', 'ndarray']: fswps = [fswps]
        # Loop through the sweeps.
        for fswp in fswps:
            # Check if only restricting to point currently in the trajectory.
            if self.cart3d.opts.get_SweepOpt(fswp, 'TrajectoryOnly'):
                # Read the data book with the trajectory as the source.
                self.ReadDataBook("trajectory")
            else:
                # Read the data book with the data book as the source.
                self.ReadDataBook("data")
            # Go to the sweep folder.
            os.chdir('sweep-%s' % fswp)
            # Get the sweeps themselves (i.e. lists of indices)
            J = self.GetSweepIndices(fswp, I, cons)
            # Loop through the subsweeps.
            for j in J:
                # Get the first folder name.
                frun = self.cart3d.DataBook.x.GetFullFolderNames(j[0])
                # Go to the folder.
                os.chdir(frun)
                # Go up one, archiving if necessary.
                self.cd('..')
                # Go back up to the sweep folder.
                os.chdir('..')
            # Go back up to report folder.
            os.chdir('..')
        
    # Function to update a sweep
    def UpdateSweep(self, fswp, I=None, cons=[]):
        """Update the pages of a sweep
        
        :Call:
            >>> R.UpdateSweep(fswp, I)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
            * 2015-06-11 ``@ddalle``: Added minimum cases per page
        """
        # Divide the cases into sweeps.
        J = self.GetSweepIndices(fswp, I, cons)
        # Minimum number of cases per page
        nMin = self.cart3d.opts.get_SweepOpt(fswp, 'MinCases')
        # Add a marker in the main document for this sweep.
        self.tex.Section['Sweeps'].insert(-1, '%%!_%s\n' % fswp)
        # Save current location
        fpwd = os.getcwd()
        # Go to report folder.
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Initialize sweep TeX handles.
        self.sweeps[fswp] = {}
        # Name of sweep folder.
        fdir = 'sweep-%s' % fswp
        # Create folder if necessary.
        if not os.path.isdir(fdir):
            os.mkdir(fdir, 0750)
        # Enter the sweep folder.
        os.chdir(fdir)
        # Loop through pages.
        for i in range(len(J)):
            # Check for enough cases to report a sweep.
            if len(J[i]) < nMin: continue
            # Update the sweep page.
            self.UpdateSweepPage(fswp, J[i])
        # Return to original directory
        os.chdir(fpwd)
        
    # Update a page for a single sweep
    def UpdateSweepPage(self, fswp, I, IT=[]):
        """Update one page of a sweep for an automatic report
        
        :Call:
            >>> R.UpdateSweepPage(fswp, I, IT=[])
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of cases in this sweep
            *IT*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of correspond indices for each target
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # --------
        # Checking
        # --------
        # Save location
        fpwd = os.getcwd()
        # Get the case names in this sweep.
        fdirs = self.cart3d.DataBook.x.GetFullFolderNames(I)
        # Split group and case name for first case in the sweep.
        fgrp, fdir = os.path.split(fdirs[0])
        # Use the first case as the name of the subsweep.
        frun = os.path.join(fgrp, fdir)
        # Status update
        print('%s/%s' % (fswp, frun))
        # Make sure group folder exists.
        if not os.path.isdir(fgrp): os.mkdir(fgrp, dmask)
        # Go into the group folder.
        os.chdir(fgrp)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            os.mkdir(fdir, dmask)
        # Go into the folder.
        self.cd(fdir)
        # Add a line to the master document.
        self.tex.Section['Sweeps'].insert(-1,
            '\\input{sweep-%s/%s/%s}\n' % (fswp, frun, self.fname))
        # Read the status file.
        fdirr, Ir, nIterr = self.ReadSweepJSONIter()
        # Extract first component.
        DBc = self.cart3d.DataBook[self.cart3d.DataBook.Components[0]]
        # Get current iteration numbers.
        nIters = list(DBc['nIter'][I])
        # Check if there's anything to do.
        if (Ir == list(I)) and (fdirr == fdirs) and (nIters == nIterr):
            # Return and quit.
            os.chdir(fpwd)
            return
        # -------------
        # Initial setup
        # -------------
        # Status update
        if fdirr is None:
            # New report
            print("  New report page")
        elif fdirr != fdirs:
            # Changing status
            print("  Updating list of cases")
        elif nIters != nIterr:
            # More iterations
            print("  Updating at least one case")
        else:
            # New case
            print("  Modifying index numbers")
        # Create the tex file.
        self.WriteSweepSkeleton(fswp, I[0])
        # Create the TeX handle.
        self.sweeps[fswp][I[0]] = tex.Tex(fname=self.fname)
        # -------
        # Figures
        # -------
        # Get the figures.
        figs = self.cart3d.opts.get_SweepOpt(fswp, "Figures")
        # Loop through the figures.
        for fig in figs:
            # Update the figure.
            self.UpdateFigure(fig, I, fswp)
        # -----
        # Write
        # -----
        # Write the updated lines.
        self.sweeps[fswp][I[0]].Write()
        # Mark the case status.
        self.SetSweepJSONIter(I)
        # Go home.
        os.chdir(fpwd)
        
        
    # Function to create the file for a case
    def UpdateCase(self, i):
        """Open, create if necessary, and update LaTeX file for a case
        
        :Call:
            >>> R.UpdateCase(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # --------
        # Checking
        # --------
        # Get the case name.
        fgrp = self.cart3d.x.GetGroupFolderNames(i)
        fdir = self.cart3d.x.GetFolderNames(i)
        # Go to the report directory if necessary.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Create the folder if necessary.
        if not os.path.isdir(fgrp): os.mkdir(fgrp, dmask)
        # Go to the group folder.
        os.chdir(fgrp)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            os.mkdir(fdir, dmask)
        # Go into the folder.
        self.cd(fdir)
        # ------------
        # Status check
        # ------------
        # Read the status file.
        nr, stsr = self.ReadCaseJSONIter()
        # Get the actual iteration number.
        n = self.cart3d.CheckCase(i)
        # Check status.
        sts = self.cart3d.CheckCaseStatus(i)
        # Call `qstat` if needed.
        if (sts == "INCOMP") and (n is not None):
            # Check the current queue
            sts = self.cart3d.CheckCaseStatus(i, auto=True)
        # Get the figure list
        if n:
            # Nominal case with some results
            figs = self.cart3d.opts.get_ReportFigList(self.rep)
        elif sts == "ERROR":
            # Get the figures for FAILed cases
            figs = self.cart3d.opts.get_ReportErrorFigList(self.rep)
        else:
            # No FAIL file, but no iterations
            figs = self.cart3d.opts.get_ReportZeroFigList(self.rep)
        # If no figures to run; exit.
        if len(figs) == 0:
            # Go home and quit.
            os.chdir(fpwd)
            return
        # Add the line to the master LaTeX file.
        self.tex.Section['Cases'].insert(-1,
            '\\input{%s/%s/%s}\n' % (fgrp, fdir, self.fname))
        # Status update
        print('%s/%s' % (fgrp, fdir))
        # Check if there's anything to do.
        if not ((nr is None) or (n>0 and nr!=n) or (stsr != sts)):
            # Go home and quit.
            os.chdir(fpwd)
            return
        # Status update
        if n != nr:
            # More iterations
            print("  Updating from iteration %s --> %s" % (nr, n))
        elif sts != stsr:
            # Changing status
            print("  Updating status %s --> %s" % (stsr, sts))
        else:
            # New case
            print("  New report at iteration %s" % n)
        # -------------
        # Initial setup
        # -------------
        # Check for the file.
        if not os.path.isfile(self.fname):
            # Make the skeleton file.
            self.WriteCaseSkeleton(i)
        # Open it.
        self.cases[i] = tex.Tex(self.fname)
        # Set the iteration number and status header.
        self.SetHeaderStatus(i)
        # -------
        # Figures
        # -------
        # Check if figures need updating.
        if not ((n==nr) and (stsr=='DONE') and (sts=='PASS')):
            # Loop through figures.
            for fig in figs:
                self.UpdateFigure(fig, i)
        # -----
        # Write
        # -----
        # Write the updated lines.
        self.cases[i].Write()
        # Mark the case status.
        self.SetCaseJSONIter(n, sts)
        # Go home.
        os.chdir(fpwd)
        
        
    # Function to write a figure.
    def UpdateFigure(self, fig, i, fswp=None):
        """Write the figure and update the contents as necessary for *fig*
        
        :Call:
            >>> R.UpdateFigure(fig, i)
            >>> R.UpdateFigure(fig, I, fswp)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of case indices
            *fswp*: :class:`str`
                Name of sweep
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
            * 2015-05-29 ``@ddalle``: Extended to include sweeps
        """
        # -----
        # Setup
        # -----
        # Check for sweep
        if fswp is None:
            # Handle for the case file.
            tx = self.cases[i]
            tf = tx.Section['Figures']
        else:
            # Transfer variable names.
            I = i; i = I[0]
            # Handle for the subsweep file.
            tx = self.sweeps[fswp][i]
            tf = tx.Section['Figures']
        # Figure header line
        ffig = '%%<%s\n' % fig
        # Check for the figure.
        if ffig in tf:
            # Get the location.
            ifig = tf.index(ffig)
            # Get the location of the end of the figure.
            ofig = ifig + tf[ifig:].index('%>\n')
            # Delete those lines (to be replaced).
            del tf[ifig:(ofig+1)]
        else:
            # Insert the figure right before the end.
            ifig = len(tf) - 1
        # --------------
        # Initialization
        # --------------
        # Initialize lines
        lines = []
        # Write the header line.
        lines.append(ffig)
        # Start the figure.
        lines.append('\\begin{figure}[!h]\n')
        # Get the optional header
        fhdr = self.cart3d.opts.get_FigHeader(fig)
        if fhdr:
            # Add the header as a semitrivial subfigure.
            lines.append('\\begin{subfigure}[t]{\\textwidth}\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\end{subfigure}\n')
        # Get figure alignment
        falgn = self.cart3d.opts.get_FigAlignment(fig)
        if falgn.lower() == "center":
            # Centering
            lines.append('\\centering\n')
        # -------
        # Subfigs
        # -------
        # Update the subfigures.
        if fswp is None:
            # Update case subfigures
            lines += self.UpdateCaseSubfigs(fig, i)
        else:
            # Update sweep subfigures
            lines += self.UpdateSweepSubfigs(fig, fswp, I)
        # -------
        # Cleanup
        # -------
        # End the figure for LaTeX
        lines.append('\\end{figure}\n')
        # pyCart report end figure marker
        lines.append('%>\n\n')
        # Add the lines to the section.
        for line in lines:
            tf.insert(ifig, line)
            ifig += 1
        # Update the section
        tx.Section['Figures'] = tf
        tx._updated_sections = True
        # Synchronize the document
        tx.UpdateLines()
        
    # Update subfig for case
    def UpdateCaseSubfigs(self, fig, i):
        """Update subfigures for a case figure *fig*
        
        :Call:
            >>> lines = R.UpdateCaseSubfigs(fig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *i*: :class:`int`
                Case index
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Get list of subfigures.
        sfigs = self.cart3d.opts.get_FigSubfigList(fig)
        # Initialize lines
        lines = []
        # Loop through subfigs.
        for sfig in sfigs:
            # Get the base type.
            btyp = self.cart3d.opts.get_SubfigBaseType(sfig)
            # Process it.
            if btyp == 'Conditions':
                # Get the content.
                lines += self.SubfigConditions(sfig, i)
            elif btyp == 'Summary':
                # Get the force and/or moment summary
                lines += self.SubfigSummary(sfig, i)
            elif btyp == 'PlotCoeff':
                # Get the force or moment history plot
                lines += self.SubfigPlotCoeff(sfig, i)
            elif btyp == 'PlotL1':
                # Get the residual plot
                lines += self.SubfigPlotL1(sfig, i)
            elif btyp == 'Tecplot3View':
                # Get the Tecplot component view
                lines += self.SubfigTecplot3View(sfig, i)
            elif btyp == 'Tecplot':
                # Get the Tecplot layout view
                lines += self.SubfigTecplotLayout(sfig, i)
        # Output
        return lines
        
    # Update subfig for a sweep
    def UpdateSweepSubfigs(self, fig, fswp, I):
        """Update subfigures for a sweep figure *fig*
        
        :Call:
            >>> lines = R.UpdateSweepSubfigs(fig, fswp, I)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray` (:class:`list`)
                List of case indices in the subsweep
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Get list of subfigures.
        sfigs = self.cart3d.opts.get_FigSubfigList(fig)
        # Initialize lines
        lines = []
        # Loop through subfigs.
        for sfig in sfigs:
            # Get the base type.
            btyp = self.cart3d.opts.get_SubfigBaseType(sfig)
            # Process it.
            if btyp == 'Conditions':
                # Get the content.
                lines += self.SubfigConditions(sfig, I)
            elif btyp == 'SweepConditions':
                # Get the variables constant in the sweep
                lines += self.SubfigSweepConditions(sfig, fswp, I[0])
            elif btyp == 'SweepCases':
                # Get the list of cases.
                lines += self.SubfigSweepCases(sfig, fswp, I)
            elif btyp == 'SweepCoeff':
                # Plot a coefficient sweep
                lines += self.SubfigSweepCoeff(sfig, fswp, I)
        # Output
        return lines
        
    # Function to write conditions table
    def SubfigConditions(self, sfig, I):
        """Create lines for a "Conditions" subfigure
        
        :Call:
            >>> lines = R.SubfigConditions(sfig, i)
            >>> lines = R.SubfigConditions(sfig, I)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of case indices
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
            * 2014-06-02 ``@ddalle``: Added range capability
        """
        # Extract the trajectory.
        try:
            # Read the data book trajectory.
            x = self.cart3d.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cart3d.x
        # Check input type.
        if type(I).__name__ in ['list', 'ndarray']:
            # Extract firs index.
            i = I[0]
        else:
            # Use index value given and make a list with one entry.
            i = I
            I = np.array([i])
        # Get the vertical alignment.
        hv = self.cart3d.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cart3d.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cart3d.opts.get_SubfigOpt(sfig, 'Header')
        if fhdr:
            # Write it.
            lines.append('\\noindent\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
        # Begin the table.
        lines.append('\\noindent\n')
        lines.append('\\begin{tabular}{ll|c}\n')
        # Header row
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Variable}} &\n')
        lines.append('\\textbf{\\textsf{Abbr.}} &\n')
        lines.append('\\textbf{\\textsf{Value}} \\\\\n')
        lines.append('\\hline\n')
        
        # Get the variables to skip.
        skvs = self.cart3d.opts.get_SubfigOpt(sfig, 'SkipVars')
        # Loop through the trajectory keys.
        for k in x.keys:
            # Check if it's a skip variable
            if k in skvs: continue
            # Write the variable name.
            line = "{\\small\\textsf{%s}}" % k.replace('_', '\_')
            # Append the abbreviation.
            line += (" & {\\small\\textsf{%s}} & " % 
                x.defns[k]['Abbreviation'].replace('_', '\_'))
            # Get values.
            v = getattr(x,k)[I]
            # Append the value.
            if x.defns[k]['Value'] in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % v[0]
            elif x.defns[k]['Value'] in ['float', 'int']:
                # Check for range.
                if max(v) > min(v):
                    # Print both values.
                    line += "$%s$, [$%s$, $%s$] \\\\\n" % (v[0],min(v),max(v))
                else:
                    # Put the value as a number.
                    line += "$%s$ \\\\\n" % v[0]
            elif x.defns[k]['Value'] in ['hex']:
                # Check for range
                if max(v) > min(v):
                    # Print min/max values.
                    line += "0x%x, [0x%x, 0x%x] \\\\\n" % (v[0],min(v),max(v))
                else:
                    # Put the value as a hex code.
                    line += "0x%x \\\\\n" % v[0]
            elif x.defns[k]['Value'] in ['oct', 'octal']:
                # Check for range
                if max(v) > min(v):
                    # Print min/max values
                    line += "0o%o, [0o%o, 0o%o] \\\\\n" % (v[0],min(v),max(v))
                else:
                    # Put the value as a hex code.
                    line += "0o%o \\\\\n" % v[0]
            else:
                # Put the virst value as string (other type)
                line += "%s \\\\\n" % v[0]
            # Add the line to the table.
            lines.append(line)
        
        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Function to write sweep conditions table
    def SubfigSweepConditions(self, sfig, fswp, i):
        """Create lines for a "SweepConditions" subfigure
        
        :Call:
            >>> lines = R.SubfigSweepConditions(sfig, fswp, I)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *i*: :class:`int`
                Case index
        :Outputs:
            *lines*: :class:`str`
                List of lines in the subfigure
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
            * 2015-06-02 ``@ddalle``: Min/max values
        """
        # Extract the trajectory.
        try:
            # Read the data book trajectory.
            x = self.cart3d.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cart3d.x
        # Get the vertical alignment.
        hv = self.cart3d.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cart3d.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cart3d.opts.get_SubfigOpt(sfig, 'Header')
        if fhdr:
            # Write the header.
            lines.append('\\noindent\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
        # Begin the table.
        lines.append('\\noindent\n')
        lines.append('\\begin{tabular}{l|cc}\n')
        # Header row
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Variable}} &\n')
        lines.append('\\textbf{\\textsf{Value}} &\n')
        lines.append('\\textbf{\\textsf{Constraint}} \\\\ \n')
        lines.append('\\hline\n')
        
        # Get equality and tolerance constraints.
        eqkeys  = self.cart3d.opts.get_SweepOpt(fswp, 'EqCons')
        tolkeys = self.cart3d.opts.get_SweepOpt(fswp, 'TolCons')
        # Loop through the trajectory keys.
        for k in x.keys:
            # Check if it's an equality value.
            if k in eqkeys:
                # Equality constraint
                scon = '$=$'
            elif k in tolkeys:
                # Tolerance constraint
                scon = '$\pm%s$' % tolkeys[k]
            else:
                # Not a constraint.
                continue
            # Write the variable name.
            line = "{\\small\\textsf{%s}} & " % k.replace('_', '\_')
            # Append the value.
            if x.defns[k]['Value'] in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % getattr(x,k)[i]
            elif x.defns[k]['Value'] in ['float', 'int']:
                # Put the value as a number
                line += "$%s$ &" % getattr(x,k)[i]
            elif x.defns[k]['Value'] in ['hex']:
                # Put the value as a hex code.
                line += "0x%x &" % getattr(x,k)[i]
            elif x.defns[k]['Value'] in ['oct', 'octal']:
                # Put the value as a hex code.
                line += "0o%o &" % getattr(x,k)[i]
            else:
                # Just put a string
                line += "%s &" % getattr(x,k)[i]
            # Append the constraint
            line += " %s \\\\ \n" % scon
            # Append the line.
            lines.append(line)
        # Index tolerance
        itol = self.cart3d.opts.get_SweepOpt(fswp, 'IndexTol')
        # Max index
        if type(itol).__name__.startswith('int'):
            # Apply the constraint.
            imax = min(i+itol, x.nCase)
        else:
            # Max index
            imax = x.nCase
        # Write the line
        lines.append("{\\small\\textit{i}} & $%i$ & $[%i,%i]$ \\\\ \n"
            % (i, i, imax))
        
        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    
        
    
    
    
    
        
    
        
    
        
        
    # Function to create coefficient plot and write figure
    def SubfigTecplot3View(self, sfig, i):
        """Create image of surface for one component using Tecplot
        
        :Call:
            >>> lines = R.SubfigTecplot3View(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-10 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cart3d.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cart3d.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Go to the Cart3D folder
        os.chdir(self.cart3d.RootDir)
        # Path to the triangulation
        ftri = os.path.join(frun, 'Components.i.tri')
        # Check for that file.
        if not os.path.isfile(ftri):
            # Try non-intersected file.
            ftri = os.path.join(frun, 'Components.c.tri')
        # Check for the triangulation file.
        if os.path.isfile(ftri):
            # Read the configuration file.
            conf = Config(opts.get_ConfigFile())
            # Read the triangulation file.
            tri = Tri(ftri)
            # Apply the configuration
            tri.conf = conf
        else:
            # Read the standard triangulation
            self.cart3d.ReadTri()
            # Make sure to start from initial position.
            self.cart3d.tri = self.cart3d.tri0.Copy()
            # Rotate for the appropriate case.
            self.cart3d.PrepareTri(i)
            # Extract the triangulation
            tri = self.cart3d.tri
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Get the vertical alignment.
        hv = opts.get_SubfigOpt(sfig, "Position")
        # Get subfigure width
        wsfig = opts.get_SubfigOpt(sfig, "Width")
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = opts.get_SubfigOpt(sfig, "Header")
        # Alignment
        algn = opts.get_SubfigOpt(sfig, "Alignment")
        # Set alignment.
        if algn.lower() == "center":
            lines.append('\\centering\n')
        # Write the header.
        if fhdr:
            # Save the line
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-6pt\n')
        # Go to the report case folder
        os.chdir(fpwd)
        # File name
        fname = "%s-%s" % (sfig, comp)
        # Create the image.
        tri.Tecplot3View(fname, comp)
        # Remove the triangulation
        os.remove('%s.tri' % fname)
        # Include the graphics.
        lines.append('\\includegraphics[width=\\textwidth]{%s/%s.png}\n'
            % (frun, fname))
        # Set the caption.
        if fcpt: lines.append('\\caption*{\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
       
    # Function to create coefficient plot and write figure
    def SubfigTecplotLayout(self, sfig, i):
        """Create image based on a Tecplot layout file
        
        :Call:
            >>> lines = R.SubfigTecplotLayout(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-10 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cart3d.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cart3d.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Get the vertical alignment.
        hv = opts.get_SubfigOpt(sfig, "Position")
        # Get subfigure width
        wsfig = opts.get_SubfigOpt(sfig, "Width")
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = opts.get_SubfigOpt(sfig, "Header")
        # Alignment
        algn = opts.get_SubfigOpt(sfig, "Alignment")
        # Set alignment.
        if algn.lower() == "center":
            lines.append('\\centering\n')
        # Write the header.
        if fhdr:
            # Save the line
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-6pt\n')
        # Go to the Cart3D folder
        os.chdir(self.cart3d.RootDir)
        # Check if the run directory exists.
        if os.path.isdir(frun):
            # Go there.
            os.chdir(frun)
            # Get the most recent PLT files.
            LinkPLT()
            # Layout file
            flay = opts.get_SubfigOpt(sfig, "Layout")
            # Full path to layout file
            fsrc = os.path.join(self.cart3d.RootDir, flay)
            # Get just the file name
            flay = os.path.split(flay)[-1]
            # Read the Mach number option
            omach = opts.get_SubfigOpt(sfig, "Mach")
            # Read the Tecplot layout
            tec = Tecscript(fsrc)
            # Set the Mach number
            try:
                tec.SetMach(getattr(self.cart3d.x,omach)[i])
            except Exception:
                pass
            # Figure width in pixels (can be ``None``).
            wfig = opts.get_SubfigOpt(sfig, "FigWidth")
            # Width in the report
            wplt = opts.get_SubfigOpt(sfig, "Width")
            # Layout file without extension
            fname = ".".join(flay.split(".")[:-1])
            # Figure file name.
            fname = "%s.png" % (sfig)
            # Check for the files.
            if (os.path.isfile('Components.i.plt') and 
                    os.path.isfile('cutPlanes.plt')):
                # Run Tecplot
                try:
                    # Copy the file into the current folder.
                    tec.Write(flay)
                    # Run the layout.
                    ExportLayout(flay, fname=fname, w=wfig)
                    # Move the file.
                    os.rename(fname, os.path.join(fpwd,fname))
                    # Form the line
                    line = (
                        '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                        % (frun, fname))
                    # Include the graphics.
                    lines.append(line)
                    # Remove the layout file.
                    #os.remove(flay)
                except Exception:
                    pass
        # Go to the report case folder
        os.chdir(fpwd)
        # Set the caption.
        if fcpt:
            lines.append('\\caption*{\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    
        
        
        
    
    
            
    
# class Report

        
