"""
Basis module for automated report interface: :mod:`cape.report`
===============================================================

The Cape module for generating automated results reports using PDFLaTeX provides
a single class :class:`cape.report.Report`, which creates a handle for the
``tex`` file and creates folders containing individual figures for each case.
The :class:`cape.report.Report` class is a sort of dual-purpose object that
contains a file interface using :class:`cape.tex.Tex` combined with a capability
to create figures for each case or sweep of cases mostly based on
:mod:`cape.dataBook`.

An automated report is a multi-page PDF generated using PDFLaTeX.  Usually, each
CFD case has one or more pages dedicated to results for that case.  The user
defines a list of figures, each with its own list of subfigures, and these are
generated for each case in the run matrix (subject to any command-line
constraints the user may specify).  Types of subfigures include

    * Table of the values of the input variables for this case
    * Table of force and moment values and statistics
    * Iterative histories of forces or moments (for one or more coefficients)
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more
    
In addition, the user may also define "sweeps," which analyze groups of cases
defined by user-specified constraints.  For example, a sweep may be used to plot
results as a function of Mach number for sets of cases having matching angle of
attack and sideslip angle.  When using a sweep, the report contains one or more
pages for each sweep (rather than one or more pages for each case).

Reports are usually created using system commands of the following format.

    .. code-block: console
    
        $ cape --report
        
The basis report class contains almost all of the capabilities needed for
generating the reports, and so the derivative classes such as
:class:`pyCart.report.Report`, :class:`pyFun.report.Report`, and
:class:`pyOver.report.Report` contain very little additional conent.

The class has an immense number of methods, which can be somewhat grouped into
bookkeeping methods and plotting methods.  The main user-facing methods are
:func:`cape.report.Report.UpdateCases` and
:func:`cape.report.Report.UpdateSweep`.  Each 
:ref:`type of subfigure <cape-json-ReportSubfigure>` has its own method, for
example :func:`cape.report.Report.SubfigPlotCoeff` for ``"PlotCoeff"``  or
:func:`cape.report.Report.SubfigPlotL2` for ``"PlotL2"``.
"""

# File system inreface
import os, json, shutil, glob
# Text processing
import re
# Numerics
import numpy as np
# Paraview Python script
from .bin import pvpython

# Local modules needed
import tex, tar


# Class to interface with report generation and updating.
class Report(object):
    """Interface for automated report generation
    
    :Call:
        >>> R = cape.report.Report(cntl, rep)
    :Inputs:
        *cntl*: :class:`cape.cntl.Cntl`
            Master Cape settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`cape.report.Report` or derivative
            Automated report interface
        *R.cntl*: :class:`cape.cntl.Cntl` or derivative
            Overall solver control interface
        *R.rep*: :class:`str`
            Name of report, same as *rep*
        *R.opts*: :class:`cape.options.Report.Report` or derivative
            Options specific to report *rep*
        *R.cases*: :class:`dict` (:class:`cape.tex.Tex`)
            Dictionary of LaTeX handles for each single-case page
        *R.sweeps*: :class:`dict` (:class:`cape.tex.Tex`)
            Dictionary of LaTeX handles for each single-sweep page
        *R.tex*: :class:`cape.tex.Tex`
            Handle to main LaTeX file
    :Versions:
        * 2015-03-07 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
        * 2015-10-15 ``@ddalle``: Basis version
    """
    # Initialization method
    def __init__(self, cntl, rep):
        """Initialization method"""
        # Save the interface
        self.cntl = cntl
        # Check for this report.
        if rep not in cntl.opts.get_ReportList():
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Go to home folder.
        fpwd = os.getcwd()
        os.chdir(cntl.RootDir)
        # Create the report folder if necessary.
        if not os.path.isdir('report'): cntl.mkdir('report')
        # Set the umask.
        os.umask(cntl.opts.get_umask())
        # Go into the report folder.
        os.chdir('report')
        # Get the options and save them.
        self.rep = rep
        self.opts = cntl.opts['Report'][rep]
        # Initialize a dictionary of handles to case LaTeX files
        self.sweeps = {}
        self.cases = {}
        # Read the file if applicable
        self.OpenMain()
        # Return
        os.chdir(fpwd)
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return '<cape.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
        
    
    # Make a folder
    def mkdir(self, fdir):
        """Create a folder with the correct umask
        
        Relies on ``R.umask``
        
        :Call:
            >>> R.mkdir(fdir)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *fdir*: :class:`str`
                Name of folder to make
        :Versions:
            * 2015-10-15 ``@ddalle``: First versoin
        """
        # Make the directory.
        self.cntl.mkdir(fdir)
        
        
    # Function to open the master latex file for this report.
    def OpenMain(self):
        """Open the primary LaTeX file or write skeleton if necessary
        
        :Call:
            >>> R.OpenMain()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get and save the report file name.
        self.fname = 'report-' + self.rep + '.tex'
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Check for the file and create if necessary.
        if not os.path.isfile(self.fname): self.WriteSkeleton()
        # Open the interface to the master LaTeX file.
        self.tex = tex.Tex(self.fname)
        # Check quality.
        if len(self.tex.SectionNames) < 5:
            raise IOError("Bad LaTeX file '%s'" % self.fname)
        # Return
        os.chdir(fpwd)
        
    # Function to create the skeleton for a master LaTeX file
    def WriteSkeleton(self):
        """Create and write preamble for master LaTeX file for report
        
        :Call:
            >>> R.WriteSkeleton()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')
        # Write the universal header.
        f.write('%$__Class\n')
        f.write('\\documentclass[letter,10pt]{article}\n\n')
        # Write the preamble.
        f.write('%$__Preamble\n')
        # Margins
        f.write('\\usepackage[margin=0.6in,top=0.7in,headsep=0.1in,\n')
        f.write('    footskip=0.15in]{geometry}\n')
        # Other packages
        f.write('\\usepackage{graphicx}\n')
        f.write('\\usepackage{caption}\n')
        f.write('\\usepackage{subcaption}\n')
        f.write('\\usepackage{hyperref}\n')
        f.write('\\usepackage{fancyhdr}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage{amssymb}\n')
        f.write('\\usepackage{times}\n')
        f.write('\\usepackage{placeins}\n')
        f.write('\\usepackage[usenames]{xcolor}\n')
        f.write('\\usepackage[T1]{fontenc}\n')
        f.write('\\usepackage[scaled]{beramono}\n\n')
        
        # Get the title and author and etc.
        fttl  = self.cntl.opts.get_ReportTitle(self.rep)
        fsttl = self.cntl.opts.get_ReportSubtitle(self.rep)
        fauth = self.cntl.opts.get_ReportAuthor(self.rep)
        fafl  = self.cntl.opts.get_ReportAffiliation(self.rep)
        flogo = self.cntl.opts.get_ReportLogo(self.rep)
        ffrnt = self.cntl.opts.get_ReportFrontispiece(self.rep)
        frest = self.cntl.opts.get_ReportRestriction(self.rep)
        # Set the title and author.
        f.write('\\title{%s}\n'  % fttl)
        f.write('\\author{%s}\n' % fauth)
        
        # Format the header and footer
        f.write('\n\\fancypagestyle{pycart}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0.4pt}%\n')
        f.write(' \\renewcommand{\\footrulewidth}{0.4pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write(' \\fancyfoot[C]{\\textbf{\\textsf{%s}}}%%\n' % frest)
        f.write(' \\fancyfoot[R]{\\thepage}%\n')
        # Check for a logo.
        if flogo is not None and  len(flogo) > 0:
            f.write(' \\fancyfoot[L]{\\raisebox{-0.32in}{%\n')
            f.write('  \\includegraphics[height=0.45in]{%s}}}%%\n' % flogo)
        # Finish this primary header/footer format
        f.write('}\n\n')
        # Empty header/footer format for first page
        f.write('\\fancypagestyle{plain}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0pt}%\n')
        f.write(' \\renewcommand{\\footrulewidth}{0pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write('}\n\n')
        
        # Small captions if needed
        f.write('\\captionsetup[subfigure]{textfont=sf}\n')
        f.write('\\captionsetup[subfigure]{skip=0pt}\n\n')
        
        # Macros for setting cases.
        f.write('\\newcommand{\\thecase}{}\n')
        f.write('\\newcommand{\\thesweep}{}\n')
        f.write('\\newcommand{\\setcase}[1]{\\renewcommand{\\thecase}{#1}}\n')
        f.write(
            '\\newcommand{\\setsweep}[1]{\\renewcommand{\\thesweep}{#1}}\n')
        
        # Actual document
        f.write('\n%$__Begin\n')
        f.write('\\begin{document}\n')
        
        # Title page
        f.write('\\pagestyle{plain}\n')
        f.write('\\begin{titlepage}\n')
        f.write('\\vskip4ex\n')
        f.write('\\raggedleft\n')
        # Write the title
        f.write('{\Huge\\sf\\textbf{\n')
        f.write('%s\n' % fttl)
        f.write('}}\n')
        # Write the subtitle
        if fsttl is not None and len(fsttl) > 0:
            f.write('\\vskip2ex\n')
            f.write('{\\Large\\sf\\textit{\n')
            f.write('%s\n' % fsttl)
            f.write('}}\\par\n')
        # Finish the title with a horizontal line
        f.write('\\rule{0.75\\textwidth}{1pt}\\par\n')
        f.write('\\vskip30ex\n')
        # Write the author
        f.write('\\raggedright\n')
        f.write('{\\LARGE\\textrm{\n')
        f.write('%s%%\n' % fauth)
        f.write('}}\n')
        # Write the affiliation
        if fafl is not None and len(fafl) > 0:
            f.write('\\vskip2ex\n')
            f.write('{\\LARGE\\sf\\textbf{\n')
            f.write('%s\n' % fafl)
            f.write('}}\n')
        # Insert the date
        f.write('\\vskip20ex\n')
        f.write('{\\LARGE\\sf\\today}\n')
        # Insert the frontispiece
        if ffrnt is not None and len(ffrnt) > 0:
            f.write('\\vskip20ex\n')
            f.write('\\raggedleft\n')
            f.write('\\includegraphics[height=2in]{%s}\n' % ffrnt)
        # Close the tile page
        f.write('\\end{titlepage}\n')
        f.write('\\pagestyle{pycart}\n\n')
        
        # Skeleton for the sweep
        f.write('%$__Sweeps\n\n')
        
        # Skeleton for the main part of the report.
        f.write('%$__Cases\n')
        
        # Termination of the report
        f.write('\n%$__End\n')
        f.write('\end{document}\n')
        
        # Close the file.
        f.close()
        
        # Return
        os.chdir(fpwd)
        
    # Function to write skeleton for a case.
    def WriteCaseSkeleton(self, i):
        """Initialize LaTeX file for case *i*
        
        :Call:
            >>> R.WriteCaseSkeleton(i, frun)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
            * 2015-10-15 ``@ddalle``: Generic version
        """
        # Get the name of the case
        frun = self.cntl.x.GetFullFolderNames(i)
        
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')

        # Make sure no spilling of figures onto other pages
        f.write('\n\\FloatBarrier\n')
        
        # Write the header.
        f.write('\\clearpage\n')
        f.write('\\setcase{%s}\n' % frun.replace('_', '\\_'))
        f.write('\\phantomsection\n')
        f.write('\\addcontentsline{toc}{section}{\\texttt{\\thecase}}\n')
        f.write('\\fancyhead[L]{\\texttt{\\thecase}}\n\n')
        
        # Empty section for the figures
        f.write('%$__Figures\n')
        f.write('\n')
        
        # Close the file.
        f.close()
        
    # Function to write skeleton for a sweep
    def WriteSweepSkeleton(self, fswp, i):
        """Initialize LaTeX file for sweep *fswp* beginning with case *i*
        
        :Call:
            >>> R.WriteSweepSkeleton(fswp, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
            * 2015-10-15 ``@ddalle``: Generic version
        """
        # Get the name of the case.
        frun = self.cntl.DataBook.x.GetFullFolderNames(i)
        
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')
        
        # Write the header.
        f.write('\n\\clearpage\n')
        f.write('\\setcase{%s}\n' % frun.replace('_', '\\_'))
        f.write('\\setsweep{%s}\n' % fswp)
        f.write('\\phantomsection\n')
        f.write('\\fancyhead[L]{\\texttt{\\thesweep/\\thecase}}\n')
        f.write('\\fancyhead[R]{}\n\n')

        # Set the table of contents entry.
        f.write('\\addcontentsline{toc}{section}' +
            '{\\texttt{\\thesweep/\\thecase}}\n')
        
        # Empty section for the figures
        f.write('%$__Figures\n')
        f.write('\n')
        
        # Close the file.
        f.close()
    
    
    # Function to update report
    def UpdateReport(self, I=None, cons=[], **kw):
        """Update a report based on the list of figures
        
        :Call:
            >>> R.UpdateReport(I)
            >>> R.UpdateReport(cons=[])
        :Inputs:
            *R*: :class:`cape.report.Report`
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
            *R*: :class:`cape.report.Report`
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
        fswps = self.cntl.opts.get_ReportSweepList(self.rep)
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
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        
    # Function to update report for several cases
    def UpdateCases(self, I=None, cons=[], **kw):
        """Update several cases and add the lines to the master LaTeX file
        
        :Call:
            >>> R.UpdateCases(I)
            >>> R.UpdateCases(cons=[])
        :Inputs:
            *R*: :class:`cape.report.Report`
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
        I = self.cntl.x.GetIndices(cons=cons, I=I)
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
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        
    # Clean up cases
    def CleanUpCases(self, I=None, cons=[]):
        """Clean up case folders
        
        :Call:
            >>> R.CleanUpCases(I=None, cons=[])
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for use of constraints instead of direct list.
        I = self.cntl.x.GetIndices(cons=cons, I=I)
        # Check for folder archiving
        if self.cntl.opts.get_ReportArchive():
            # Loop through folders.
            for frun in self.cntl.x.GetFullFolderNames(I):
                # Check for the folder (has trouble if a case is repeated)
                if not os.path.isdir(frun): continue
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
            *R*: :class:`cape.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for folder archiving
        if not self.cntl.opts.get_ReportArchive(): return
        # Get sweep list
        fswps = self.opts.get('Sweeps', [])
        # Check type.
        if type(fswps).__name__ not in ['list', 'ndarray']: fswps = [fswps]
        # Loop through the sweeps.
        for fswp in fswps:
            # Check if only restricting to point currently in the trajectory.
            if self.cntl.opts.get_SweepOpt(fswp, 'TrajectoryOnly'):
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
                # Skip empty sweeps
                if len(j) == 0: continue
                # Get the first folder name.
                frun = self.cntl.DataBook.x.GetFullFolderNames(j[0])
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
            *R*: :class:`cape.report.Report`
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
        nMin = self.cntl.opts.get_SweepOpt(fswp, 'MinCases')
        # Add a marker in the main document for this sweep.
        self.tex.Section['Sweeps'].insert(-1, '%%!_%s\n' % fswp)
        # Save current location
        fpwd = os.getcwd()
        # Go to report folder.
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Initialize sweep TeX handles.
        self.sweeps[fswp] = {}
        # Name of sweep folder.
        fdir = 'sweep-%s' % fswp
        # Create folder if necessary.
        if not os.path.isdir(fdir):
            self.mkdir(fdir)
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
            *R*: :class:`cape.report.Report`
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
        fdirs = self.cntl.DataBook.x.GetFullFolderNames(I)
        # Split group and case name for first case in the sweep.
        fgrp, fdir = os.path.split(fdirs[0])
        # Use the first case as the name of the subsweep.
        frun = os.path.join(fgrp, fdir)
        # Status update
        print('%s/%s' % (fswp, frun))
        # Make sure group folder exists.
        if not os.path.isdir(fgrp): self.mkdir(fgrp)
        # Go into the group folder.
        os.chdir(fgrp)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            self.mkdir(fdir)
        # Go into the folder.
        self.cd(fdir)
        # Add a line to the master document.
        self.tex.Section['Sweeps'].insert(-1,
            '\\input{sweep-%s/%s/%s}\n' % (fswp, frun, self.fname))
        # Read the status file.
        fdirr, Ir, nIterr = self.ReadSweepJSONIter()
        # Extract first component.
        DBc = self.cntl.DataBook[self.cntl.DataBook.Components[0]]
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
        figs = self.cntl.opts.get_SweepOpt(fswp, "Figures")
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
            *R*: :class:`cape.report.Report`
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
        fgrp = self.cntl.x.GetGroupFolderNames(i)
        fdir = self.cntl.x.GetFolderNames(i)
        # Go to the report directory if necessary.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Create the folder if necessary.
        if not os.path.isdir(fgrp): self.mkdir(fgrp)
        # Go to the group folder.
        os.chdir(fgrp)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            self.mkdir(fdir)
        # Go into the folder.
        self.cd(fdir)
        # ------------
        # Status check
        # ------------
        # Read the status file.
        nr, stsr = self.ReadCaseJSONIter()
        # Get the actual iteration number.
        n = self.cntl.CheckCase(i)
        # Check status.
        sts = self.cntl.CheckCaseStatus(i)
        # Call `qstat` if needed.
        if (sts == "INCOMP") and (n is not None):
            # Check the current queue
            sts = self.cntl.CheckCaseStatus(i, auto=True)
        # Get the figure list
        if n:
            # Nominal case with some results
            figs = self.cntl.opts.get_ReportFigList(self.rep)
        elif sts == "ERROR":
            # Get the figures for FAILed cases
            figs = self.cntl.opts.get_ReportErrorFigList(self.rep)
        else:
            # No FAIL file, but no iterations
            figs = self.cntl.opts.get_ReportZeroFigList(self.rep)
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
            *R*: :class:`cape.report.Report`
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
        fhdr = self.cntl.opts.get_FigHeader(fig)
        if fhdr:
            # Add the header as a semitrivial subfigure.
            lines.append('\\begin{subfigure}[t]{\\textwidth}\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\end{subfigure}\n')
        # Get figure alignment
        falgn = self.cntl.opts.get_FigAlignment(fig)
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
        # cape report end figure marker
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
            *R*: :class:`cape.report.Report`
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
        sfigs = self.cntl.opts.get_FigSubfigList(fig)
        # Initialize lines
        lines = []
        # Loop through subfigs.
        for sfig in sfigs:
            # Get the base type.
            btyp = self.cntl.opts.get_SubfigBaseType(sfig)
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
            elif btyp == 'PlotLineLoad':
                # Plot a sectional loads plot
                lines += self.SubfigPlotLineLoad(sfig, i)
            elif btyp == 'PlotL1':
                # Get the residual plot
                lines += self.SubfigPlotL1(sfig, i)
            elif btyp == 'PlotL2':
                # Get the global residual plot
                lines += self.SubfigPlotL2(sfig, i)
            elif btyp == ['PlotResid', 'PlotTurbResid', 'PlotSpeciesResid']:
                # Plot generic residual
                lines += self.SubfigPlotResid(sfig, i)
            elif btyp == 'Paraview':
                # Get the Paraview layout view
                lines += self.SubfigParaviewLayout(sfig, i)
        # Output
        return lines
        
    # Update subfig for a sweep
    def UpdateSweepSubfigs(self, fig, fswp, I):
        """Update subfigures for a sweep figure *fig*
        
        :Call:
            >>> lines = R.UpdateSweepSubfigs(fig, fswp, I)
        :Inputs:
            *R*: :class:`cape.report.Report`
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
        sfigs = self.cntl.opts.get_FigSubfigList(fig)
        # Initialize lines
        lines = []
        # Loop through subfigs.
        for sfig in sfigs:
            # Get the base type.
            btyp = self.cntl.opts.get_SubfigBaseType(sfig)
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
        
    # Function to initialize a subfigure
    def SubfigInit(self, sfig):
        """Create the initial lines of a subfigure
        
        :Call:
            >>> lines = R.SubfigInit(sfig)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to initialize
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                Formatting lines to initialize a subfigure common to all types
        :Versions:
            * 2016-01-16 ``@ddalle``: First version
        """
        # Extract options
        opts = self.cntl.opts
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
        # Output
        return lines
    
    # Function to write conditions table
    def SubfigConditions(self, sfig, I):
        """Create lines for a "Conditions" subfigure
        
        :Call:
            >>> lines = R.SubfigConditions(sfig, i)
            >>> lines = R.SubfigConditions(sfig, I)
        :Inputs:
            *R*: :class:`cape.report.Report`
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
            x = self.cntl.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cntl.x
        # Check input type.
        if type(I).__name__ in ['list', 'ndarray']:
            # Extract firs index.
            i = I[0]
        else:
            # Use index value given and make a list with one entry.
            i = I
            I = np.array([i])
        # Get the vertical alignment.
        hv = self.cntl.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cntl.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cntl.opts.get_SubfigOpt(sfig, 'Header')
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
        skvs = self.cntl.opts.get_SubfigOpt(sfig, 'SkipVars')
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
                line += "{\\small\\textsf{%s}} \\\\\n"%v[0].replace('_','\_')
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
                line += "%s \\\\\n" % v[0].replace('_','\_')
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
            *R*: :class:`cape.report.Report`
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
            x = self.cntl.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cntl.x
        # Get the vertical alignment.
        hv = self.cntl.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cntl.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cntl.opts.get_SubfigOpt(sfig, 'Header')
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
        eqkeys  = self.cntl.opts.get_SweepOpt(fswp, 'EqCons')
        tolkeys = self.cntl.opts.get_SweepOpt(fswp, 'TolCons')
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
                line += "{\\small\\textsf{%s}} \\\\\n" % (
                    getattr(x,k)[i].replace('_', '\_'))
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
                line += "%s &" % getattr(x,k)[i].replace('_','\_')
            # Append the constraint
            line += " %s \\\\ \n" % scon
            # Append the line.
            lines.append(line)
        # Index tolerance
        itol = self.cntl.opts.get_SweepOpt(fswp, 'IndexTol')
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
        
    # Function to write sweep conditions table
    def SubfigSweepCases(self, sfig, fswp, I):
        """Create lines for a "SweepConditions" subfigure
        
        :Call:
            >>> lines = R.SubfigSweepCases(sfig, fswp, I)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray` (:class:`int`)
                Case indices
        :Outputs:
            *lines*: :class:`str`
                List of lines in the subfigure
        :Versions:
            * 2015-06-02 ``@ddalle``: First version
        """
        # Extract the trajectory.
        try:
            # Read the data book trajectory.
            x = self.cntl.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cntl.x
        # Get the vertical alignment.
        hv = self.cntl.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cntl.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cntl.opts.get_SubfigOpt(sfig, 'Header')
        if fhdr:
            # Write the header.
            lines.append('\\noindent\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
        # Begin the table.
        lines.append('\\noindent\n')
        lines.append('\\begin{tabular}{c|l}\n')
        # Header row
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Index}} &\n')
        lines.append('\\textbf{\\textsf{Case}} \\\\ \n')
        lines.append('\\hline\n')
        
        # Get the cases.
        fruns = x.GetFullFolderNames(I)
        # Loop through the cases.
        for j in range(len(I)):
            # Extract index and folder name.
            i = I[j]
            frun = fruns[j].replace('_', '\_')
            # Add the index and folder name.
            lines.append('\\texttt{%i} & \\texttt{%s} \\\\ \n' % (i, frun))
        
        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Function to create coefficient plot and write figure
    def SubfigPlotCoeff(self, sfig, i):
        """Create plot for a coefficient and input lines int LaTeX file
        
        :Call:
            >>> lines = R.SubfigPlotCoeff(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-03-09 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of components
        if type(comp).__name__ in ['list', 'ndarray']:
            # List of components
            nCoeff = max(nCoeff, len(comp))
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption. 
        if fcpt is None:
            # Check for a list.
            if type(comp).__name__ in ['list']:
                # Join them, e.g. "[RSRB,LSRB]/CA"
                fcpt = "[" + ",".join(comp) + "]"
            else:
                # Use the coefficient.
                fcpt = comp
            # Defaut: Wing/CY
            fcpt = ("%s/%s" % (fcpt, coeff))
            # Ensure there are no underscores.
            fcpt = fcpt.replace('_', '\_')
        # First lines.
        lines = self.SubfigInit(sfig)
        # Loop through plots.
        for k in range(nCoeff):
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Numbers of iterations
            nStats = opts.get_SubfigOpt(sfig, "nStats",    k)
            nMin   = opts.get_SubfigOpt(sfig, "nMinStats", k)
            nMax   = opts.get_SubfigOpt(sfig, "nMaxStats", k)
            # Default to databook options
            if nStats is None: nStats = opts.get_nStats()
            if nMin   is None: nMin   = opts.get_nMin()
            if nMax   is None: nMax   = opts.get_nMaxStats()
            # Numbers of iterations for plots
            nPlotIter  = opts.get_SubfigOpt(sfig, "nPlot",      k)
            nPlotFirst = opts.get_SubfigOpt(sfig, "nPlotFirst", k)
            nPlotLast  = opts.get_SubfigOpt(sfig, "nPlotLast",  k)
            # Check for defaults
            if nPlotIter  is None: nPlotIter  = opts.get_nPlotIter(comp)
            if nPlotFirst is None: nPlotFirst = opts.get_nPlotFirst(comp)
            if nPlotLast  is None: nPlotLast  = opts.get_nPlotLast(comp)
            # Check if there are iterations.
            if nIter < 2: continue
            # Don't use iterations before *nMin*
            nMax = min(nMax, nIter-nMin)
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Read the Aero history.
            FM = self.ReadCaseFM(comp)
            # Loop through the transformations.
            for topts in opts.get_DataBookTransformations(comp):
                # Apply the transformation.
                FM.TransformFM(topts, self.cntl.x, i)
            # Get the statistics.
            s = FM.GetStats(nStats=nStats, nMax=nMax, nLast=nPlotLast)
            # Get the manual range to show
            dc = opts.get_SubfigOpt(sfig, "Delta", k)
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation", k)
            # Get the multiple of iterative error to show
            uerr = opts.get_SubfigOpt(sfig, "IterativeError", k)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigHeight", k)
            # Plot options
            kw_p = opts.get_SubfigPlotOpt(sfig, "LineOptions",   k)
            kw_m = opts.get_SubfigPlotOpt(sfig, "MeanOptions",   k)
            kw_s = opts.get_SubfigPlotOpt(sfig, "StDevOptions",  k)
            kw_u = opts.get_SubfigPlotOpt(sfig, "ErrPltOptions", k)
            kw_d = opts.get_SubfigPlotOpt(sfig, "DeltaOptions",  k)
            # Label options
            sh_m = opts.get_SubfigOpt(sfig, "ShowMu", k)
            sh_s = opts.get_SubfigOpt(sfig, "ShowSigma", k)
            sh_d = opts.get_SubfigOpt(sfig, "ShowDelta", k)
            sh_e = opts.get_SubfigOpt(sfig, "ShowEpsilon", k)
            # Label formats
            fmt_m = opts.get_SubfigOpt(sfig, "MuFormat", k)
            fmt_s = opts.get_SubfigOpt(sfig, "SigmaFormat", k)
            fmt_d = opts.get_SubfigOpt(sfig, "DeltaFormat", k)
            fmt_e = opts.get_SubfigOpt(sfig, "EpsilonFormat", k)
            # Draw the plot.
            h = FM.PlotCoeff(coeff, n=nPlotIter, nAvg=s['nStats'],
                nFirst=nPlotFirst, nLast=nPlotLast,
                LineOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
                ShowMu=sh_m, MuFormat=fmt_m,
                ShowDelta=sh_d, DeltaFormat=fmt_d,
                ShowSigma=sh_s, SigmaFormat=fmt_s,
                ShowEspsilon=sh_e, EpsilonFormat=fmt_e,
                FigWidth=figw, FigHeight=figh)
        # Change back to report folder.
        os.chdir(fpwd)
        # Check for a figure to write.
        if nIter >= 2:
            # Get the file formatting
            fmt = opts.get_SubfigOpt(sfig, "Format")
            dpi = opts.get_SubfigOpt(sfig, "DPI")
            # Figure name
            fimg = '%s.%s' % (sfig, fmt)
            fpdf = '%s.pdf' % sfig
            # Save the figure.
            if fmt.lower() in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt.lower() in ['svg']:
                # Save as PDF and SVG
                h['fig'].savefig(fimg)
                h['fig'].savefig(fpdf)
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
                h['fig'].savefig(fpdf)
            # Close the figure.
            h['fig'].clf()
            # Include the graphics.
            lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Function to create coefficient plot and write figure
    def SubfigPlotLineLoad(self, sfig, i):
        """Create plot for a sectional loads profile
        
        :Call:
            >>> lines = R.SubfigPlotLineLoad(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-06-10 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # And their types
        targ_types = {}
        print("Label 004")
        LL = self.ReadLineLoad(comp, i, update=False)
        print("Label 005: LL=%s" % LL)
        # Loop through targets.
        for targ in targs:
            # Try to read the line loads
            try:
                print("Label 015: targ='%s'" % targ)
                os.system('ls /nobackup/ddalle/sls/10008/c3_geom/report/rsrb-ring/m0.50a0.0_M')
                # Read line loads
                self.ReadLineLoad(comp, i, targ=targ, update=False)
                # If read successfully, duplicate data book target
                targ_types[targ] = 'cape'
            except Exception:
                # Read failed
                targ_types[targ] = 'generic'
        print("Label 019")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of components
        if type(comp).__name__ in ['list', 'ndarray']:
            # List of components
            nCoeff = max(nCoeff, len(comp))
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption. 
        if fcpt is None:
            # Check for a list.
            if type(comp).__name__ in ['list']:
                # Join them, e.g. "[RSRB,LSRB]/CA"
                fcpt = "[" + ",".join(comp) + "]"
            else:
                # Use the coefficient.
                fcpt = comp
            # Defaut: Wing/CY
            fcpt = ("%s/%s" % (fcpt, coeff))
            # Ensure there are no underscores.
            fcpt = fcpt.replace('_', '\_')
        # First lines.
        lines = self.SubfigInit(sfig)
        # Initialize plot count
        nPlot = 0
        # Loop through plots.
        for k in range(nCoeff):
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Auto-update flag
            q_auto = opts.get_SubfigOpt(sfig, "AutoUpdate", k)
            # Read the line load data book and read case *i* if possible
            LL = self.ReadLineLoad(comp, i, update=q_auto)
            # Check for case
            if LL is None: continue
            # Add to plot count
            nPlot += 1
            # Loop through the transformations.
            for topts in opts.get_DataBookTransformations(comp):
                pass
                # Apply the transformation.
                # LL.TransformFM(topts, self.cntl.x, i)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigHeight", k)
            # Rotate this figure?
            orient = opts.get_SubfigOpt(sfig, "Orientation", k)
            # Plot label
            lbl = opts.get_SubfigOpt(sfig, "Label", k)
            # Plot options
            kw_p = opts.get_SubfigPlotOpt(sfig, "LineOptions",   k)
            kw_s = opts.get_SubfigPlotOpt(sfig, "SeamOptions",   k)
            # Seam curve options
            sm_ax  = opts.get_SubfigOpt(sfig, "SeamCurves", k)
            sm_loc = opts.get_SubfigOpt(sfig, "SeamLocations", k)
            # Margins
            adj_l = opts.get_SubfigOpt(sfig, 'AdjustLeft',   k)
            adj_r = opts.get_SubfigOpt(sfig, 'AdjustRight',  k)
            adj_t = opts.get_SubfigOpt(sfig, 'AdjustTop',    k)
            adj_b = opts.get_SubfigOpt(sfig, 'AdjustBottom', k)
            # Subplot margin
            w_sfig = opts.get_SubfigOpt(sfig, 'SubplotMargin', k)
            # Axes padding options
            kw_pad = {
                'xpad': opts.get_SubfigOpt(sfig, 'XPad', k),
                'ypad': opts.get_SubfigOpt(sfig, 'YPad', k)
            }
            # Asymmetric padding
            xp = opts.get_SubfigOpt(sfig, 'XPlus', k)
            yp = opts.get_SubfigOpt(sfig, 'YPlus', k)
            xm = opts.get_SubfigOpt(sfig, 'XMinus', k)
            ym = opts.get_SubfigOpt(sfig, 'YMinus', k)
            # Apply asymmetric padding
            if xm is not None: kw_pad['xm'] = xm
            if xp is not None: kw_pad['xp'] = xp
            if ym is not None: kw_pad['ym'] = ym
            if yp is not None: kw_pad['yp'] = yp
            # Draw the plot.
            h = LL.Plot(coeff, 
                Seams=sm_ax, SeamLocation=sm_loc,
                LineOptions=kw_p, SeamOptions=kw_s,
                Label=lbl,
                FigWidth=figw, FigHeight=figh,
                AdjustLeft=adj_l, AdjustRight=adj_r,
                AdjustTop=adj_t, Adjust_Bottom=adj_b,
                SubplotMargin=w_sfig, **kw_pad)
            # Loop through targets
            for targ in targs:
                # Check for generic target
                print("Label 030: targ='%s'" % targ)
                os.system('ls /nobackup/ddalle/sls/10008/c3_geom/report/rsrb-ring/m0.50a0.0_M')
                if targ_types[targ] != 'cape': continue
                # Read the line load data book and read case *i* if possible
                LLT = self.ReadLineLoad(comp, i, targ=targ, update=False)
                print("Label 040")
                os.system('ls /nobackup/ddalle/sls/10008/c3_geom/report/rsrb-ring/m0.50a0.0_M')
                # Check for a find.
                if LLT is None: continue
                # Get target plot label.
                tlbl = self.SubfigTargetPlotLabel(sfig, k, targ)
                # Don't start with comma.
                tlbl = tlbl.lstrip(", ")
                # Specified target plot options
                kw_t = opts.get_SubfigPlotOpt(sfig, "TargetOptions",
                    targs.index(targ))
                # Initialize target plot options.
                kw_l = kw_p
                # Apply non-default options
                for k_i in kw_t: kw_l[k_i] = kw_t[k_i]
                # Draw the plot
                LLT.Plot(coeff, LineOptions=kw_l,
                    Label=tlbl, Legend=True,
                    FigWidth=figw, FigHeight=figh)
        # Change back to report folder.
        os.chdir(fpwd)
        # Check for a figure to write.
        if nPlot > 0:
            # Get the file formatting
            fmt = opts.get_SubfigOpt(sfig, "Format")
            dpi = opts.get_SubfigOpt(sfig, "DPI")
            # Figure name
            fimg = '%s.%s' % (sfig, fmt)
            fpdf = '%s.pdf' % sfig
            # Save the figure.
            if fmt.lower() in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt.lower() in ['svg']:
                # Save as PDF and SVG
                h['fig'].savefig(fimg)
                h['fig'].savefig(fpdf)
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
                h['fig'].savefig(fpdf)
            # Close the figure.
            h['fig'].clf()
            # Include the graphics.
            lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
    
    # Function to plot mean coefficient for a sweep
    def SubfigSweepCoeff(self, sfig, fswp, I):
        """Plot a sweep of a coefficient over several cases
        
        :Call:
            >>> R.SubfigSweepCoeff(sfig, fswp, I)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices in the sweep
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Carpet constraints
        CEq = opts.get_SweepOpt(fswp, "CarpetEqCons")
        CTol = opts.get_SweepOpt(fswp, "CarpetTolCons")
        # Check for carpet constraints.
        if CEq or CTol:
            # Divide sweep into subsweeps.
            J = x.GetSweeps(I=I, EqCons=CEq, TolCons=CTol)
        else:
            # Single sweep
            J = [I]
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # Initialize target sweeps
        JT = {}
        # Loop through targets.
        for targ in targs:
            # Read the target if not present.
            self.cntl.DataBook.ReadTarget(targ)
            # Initialize target sweeps.
            jt = []
            # Loop through carpet subsweeps
            for I in J:
                # Get co-sweep
                jt.append(self.GetTargetSweepIndices(fswp, I[0], targ))
            # Save the sweeps.
            JT[targ] = jt
        # Horizontal axis variable
        xk = opts.get_SweepOpt(fswp, "XAxis")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of components
        if type(comp).__name__ in ['list', 'ndarray']:
            # List of components
            nCoeff = max(nCoeff, len(comp))
        # Number of sweeps
        nSweep = len(J)
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption. 
        if fcpt is None:
            # Check for a list.
            if type(comp).__name__ in ['list']:
                # Join them, e.g. "[RSRB,LSRB]/CA"
                fcpt = "[" + ",".join(comp) + "]"
            else:
                # Use the coefficient.
                fcpt = comp
            # Default format: RSRB/CLM
            fcpt = "%s/%s" % (fcpt, coeff)
        # Ensure there are no underscores.
        fcpt = fcpt.replace("_", "\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Loop through plots.
        for i in range(nSweep*nCoeff):
            # Coefficient index
            k = i % nCoeff
            # Sweep index
            j = i / nCoeff
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Plot label (for legend)
            lbl = self.SubfigPlotLabel(sfig, k)
            # Carpet label appendix
            clbl = ""
            # Append carpet constraints to label if appropriate.
            for kx in CEq:
                # Value of the key or modified key for all points.
                V = eval('x.%s' % kx)
                # Print the subsweep equality constraint in the label.
                clbl += ", %s=%s" % (kx, V[J[j][0]])
            # More carpet constraints
            for kx in CTol:
                # Value of the key or modified key for all points.
                V = eval('x.%s' % kx)
                # Print the subsweep tolerance constraint in the label.
                clbl += u", %s=%s\u00B1%s" % (kx, V[J[j][0]], CTol[kx])
            # Add appendix to label.
            lbl += clbl
            # Don't start with a comma!
            lbl = lbl.lstrip(", ")
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation", k)
            qmmx = opts.get_SubfigOpt(sfig, "MinMax", k)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigHeight", k)
            # Plot options
            kw_p = opts.get_SubfigPlotOpt(sfig, "LineOptions",   i)
            kw_s = opts.get_SubfigPlotOpt(sfig, "StDevOptions",  i)
            kw_m = opts.get_SubfigPlotOpt(sfig, "MinMaxOptions", i)
            # Draw the plot.
            h = self.cntl.DataBook.PlotCoeff(comp, coeff, J[j], x=xk,
                Label=lbl, LineOptions=kw_p,
                StDev=ksig, StDevOptions=kw_s,
                MinMax=qmmx, MinMaxOptions=kw_m,
                FigWidth=figw, FigHeight=figh)
            # Loop through targets
            for targ in targs:
                # Get the target handle.
                DBT = self.cntl.DataBook.GetTargetByName(targ)
                # Check for results to plot.
                if len(JT[targ][j]) == 0:
                    continue
                # Check if the *comp*/*coeff* combination is available.
                if (comp not in DBT.ckeys) or (coeff not in DBT.ckeys[comp]):
                    continue
                # Get target plot label.
                tlbl = self.SubfigTargetPlotLabel(sfig, k, targ) + clbl
                # Don't start with comma.
                tlbl = tlbl.lstrip(", ")
                # Specified target plot options
                kw_t = opts.get_SubfigPlotOpt(sfig, "TargetOptions",
                    targs.index(targ))
                # Initialize target plot options.
                kw_l = kw_p
                # Apply non-default options
                for k_i in kw_t: kw_l[k_i] = kw_t[k_i]
                # Draw the plot
                DBT.PlotCoeff(comp, coeff, JT[targ][j], x=xk,
                    Label=tlbl, LineOptions=kw_l,
                    FigWidth=figw, FigHeight=figh)
        # Check for manually specified axes labels.
        xlbl = opts.get_SubfigOpt(sfig, "XLabel")
        ylbl = opts.get_SubfigOpt(sfig, "YLabel")
        # Specify x-axis label if given.
        if xlbl is not None:
            h['ax'].set_xlabel(xlbl)
        # Specify x-axis label if given.
        if ylbl is not None:
            h['ax'].set_ylabel(ylbl)
        # Change back to report folder.
        os.chdir(fpwd)
        # Get the file formatting
        fmt = opts.get_SubfigOpt(sfig, "Format")
        dpi = opts.get_SubfigOpt(sfig, "DPI")
        # Figure name
        fimg = '%s.%s' % (sfig, fmt)
        # PDF version
        fpdf = '%s.pdf' % sfig
        # Save the figure.
        if fmt.lower() in ['pdf']:
            # Save as vector-based image
            h['fig'].savefig(fimg)
        elif fmt.lower() in ['svg']:
            # Save as SVG and PDF
            h['fig'].savefig(fimg)
            h['fig'].savefig(fpdf)
        else:
            # Save with resolution.
            h['fig'].savefig(fimg, dpi=dpi)
            h['fig'].savefig(fpdf)
        # Close the figure.
        h['fig'].clf()
        # Include the graphics.
        lines.append('\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Get subfig label for plot *k*
    def SubfigPlotLabel(self, sfig, k):
        """Get line label for subfigure plot
        
        :Call:
            >>> lbl = R.SubfigPlotLabel(sfig, k)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *k*: :class:`int`
                Plot index
        :Outputs:
            *lbl*: :class:`str`
                Plot label
        :Versions:
            * 2015-06-04 ``@ddalle``: First version
        """
        # Get options
        opts = self.cntl.opts
        # Get the label if specified.
        lbl = opts.get_SubfigOpt(sfig, "Label", k)
        # Check.
        if lbl is not None: return lbl
        # Component name
        comp = opts.get_SubfigOpt(sfig, "Component", k)
        # List of coefficients
        coeffs = opts.get_SubfigOpt(sfig, "Coefficient")
        # Number of coefficients.
        if type(coeffs).__name__ in ['list']:
            # Coefficient name
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Include coefficient in default label.
            return '%s/%s' % (comp, coeff)
        else:
            # Just use the component
            return comp
        
    # Get subfig label for target plot *k*
    def SubfigTargetPlotLabel(self, sfig, k, targ):
        """Get line label for subfigure plot
        
        :Call:
            >>> lbl = R.SubfigPlotLabel(sfig, k, targ)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *k*: :class:`int`
                Plot index
            *targ*: :class:`str`
                Name of target
        :Outputs:
            *lbl*: :class:`str`
                Plot label for target plot
        :Versions:
            * 2015-06-04 ``@ddalle``: First version
        """
        # Extract options
        opts = self.cntl.opts
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # Target index among list of targets for this subfigure
        kt = targs.index(targ)
        # Get the label if specified.
        lbl = opts.get_SubfigOpt(sfig, "TargetLabel", kt)
        # Check.
        if lbl is not None: return lbl
        # List of components
        comps = opts.get_SubfigOpt(sfig, "Component")
        # List of coefficients
        coeffs = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get the target handle.
        DBT = self.cntl.DataBook.GetTargetByName(targ)
        # Get the nominal label of the target
        tlbl = DBT.topts.get_TargetLabel()
        # Number of coefficients.
        if type(coeffs).__name__ in ['list']:
            # Coefficient name
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Check number of components
            if type(comps).__name__ in ['list']:
                # Component name
                comp = opts.get_SubfigOpt(sfig, "Component", k)
                # Include component and coefficient in label.
                return '%s %s/%s' % (tlbl, comp, coeff)
            else:
                # Include coefficient in label.
                return '%s %s' % (tlbl, coeff)
        elif type(comps).__name__ in ['list']:
            # Component name
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            # Include component in label.
            return '%s %s' % (tlbl, comp)
        else:
            # Just use target
            return tlbl
      
    # Function to get the list of targets for a subfigure
    def SubfigTargets(self, sfig):
        """Return list of targets (by name) for a subfigure
        
        :Call:
            >>> targs = R.SubfigTargets(sfig)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
        :Outputs:
            *targs*: :class:`list` (:class:`str`)
                List of target names
        :Versions:
            * 2015-06-04 ``@ddalle``: First version
        """
        # Target option for this subfigure (defaults to all targets)
        otarg = self.cntl.opts.get_SubfigOpt(sfig, "Target")
        # Process list of targets.
        if type(otarg).__name__ in ['list', 'ndarray']:
            # List of targets directly specified
            targs = otarg
        elif type(otarg).__name__ in ['str', 'unicode']:
            # Single target
            targs = [otarg]
        elif otarg:
            # All targets
            targs = [DBT.Name for DBT in self.cntl.DataBook]
        else:
            # No targets
            targs = []
        # Output
        return targs
        
    # Function to create coefficient plot and write figure
    def SubfigPlotL1(self, sfig, i):
        """Create plot for L1 residual
        
        :Call:
            >>> lines = R.SubfigPlotL1(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-09 ``@ddalle``: First version
            * 2015-11-25 ``@ddalle``: Moved contents to :func:`SubfigPlotResid`
        """
        return self.SubfigPlotResid(sfig, i, c='L1')
        
    # Function to create L2 plot and write figure
    def SubfigPlotL2(self, sfig, i):
        """Create plot for L2 residual
        
        :Call:
            >>> lines = R.SubfigPlotL2(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-11-25 ``@ddalle``: First version
        """
        return self.SubfigPlotResid(sfig, i, c='L2')
        
    # Function to create L2 plot and write figure
    def SubfigPlotTurbResid(self, sfig, i):
        """Create plot for turbulence residual
        
        :Call:
            >>> lines = R.SubfigPlotTurbResid(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-11-25 ``@ddalle``: First version
        """
        return self.SubfigPlotResid(sfig, i, c='TurbResid')
    
    # Function to create coefficient plot and write figure
    def SubfigPlotResid(self, sfig, i, c=None):
        """Create plot for named residual
        
        :Call:
            >>> lines = R.SubfigPlotResid(sfig, i, r=None)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *r*: :class:`str`
                Name of residual to plot (defaults to option from JSON)
        :Versions:
            * 2014-03-09 ``@ddalle``: First version
            * 2015-11-25 ``@ddalle``: Forked from :func:`SubfigPlotL1`
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Numbers of iterations for plots
        nPlotIter  = opts.get_SubfigOpt(sfig, "nPlot")
        nPlotFirst = opts.get_SubfigOpt(sfig, "nPlotFirst")
        nPlotLast  = opts.get_SubfigOpt(sfig, "nPlotLast")
        # Check for defaults.
        if nPlotIter  is None: nPlotIter  = opts.get_nPlotIter(comp)
        if nPlotFirst is None: nPlotFirst = opts.get_nPlotFirst(comp)
        if nPlotLast  is None: nPlotLast  = opts.get_nPlotLast(comp)
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # First lines.
        lines = self.SubfigInit(sfig)
        # Create plot if possible
        if nIter >= 2:
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Get figure width
            figw = opts.get_SubfigOpt(sfig, "FigWidth")
            figh = opts.get_SubfigOpt(sfig, "FigHeight")
            # Read the Aero history.
            H = self.ReadCaseResid(sfig)
            # Options dictionary
            kw_p = {"nFirst": nPlotFirst, "nLast": nPlotLast,
                "FigWidth": figw, "FigHeight": figh}
            # Determine which function to call
            if c == "L1":
                # Draw the "L1" plot
                h = H.PlotL1(n=nPlotIter, **kw_p)
            elif c == "L2":
                # Plot global L2 residual
                h = H.PlotL2(n=nPlotIter, **kw_p)
            elif c == "TurbResid":
                # Plot turbulence residual
                h = H.PlotTurbResid(n=nPlotIter, **kw_p)
            else:
                # Plot named residual
                # Get y-axis label
                kw_p["YLabel"] = opts.get_SubfigOpt(sfig, 'YLabel')
                # Get coefficient
                cr = opts.get_SubfigOpt(sfig, "Residual")
                # Plot it
                h = H.PlotResid(c=cr, n=nPlotIter, **kw_p)
            # Change back to report folder.
            os.chdir(fpwd)
            # Get the file formatting
            fmt = opts.get_SubfigOpt(sfig, "Format")
            dpi = opts.get_SubfigOpt(sfig, "DPI")
            # Figure name
            fimg = '%s.%s' % (sfig, fmt)
            fpdf = '%s.pdf' % sfig
            # Save the figure.
            if fmt.lower() in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt.lower() in ['svg']:
                # Save as PDF and SVG
                h['fig'].savefig(fimg)
                h['fig'].savefig(fpdf)
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
                h['fig'].savefig(fpdf)
            # Close the figure.
            h['fig'].clf()
            # Include the graphics.
            lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fpdf))
        # Set the caption.
        if fcpt: lines.append('\\caption*{\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Read iterative history
    def ReadCaseFM(self, comp):
        """Read iterative history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> FM = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *FM*: ``None`` or :class:`cape.dataBook.CaseFM` derivative
                Case iterative force & moment history for one component
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return None
        
    # Read residual history
    def ReadCaseResid(self, sfig=None):
        """Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Outputs:
            *hist*: ``None`` or :class:`cape.dataBook.CaseResid` derivative
                Case iterative residual history for one case
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return None
        
    # Function to create coefficient plot and write figure
    def SubfigParaviewLayout(self, sfig, i):
        """Create image based on a Paraview Python script
        
        :Call:
            >>> lines = R.SubfigParaviewLayout(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-11-22 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
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
        os.chdir(self.cntl.RootDir)
        # Check if the run directory exists.
        if os.path.isdir(frun):
            # Go there.
            os.chdir(frun)
            # Get the most recent PLT files.
            self.LinkVizFiles()
            # Layout file
            flay = opts.get_SubfigOpt(sfig, "Layout")
            # Full path to layout file
            fsrc = os.path.join(self.cntl.RootDir, flay)
            # Get just the file name
            flay = os.path.split(flay)[-1]
            # Figure width in pixels (can be ``None``).
            wfig = opts.get_SubfigOpt(sfig, "FigWidth")
            # Width in the report
            wplt = opts.get_SubfigOpt(sfig, "Width")
            # Figure extension
            fext = opts.get_SubfigOpt(sfig, "Format")
            # Figure file name.
            fname = "%s.%s" % (sfig, fext)
            # Layout file name
            fout = opts.get_SubfigOpt(sfig, "ImageFile")
            # Name of executable
            fcmd = opts.get_SubfigOpt(sfig, "Command")
            # Run Paraview
            try:
                # Copy the file into the current folder.
                shutil.copy(fsrc, '.')
                # Run the layout.
                pvpython(flay, cmd=fcmd)
                # Move the file to the location this subfig was built in
                os.rename(fout, os.path.join(fpwd,fname))
                # Form the line
                line = (
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fname))
                # Include the graphics.
                lines.append(line)
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
        
    # Function to write summary table
    def SubfigSummary(self, sfig, i):
        """Create lines for a "Summary" subfigure
        
        :Call:
            >>> lines = R.SubfigSummary(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-09 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options
        opts = self.cntl.opts
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Numbers of iterations for statistics
        nStats = opts.get_SubfigOpt(sfig, "nStats")
        nMin   = opts.get_SubfigOpt(sfig, "nMinStats")
        nMax   = opts.get_SubfigOpt(sfig, "nMaxStats")
        # Get the status and data book options
        if nStats is None: nStats = opts.get_nStats()
        if nMin   is None: nMin   = opts.get_nMin()
        if nMax   is None: nMax   = opts.get_nMaxStats()
        # Pure defaults
        if nStats is None: nStats = 1
        # Iteration at which to build table
        nOpt = opts.get_SubfigOpt(sfig, "Iteration")
        # Make sure current progress is a number
        if nIter is None: nIter = 0
        # Get the components.
        comps = opts.get_SubfigOpt(sfig, "Components")
        # Translate into absolute iteration number if relative.
        if nOpt == 0:
            # Use last iteration (standard)
            nCur = nIter
        elif nOpt < 0:
            # Use iteration relative to the end
            nCur = nIter + 1 + nOpt
        else:
            # Use the number
            nCur = nOpt
        # Initialize statistics
        S = {}
        # Get statistics if possible.
        if nCur >= max(1, nMin+nStats):
            # Don't use iterations before *nMin*
            nMax = min(nMax, nCur-nMin)
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(self.cntl.x.GetFullFolderNames(i))
            # Loop through components
            for comp in comps:
                # Read the Aero history.
                FM = self.ReadCaseFM(comp)
                # Loop through the transformations.
                for topts in opts.get_DataBookTransformations(comp):
                    # Apply the transformation.
                    FM.TransformFM(topts, self.cntl.x, i)
                # Get the statistics.
                S[comp] = FM.GetStats(nStats=nStats, nMax=nMax, nLast=nCur)
        # Go back to original folder.
        os.chdir(fpwd)
        # Get the vertical alignment.
        hv = self.cntl.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cntl.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cntl.opts.get_SubfigOpt(sfig, 'Header')
        # Add the iteration number to header
        line = '\\textbf{\\textit{%s}} (Iteration %i' % (fhdr, nCur)
        # Add number of iterations used for statistics
        if len(S)>0:
            # Add stats count.
            line += (', {\\small\\texttt{nStats=%i}}' % S[comps[0]]['nStats'])
        # Close parentheses
        line += ')\\par\n'
        # Write the header.
        lines.append('\\noindent\n')
        lines.append(line)
        # Begin the table with the right amount of columns.
        lines.append('\\begin{tabular}{l' + ('|c'*len(comps)) + '}\n')
        # Write headers.
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Coefficient}}\n')
        for comp in comps:
            lines.append(' & {\\small\\texttt{%s}} \n'
                % comp.replace('_', '\_'))
        lines.append('\\\\\n')
        # Loop through coefficients
        for c in self.cntl.opts.get_SubfigOpt(sfig, "Coefficients"):
            # Convert coefficient title to symbol
            if c in ['CA', 'CY', 'CN']:
                # Just add underscore
                fc = c[0] + '_' + c[1]
            elif c in ['CLL']:
                # Special rolling moment
                fc = 'C_\ell'
            elif c in ['CLM', 'CLN']:
                # Other moments
                fc = 'C_%s' % c[-1].lower()
            else:
                # What?
                fc = 'C_{%s}' % c[1:]
            # Print horizontal line
            lines.append('\\hline\n')
            # Loop through statistical varieties.
            for fs in self.cntl.opts.get_SubfigOpt(sfig, c):
                # Write the description
                if fs == 'mu':
                    # Mean
                    lines.append('\\textit{%s} mean, $\\mu(%s)$\n' % (c, fc))
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'MuFormat')
                elif fs == 'std':
                    # Standard deviation
                    lines.append(
                        '\\textit{%s} standard deviation, $\\sigma(%s)$\n'
                        % (c, fc))
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'SigmaFormat')
                elif fs == 'err':
                    # Uncertainty
                    lines.append('\\textit{%s} iterative uncertainty, ' % c 
                        + '$\\varepsilon(%s)$\n' % fc)
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'EpsFormat')
                elif fs == 'min':
                    # Min value
                    lines.append(
                        '\\textit{%s} minimum, $\\min(%s)$\n' % (c, fc))
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'MuFormat')
                elif fs == 'max':
                    # Min value
                    lines.append(
                        '\\textit{%s} maximum, $\\max(%s)$\n' % (c, fc))
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'MuFormat')
                elif fs == "t":
                    # Target value
                    lines.append(
                        '\\textit{%s} target, $t(%s)$\n' % (c, fc))
                    # Format
                    ff = self.cntl.opts.get_SubfigOpt(sfig, 'MuFormat')
                # Downselect format flag specific to *c* if appropriate
                if type(ff).__name__ == 'dict':
                    # Check for coefficient
                    if c in ff: ff = ff[c]
                # Initialize line
                line = ''
                # Loop through components.
                for comp in comps:
                    # Downselect format flag to *comp* if appropriate
                    if type(ff).__name__ == 'dict':
                        # Select component
                        ffc = ff[comp]
                    else:
                        # Use non-dictionary value
                        ffc = ff
                    # Check for iterations.
                    if nCur <= 0 or comp not in S:
                        # No iterations
                        word = '& $-$ '
                    elif fs == 'mu':
                        # Process value.
                        word = (('& $%s$ ' % ffc) % S[comp][c])
                    elif (fs in ['min', 'max']) or (S[comp]['nStats'] > 1):
                        # Present?
                        if (c+'_'+fs) in S[comp]:
                            # Process min/max or statistical value
                            word = (('& $%s$ ' % ffc) % S[comp][c+'_'+fs])
                        else:
                            # Missing
                            word = '& $-$ '
                    else:
                        # No statistics
                        word = '& $-$ '
                    # Process exponential notation
                    m = re.search('[edED]([+-][0-9]+)', word)
                    # Check for a match to 'e+09', etc.
                    if m is not None:
                        # Existing text from exponent, e.g. 'e+09', 'D-13'
                        txt = m.group(1)
                        # Process the actual exponent text; strip '+' and '0'
                        exp = txt[0].lstrip('+') + txt[1:].lstrip('0')
                        # Replace text
                        word = word.replace(m.group(0), '\\times10^{%s}' % exp)
                    # Add this value to the line
                    line += word
                # Finish the line and append it.
                line += '\\\\\n'
                lines.append(line)
        # Finish table and subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Read the iteration to which the figures for this report have been updated
    def ReadCaseJSONIter(self):
        """Read JSON file to determine the current iteration for the report
        
        The status is read from the present working directory
        
        :Call:
            >>> n, sts = R.ReadCaseJSONIter()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Outputs:
            *n*: :class:`int` or :class:`float`
                Iteration at which the figure has been created
            *sts*: :class:`str`
                Status for the case
        :Versions:
            * 2014-10-02 ``@ddalle``: First version
        """
        # Check for the file.
        if not os.path.isfile('report.json'): return [None, None]
        # Open the file
        f = open('report.json')
        # Read the settings.
        try:
            # Read from file
            opts = json.load(f)
        except Exception:
            # Problem with the file.
            f.close()
            return [None, None]
        # Close the file.
        f.close()
        # Read the status for this iteration
        return tuple(opts.get(self.rep, [None, None]))
        
    # Read the iteration numbers and cases in a sweep
    def ReadSweepJSONIter(self):
        """
        Read JSON file to determine the current iteration and list of cases for
        a sweep report
        
        :Call:
            >>> fruns, I, nIter = R.ReadSweepJSONIter()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Outputs:
            *fruns*: :class:`list` (:class:`str`)
                List of case names
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *nIter*: :class:`list` (:class:`int`)
                Number of iterations for each case
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for the file.
        if not os.path.isfile('report.json'): return (None, None, None)
        # Open the file
        f = open('report.json')
        # Read the settings.
        try:
            # Rread from file
            opts = json.load(f)
        except Exception:
            # Problem with the file
            f.close()
            return (None, None, None)
        # Close the file.
        f.close()
        # Read the status for this iteration.
        return (opts.get("Cases"), opts.get("I"), opts.get("nIter"))
        
    # Set the iteration of the JSON file.
    def SetCaseJSONIter(self, n, sts):
        """Mark the JSON file to say that the figure is up to date
        
        :Call:
            >>> R.SetCaseJSONIter(n, sts)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *n*: :class:`int` or :class:`float`
                Iteration at which the figure has been created
            *sts*: :class:`str`
                Status for the case
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the file.
        if os.path.isfile('report.json'):
            # Open the file.
            f = open('report.json')
            # Read the settings.
            try:
                opts = json.load(f)
            except Exception:
                # Problem with the file.
                opts = {}
            # Close the file.
            f.close()
        else:
            # Create empty settings.
            opts = {}
        # Write 0 instead of null
        if n is None: n = 0
        # Set the iteration number.
        opts[self.rep] = [n, sts]
        # Create the updated JSON file
        f = open('report.json', 'w')
        # Write the contents.
        json.dump(opts, f)
        # Close file.
        f.close()   
        
    # Set the iteration numbers for a sweep
    def SetSweepJSONIter(self, I):
        """Mark the JSON file to state what iteration each case is at
        
        :Call:
            >>> R.SetSweepJSONIter
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *I*: :class:`list` (:class:`int`)
                List of case indices in a sweep
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check for the file.
        if os.path.isfile('report.json'):
            # Open the file.
            f = open('report.json')
            # Read the settings.
            try:
                opts = json.load(f)
            except Exception:
                opts = {}
            # Close the file.
            f.close()
        else:
            # Create empty settings.
            opts = {}
        # Get case names.
        fruns = self.cntl.DataBook.x.GetFullFolderNames(I)
        # Get first component
        DBc = self.cntl.DataBook[self.cntl.DataBook.Components[0]]
        # Get current iteration numbers.
        nIter = list(DBc['nIter'][I])
        # Loop through the cases in the sweep.
        opts = {
            "Cases": fruns,
            "I": list(I),
            "nIter": nIter
        }
        # Create the updated JSON file
        f = open('report.json', 'w')
        # Write the contents.
        json.dump(opts, f, indent=1)
        # Close file.
        f.close()
        
    # Function to set the upper-right header
    def SetHeaderStatus(self, i):
        """Set header to state iteration progress and summary status
        
        :Call:
            >>> R.WriteCaseSkeleton(i)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Get case current iteration
        n = self.cntl.CheckCase(i)
        # Get case number of required iterations
        nMax = self.cntl.GetLastIter(i)
        # Get status
        sts = self.cntl.CheckCaseStatus(i)
        # Form iteration string
        if n is None:
            # Unknown.
            fitr = '-/%s' % self.cntl.opts.get_PhaseIters(-1)
        elif int(n) == n:
            # Use an integer iteration number
            fitr = '%i/%s' % (int(n), nMax)
        else:
            # Use the values.
            fitr = '%s/%s' % (n, nMax)
        # Form string for the status type
        if sts == "PASS":
            # Set green
            fsts = '\\color{green}\\textbf{PASS}'
        elif sts == "ERROR":
            # Set red
            fsts = '\\color{red}\\textbf{ERROR}'
        else:
            # No color (black)
            fsts = '\\textbf{%s}' % sts
        # Form the line.
        line = '\\fancyhead[R]{\\textsf{%s, \large%s}}\n' % (fitr, fsts)
        # Put the line into the text.
        self.cases[i].ReplaceOrAddLineToSectionStartsWith(
            '_header', '\\fancyhead[R]', line, -1)
        # Update sections.
        self.cases[i].UpdateLines()
            
    # Check for any case figures
    def HasCaseFigures(self):
        """Check if there are any case figures for this report
        
        :Call:
            >>> q = R.HasCaseFigures()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Outputs:
            *q*: :class:`bool`
                Whether or not any of report figure lists has nonzero length
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Get the three sets of lists.
        cfigs = self.cntl.opts.get_ReportFigList(self.rep)
        efigs = self.cntl.opts.get_ReportErrorFigList(self.rep)
        zfigs = self.cntl.opts.get_ReportZeroFigList(self.rep)
        # Check if any of them have nozero length.
        return (len(cfigs)>0) or (len(efigs)>0) or (len(zfigs)>0)
    
    # Function to get update sweeps
    def GetSweepIndices(self, fswp, I=None, cons=[]):
        """Divide cases into individual sweeps
        
        :Call:
            >>> J = R.GetSweepIndices(fswp, I=None, cons=[])
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`list` (:class:`int`)
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to update
        :Outputs:
            *J*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of sweep index lists
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check if only restricting to point currently in the trajectory.
        if self.cntl.opts.get_SweepOpt(fswp, 'TrajectoryOnly'):
            # Read the data book with the trajectory as the source.
            self.ReadDataBook("trajectory")
        else:
            # Read the data book with the data book as the source.
            self.ReadDataBook("data")
            # Do not restrict indexes.
            I = np.arange(self.cntl.DataBook.x.nCase)
        # Extract options
        opts = self.cntl.opts
        # Sort variable
        xk = opts.get_SweepOpt(fswp, 'XAxis')
        # Sweep constraints
        EqCons = opts.get_SweepOpt(fswp, 'EqCons')
        TolCons = opts.get_SweepOpt(fswp, 'TolCons')
        IndexTol = opts.get_SweepOpt(fswp, 'IndexTol')
        # Global constraints
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        Indices  = opts.get_SweepOpt(fswp, 'Indices')
        # Extract trajectory.
        x = self.cntl.DataBook.x
        # Turn command-line constraints into indices.
        I0 = x.GetIndices(cons=cons, I=I)
        # Turn report sweep definition into indices.
        I1 = x.GetIndices(cons=GlobCons, I=Indices)
        # Restrict Indices
        I = np.intersect1d(I0, I1)
        # Divide the cases into individual sweeps.
        J = x.GetSweeps(I=I, SortVar=xk,
            EqCons=EqCons, TolCons=TolCons, IndexTol=IndexTol)
        # Output
        return J
        
    # Function to get subset of target catches matching a sweep
    def GetTargetSweepIndices(self, fswp, i0, targ, cons=[]):
        """
        Return indices of a target data set that correspond to sweep constraints
        from a data book point
        
        :Call:
            >>> I = R.GetTargetSweepIndices(fswp, i0, targ)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *i0*: :class:`int`
                Index of point in *R.cntl.DataBook.x* to use as reference
            *targ*: :class:`int`
                Name of the target in data book to use
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of target data indices
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Extract the target interface.
        DBT = self.cntl.DataBook.GetTargetByName(targ)
        # Extract options
        opts = self.cntl.opts
        # Sort variable
        xk = opts.get_SweepOpt(fswp, 'XAxis')
        # Sweep constraints
        EqCons = opts.get_SweepOpt(fswp, 'EqCons')
        TolCons = opts.get_SweepOpt(fswp, 'TolCons')
        # Global constraints
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        # Turn command-line constraints into indices.
        I0 = DBT.x.GetIndices(cons=cons)
        # Turn report sweep definition into indices.
        I1 = DBT.x.GetIndices(cons=GlobCons)
        # Restrict Indices
        I = np.intersect1d(I0, I1)
        # Get the matching sweep.
        I = DBT.x.GetCoSweep(self.cntl.DataBook.x, i0,
            SortVar=xk, EqCons=EqCons, TolCons=TolCons, I=I)
        # Output
        return I
        
    # Function to read the data book and reread it if necessary
    def ReadDataBook(self, fsrc="data"):
        """Read the data book if necessary for a specific sweep
        
        :Call:
            >>> R.ReadDataBook(fsrc="data")
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *fsrc*: :class:`str` [{data} | trajectory]
                Data book source
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
        """
        # Check if there's a data book at all.
        try:
            # Reference the data book
            self.cntl.DataBook
            # Set source to "data"
            try: 
                # Check data book source
                self.cntl.DataBook.source
            except Exception:
                # Set source to none.
                self.cntl.DataBook.source = 'none'
        except Exception:
            # Read the data book.
            self.cntl.ReadDataBook()
            # Set the source to "inconsistent".
            self.cntl.DataBook.source = 'none'
        # Check the existing source.
        if fsrc == self.cntl.DataBook.source:
            # Everything is good.
            return
        elif self.cntl.DataBook.source != 'none':
            # Delete the data book.
            del self.cntl.DataBook
            # Reread
            self.cntl.ReadDataBook()
            # Set the source to "inconsistent".
            self.cntl.DataBook.source = 'none'
        # Check the requested source.
        if fsrc == "trajectory":
            # Match the data book to the trajectory
            self.cntl.DataBook.MatchTrajectory()
            # Save the data book source.
            self.cntl.DataBook.source = "trajectory"
        else:
            # Match the trajectory to the data book.
            self.cntl.DataBook.UpdateTrajectory()
            # Save the data book source.
            self.cntl.DataBook.source = "data"
            
    # Function to link appropriate visualization files
    def LinkVizFiles(self):
        """Create links to appropriate visualization files
        
        :Call:
            >>> R.LinkVizFiles()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        pass
        
    # Function to go into a folder, respecting archive option
    def cd(self, fdir):
        """Interface to :func:`os.chdir`, respecting "Archive" option
        
        This function can only change one directory at a time.
        
        :Call:
            >>> R.cd(fdir)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *fdir*: :class:`str`
                Name of directory to change to
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get archive option.
        q = self.cntl.opts.get_ReportArchive()
        # Check direction.
        if fdir.startswith('..'):
            # Check archive option.
            if q:
                # Tar and clean up if necessary.
                tar.chdir_up()
            else:
                # Go up a folder.
                os.chdir('..')
        else:
            # Untar if necessary
            tar.chdir_in(fdir)
# class Report


