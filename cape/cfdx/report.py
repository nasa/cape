r"""
:mod:`cape.cfdx.report`: Automated CFD report interface
========================================================

The Cape module for generating automated results reports using PDFLaTeX
provides a single class :class:`cape.cfdx.report.Report`, which creates
a handle for the ``tex`` file and creates folders containing individual
figures for each casecntl. The :class:`cape.cfdx.report.Report` class is
a sort of dual-purpose object that contains a file interface using
:class:`cape.filecntl.texfile.Tex` combined with a capability to create
figures for each case or sweep of cases mostly based on
:mod:`cape.cfdx.dataBook`.

An automated report is a multi-page PDF generated using PDFLaTeX.
Usually, each CFD case has one or more pages dedicated to results for
that casecntl. The user defines a list of figures, each with its own
list of subfigures, and these are generated for each case in the run
matrix (subject to any command-line constraints the user may specify).
Types of subfigures include

    * Table of the values of the input variables for this case
    * Table of force and moment values and statistics
    * Iterative histories of forces or moments (one or more coeffs)
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more

In addition, the user may also define "sweeps," which analyze groups of
cases defined by user-specified constraints. For example, a sweep may be
used to plot results as a function of Mach number for sets of cases
having matching angle of attack and sideslip angle. When using a sweep,
the report contains one or more pages for each sweep (rather than one or
more pages for each case).

Reports are usually created using system commands of the following
format.

    .. code-block: console

        $ cape --report

The basis report class contains almost all of the capabilities needed
for generating the reports, and so the derivative classes such as
:class:`cape.pycart.report.Report`, :class:`caep.pyfun.report.Report`,
and :class:`cape.pyover.report.Report` contain very little additional
content.

The class has an immense number of methods, which can be somewhat
grouped into bookkeeping methods and plotting methods. The main
user-facing methods are :func:`cape.cfdx.report.Report.UpdateCases` and
:func:`cape.cfdx.report.Report.UpdateSweep`. Each
:ref:`type of subfigure <cape-json-ReportSubfigure>` has its own method,
for example :func:`cape.cfdx.report.Report.SubfigPlotCoeff` for
``"PlotCoeff"``  or :func:`cape.cfdx.report.Report.SubfigPlotL2` for
``"PlotL2"``.

:See also:
    * :mod:`cape.cfdx.options.reportopts`
    * :class:`cape.cfdx.databook.FMDataBook`
    * :class:`cape.cfdx.databook.CaseFM`
    * :class:`cape.cfdx.lineload.LineLoadDataBook`

"""

# Standard library modules
import ast
import glob
import json
import os
import re
import shutil

# Third-party modules
import numpy as np
from numpy import sqrt, sin, cos, tan, exp

# Local modules
from .. import tar
from .cmdrun import pvpython
from .cntlbase import CntlBase
from ..filecntl import texfile
from ..filecntl.tecfile import ExportLayout, Tecscript, convert_vtk
from .. import pltfile


# List of math functions
_MATH_FUNCTIONS = (sqrt, sin, cos, tan, exp)


# Class to interface with report generation and updating.
class Report(object):
    r"""Interface for automated report generation

    :Call:
        >>> R = Report(cntl, rep)
    :Inputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            CAPE run matrix control instance
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`cape.cfdx.report.Report`
            Automated report interface
    :Attributes:
        * :attr:`cntl`
        * :attr:`cases`
        * :attr:`i`
        * :attr:`rep`
        * :attr:`sweeps`
        * :attr:`tex`
    :Versions:
        * 2015-03-10 ``@ddalle``: v1.0
        * 2015-10-15 ``@ddalle``: v1.1; ``cfdx`` version
    """
  # === __dunder__ ===
    # Initialization method
    def __init__(self, cntl: CntlBase, rep: str):
        r"""Initialization method"""
        #: :class:`cape.cfdx.cntl.Cntl`
        #: Run matrix control instance
        self.cntl = cntl
        #: :class:`int`
        #: Current case index
        self.i = None
        # Check for this report.
        if rep not in cntl.opts.get_ReportList():
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Go to home folder.
        fpwd = os.getcwd()
        os.chdir(cntl.RootDir)
        # Create the report folder if necessary.
        if not os.path.isdir('report'):
            os.mkdir('report')
        # Set the umask.
        os.umask(cntl.opts.get_umask())
        # Go into the report folder.
        os.chdir('report')
        #: :class:`str`
        #: Name of report
        self.rep = rep
        self.opts = cntl.opts['Report'][rep]
        #: :class:`dict`\ [:class:`cape.filecntl.texfile.Tex`]
        #: Dictionary of LaTeX handles for each sweep page
        self.sweeps = {}
        #: :class:`dict`\ [:class:`cape.filecntl.texfile.Tex`]
        #: Dictionary of LaTeX handles for each single-case page
        self.cases = {}
        #: :class:`cape.filecntl.texfile.Tex`
        #: Main LaTeX file interface
        self.tex = None
        # Read the file if applicable
        self.OpenMain()
        #: :class:`bool`
        #: Option to overwrite existing subfigures
        self.force_update = False
        # Return
        os.chdir(fpwd)

    # String conversion
    def __repr__(self):
        r"""String/representation method

        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0
        """
        return '<cape.cfdx.report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__

  # === Folder Functions ===
    # Function to go into a folder, respecting archive option
    def cd(self, fdir: str):
        r"""Interface to :func:`os.chdir`, respecting "Archive" option

        This function can only change one directory at a time.

        :Call:
            >>> R.cd(fdir)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fdir*: :class:`str`
                Name of directory to change to
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # Get archive option.
        q = self.cntl.opts.get_ReportOpt(self.rep, "Archive")
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

  # === LaTeX Files ===
   # --- Main .tex File ---
    # Function to open the master latex file for this report.
    def OpenMain(self):
        r"""Open the primary LaTeX file or write skeleton if necessary

        :Call:
            >>> R.OpenMain()
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # Get and save the report file name.
        self.fname = 'report-' + self.rep + '.tex'
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Check for the file and create if necessary.
        if not os.path.isfile(self.fname):
            self.WriteSkeleton()
        # Open the interface to the master LaTeX file.
        self.tex = texfile.Tex(self.fname)
        # Check quality.
        if len(self.tex.SectionNames) < 5:
            raise IOError("Bad LaTeX file '%s'" % self.fname)
        # Return
        os.chdir(fpwd)

    # Function to create the skeleton for a master LaTeX file
    def WriteSkeleton(self):
        r"""Create and write preamble for master LaTeX file for report

        :Call:
            >>> R.WriteSkeleton()
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
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
        fttl  = self.cntl.opts.get_ReportOpt(self.rep, "Title")
        fsttl = self.cntl.opts.get_ReportOpt(self.rep, "Subtitle")
        fauth = self.cntl.opts.get_ReportOpt(self.rep, "Author")
        fafl  = self.cntl.opts.get_ReportOpt(self.rep, "Affiliation")
        flogo = self.cntl.opts.get_ReportOpt(self.rep, "Logo")
        ffrnt = self.cntl.opts.get_ReportOpt(self.rep, "Frontispiece")
        frest = self.cntl.opts.get_ReportOpt(self.rep, "Restriction")
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
        if flogo is not None and len(flogo) > 0:
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
        f.write('{\\Huge\\sf\\textbf{\n')
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
        f.write('\\end{document}\n')

        # Close the file.
        f.close()

        # Return
        os.chdir(fpwd)

   # --- Folder .tex Files ---
    # Function to write skeleton for a casecntl.
    def WriteCaseSkeleton(self, i):
        r"""Initialize LaTeX file for case *i*

        :Call:
            >>> R.WriteCaseSkeleton(i, frun)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: v1.0
            * 2015-10-15 ``@ddalle``: v1.1; ``cfdx`` version
        """
        # Get the name of the case
        frun = self.cntl.x.GetFullFolderNames(i)
        # Check for the ShowCase option
        qnum = self.cntl.opts.get_ReportOpt(self.rep, "ShowCaseNumber")

        # Create the file (delete if necessary)
        f = open(self.fname, 'w')

        # Make sure no spilling of figures onto other pages
        f.write('\n\\FloatBarrier\n')

        # Write the header.
        f.write('\\clearpage\n')
        f.write('\\setcase{%s}\n' % frun.replace('_', '\\_'))
        f.write('\\phantomsection\n')
        f.write('\\addcontentsline{toc}{section}{\\texttt{\\thecase}}\n')
        # Check if we should write the case number
        if qnum:
            # Add case number
            f.write(
                '\\fancyhead[L]{(\\textbf{Case %s}) \\texttt{\\thecase}}\n\n'
                % i)
        else:
            # Write case name without case number
            f.write('\\fancyhead[L]{\\texttt{\\thecase}}\n\n')

        # Empty section for the figures
        f.write('%$__Figures\n')
        f.write('\n')

        # Close the file.
        f.close()

    # Function to write skeleton for a sweep
    def WriteSweepSkeleton(self, fswp, i):
        r"""Initialize LaTeX file for sweep *fswp* beginning w/ case *i*

        :Call:
            >>> R.WriteSweepSkeleton(fswp, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
            * 2015-10-15 ``@ddalle``: Generic version
        """
        # Get the name of the casecntl.
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
        f.write('\\addcontentsline{toc}{section}')
        f.write('{\\texttt{\\thesweep/\\thecase}}\n')

        # Empty section for the figures
        f.write('%$__Figures\n')
        f.write('\n')

        # Close the file.
        f.close()

    # Function to set the upper-right header
    def SetHeaderStatus(self, i):
        r"""Set header to state iteration progress and summary status

        :Call:
            >>> R.SetHeaderStatus(i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: v1.0
        """
        # Get case current iteration
        n = self.cntl.CheckCase(i)
        # Get case number of required iterations
        nMax = self.cntl.GetLastIter(i)
        # Get status
        sts = self.cntl.check_case_status(i)
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
        line = '\\fancyhead[R]{\\textsf{%s, \\large%s}}\n' % (fitr, fsts)
        # Put the line into the text.
        self.cases[i].ReplaceOrAddLineToSectionStartsWith(
            '_header', '\\fancyhead[R]', line, -1)
        # Update sections.
        self.cases[i].UpdateLines()

  # === Update Functions ===
   # --- General Updates ---
    # Function to update report
    def UpdateReport(self, **kw):
        r"""Update a report based on the list of figures

        :Call:
            >>> R.UpdateReport(I)
            >>> R.UpdateReport(cons=[])
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-22 ``@ddalle``: v1.0
        """
        # Get list of indices.
        I = self.cntl.x.GetIndices(**kw)
        # Update any sweep figures.
        self.UpdateSweeps(I)
        # Update any case-by-case figures.
        if self.HasCaseFigures():
            self.UpdateCases(I)
        # Check for no compile
        compil = kw.get("compile", True)
        # If compile requested
        if compil:
            # Write the file.
            self.tex.Write()
            # Compile it.
            print("Compiling...")
            self.tex.Compile(False)
            # Need to compile twice for links
            print("Compiling...")
            self.tex.Compile(False)
        # Clean up
        print("Cleaning up...")
        # Clean up sweeps
        self.CleanUpSweeps(I=I)
        # Clean up cases
        if self.HasCaseFigures():
            self.CleanUpCases(I=I)
        # Get other 'report-*.*' files.
        fglob = glob.glob('%s*' % self.fname[:-3])
        # Delete most of them
        for f in fglob:
            # Check for the two good ones
            if f[-3:] in ('tex', 'pdf'):
                continue
            # Else remove it
            os.remove(f)
        # Return to top-level folder
        os.chdir(self.cntl.RootDir)

    # Check for any case figures
    def HasCaseFigures(self):
        r"""Check if there are any case figures for this report

        :Call:
            >>> q = R.HasCaseFigures()
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
        :Outputs:
            *q*: :class:`bool`
                Whether any of report's figure lists has nonzero length
        :Versions:
            * 2015-06-03 ``@ddalle``: v1.0
            * 2025-05-15 ``@ddalle``: v1.1; fix for fig lists == None
        """
        # Get the three sets of lists.
        cfigs = self.cntl.opts.get_ReportOpt(self.rep, "Figures", vdef=())
        zfigs = self.cntl.opts.get_ReportOpt(self.rep, "ZeroFigures", vdef=())
        efigs = self.cntl.opts.get_ReportOpt(self.rep, "ErrorFigures", vdef=())
        # De-None if necessary
        nc = 0 if cfigs is None else len(cfigs)
        nz = 0 if zfigs is None else len(zfigs)
        ne = 0 if efigs is None else len(efigs)
        # Check if any of them have nozero length
        return (nc > 0) or (nz > 0) or (ne > 0)

    # Function to update sweeps
    def UpdateSweeps(self, I=None, cons=[], **kw):
        r"""Update pages of the report related to data book sweeps

        :Call:
            >>> R.UpdateSweeps(I=None, cons=[], **kw)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
        # Clear out the lines.
        if 'Sweeps' in self.tex.Section:
            del self.tex.Section['Sweeps'][1:-1]
        # Get sweeps
        fswps = self.cntl.opts.get_ReportOpt(self.rep, "Sweeps")
        # Exit if no sweeps
        if fswps is None:
            return
        # Check for a list
        if not isinstance(fswps, (list, np.ndarray)):
            fswps = [fswps]
        # Loop through the sweep figures
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
    def UpdateCases(self, I=None, **kw):
        r"""Update several cases and add the lines to the master LaTeX file

        :Call:
            >>> R.UpdateCases(I=None, **kw)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-03-10 ``@ddalle``: v1.0
            * 2015-05-22 ``@ddalle``: v1.1; move compile call
        """
        # Check for use of constraints instead of direct list.
        I = self.cntl.x.GetIndices(I=I, **kw)
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

   # --- Case/Sweep Updaters ---
    # Function to update a sweep
    def UpdateSweep(self, fswp, I=None, cons=[]):
        r"""Update the pages of a sweep

        :Call:
            >>> R.UpdateSweep(fswp, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
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
            os.mkdir(fdir)
        # Enter the sweep folder.
        os.chdir(fdir)
        # Loop through pages.
        for i in range(len(J)):
            # Check for enough cases to report a sweep.
            if len(J[i]) < nMin:
                continue
            # Update the sweep page.
            self.UpdateSweepPage(fswp, J[i])
        # Return to original directory
        os.chdir(fpwd)

    # Update a page for a single sweep
    def UpdateSweepPage(self, fswp, I, IT=[]):
        r"""Update one page of a sweep for an automatic report

        :Call:
            >>> R.UpdateSweepPage(fswp, I, IT=[])
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of cases in this sweep
            *IT*: :class:`list` (:class:`numpy.ndarray`\ [:class:`int`])
                List of correspond indices for each target
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
            * 2023-10-21 ``@ddalle``: v1.1; allow arbitrary depth
        """
        # --------
        # Checking
        # --------
        # Save location
        fpwd = os.getcwd()
        # Get the case names in this sweep
        fdirs = self.cntl.DataBook.x.GetFullFolderNames(I)
        # Split group and case name for first case in the sweep
        fgrp, fdir = os.path.split(fdirs[0])
        # Use the first case as the name of the subsweep
        frun = os.path.join(fgrp, fdir)
        # Status update
        print(f"{fswp}{os.sep}{frun}")
        # Make sure group folder exists
        for fpart in fgrp.split(os.sep):
            # Create part if needed
            if not os.path.isdir(fpart):
                os.mkdir(fpart)
            # Enter folder
            os.chdir(fpart)
        # Create the case folder if necessary
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            os.mkdir(fdir)
        # Go into the folder
        self.cd(fdir)
        # Add a line to the master document
        self.tex.Section['Sweeps'].insert(
            -1, '\\input{sweep-%s/%s/%s}\n' % (fswp, frun, self.fname))
        # -------------
        # Initial setup
        # -------------
        # Create the tex file
        self.WriteSweepSkeleton(fswp, I[0])
        # Create the TeX handle
        self.sweeps[fswp][I[0]] = texfile.Tex(fname=self.fname)
        # -------
        # Figures
        # -------
        # Save the subfigures
        self.SaveSubfigs(I, fswp)
        # Get the figures
        figs = self.cntl.opts.get_SweepOpt(fswp, "Figures")
        # Loop through the figures
        for fig in figs:
            # Update the figure
            self.UpdateFigure(fig, I, fswp)
        # -----
        # Write
        # -----
        # Write the updated lines
        self.sweeps[fswp][I[0]].Write()
        # Go home
        os.chdir(fpwd)

    # Get appropriate list of figures
    def GetFigureList(self, i, fswp=None):
        r"""Get list of figures for a report or sweep page

        :Call:
            >>> figs = R.GetFigureList(i)
            >>> figs = R.GetFigureList(I, fswp)
        :Inputs:
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
            *fswp*: :class:`str`
                Name of sweep
        :Outputs:
            *figs*: :class:`list`
                List of figure names
        :Versions:
            * 2017-05-27 ``@ddalle``: v1.0
        """
        # Check for sweep
        if fswp is None:
            # Get the actual iteration number.
            n = self.cntl.CheckCase(i)
            # Get required number of iterations for report
            nMin = self.cntl.opts.get_ReportOpt(self.rep, "MinIter")
            # Move on if iteration count not yet achieved
            if (nMin is not None) and ((n is None) or (n < nMin)):
                # Go home and quit.
                return []
            # Check status.
            sts = self.cntl.check_case_status(i)
            # Call `qstat` if needed.
            if (sts == "INCOMP") and (n is not None):
                # Check the current queue
                sts = self.cntl.check_case_status(i, auto=True)
            # Get the figure list
            if n:
                # Nominal case with some results
                figs = self.cntl.opts.get_ReportOpt(self.rep, "Figures")
            elif sts == "ERROR":
                # Get the figures for FAILed cases
                figs = self.cntl.opts.get_ReportErrorFigures(self.rep)
            else:
                # No FAIL file, but no iterations
                figs = self.cntl.opts.get_ReportOpt(self.rep, "ZeroFigures")
        else:
            # Get the list of sweep figures
            figs = self.cntl.opts.get_SweepOpt(fswp, "Figures")
        # Output
        return figs

    # Function to create the file for a case
    def UpdateCase(self, i):
        r"""Open, create if necessary, and update LaTeX file for a case

        :Call:
            >>> R.UpdateCase(i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2023-10-21 ``@ddalle``: v1.1; allow arbitray depth
        """
        # --------
        # Checking
        # --------
        # Get the case name
        frun = self.cntl.x.GetFullFolderNames(i)
        # Get just the last level of case
        fgrp, fdir = os.path.split(frun)
        # Go to the report directory if necessary.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Loop through parts of *fgrp*
        for fpart in fgrp.split(os.sep):
            # Create the folder if necessary
            if not os.path.isdir(fpart):
                os.mkdir(fpart)
            # Enter the folder
            os.chdir(fpart)
        # Create the case folder if necessary.
        if not (os.path.isfile(fdir+'.tar') or os.path.isdir(fdir)):
            os.mkdir(fdir)
        # Go into the folder.
        self.cd(fdir)
        # ------------
        # Status check
        # ------------
        # Get the actual iteration number.
        n = self.cntl.CheckCase(i)
        # Get required number of iterations for report
        nMin = self.cntl.opts.get_ReportOpt(self.rep, "MinIter")
        # Move on if iteration count not yet achieved
        if (nMin is not None) and ((n is None) or (n < nMin)):
            # Go home and quit.
            os.chdir(fpwd)
            return
        # Check status.
        sts = self.cntl.check_case_status(i)
        # Call `qstat` if needed.
        if (sts == "INCOMP") and (n is not None):
            # Check the current queue
            sts = self.cntl.check_case_status(i, auto=True)
        # Get the figure list
        if n:
            # Nominal case with some results
            figs = self.cntl.opts.get_ReportOpt(self.rep, "Figures")
        elif sts == "ERROR":
            # Get the figures for FAILed cases
            figs = self.cntl.opts.get_ReportErrorFigures(self.rep)
        else:
            # No FAIL file, but no iterations
            figs = self.cntl.opts.get_ReportOpt(self.rep, "ZeroFigures")
        # If no figures to run; exit.
        if (figs is None) or (len(figs) == 0):
            # Go home and quit.
            os.chdir(fpwd)
            return
        # Add the line to the master LaTeX file.
        self.tex.Section['Cases'].insert(
            -1, '\\input{%s/%s/%s}\n' % (fgrp, fdir, self.fname))
        # Status update
        print('%s/%s' % (fgrp, fdir))
        # -------------
        # Initial setup
        # -------------
        # Save case number
        self.i = i
        # Check for the file
        if not os.path.isfile(self.fname):
            # Make the skeleton file.
            self.WriteCaseSkeleton(i)
        # Open it.
        self.cases[i] = texfile.Tex(self.fname)
        # Set the iteration number and status header.
        self.SetHeaderStatus(i)
        # -------
        # Figures
        # -------
        # Save the subfigures
        self.SaveSubfigs(i)
        # Loop through figures.
        for fig in figs:
            self.UpdateFigure(fig, i)
        # -----
        # Write
        # -----
        # Write the updated lines.
        self.cases[i].Write()
        # Go home.
        os.chdir(fpwd)

   # --- Figure/Subfigure Updaters ---
    # Function to save the subfigures
    def SaveSubfigs(self, i, fswp=None):
        r"""Save the current text of subfigures

        :Call:
            >>> R.SaveSubfigs(i)
            >>> R.SaveSubfigs(I, fswp)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
            *fswp*: :class:`str`
                Name of sweep
        :Attributes:
            *R.subfigs*: :class:`dict`\ [:class:`list`]
                List of LaTeX lines in each subfigure by name
        :Versions:
            * 2017-05-27 ``@ddalle``: v1.0
        """
        # -----
        # Setup
        # -----
        # Check for sweep
        if fswp is None:
            # Handle for the case file.
            tx = self.cases[i]
        else:
            # Transfer variable names
            I = i
            i = I[0]
            # Handle for the subsweep file
            tx = self.sweeps[fswp][i]
        # Initialize holder
        self.subfigs = {}
        # Find line numbers of start and end of each subfig
        sfiga = tx.GetIndexStartsWith("\\begin{subfigure}")
        sfigb = tx.GetIndexStartsWith("\\end{subfigure}")
        # Initialize number of subfigs
        nsfig = 0
        # List of figures
        figs = self.GetFigureList(i, fswp=fswp)
        # Loop through figs
        for fig in figs:
            # Loop through subfigs
            for sfig in self.cntl.opts.get_FigOpt(fig, "Subfigures"):
                # Check if the subfigure existed
                if nsfig >= len(sfiga):
                    # No subfigure yet
                    self.subfigs[sfig] = []
                    continue
                # Save lines
                self.subfigs[sfig] = tx.lines[sfiga[nsfig]:sfigb[nsfig]+1]
                # Increase count
                nsfig += 1

    # Function to write a figure.
    def UpdateFigure(self, fig, i, fswp=None):
        r"""Write the figure and update the contents as needed for *fig*

        :Call:
            >>> R.UpdateFigure(fig, i)
            >>> R.UpdateFigure(fig, I, fswp)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
            *fswp*: :class:`str`
                Name of sweep
        :Versions:
            * 2014-03-08 ``@ddalle``: v1.0
            * 2015-05-29 ``@ddalle``: v1.1; include sweeps
            * 2016-10-25 ``@ddalle``: v1.2; add checks
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
            I = i
            i = I[0]
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
        fhdr = self.cntl.opts.get_FigOpt(fig, "Header")
        if fhdr:
            # Add the header as a semitrivial subfigure.
            lines.append('\\begin{subfigure}[t]{\\textwidth}\n')
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\end{subfigure}\n')
        # Get figure alignment
        falgn = self.cntl.opts.get_FigOpt(fig, "Alignment")
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
        r"""Update subfigures for a case figure *fig*

        :Call:
            >>> lines = R.UpdateCaseSubfigs(fig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *i*: :class:`int`
                Case index
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
        """
        # Get list of subfigures.
        sfigs = self.cntl.opts.get_FigOpt(fig, "Subfigures")
        # Initialize lines
        lines = []
        # Get status
        n = self.cntl.CheckCase(i)
        # Read settings
        rc = self.ReadCaseJSON()
        # Loop through subfigs.
        for sfig in sfigs:
            # Check the status (also prints status update)
            q = self.CheckSubfigStatus(sfig, rc, n)
            # Use a separate function to find the right updater
            lines = self.SubfigSwitch(sfig, i, lines, q)
            # Update the settings
            rc["Status"][sfig] = n
            rc["Subfigures"][sfig] = self.cntl.opts.get_SubfigCascade(sfig)
        # Write the new settings
        self.WriteCaseJSON(rc)
        # Output
        return lines

    # Check status of a subfigure and give status update
    def CheckSubfigStatus(self, sfig, rc, n):
        r"""Check whether or not to update a subfigure and print status

        :Call:
            >>> q = R.CheckSubfigStatus(sfig, rc, n)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to check
            *rc*: :class:`dict`
                Dictionary from ``report.json``
            *n*: :class:`int` | ``None``
                Current iteration number
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not to update the subfigure
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0
        """
        # Get the status options
        stsr = rc.get("Status", {})
        # Check if the subfigure has been handled at all
        if sfig not in stsr:
            # New subfigure
            print("  %s: New subfig at iteration %s" % (sfig, n))
            return True
        # Get the iteration status
        nr = stsr.get(sfig)
        # Check the iteration
        if n != nr:
            # Iteration update
            print("  %s: Update %s --> %s" % (sfig, nr, n))
            return True
        # Get the definition last used to generate this subfig
        defr = rc.get("Subfigures", {}).get(sfig, {})
        # Get the present definition
        defo = self.cntl.opts.get_SubfigCascade(sfig)
        # Compare the subfig definitions
        if defr != defo:
            # Definition changed
            print("  %s: Definition updated" % sfig)
            return True
        # Check for forced
        if self.force_update:
            # Forced update
            print(" %s: Update forced" % sfig)
            return True
        # If reached this point, no update
        return False

    # Point to the correct subfigure updater
    def SubfigSwitch(self, sfig, i, lines, q):
        r"""Switch function to find the correct subfigure function

        This function may need to be defined for each CFD solver

        :Call:
            >>> lines = R.SubfigSwitch(sfig, i, lines, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *lines*: :class:`list`\ [:class:`str`]
                List of lines already in LaTeX file
            *q*: ``True`` | ``False``
                Whether or not to regenerate subfigure
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Updated list of lines for LaTeX file
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0, from :func:`UpdateSubfig`
        """
        # Get the base type.
        btyp = self.cntl.opts.get_SubfigBaseType(sfig)
        # Process it.
        if btyp == 'Conditions':
            # Get the content.
            lines += self.SubfigConditions(sfig, i, q)
        elif btyp == 'Summary':
            # Get the force and/or moment summary
            lines += self.SubfigSummary(sfig, i, q)
        elif btyp == 'PlotCoeff':
            # Get the force or moment history plot
            lines += self.SubfigPlotCoeff(sfig, i, q)
        elif btyp == 'PlotLineLoad':
            # Plot a sectional loads plot
            lines += self.SubfigPlotLineLoad(sfig, i, q)
        elif btyp == 'PlotL1':
            # Get the residual plot
            lines += self.SubfigPlotL1(sfig, i, q)
        elif btyp == 'PlotL2':
            # Get the global residual plot
            lines += self.SubfigPlotL2(sfig, i, q)
        elif btyp == ['PlotResid', 'PlotTurbResid', 'PlotSpeciesResid']:
            # Plot generic residual
            lines += self.SubfigPlotResid(sfig, i, q)
        elif btyp == 'Paraview':
            # Get the Paraview layout view
            lines += self.SubfigParaviewLayout(sfig, i, q)
        elif btyp == 'Tecplot':
            # Use a Tecplot layout
            lines += self.SubfigTecplotLayout(sfig, i, q)
        elif btyp == 'Image':
            # Coy an image
            lines += self.SubfigImage(sfig, i, q)
        else:
            print("  %s: No function for subfigure type '%s'" % (sfig, btyp))
        # Output
        return lines

    # Update subfig for a sweep
    def UpdateSweepSubfigs(self, fig, fswp, I):
        r"""Update subfigures for a sweep figure *fig*

        :Call:
            >>> lines = R.UpdateSweepSubfigs(fig, fswp, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices in the subsweep
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
            * 2016-10-25 ``@ddalle``: v1.1; move some to SubfigSwitch()
        """
        # Get list of subfigures.
        sfigs = self.cntl.opts.get_FigOpt(fig, "Subfigures")
        # Initialize lines
        lines = []
        # Read settings
        rc = self.ReadCaseJSON()
        # Get primary constraints
        EqCons   = self.cntl.opts.get_SweepOpt(fswp, "EqCons")
        TolCons  = self.cntl.opts.get_SweepOpt(fswp, "TolCons")
        GlobCons = self.cntl.opts.get_SweepOpt(fswp, "GlobalCons")
        # Loop through subfigs.
        for sfig in sfigs:
            # Get component for this component
            DBc = self.GetSubfigRefComponent(sfig)
            # Get the co sweep
            J = DBc.FindCoSweep(
                self.cntl.DataBook.x, I[0],
                EqCons, TolCons, GlobCons)
            # Match up trajectory
            DBc.UpdateRunMatrix()
            # Get the list of cases for the subfigures
            fruns = DBc.x.GetFullFolderNames(J)
            # Get current iteration number
            nIter = list(DBc['nIter'][J])
            # Check the status (also prints status update)
            q = self.CheckSweepSubfigStatus(sfig, rc, fruns, nIter)
            # Process the subfigure
            lines = self.SweepSubfigSwitch(sfig, fswp, I, lines, q)
            # Save the status
            rc["Status"][sfig] = {
                "Cases": fruns,
                "nIter": nIter
            }
            # Save the definition
            rc["Subfigures"][sfig] = self.cntl.opts.get_SubfigCascade(sfig)
        # Write the new settings
        self.WriteCaseJSON(rc)
        # Output
        return lines

    # Get handle for component
    def GetSubfigRefComponent(self, sfig):
        r"""Get handle for data book component for a sweep subfigure

        :Call:
            >>> DBc = R.GetSubfigRefComponent(sfig)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Component data book
        :Versions:
            * 2017-04-23 ``@ddalle``: v1.0
        """
        # Get the list of components
        comp = self.cntl.opts.get_SubfigOpt(sfig, "Component")
        # Check type
        if comp is None:
            # No component... just global ref component
            return self.cntl.DataBook.GetRefComponent()
        elif type(comp).__name__ in ["list", "ndarray"]:
            # List... use the first component in list
            return self.ReadFM(comp[0])
        else:
            # Use single component
            return self.ReadFM(comp)

    # Point to the correct subfigure updater
    def SweepSubfigSwitch(self, sfig, fswp, I, lines, q):
        r"""Switch function to find the correct subfigure function

        This function may need to be defined for each CFD solver

        :Call:
            >>> lines = R.SubfigSwitch(sfig, fswp, I, lines)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices in the subsweep
            *lines*: :class:`list`\ [:class:`str`]
                List of lines already in LaTeX file
            *q*: ``True`` | ``False``
                Whether or not to regenerate subfigure
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Updated list of lines for LaTeX file
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0, from :func:`UpdateSubfig`
        """
        # Get the base type.
        btyp = self.cntl.opts.get_SubfigBaseType(sfig)
        # Process it.
        if btyp == 'Conditions':
            # Get the content.
            lines += self.SubfigConditions(sfig, I, q)
        elif btyp == 'SweepConditions':
            # Get the variables constant in the sweep
            lines += self.SubfigSweepConditions(sfig, fswp, I[0], q)
        elif btyp == 'SweepCases':
            # Get the list of cases.
            lines += self.SubfigSweepCases(sfig, fswp, I, q)
        elif btyp == 'SweepCoeff':
            # Plot a coefficient sweep
            lines += self.SubfigSweepCoeff(sfig, fswp, I, q)
        elif btyp == 'SweepCoeffHist':
            # Plot a coefficient histogram
            lines += self.SubfigSweepCoeffHist(sfig, fswp, I, q)
        elif btyp == 'ContourCoeff':
            # Contour plot of slice results
            lines += self.SubfigContourCoeff(sfig, fswp, I, q)
        elif btyp == 'SweepLineLoad':
            # Group of line loads
            lines += self.SubfigPlotLineLoadGroup(sfig, fswp, I, q)
        else:
            # No figure found
            print("  %s: No function for subfigure type '%s'" % (sfig, btyp))
        # Output
        return lines

    # Check status of a subfigure and give status update
    def CheckSweepSubfigStatus(self, sfig, rc, fruns, nIter):
        r"""Check whether or not to update a subfigure and print status

        :Call:
            >>> q = R.CheckSweepSubfigStatus(sfig, I, rc, fruns, nIter)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *sfig*: :class:`str`
                Name of subfigure to check
            *rc*: :class:`dict`
                Dictionary from ``report.json``
            *fruns*: :class:`list`\ [:class:`str`]
                List of cases in the sweep
            *nIter*: :class:`list`\ [:class:`int`]
                List of iterations for each case
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not to update the subfigure
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0
        """
        # Get the status options
        stsr = rc.get("Status", {})
        # Check if the subfigure has been handled at all
        if sfig not in stsr:
            # New subfigure
            print("  %s: New subfig" % sfig)
            return True
        # Get the iteration status
        stsf = stsr.get(sfig)
        # Get the list of cases and current iterations
        frunsr = stsf.get("Cases", [])
        nIterr = stsf.get("nIter", [])
        # Check the data book status
        if fruns != frunsr:
            # Case list update
            print("  %s: Updating list of cases" % sfig)
            return True
        # Check iterations for data book
        if nIterr != nIter:
            # Iteration update
            print("  %s: Data book update to at least one case" % sfig)
            return True
        # Get the definition last used to generate this subfig
        defr = rc.get("Subfigures", {}).get(sfig, {})
        # Get the present definition
        defo = self.cntl.opts.get_SubfigCascade(sfig)
        # Compare the subfig definitions
        if defr != defo:
            # Definition changed
            print("  %s: Definition updated" % sfig)
            return True
        # Check for force
        if self.force_update:
            # Forced update
            print(" %s: Update forced" % sfig)
            return True
        # If reached this point, no update
        return False

  # === Cleanup ===
    # Clean up cases
    def CleanUpCases(self, I=None, cons=[]):
        r"""Clean up case folders

        :Call:
            >>> R.CleanUpCases(I=None, cons=[])
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
        """
        # Check for use of constraints instead of direct list.
        I = self.cntl.x.GetIndices(cons=cons, I=I)
        # Check for folder archiving
        if self.cntl.opts.get_ReportOpt(self.rep, "Archive"):
            # Loop through folders.
            for frun in self.cntl.x.GetFullFolderNames(I):
                # Check for the folder (has trouble if a case is repeated)
                if not os.path.isdir(frun):
                    continue
                # Go to the folder.
                os.chdir(frun)
                # Go up one, archiving if necessary.
                self.cd('..')
                # Go back to root directory.
                os.chdir('..')

    # Clean up sweeps
    def CleanUpSweeps(self, I=None, cons=[]):
        r"""Clean up the folders for all sweeps

        :Call:
            >>> R.CleanUpSweeps(I=None, cons=[])
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
        """
        # Check for folder archiving
        if not self.cntl.opts.get_ReportOpt(self.rep, "Archive"):
            return
        # Get sweep list
        fswps = self.opts.get('Sweeps', [])
        # Check type.
        if not isinstance(fswps, (list, np.ndarray)):
            fswps = [fswps]
        # Loop through the sweeps.
        for fswp in fswps:
            # Check if only restricting to point currently in the trajectory.
            if self.cntl.opts.get_SweepOpt(fswp, 'RunMatrixOnly'):
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
                if len(j) == 0:
                    continue
                # Get the first folder name.
                frun = self.cntl.DataBook.x.GetFullFolderNames(j[0])
                # Check if the folder exists (may have been cleaned already)
                if not os.path.isdir(frun):
                    continue
                # Go to the folder.
                os.chdir(frun)
                # Go up one, archiving if necessary.
                self.cd('..')
                # Go back up to the sweep folder.
                os.chdir('..')
            # Go back up to report folder.
            os.chdir('..')

  # === Removal ===
    def RemoveCases(self, I=None, cons=[], **kw):
        r"""Remove case folders or tars

        :Call:
            >>> R.RemoveCases(I=None, cons=[])
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints to define what cases to remove
        :Versions:
            * 2023-06-06 ``@aburkhea``: v1.0
        """
        # Check for use of constraints instead of direct list.
        I = self.cntl.x.GetIndices(I=I, **kw)
        # Loop through those cases.
        for i in I:
            # Remove the case
            self.RemoveCase(i)

    # Remove case folders
    def RemoveCase(self, i):
        r"""Remove case folder or tar

        :Call:
            >>> R.RemoveCase(i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2023-06-06 ``@aburkhea``: v1.0
        """
        # Get the case name.
        fgrp = self.cntl.x.GetGroupFolderNames(i)
        fdir = self.cntl.x.GetFolderNames(i)
        # Go to the report directory if necessary.
        fpwd = os.getcwd()
        os.chdir(self.cntl.RootDir)
        os.chdir('report')
        # Do nothing if no group folder
        if not os.path.isdir(fgrp):
            # Go home.
            os.chdir(fpwd)
            return
        # Go to the group folder.
        os.chdir(fgrp)
        # Remove the case folder tar if exists.
        if os.path.isfile(fdir + '.tar'):
            print(" Removing %s" % fdir + ".tar")
            os.remove(fdir + ".tar")
        # Remove the case folder if exists.
        if os.path.isdir(fdir):
            print(" Removing %s" % fdir)
            shutil.rmtree(fdir)
        # Go home.
        os.chdir(fpwd)

  # === Subfigures ===
   # --- Config ---
    # Function to initialize a subfigure
    def SubfigInit(self, sfig):
        r"""Create the initial lines of a subfigure

        :Call:
            >>> lines = R.SubfigInit(sfig)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to initialize
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Formatting lines to initialize a subfigure
        :Versions:
            * 2016-01-16 ``@ddalle``: v1.0
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

    # Apply a generic Python function for a subfigure
    def SubfigFunction(self, sfig, I):
        r"""Apply a generic Python function to a subfigure definition

        :Call:
            >>> R.SubfigFunction(sfig, i)
            >>> R.SubfigFunction(sfig, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to initialize
            *i*: :class:`int`
                Case index
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Case index
        :Versions:
            * 2017-06-22 ``@ddalle``: v1.0
        """
        # Extract control object
        cntl = self.cntl
        # Get the functions
        funcs = cntl.opts.get_SubfigOpt(sfig, "Function")
        # Exit if none
        if not funcs:
            return
        # Ensure list
        funcs = list(np.array(funcs).flatten())
        # Get first index
        if I.__class__.__name__ in ["list", "ndarray", "tuple"]:
            # Ensure array
            I = list(I)
            # Get first index
            i = I[0]
        else:
            # Already scalar
            i = I
        # Loop through functions
        for func in funcs:
            # Status update
            print('    Subfig function: %s("%s", %s)' % (func, sfig, i))
            # Run the function
            exec("cntl.%s(cntl, sfig, I)" % (func))

    # Function to get the list of targets for a subfigure
    def SubfigTargets(self, sfig):
        r"""Return list of targets (by name) for a subfigure

        :Call:
            >>> targs = R.SubfigTargets(sfig)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
        :Outputs:
            *targs*: :class:`list`\ [:class:`str`]
                List of target names
        :Versions:
            * 2015-06-04 ``@ddalle``: v1.0
        """
        # Target option for this subfigure (defaults to all targets)
        otarg = self.cntl.opts.get_SubfigOpt(sfig, "Target")
        # Process list of targets.
        if isinstance(otarg, (list, np.ndarray)):
            # List of targets directly specified
            targs = otarg
        elif isinstance(otarg, str):
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

    # Process caption
    def SubfigCaption(self, sfig, cdef=None):
        r"""Get a caption for a subfigure

        This relies on the options *Caption* and *CaptionComponent*.  A
        *Caption* specification creates a full caption, while the
        *CaptionComponent* creates a prefix.  The default caption has
        the format *Component*/*Coefficient* if applicable, and
        otherwise uses *sfig*.

        :Call:
            >>> fcpt = R.SubfigCaption(sfig, cdef=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *cdef*: {``None``} | :class:`str`
                Default caption
        :Outputs:
            *fcpt*: :class:`str`
                Caption
        :Versions:
            * 2017-03-31 ``@ddalle``: v1.0
        """
        # Get caption.
        fcpt = self.cntl.opts.get_SubfigOpt(sfig, "Caption")
        # Check for non-default
        if (fcpt is not None):
            # User has specified a caption
            return fcpt
        elif (cdef is not None):
            # Default caption specified for this type of subfigure
            return cdef
        # Get prefix
        fcptb = self.cntl.opts.get_SubfigOpt(sfig, "CaptionComponent")
        # Get component and coefficients
        comp = self.cntl.opts.get_SubfigOpt(sfig, "Component")
        coeff = self.cntl.opts.get_SubfigOpt(sfig, "Coefficient")
        # Types
        tcomp = type(comp).__name__
        tcoef = type(coeff).__name__
        # Check for a list.
        if (fcptb is not None):
            # Use the user-specified base to start the caption
            # Deal with any unbound underscores
            fcptb = re.sub(r"([^\\])_", r"\1\_", fcptb)
            # e.g. [RSRB (black), LSRB(g)]/CA"
            fcpt = fcptb
        elif (tcomp == "list") and (len(comp) > 1):
            # Join them, e.g. "[RSRB,LSRB]/CA"
            fcpt = "[" + ",".join(comp) + "]"
            # Eliminate underscores
            fcpt = fcpt.replace('_', r'\_')
        elif (tcomp == "list"):
            # Single list, e.g. "Component": ["RSRB_No_Base"]
            fcpt = comp[0]
            # Eliminate underscores
            fcpt = fcpt.replace('_', r'\_')
        elif (comp is not None):
            # Use the coefficient.
            fcpt = comp
            # Eliminate underscores
            fcpt = fcpt.replace('_', r'\_')
        else:
            # Use the name of the subfigure
            fcpt = sfig.replace('_', r'\_')
        # Check for coefficient
        if (tcoef == "list") and (len(coeff) > 1):
            # Join them, e.g. "RSRB/[CY,CN]"
            fcpt += ("/[%s]" % (",".join(coeff)).replace("_", r"\_"))
        elif (tcoef == "list"):
            # Singleton coefficient list
            fcpt += ("/%s" % coeff[0].replace('_', r'\_'))
        elif (coeff is not None):
            # Single coefficient
            fcpt += ("/%s" % coeff.replace('_', r'\_'))
        # Output
        return fcpt

   # --- Formatting ---
    # Function to turn on grid, turn off ticks, etc.
    def SubfigFormatAxes(self, sfig, ax):
        r"""Apply formatting options to an :class:`AxesSubplot` instance

        :Call:
            >>> R.SubfigFormatAxes(sfig, ax)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
                Axis handle
        :Versions:
            * 2017-10-12 ``@ddalle``: v1.0
        """
        # Get the options
        opts = self.cntl.opts
       # -----------
       # Grid Lines
       # -----------
        # Grid options
        mjgrid = opts.get_SubfigOpt(sfig, "Grid")
        kw_gmj = opts.get_SubfigOpt(sfig, "GridStyle")
        mngrid = opts.get_SubfigOpt(sfig, "MinorGrid")
        kw_gmn = opts.get_SubfigOpt(sfig, "MinorGridStyle")
        # Remove ``None`` for options
        if not isinstance(kw_gmj, dict):
            kw_gmj = {}
        if not isinstance(kw_gmn, dict):
            kw_gmn = {}
        # Format | turn on | turn off grid
        if mjgrid is None:
            # Leave it as it currently is
            pass
        elif mjgrid:
            # Ensure that the axis is below
            ax.set_axisbelow(True)
            # Add the grid
            ax.grid(**kw_gmj)
        # Format | turn on | turn off minor grid
        if mngrid is None:
            # Leave it as it currently is
            pass
        elif mngrid:
            # Ensure that the axis is below
            ax.set_axisbelow(True)
            # Minor ticks are required
            ax.minorticks_on()
            # Add the grid
            ax.grid(which="minor", **kw_gmn)
       # ------------
       # Tick Labels
       # ------------
        # Get tick label options
        TL = opts.get_SubfigOpt(sfig, "TickLabels")
        xTL = opts.get_SubfigOpt(sfig, "XTickLabels")
        yTL = opts.get_SubfigOpt(sfig, "YTickLabels")
        # Formattiong
        kw_TL  = opts.get_SubfigOpt(sfig, "TickLabelOptions")
        kw_xTL = opts.get_SubfigOpt(sfig, "XTickLabelOptions")
        kw_yTL = opts.get_SubfigOpt(sfig, "YTickLabelOptions")
        # Ensure dictionary
        if not isinstance(kw_TL, dict):
            kw_TL = {}
        if not isinstance(kw_xTL, dict):
            kw_xTL = {}
        if not isinstance(kw_yTL, dict):
            kw_yTL = {}
        # Apply *kw_TL* to others as defaults
        for k in kw_TL:
            kw_xTL.setdefault(kw_TL[k])
            kw_yTL.setdefault(kw_TL[k])
        # Check for x-axis tick label options
        if xTL is False:
            # Turn off tick labels
            ax.set_xticklabels([])
        elif xTL:
            # Explicit list of x-tick labels
            ax.set_xticklabels(xTL, **kw_xTL)
        elif kw_xTL:
            # Get the current labels
            xTL = ax.get_xticklabels()
            # Put them back in with additional formatting options
            ax.set_xticklabels(xTL, **kw_xTL)
        # Check for y-axis tick label options
        if yTL is False:
            # Turn off tick labels
            ax.set_yticklabels([])
        elif yTL:
            # Explicit list of y-tick labels
            ax.set_yticklabels(yTL, **kw_yTL)
        elif kw_yTL:
            # Get the current labels
            yTL = ax.get_yticklabels()
            # Put them back in with additional formatting options
            ax.set_yticklabels(xTL, **kw_yTL)
        # Check for overall tick label option
        if TL is False:
            # Turn off both sets
            ax.set_xticklabels([])
            ax.set_yticklabels([])
       # ----------
       # Limits
       # ----------
        # Get options related to axis limits
        xlim = opts.get_SubfigOpt(sfig, "XLim")
        xlimmax = opts.get_SubfigOpt(sfig, "XLimMax")
        xmin = opts.get_SubfigOpt(sfig, "XMin")
        xmax = opts.get_SubfigOpt(sfig, "XMax")
        ylim = opts.get_SubfigOpt(sfig, "YLim")
        ylimmax = opts.get_SubfigOpt(sfig, "YLimMax")
        ymin = opts.get_SubfigOpt(sfig, "YMin")
        ymax = opts.get_SubfigOpt(sfig, "YMax")
        # Ensure *lim* vars are all pairs
        if xlimmax is None:
            # Null option
            xminlim, xmaxlim = [None, None]
        else:
            # Unpack outer limits for xmin, xmax
            xminlim, xmaxlim = xlimmax
        if ylimmax is None:
            # Null option
            yminlim, ymaxlim = [None, None]
        else:
            # Unpack outer limits for ymin, ymax
            yminlim, ymaxlim = ylimmax
        if xlim is None:
            xlim = [None, None]
        if ylim is None:
            ylim = [None, None]
        # Get current limits
        xmin0, xmax0 = ax.get_xlim()
        ymin0, ymax0 = ax.get_ylim()
        # De-None the outer limits
        xminlim = xmin0 if xminlim is None else xminlim
        xmaxlim = xmax0 if xmaxlim is None else xmaxlim
        yminlim = ymin0 if yminlim is None else yminlim
        ymaxlim = ymax0 if ymaxlim is None else ymaxlim
        # Default
        if xmin is None:
            xmin = xlim[0]
        if xmax is None:
            xmax = xlim[1]
        if ymin is None:
            ymin = ylim[0]
        if ymax is None:
            ymax = ylim[1]
        # Apply non-specific outer limits
        if xmin is None:
            xmin = max(xmin0, xminlim)
        if xmax is None:
            xmax = min(xmax0, xmaxlim)
        if ymin is None:
            ymin = max(ymin0, yminlim)
        if ymax is None:
            ymax = min(ymax0, ymaxlim)
        # Apply limits (even if same)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
       # ------
       # Ticks
       # ------
        # Get tick label options
        Ticks = opts.get_SubfigOpt(sfig, "Ticks")
        XTicks = opts.get_SubfigOpt(sfig, "XTicks")
        YTicks = opts.get_SubfigOpt(sfig, "YTicks")
        # Check for x-axis tick options
        if XTicks is False:
            # Turn off tick labels
            ax.set_xticks([])
        elif XTicks:
            # Explicit list of x-tick labels
            ax.set_xticks(XTicks)
        # Check for y-axis tick options
        if YTicks is False:
            # Turn off tick labels
            ax.set_yticks([])
        elif YTicks:
            # Explicit list of y-tick labels
            ax.set_yticks(YTicks)
        # Check for overall ticks option
        if Ticks is False:
            # Turn off both sets
            ax.set_xticks([])
            ax.set_yticks([])
       # ------------
       # Axes Labels
       # ------------
        # Check for manually specified axes labels.
        xlbl = opts.get_SubfigOpt(sfig, "XLabel")
        ylbl = opts.get_SubfigOpt(sfig, "YLabel")
        # Formatting options
        xlblopts = opts.get_SubfigOpt(sfig, "XLabelOptions")
        ylblopts = opts.get_SubfigOpt(sfig, "YLabelOptions")
        # Ensure dictionary
        if not isinstance(xlblopts, dict):
            xlblopts = {}
        if not isinstance(ylblopts, dict):
            ylblopts = {}
        # Specify x-axis label if given.
        if xlbl is not None:
            ax.set_xlabel(xlbl)
        # Specify x-axis label if given.
        if ylbl is not None:
            ax.set_ylabel(ylbl)
        # Get handles
        hx = ax.get_xaxis().get_label()
        hy = ax.get_yaxis().get_label()
        # Reissue commands with options
        ax.set_xlabel(hx.get_text(), **xlblopts)
        ax.set_ylabel(hy.get_text(), **ylblopts)
       # ------------
       # Restriction
       # ------------
        # Get restriction option
        fres = opts.get_SubfigOpt(sfig, "Restriction")
        # Get position
        xres = opts.get_SubfigOpt(sfig, "RestrictionXPosition")
        yres = opts.get_SubfigOpt(sfig, "RestrictionYPosition")
        # Location
        locres = opts.get_SubfigOpt(sfig, "RestrictionLoc")
        # Other options
        kw_res = opts.get_SubfigOpt(sfig, "RestrictionOptions")
        # Ensure dictionary
        if kw_res.__class__.__name__ != "dict":
            kw_res = {}
        # Process default options
        kw_res.setdefault("fontsize", 10)
        kw_res.setdefault("fontweight", "bold")
        kw_res.setdefault("transform", ax.transAxes)
        # Default options
        if locres in ["top left", "upper left"]:
            # Default coords for upper left label
            xresdef = 0.01
            yresdef = 1.01
            # Default alignments
            kw_res.setdefault('horizontalalignment', 'left')
            kw_res.setdefault('verticalalignment', 'bottom')
        elif locres in ["top right", "upper right"]:
            # Default coords for upper right label
            xresdef = 0.99
            yresdef = 1.01
            # Default alignments
            kw_res.setdefault('horizontalalignment', 'right')
            kw_res.setdefault('verticalalignment', 'bottom')
        elif locres in ["right"]:
            # Default coords for middle right label
            xresdef = 0.99
            yresdef = 0.50
            # Default alignment
            kw_res.setdefault('horizontalalignment', 'right')
            kw_res.setdefault('verticalalignment', 'center')
        elif locres in ["bottom right", "lower right"]:
            # Default coords for bottom right label
            xresdef = 0.99
            yresdef = 0.01
            # Default alignment
            kw_res.setdefault('horizontalalignment', 'right')
            kw_res.setdefault('verticalalignment', 'bottom')
        elif locres in ["bottom"]:
            # Default coords for bottom label
            xresdef = 0.5
            yresdef = 0.01
            # Default alignment
            kw_res.setdefault('horizontalalignment', 'center')
            kw_res.setdefault('verticalalignment', 'bottom')
        elif locres in ["bottom left", "lower left"]:
            # Default coords for bottom left label
            xresdef = 0.01
            yresdef = 0.01
            # Default alignment
            kw_res.setdefault('horizontalalignment', 'left')
            kw_res.setdefault('verticalalignment', 'bottom')
        else:
            # Default coords for top middle label
            xresdef = 0.5
            yresdef = 1.01
            # Default alignments
            kw_res.setdefault('horizontalalignment', 'center')
            kw_res.setdefault('verticalalignment', 'bottom')
        # Final coordinates
        if xres is None:
            xres = xresdef
        if yres is None:
            yres = yresdef
        # Write label if nontrivial
        if fres:
            pltfile.text(xres, yres, fres, **kw_res)

   # --- Tables ---
    # Function to write conditions table
    def SubfigConditions(self, sfig, I, q=True):
        r"""Create lines for a "Conditions" subfigure

        :Call:
            >>> lines = R.SubfigConditions(sfig, i, q=True)
            >>> lines = R.SubfigConditions(sfig, I, q=True)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of case indices
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2014-03-08 ``@ddalle``: v1.0
            * 2014-06-02 ``@ddalle``: Added range capability
        """
        # Extract the trajectory.
        try:
            # Read the data book trajectory.
            x = self.cntl.DataBook.x
        except Exception:
            # Use the run matrix trajectory.
            x = self.cntl.x
        # Check input type
        if isinstance(I, (int, np.int32, np.int64)):
            # Use index value given and make a list with one entry.
            i = int(I)
            I = np.array([i])
        else:
            # Extract first index
            i = I[0]
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

        # Get the variables to skip
        skvs = self.cntl.opts.get_SubfigOpt(sfig, 'SkipVars')
        # Replace None -> []
        skvs = [] if skvs is None else skvs
        # Loop through the trajectory keys.
        for k in x.cols:
            # Check if it's a skip variable
            if k in skvs:
                continue
            # Write the variable name.
            line = "{\\small\\textsf{%s}}" % k.replace('_', r'\_')
            # Append the abbreviation.
            abbrv = x.defns[k].get('Abbreviation', k)
            line += (
                " & {\\small\\textsf{%s}} & " %
                abbrv.replace('_', r'\_'))
            # Get values.
            v = x[k][I]
            # Nominal, min, and max
            v0 = v[0]
            vmin = min(v)
            vmax = max(v)
            # Append the value.
            if x.defns[k]['Value'] in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % (
                    v[0].replace('_', r'\_'))
            elif x.defns[k]['Value'] in ['float', 'int']:
                # Check for range.
                if max(v) > min(v):
                    # Perform substitutions
                    w0 = self.WriteScientific('%.5g' % v0)
                    wmin = self.WriteScientific('%.5g' % vmin)
                    wmax = self.WriteScientific('%.5g' % vmax)
                    # Print both values.
                    line += "$%s$, [$%s$, $%s$] \\\\\n" % (
                        w0, wmin, wmax)
                else:
                    # Substitution
                    w0 = self.WriteScientific('%.5g' % v0)
                    # Put the value as a number.
                    line += "$%s$ \\\\\n" % w0
            elif x.defns[k]['Value'] in ['hex']:
                # Check for range
                if max(v) > min(v):
                    # Print min/max values.
                    line += "0x%x, [0x%x, 0x%x] \\\\\n" % (
                        v[0], min(v), max(v))
                else:
                    # Put the value as a hex code.
                    line += "0x%x \\\\\n" % v[0]
            elif x.defns[k]['Value'] in ['oct', 'octal']:
                # Check for range
                if max(v) > min(v):
                    # Print min/max values
                    line += "0o%o, [0o%o, 0o%o] \\\\\n" % (
                        v[0], min(v), max(v))
                else:
                    # Put the value as a hex code.
                    line += "0o%o \\\\\n" % v[0]
            else:
                # Put the virst value as string (other type)
                line += "%s \\\\\n" % v[0].replace('_', r'\_')
            # Add the line to the table.
            lines.append(line)
        # List of "special" variables
        spvars = self.cntl.opts.get_SubfigOpt(sfig, "SpecialVars")
        # Dictionary of recognized special keys
        spdict = {
            "AngleOfAttack":      ["a",    "GetAlpha"],
            "a":                  ["a",    "GetAlpha"],
            "aoa":                ["a",    "GetAlpha"],
            "alpha":              ["a",    "GetAlpha"],
            "AlphaTotal":         ["a",    "GetAlphaTotal"],
            "TotalAngleOfAttack": ["a",    "GetAlphaTotal"],
            "alpha_t":            ["a",    "GetAlphaTotal"],
            "aoav":               ["a",    "GetAlphaTotal"],
            "Sideslip":           ["b",    "GetBeta"],
            "AngleOfSideslip":    ["b",    "GetBeta"],
            "b":                  ["b",    "GetBeta"],
            "beta":               ["b",    "GetBeta"],
            "aos":                ["b",    "GetBeta"],
            "Phi":                ["phi",  "GetPhi"],
            "phi":                ["phi",  "GetPhi"],
            "phiv":               ["phi",  "GetPhi"],
            "DynamicPressure":    ["q",    "GetDynamicPressure"],
            "q":                  ["q",    "GetDynamicPressure"],
            "qbar":               ["q",    "GetDynamicPressure"],
            "qinf":               ["q",    "GetDynamicPressure"],
            "Pressure":           ["p",    "GetPressure"],
            "p":                  ["p",    "GetPressure"],
            "pinf":               ["p",    "GetPressure"],
            "TotalPressure":      ["p0",   "GetTotalPressure"],
            "p0":                 ["p0",   "GetTotalPressure"],
            "ReynoldsNumber":     ["Re",   "GetReynoldsNumber"],
            "Reynolds":           ["Re",   "GetReynoldsNumber"],
            "Re":                 ["Re",   "GetReynoldsNumber"],
            "Rey":                ["Re",   "GetReynoldsNumber"],
            "REY":                ["Re",   "GetReynoldsNumber"],
            "Temperature":        ["T",    "GetTemperature"],
            "T":                  ["T",    "GetTemperature"],
            "TotalTemperature":   ["T0",   "GetTotalTemperature"],
            "T0":                 ["T0",   "GetTotalTemperature"],
        }
        # De-none special vars
        if spvars is None:
            spvars = []
        # Loop through special variables
        for k in spvars:
            # Get the information on this parameter
            abrv, func = spdict.get(k, ["", None])
            # Get value
            if k in x.defns:
                # Get reference value(s)
                v = x.GetValue(k, I)
                # Local definitions
                defns = x.defns.get(k, {})
                # Get abbreviation
                abrv = defns.get("Abbreviation", k)
            elif func is None:
                # No value to get; skip
                continue
            else:
                # Evaluate the function
                v = eval("x.%s(np.%s)" % (func, I.__repr__()))
            # Round principal value
            v = np.around(v, decimals=8)
            # Get first value and min/max
            v0 = v[0]
            vmin = min(v)
            vmax = max(v)
            # Write the variable name
            line = "{\\small\\textsf{%s}}" % k.replace('_', r'\_')
            # Append the abbreviation
            line += " & {\\small\\textsf{%s}} & " % abrv.replace('_', r'\_')
            # Type
            tv = type(vmin).__name__
            # Append the value.
            if tv in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % (
                    v[0].replace('_', r'\_'))
            elif isinstance(vmin, (float, int)):
                # Check for range.
                if vmax > vmin:
                    # Perform substitutions
                    w0 = self.WriteScientific('%.5g' % v0)
                    wmin = self.WriteScientific('%.5g' % vmin)
                    wmax = self.WriteScientific('%.5g' % vmax)
                    # Print both values.
                    line += "$%s$, [$%s$, $%s$] \\\\\n" % (w0, wmin, wmax)
                else:
                    # Put the value as a number.
                    w0 = self.WriteScientific('%.5g' % v0)
                    line += "$%s$ \\\\\n" % w0
            else:
                # Put the virst value as string (other type)
                line += "\\\\\n"
            # Add the line to the table.
            lines.append(line)
        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to write sweep conditions table
    def SubfigSweepConditions(self, sfig, fswp, i, q=True):
        r"""Create lines for a "SweepConditions" subfigure

        :Call:
            >>> lines = R.SubfigSweepConditions(sfig, fswp, I, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Outputs:
            *lines*: :class:`str`
                List of lines in the subfigure
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
            * 2015-06-02 ``@ddalle``: v1.1; add min/max values
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
        for k in x.cols:
            # Check if it's an equality value.
            if k in eqkeys:
                # Equality constraint
                scon = '$=$'
            elif k in tolkeys:
                # Tolerance constraint
                scon = r'$\pm%s$' % tolkeys[k]
            else:
                # Not a constraint.
                continue
            # Write the variable name.
            line = "{\\small\\textsf{%s}} & " % k.replace('_', r'\_')
            # Append the value.
            if x.defns[k]['Value'] in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % (
                    x[k][i].replace('_', r'\_'))
            elif x.defns[k]['Value'] in ['float', 'int']:
                # Put the value as a number
                line += "$%s$ &" % x[k][i]
            elif x.defns[k]['Value'] in ['hex']:
                # Put the value as a hex code.
                line += "0x%x &" % x[k][i]
            elif x.defns[k]['Value'] in ['oct', 'octal']:
                # Put the value as a hex code.
                line += "0o%o &" % x[k][i]
            else:
                # Just put a string
                line += "%s &" % x[k][i].replace('_', r'\_')
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
        lines.append(
            "{\\small\\textit{i}} & $%i$ & $[%i,%i]$ \\\\ \n"
            % (i, i, imax))
        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to write sweep conditions table
    def SubfigSweepCases(self, sfig, fswp, I, q=True):
        r"""Create lines for a "SweepConditions" subfigure

        :Call:
            >>> lines = R.SubfigSweepCases(sfig, fswp, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                Case indices
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Outputs:
            *lines*: :class:`str`
                List of lines in the subfigure
        :Versions:
            * 2015-06-02 ``@ddalle``: v1.0
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
            frun = fruns[j].replace('_', r'\_')
            # Add the index and folder name.
            lines.append('\\texttt{%i} & \\texttt{%s} \\\\ \n' % (i, frun))

        # Finish the subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to write summary table
    def SubfigSummary(self, sfig, i, q=True):
        r"""Create lines for a "Summary" subfigure

        :Call:
            >>> lines = R.SubfigSummary(sfig, i, q=True)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2014-03-09 ``@ddalle``: v1.0
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options
        opts = self.cntl.opts
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Numbers of iterations for statistics
        nStats = opts.get_SubfigOpt(sfig, "NStats")
        nMin = opts.get_SubfigOpt(sfig, "NMinStats")
        nMax = opts.get_SubfigOpt(sfig, "NMaxStats")
        # Get the status and data book options
        if nStats is None:
            nStats = opts.get_DataBookNStats()
        if nMin is None:
            nMin = opts.get_DataBookNMin()
        if nMax is None:
            nMax = opts.get_DataBookNMaxStats()
        # Pure defaults
        if nStats is None:
            nStats = 1
        # Iteration at which to build table
        nOpt = opts.get_SubfigOpt(sfig, "Iteration")
        # De-None
        if nOpt is None:
            nOpt = 0
        # Make sure current progress is a number
        if nIter is None:
            nIter = 0
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
            if nMax is None:
                nMax = nCur - nMin
            else:
                nMax = min(nMax, nCur-nMin)
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(self.cntl.x.GetFullFolderNames(i))
            # Loop through components
            for comp in comps:
                # Component label
                # Read the Aero history.
                FM = self.ReadCaseFM(comp)
                # Get iters
                iters = FM.get_values("i")
                # Check for trivial
                if iters.size == 0:
                    # Warning
                    print("  No iterations for comp '%s'" % comp)
                    # Save empty stats
                    S[comp] = {}
                    # Go to next component
                    continue
                # Get transformations
                tforms = opts.get_DataBookTransformations(comp)
                tforms = [] if tforms is None else tforms
                # Loop through the transformations
                for topts in tforms:
                    # Get type
                    ttyp = topts.get("Type")
                    # Only apply to "ShiftMRP"
                    if ttyp == "ShiftMRP":
                        # Use a copy to avoid changing cntl.opts
                        topts = dict(topts)
                        # Component to use for current MRP
                        compID = self.cntl.opts.get_DataBookCompID(comp)
                        if isinstance(compID, list):
                            compID = compID[0]
                        # Reset points for default *FromMRP*
                        self.cntl.opts.reset_Points()
                        # Use MRP prior to trans as default *FromMRP*
                        x0 = self.cntl.opts.get_RefPoint(comp)
                        # Ensure points are calculated
                        self.cntl.PreparePoints(i)
                        # Use post-transformation MRP as default *ToMRP*
                        x1 = self.cntl.opts.get_RefPoint(comp)
                        # Get current Lref
                        Lref = self.cntl.opts.get_RefLength(comp)
                        # Set those as defaults in transformation
                        x0 = topts.setdefault("FromMRP", x0)
                        x1 = topts.setdefault("ToMRP", x1)
                        topts.setdefault("RefLength", Lref)
                        # Expand if *x0* is a string
                        topts["FromMRP"] = self.cntl.opts.expand_Point(x0)
                        topts["ToMRP"] = self.cntl.opts.expand_Point(x1)
                    # Apply the transformation
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
        if len(S) > 0:
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
            lines.append(
                ' & {\\small\\texttt{%s}} \n'
                % comp.replace('_', r'\_'))
        lines.append('\\\\\n')
        # Get coefficient list
        coeffs = self.cntl.opts.get_SubfigOpt(sfig, "Coefficients")
        # Default coeffs: None -> ["CA", "CY", "CN"]
        coeffs = ["CA", "CY", "CN"] if coeffs is None else coeffs
        # Loop through coefficients
        for c in coeffs:
            # Convert coefficient title to symbol
            if c in ['CA', 'CY', 'CN']:
                # Just add underscore
                fc = c[0] + '_' + c[1]
            elif c in ['CLL']:
                # Special rolling moment
                fc = r'C_\ell'
            elif c in ['CLM', 'CLN']:
                # Other moments
                fc = 'C_%s' % c[-1].lower()
            else:
                # What?
                fc = 'C_{%s}' % c[1:]
            # Print horizontal line
            lines.append('\\hline\n')
            # Get statistics for this coefficient
            statcols = ["mu", "std"]
            # Loop through statistical varieties
            for fs in statcols:
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
                    lines.append(
                        ('\\textit{%s} iterative uncertainty, ' % c) +
                        '$\\varepsilon(%s)$\n' % fc)
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
                if isinstance(ff, dict):
                    # Check for coefficient
                    if c in ff:
                        ff = ff[c]
                # Initialize line
                line = ''
                # Loop through components.
                for comp in comps:
                    # Downselect format flag to *comp* if appropriate
                    if isinstance(ff, dict):
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
                    word = self.WriteScientific(word)
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

    # Function to redo scientific notation
    def WriteScientific(self, v, decimals=8):
        r"""Convert value or string to scientific notation

        The typical behavior is ``1.4e-5`` --> ``1.4\times10^{-5}``

        :Call:
            >>> word = R.WriteScientific(v, decimals=8)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *v*: :class:`str` | :class:`float` | :class:`int`
                Value to be translated, preferably a string
            *decimals*: {``8``} | :class:`int` > 4
                Round floats at :math:`10^{-d}`
        :Outputs:
            *word*: :class:`str`
                String with substitutions made
        :Versions:
            * 2017-04-11 ``@ddalle``: v1.0
        """
        # Check type
        t = type(v).__name__
        # Truncate if small number
        if t.startswith('str') or t.startswith('unicode'):
            # Already string
            word = v
        else:
            # Round and convert to string
            word = str(np.around(v, decimals=decimals))
        # Process exponential notation
        m = re.search('[edED]([+-][0-9]+)', word)
        # Check for a match to 'e+09', etc.
        if m is not None:
            # Existing text from exponent, e.g. 'e+09', 'D-13'
            txt = m.group(1)
            # Process the actual exponent text; strip '+' and '0'
            exp = txt[0].lstrip('+') + txt[1:].lstrip('0')
            # Handle '0'
            exp = '0' if exp == '' else exp
            # Replace text
            word = word.replace(m.group(0), '{\\times}10^{%s}' % exp)
        # Output
        return word

   # --- PlotCoeff ---
    # Function to create coefficient plot and write figure
    def SubfigPlotCoeff(self, sfig, i, q):
        r"""Create plot for a coefficient and input lines int LaTeX file

        :Call:
            >>> lines = R.SubfigPlotCoeff(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-03-09 ``@ddalle``: v1.0
        """
       # ------
       # Setup
       # ------
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
        fcpt = self.SubfigCaption(sfig)
        # First lines.
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
       # ---------
       # Plotting
       # ---------
        # Loop through plots.
        for k in range(nCoeff):
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Numbers of iterations
            nStats = opts.get_SubfigOpt(sfig, "nNtats", k)
            dn = opts.get_SubfigOpt(sfig, "DNStats", k)
            nMin = opts.get_SubfigOpt(sfig, "NMinStats", k)
            nMax = opts.get_SubfigOpt(sfig, "NMaxStats", k)
            # Default to databook options
            if nStats is None:
                nStats = opts.get_DataBookNStats()
            if dn is None:
                dn = opts.get_DataBookDNStats()
            if nMin is None:
                nMin = opts.get_DataBookNMin()
            if nMax is None:
                nMax = opts.get_DataBookNMaxStats()
            # Numbers of iterations for plots
            nPlotIter  = opts.get_SubfigOpt(sfig, "NPlotIters", k)
            nPlotFirst = opts.get_SubfigOpt(sfig, "NPlotFirst", k)
            nPlotLast  = opts.get_SubfigOpt(sfig, "NPlotLast",  k)
            # Check if there are iterations.
            if nIter < 2:
                continue
            # Don't use iterations before *nMin*
            if nMax is None:
                nMax = nIter - nMin
            else:
                nMax = min(nMax, nIter-nMin)
            # ... unless it's needed because *nIter* is too low
            if nMax < nStats:
                # Try as hard as possible to get *nStats* in window
                nMax = nStats if nIter >= nStats else nIter
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Read the Aero history
            FM = self.ReadCaseFM(comp)
            # Get iters
            iters = FM.get_values("i")
            # Check for missing history
            if iters is None or iters.size == 0:
                raise AttributeError(
                    "Comp '%s' in subfig '%s' has no history found"
                    % (comp, sfig))
            # Get transformations
            tforms = opts.get_DataBookTransformations(comp)
            tforms = [] if tforms is None else tforms
            # Loop through the transformations
            for topts in tforms:
                # Get type
                ttyp = topts.get("Type")
                # Only apply to "ShiftMRP"
                if ttyp == "ShiftMRP":
                    # Use a copy to avoid changing cntl.opts
                    topts = dict(topts)
                    # Component to use for current MRP
                    compID = self.cntl.opts.get_DataBookCompID(comp)
                    if isinstance(compID, list):
                        compID = compID[0]
                    # Reset points for default *FromMRP*
                    self.cntl.opts.reset_Points()
                    # Use MRP prior to transfformations as default *FromMRP*
                    x0 = self.cntl.opts.get_RefPoint(comp)
                    # Ensure points are calculated
                    self.cntl.PreparePoints(i)
                    # Use post-transformation MRP as default *ToMRP*
                    x1 = self.cntl.opts.get_RefPoint(comp)
                    # Get current Lref
                    Lref = self.cntl.opts.get_RefLength(comp)
                    # Set those as defaults in transformation
                    x0 = topts.setdefault("FromMRP", x0)
                    x1 = topts.setdefault("ToMRP", x1)
                    topts.setdefault("RefLength", Lref)
                    # Expand if *x0* is a string
                    topts["FromMRP"] = self.cntl.opts.expand_Point(x0)
                    topts["ToMRP"] = self.cntl.opts.expand_Point(x1)
                # Apply the transformation.
                FM.TransformFM(topts, self.cntl.x, i)
            # Get the manual range to show
            dc = opts.get_SubfigOpt(sfig, "Delta", k)
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation", k)
            # Get the multiple of iterative error to show
            uerr = opts.get_SubfigOpt(sfig, "IterativeError", k)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "PlotOptions",   k)
            kw_m = opts.get_SubfigOpt(sfig, "MeanOptions",   k)
            kw_s = opts.get_SubfigOpt(sfig, "StDevOptions",  k)
            kw_u = opts.get_SubfigOpt(sfig, "ErrPltOptions", k)
            kw_d = opts.get_SubfigOpt(sfig, "DeltaOptions",  k)
            # Label options
            sh_m = opts.get_SubfigOpt(sfig, "ShowMu", k)
            sh_s = opts.get_SubfigOpt(sfig, "ShowSigma", k)
            sh_d = opts.get_SubfigOpt(sfig, "ShowDelta", k)
            sh_e = opts.get_SubfigOpt(sfig, "ShowEpsilon", k)
            # Label formats
            fmt_m = opts.get_SubfigOpt(sfig, "MuFormat", k)
            fmt_s = opts.get_SubfigOpt(sfig, "SigmaFormat", k)
            fmt_d = opts.get_SubfigOpt(sfig, "DeltaFormat", k)
            fmt_e = opts.get_SubfigOpt(sfig, "ErrorFormat", k)
            # Draw the plot.
            h = FM.PlotCoeff(
                coeff, n=nPlotIter,
                nFirst=nPlotFirst, nLast=nPlotLast,
                nStats=nStats, nMaxStats=nMax, dnStats=dn,
                PlotOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
                ShowMu=sh_m, MuFormat=fmt_m,
                ShowDelta=sh_d, DeltaFormat=fmt_d,
                ShowSigma=sh_s, SigmaFormat=fmt_s,
                ShowError=sh_e, ErrorFormat=fmt_e,
                FigureWidth=figw, FigureHeight=figh)
       # ----------------
       # Post Formatting
       # ----------------
        # Additional formatting
        self.SubfigFormatAxes(sfig, h['ax'])
       # --------
       # Config
       # --------
        # Change back to report folder.
        os.chdir(fpwd)
        # Check for a figure to write.
        if nIter >= 2:
            # Save the figure
            fimg = self.save_figure(sfig, h)
            # Include the graphics.
            lines.append(
                '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- Other DataBook Types ---
    # Function to create coefficient plot and write figure
    def SubfigPlotLineLoad(self, sfig, i, q):
        """Create plot for a sectional loads profile

        :Call:
            >>> lines = R.SubfigPlotLineLoad(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2016-06-10 ``@ddalle``: v1.0
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
        # Loop through targets.
        for targ in targs:
            # Try to read the line loads
            try:
                # Read line loads
                self.ReadLineLoad(comp, i, targ=targ, update=False)
                # If read successfully, duplicate data book target
                targ_types[targ] = 'cape'
            except Exception:
                # Read failed
                print(
                    "    WARNING: " +
                    ("failed to read target line load '%s'" % targ))
                raise IOError
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
        # Get caption.
        fcpt = self.SubfigCaption(sfig)
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
            fcpt = fcpt.replace('_', r'\_')
        # First lines.
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
        # Initialize plot count
        nPlot = 0
        # Loop through plots.
        for k in range(nCoeff):
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Read the line load data book and read case *i* if possible
            LL = self.ReadLineLoad(comp, i, update=True)
            # Check for case
            if LL is None:
                continue
            elif LL.x.size == 0:
                continue
            # Add to plot count
            nPlot += 1
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
            # Plot label
            lbl = opts.get_SubfigOpt(sfig, "Label", k)
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "PlotOptions",   k)
            kw_s = opts.get_SubfigOpt(sfig, "SeamOptions",   k)
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
            if xm is not None:
                kw_pad['xm'] = xm
            if xp is not None:
                kw_pad['xp'] = xp
            if ym is not None:
                kw_pad['ym'] = ym
            if yp is not None:
                kw_pad['yp'] = yp
            # Draw the plot.
            h = LL.Plot(
                coeff,
                Seams=sm_ax, SeamLocation=sm_loc,
                PlotOptions=kw_p, SeamOptions=kw_s,
                Label=lbl,
                FigureWidth=figw, FigureHeight=figh,
                AdjustLeft=adj_l, AdjustRight=adj_r,
                AdjustTop=adj_t, Adjust_Bottom=adj_b,
                SubplotMargin=w_sfig, **kw_pad)
            # Loop through targets
            for targ in targs:
                # Check for generic target
                if targ_types[targ] != 'cape':
                    continue
                # Read the line load data book and read case *i* if possible
                LLT = self.ReadLineLoad(comp, i, targ=targ, update=False)
                # Check for a find.
                if LLT is None:
                    continue
                # Get target plot label.
                tlbl = self.SubfigTargetPlotLabel(sfig, k, targ)
                # Don't start with comma.
                tlbl = tlbl.lstrip(", ")
                # Specified target plot options
                kw_t = opts.get_SubfigOpt(
                    sfig, "TargetOptions", targs.index(targ) + k)
                # Initialize target plot options.
                kw_l = kw_p
                # Apply non-default options
                for k_i in kw_t:
                    kw_l[k_i] = kw_t[k_i]
                # Draw the plot
                LLT.Plot(
                    coeff, PlotOptions=kw_l,
                    Label=tlbl, Legend=True,
                    FigureWidth=figw, FigureHeight=figh)
        # Change back to report folder.
        os.chdir(fpwd)
        # Apply other options to axes
        self.SubfigFormatAxes(sfig, h['ax'])
        # Check for a figure to write.
        if nPlot > 0:
            # Additional formatting
            self.SubfigFormatAxes(sfig, h['ax'])
            # Save the figure
            fimg = self.save_figure(sfig, h)
            # Include the graphics.
            lines.append(
                '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Function to plot mean coefficient for a sweep
    def SubfigPlotLineLoadGroup(self, sfig, fswp, I, q):
        r"""Plot a line load for a group of cases

        :Call:
            >>> R.SubfigPlotLineLoadGroup(sfig, fswp, I, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices in the sweep
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2017-04-23 ``@ddalle``: v1.0
        """
       # -------
       # Options
       # -------
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(I[0])
        # Extract options
        opts = self.cntl.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption.
        if fcpt is None:
            # Defaut: Wing/CY
            fcpt = ("%s/%s" % (comp, coeff))
            # Ensure there are no underscores.
            fcpt = fcpt.replace('_', r'\_')
        # First lines.
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
       # ---------
       # Plotting
       # ---------
        # Initialize plot count
        k = 0
        # Initialize list
        h = []
        # Loop through plots.
        for i in I:
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Auto-update flag
            q_auto = opts.get_SubfigOpt(sfig, "AutoUpdate", k)
            # Read the line load data book and read case *i* if possible
            LL = self.ReadLineLoad(comp, i, update=q_auto)
            # Check for case
            if LL is None:
                continue
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
            # Plot label
            lbl = opts.get_SubfigOpt(sfig, "Label", k)
            # Default label
            if lbl is None:
                lbl = self.cntl.DataBook.x.GetFullFolderNames(i)
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "PlotOptions",   k)
            kw_s = opts.get_SubfigOpt(sfig, "SeamOptions",   k)
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
            if xm is not None:
                kw_pad['xm'] = xm
            if xp is not None:
                kw_pad['xp'] = xp
            if ym is not None:
                kw_pad['ym'] = ym
            if yp is not None:
                kw_pad['yp'] = yp
            # Draw the plot.
            if k == 0:
                # First plot: ok to add seams
                h.append(LL.Plot(
                    coeff,
                    Seams=sm_ax, SeamLocation=sm_loc,
                    PlotOptions=kw_p, SeamOptions=kw_s,
                    Label=lbl,
                    FigureWidth=figw, FigureHeight=figh,
                    AdjustLeft=adj_l, AdjustRight=adj_r,
                    AdjustTop=adj_t, Adjust_Bottom=adj_b,
                    SubplotMargin=w_sfig, **kw_pad))
            else:
                # No seams after first plot
                h.append(LL.Plot(
                    coeff,
                    PlotOptions=kw_p, Label=lbl,
                    FigureWidth=figw, FigureHeight=figh,
                    Legend=True,
                    AdjustLeft=adj_l, AdjustRight=adj_r,
                    AdjustTop=adj_t, Adjust_Bottom=adj_b,
                    SubplotMargin=w_sfig, **kw_pad))
            # Add to plot count
            k += 1
       # ---------
       # Finalize
       # ---------
        # Change back to report folder.
        os.chdir(fpwd)
        # Apply other options to axes
        self.SubfigFormatAxes(sfig, h['ax'])
        # Check for a figure to write.
        if k > 0:
            # Save the figure
            fimg = self.save_figure(sfig, h)
            # Include the graphics.
            lines.append(
                '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- SweepCoeff ---
    # Function to plot mean coefficient for a sweep
    def SubfigSweepCoeff(self, sfig, fswp, I, q):
        r"""Plot a sweep of a coefficient over several cases

        :Call:
            >>> R.SubfigSweepCoeff(sfig, fswp, I, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices in the sweep
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
       # ------------------
       # Options and Config
       # ------------------
        # Apply case functions
        self.SubfigFunction(sfig, I[0])
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
       # ----------------
       # Index Processing
       # ----------------
        # Carpet constraints
        CEq = opts.get_SweepOpt(fswp, "CarpetEqCons")
        CTol = opts.get_SweepOpt(fswp, "CarpetTolCons")
        # Get primary constraints
        EqCons = opts.get_SweepOpt(fswp, "EqCons")
        TolCons = opts.get_SweepOpt(fswp, "TolCons")
        GlobCons = opts.get_SweepOpt(fswp, "GlobalCons")
        # Check for carpet constraints.
        if CEq or CTol:
            # Divide sweep into subsweeps.
            J = x.GetSweeps(I=I, EqCons=CEq, TolCons=CTol)
        else:
            # Single sweep
            J = [I]
        # Check for carpet equality constraints
        if CEq:
            # Combine carpet cons
            EqCons += CEq
        # Check for carpet tolerance contraints
        if CTol:
            # Add to tolerance constraints
            for kx in CTol:
                TolCons[kx] = CTol[kx]
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # Number of targets
        nTarg = len(targs)
       # --------------
       # Format Options
       # --------------
        # Horizontal axis variable
        xk = opts.get_SweepOpt(fswp, "XCol")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of components
        if isinstance(comp, (list, np.ndarray)):
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
        fcpt = fcpt.replace("_", r"\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
                    % (fswp, frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
       # --------
       # Plotting
       # --------
        # Number of targets plotted
        j_t = 0
        # Loop through plots.
        for i in range(nSweep*nCoeff):
            # Coefficient index
            k = i % nCoeff
            # Sweep index
            j = i // nCoeff
            # Get the component and coefficient.
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Check for patch delimiter
            if "/" in comp:
                # Format: MPCV_Camera_patch/front
                compo, patch = comp.split("/")
            elif "." in comp:
                # Format: MPCV_Camera_patch.front
                compo, patch = comp.split(".")
            else:
                # Only comp given; use total of patches
                compo = comp
                patch = None
            # Read the component
            DBc = self.ReadFM(comp)
            # Get matches
            Jj = DBc.FindCoSweep(x, J[j][0], EqCons, TolCons, GlobCons)
            # Plot label (for legend)
            lbl = self.SubfigPlotLabel(sfig, k)
            # Carpet label appendix
            clbl = ""
            # Append carpet constraints to label if appropriate.
            if CEq:
                for kx in CEq:
                    # Value of the key or modified key for all points.
                    V = x[kx]
                    # Print the subsweep equality constraint in the label.
                    clbl += ", %s=%s" % (kx, V[Jj[0]])
                    # Add appendix to label.
                    lbl += clbl
            if CTol:
                # More carpet constraints
                for kx in CTol:
                    # Value of the key or modified key for all points.
                    V = x[xk]
                    # Print the subsweep tolerance constraint in the label.
                    clbl += u", %s=%s\u00B1%s" % (kx, V[Jj[0]], CTol[kx])
                    # Add appendix to label.
                    lbl += clbl
            # Don't start with a comma!
            lbl = lbl.lstrip(", ")
            # Moment reference center parameters
            xmrp = opts.get_SubfigOpt(sfig, "XMRP", k)
            dxmrp = opts.get_SubfigOpt(sfig, "DXMRP", k)
            fxmrp = opts.get_SubfigOpt(sfig, "XMRPFunction", k)
            # Get callable function if specified
            if fxmrp is not None:
                fxmrp = eval("self.cntl.%s" % fxmrp)
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation", k)
            qmmx = opts.get_SubfigOpt(sfig, "MinMax", k)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
            # Check for hard-coded axis limits
            xmin = opts.get_SubfigOpt(sfig, "XMin", k)
            xmax = opts.get_SubfigOpt(sfig, "XMax", k)
            ymin = opts.get_SubfigOpt(sfig, "YMin", k)
            ymax = opts.get_SubfigOpt(sfig, "YMax", k)
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "PlotOptions", i)
            kw_s = opts.get_SubfigOpt(sfig, "StDevOptions", i)
            kw_m = opts.get_SubfigOpt(sfig, "MinMaxOptions", i)
            # Draw the plot
            h = DBc.PlotCoeff(
                coeff, Jj, x=xk,
                XMRP=xmrp, DXMRP=dxmrp,
                XMRPFunction=fxmrp,
                Label=lbl, PlotOptions=kw_p,
                StDev=ksig, StDevOptions=kw_s,
                MinMax=qmmx, MinMaxOptions=kw_m,
                FigureWidth=figw, FigureHeight=figh,
                XMin=xmin, XMax=xmax, YMin=ymin, YMax=ymax)
            # Loop through targets
            for targ in targs:
                # Get the target handle.
                DBTc = self.ReadFM(comp, targ=targ)
                # Exit if not found
                if DBTc is None:
                    print(
                        ("    Skipping target '%s': " % targ) +
                        (" failed to read"))
                # Target type
                typt = opts.get_TargetDataBookType(targ).lower()
                # Process type
                qdup = (typt in ['duplicate', 'cape']) or typt.startswith("py")
                # Target options
                topts = opts.get_TargetDataBookByName(targ)
                # Check if the *comp*/*coeff* combination is available.
                if qdup:
                    # Check if we have the data
                    if (coeff not in DBTc) and (
                            coeff not in ["cp", "CP", "cpy", "CPY", "CT"]):
                        print(
                            ("    Skipping target '%s': " % targ) +
                            ("coeff '%s/%s' not in target" % (comp, coeff)))
                        continue
                else:
                    # Target coefficient
                    if patch is None:
                        # Simple comp/coeff pair
                        ccoeff = coeff
                    else:
                        # Format the key lookup as pt.coeff
                        ccoeff = "%s.%s" % (patch, coeff)
                    # Check *DBT* as a TargetDataBook
                    if compo not in DBTc.ckeys:
                        print(
                            ("    Skipping target '%s': " % targ) +
                            ("comp '%s' not in target" % compo))
                        continue
                    elif ccoeff in ["cp", "CP"] and "CLM" in DBTc.ckeys[comp]:
                        # Can reconstruct a center of pressure (probably)
                        pass
                    elif ccoeff in ["cpy", "CPY"] and (
                            "CLN" in DBTc.ckeys[comp]):
                        # Can reconstruct a center of pressure (probably)
                        pass
                    elif ccoeff not in DBTc.ckeys[compo]:
                        print(
                            ("    Skipping target '%s': " % targ) +
                            ("coeff '%s' not in target" % ccoeff))
                        continue
                # Get any translation keys
                xkeys = topts.get("RunMatrix", {})
                # Get matches
                JTj = DBTc.FindCoSweep(
                    x, J[j][0],
                    EqCons=EqCons, TolCons=TolCons,
                    GlobCons=GlobCons, xkeys=xkeys)
                # Check for results to plot.
                if len(JTj) == 0:
                    print(
                        ("    Skipping target '%s' " % targ) +
                        ("coeff %s/%s: no matching cases" % (comp, coeff)))
                    continue
                # Get target plot label.
                tlbl = self.SubfigTargetPlotLabel(sfig, k, targ) + clbl
                # Don't start with comma.
                tlbl = tlbl.lstrip(", ")
                # Specified target plot options
                kw_t = opts.get_SubfigOpt(
                    sfig, "TargetOptions",
                    k*nTarg + targs.index(targ))
                # Target options index
                j_t += 1
                # Initialize target plot options.
                kw_l = kw_p
                # Apply non-default options
                for k_i in kw_t:
                    kw_l[k_i] = kw_t[k_i]
                # Draw the plot
                if qdup:
                    # Separate object for each component
                    DBTc.PlotCoeff(
                        coeff, JTj, x=xk,
                        XMRP=xmrp, DXMRP=dxmrp,
                        XMRPFunction=fxmrp,
                        Label=tlbl, PlotOptions=kw_l,
                        FigureWidth=figw, FigureHeight=figh,
                        XMin=xmin, XMax=xmax, YMin=ymin, YMax=ymax)
                else:
                    # All components in one object; need to say comp
                    DBTc.PlotCoeff(
                        comp, coeff, JTj, x=xk,
                        XMRP=xmrp, DXMRP=dxmrp,
                        XMRPFunction=fxmrp,
                        Label=tlbl, PlotOptions=kw_l,
                        FigureWidth=figw, FigureHeight=figh,
                        XMin=xmin, XMax=xmax, YMin=ymin, YMax=ymax)
       # ----------
       # Formatting
       # ----------
        # Apply other options to axes
        self.SubfigFormatAxes(sfig, h['ax'])
        # Save the figure
        fimg = self.save_figure(sfig, h)
        # Include the graphics.
        lines.append(
            '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Get subfig label for plot *k*
    def SubfigPlotLabel(self, sfig, k):
        r"""Get line label for subfigure plot

        :Call:
            >>> lbl = R.SubfigPlotLabel(sfig, k)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *k*: :class:`int`
                Plot index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Outputs:
            *lbl*: :class:`str`
                Plot label
        :Versions:
            * 2015-06-04 ``@ddalle``: v1.0
        """
        # Get options
        opts = self.cntl.opts
        # Get the label if specified.
        lbl = opts.get_SubfigOpt(sfig, "Label", k)
        # Check.
        if lbl is not None:
            return lbl
        # Component name
        comp = opts.get_SubfigOpt(sfig, "Component", k)
        # List of coefficients
        coeffs = opts.get_SubfigOpt(sfig, "Coefficient")
        # Check for patch delimiter
        if "/" in comp:
            # Format: MPCV_Camera_patch/front
            compo, patch = comp.split("/")
        elif "." in comp:
            # Format: MPCV_Camera_patch.front
            compo, patch = comp.split(".")
        else:
            # Only comp given; use total of patches
            compo = comp
            patch = None
        # Get DB type
        ctyp = opts.get_DataBookType(compo)
        # Check for special type
        if ctyp == "TriqFM":
            # Get prefix
            fpre = opts.get_DataBookPrefix(compo)
            # Build up name
            if patch is None:
                # Mark the TriqFM patch
                comp = "%s(total)" % fpre
            else:
                # Mark the patch name
                comp = "%s(%s)" % (fpre, patch)
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
        r"""Get line label for subfigure plot

        :Call:
            >>> lbl = R.SubfigPlotLabel(sfig, k, targ)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
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
            * 2015-06-04 ``@ddalle``: v1.0
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
        if lbl is not None:
            return lbl
        # List of components
        comps = opts.get_SubfigOpt(sfig, "Component")
        # List of coefficients
        coeffs = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get the target handle.
        DBT = self.cntl.DataBook.GetTargetByName(targ)
        # Get the nominal label of the target
        tlbl = DBT.topts.get('Label', targ)
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
        elif isinstance(comps, list):
            # Component name
            comp = opts.get_SubfigOpt(sfig, "Component", k)
            # Include component in label.
            return '%s %s' % (tlbl, comp)
        else:
            # Just use target
            return tlbl

   # --- SweepCoeffHist ---
    # Function to plot coefficient histogram for a sweep
    def SubfigSweepCoeffHist(self, sfig, fswp, I, q):
        r"""Plot a histogram of a coefficient over several cases

        :Call:
            >>> R.SubfigSweepCoeffHist(sfig, fswp, I)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices in the sweep
            *q*: ``True`` | ``False``
                Whether or not to redraw images
        :Versions:
            * 2016-01-16 ``@ddalle``: v1.0 (SubfigSweepPointHist)
            * 2018-02-13 ``@ddalle``: v1.0
        """
       # ------------------
       # Options and Config
       # ------------------
        # Save current folder.
        fpwd = os.getcwd()
        # Apply case functions
        self.SubfigFunction(sfig, I[0])
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get histogram type
        htyp = opts.get_SubfigOpt(sfig, "HistogramType")
       # --------------
       # Format Options
       # --------------
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption
        if fcpt is None:
            # Use the point name and the coefficient
            fcpt = "%s.%s" % (comp.replace(".", "/"), coeff)
        # Ensure that there are not underscores.
        fcpt = fcpt.replace("_", r"\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
                    % (fswp, frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
       # --------
       # Plotting
       # --------
        # Read the component
        DBc = self.ReadFM(comp)
        # Get the targets
        targs = self.SubfigTargets(sfig)
        # Number of targets
        ntarg = len(targs)
        # Target labels
        ltarg = opts.get_SubfigOpt(sfig, "TargetLabel")
        # Process targets according to histogram type
        if htyp.lower() == "value":
            # Ensure list
            if type(ltarg).__name__ != 'list':
                ltarg = [ltarg]
            # Initialize target values
            vtarg = []
            # Loop through targets.
            for i in range(ntarg):
                # Select the target
                targ = targs[i]
                # Get the target handle.
                DBT = self.ReadFM(comp, targ=targ)
                # Get the target co-sweep
                jt = DBc.FindTargetMatch(DBT, I[0], {}, keylist="tol")
                # Check for match
                if len(jt) == 0:
                    # No value to match
                    vtarg.append(None)
                # Otherwise, get the mean value
                vtarg.append(np.mean(DBT.GetCoeff(comp, coeff, jt)))
                # Default label
                if len(ltarg) < i or (ltarg[i] is None):
                    ltarg[i] = targ
        elif ntarg < 1:
            raise ValueError(
                "Cannot plot delta or range histogram " +
                ("without one or more target (received %s)" % ntarg))
        else:
            # Read the target
            DBT = [self.ReadFM(comp, targ=targ) for targ in targs]
        # Form and set universal options for histogram
        kw_h = {
            # Reference values
            "StDev":           opts.get_SubfigOpt(sfig, "StandardDeviation"),
            "Delta":           opts.get_SubfigOpt(sfig, "Delta"),
            "PlotMu":          opts.get_SubfigOpt(sfig, "PlotMu"),
            "OutlierSigma":    opts.get_SubfigOpt(sfig, "OutlierSigma"),
            "Range":           opts.get_SubfigOpt(sfig, "Range"),
            # Plot label information
            "Label":           opts.get_SubfigOpt(sfig, "Label"),
            "TargetLabel":     ltarg,
            # Axis labels
            "XLabel":          opts.get_SubfigOpt(sfig, "XLabel"),
            "YLabel":          opts.get_SubfigOpt(sfig, "YLabel"),
            # Figure dimensions
            "FigureWidth":     opts.get_SubfigOpt(sfig, "FigureWidth"),
            "FigureHeight":    opts.get_SubfigOpt(sfig, "FigureHeight"),
            # Text labels of reference values
            "ShowMu":          opts.get_SubfigOpt(sfig, "ShowMu"),
            "ShowSigma":       opts.get_SubfigOpt(sfig, "ShowSigma"),
            "ShowDelta":       opts.get_SubfigOpt(sfig, "ShowDelta"),
            "ShowTarget":      opts.get_SubfigOpt(sfig, "ShowTarget"),
            # Format flags
            "MuFormat":        opts.get_SubfigOpt(sfig, "MuFormat"),
            "SigmaFormat":     opts.get_SubfigOpt(sfig, "SigmaFormat"),
            "DeltaFormat":     opts.get_SubfigOpt(sfig, "DeltaFormat"),
            "TargetFormat":    opts.get_SubfigOpt(sfig, "TargetFormat"),
            # Alternative labels
            "MuLabel":         opts.get_SubfigOpt(sfig, "MuLabel"),
            "SigmaLabel":      opts.get_SubfigOpt(sfig, "SigmaLabel"),
            # Extra plotting switches
            "PlotGaussian":    opts.get_SubfigOpt(sfig, "PlotGaussian"),
            "PlotMean":        opts.get_SubfigOpt(sfig, "PlotMean"),
            "PlotSigma":       opts.get_SubfigOpt(sfig, "PlotSigma"),
            # Plot options
            "HistOptions":     opts.get_SubfigOpt(sfig, "HistOptions"),
            "MeanOptions":     opts.get_SubfigOpt(sfig, "MeanOptions"),
            "StDevOptions":    opts.get_SubfigOpt(sfig, "StDevOptions"),
            "DeltaOptions":    opts.get_SubfigOpt(sfig, "DeltaOptions"),
            "TargetOptions":   opts.get_SubfigOpt(sfig, "TargetOptions"),
            "GaussianOptions": opts.get_SubfigOpt(sfig, "GaussianOptions"),
        }
        # Non-default options
        if htyp.lower() == "value":
            # Set the *TargetValue*
            kw_h["TargetValue"]  = vtarg
            # Plot the histogram with labels
            h = DBc.PlotHist(coeff, I, **kw_h)
        elif htyp.lower() == "range":
            # Range histogram
            kw_h["Target"] = DBT
            # Plot the range histogram
            h = DBc.PlotRangeHist(coeff, I, **kw_h)
        else:
            # Delta histogram
            # Add the target as a keyword input
            kw_h["Target"] = DBT
            # Plot the histogram with labels
            h = DBc.PlotHist(coeff, I, **kw_h)
       # ------------
       # Finalization
       # ------------
        # Apply other options to axes
        self.SubfigFormatAxes(sfig, h['ax'])
        # Change back to report folder.
        os.chdir(fpwd)
        # Save the figure
        fimg = self.save_figure(sfig, h)
        # Include the graphics.
        lines.append(
            '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- ContourCoeff ---
    # Function to plot mean coefficient for a sweep
    def SubfigContourCoeff(self, sfig, fswp, I, q):
        r"""Create a contour plot of results from several cases

        :Call:
            >>> R.SubfigSweepCoeff(sfig, fswp, I, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *fswp*: :class:`str`
                Name of sweep
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices in the sweep
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2017-04-17 ``@ddalle``: v1.0
        """
       # ------------------
       # Options and Config
       # ------------------
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
       # ----------------
       # Format Options
       # ----------------
        # Axis variables
        xk = opts.get_SweepOpt(fswp, "XAxis")
        yk = opts.get_SweepOpt(fswp, "YAxis")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            raise ValueError(
                "Can only create contour plot of one coefficient; " +
                ("received '%s'" % coeff))
        # Check for list of components
        if type(comp).__name__ in ['list', 'ndarray']:
            # List of components
            raise ValueError(
                "Can only create contour plot of one component; " +
                ("received '%s'" % comp))
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption.
        if fcpt is None:
            # Default format: RSRB/CLM
            fcpt = "%s/%s" % (comp, coeff)
        # Ensure there are no underscores.
        fcpt = fcpt.replace("_", r"\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
                    % (fswp, frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
       # --------
       # Plotting
       # --------
        # Read the data book component
        DBc = self.ReadFM(comp)
        # Sweep constraints
        EqCons = opts.get_SweepOpt(fswp, 'EqCons')
        TolCons = opts.get_SweepOpt(fswp, 'TolCons')
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        # Tolerate target? ...

        # Get co-sweep
        J = DBc.FindCoSweep(x, I[0], EqCons, TolCons, GlobCons)
        # Don't generate void plot
        if len(J) > 2:
            # Plot label (for legend)
            lbl = self.SubfigPlotLabel(sfig, 0)
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth")
            figh = opts.get_SubfigOpt(sfig, "FigureHeight")
            # Plot options
            kw_c = opts.get_SubfigOpt(sfig, "ContourOptions", 0)
            kw_p = opts.get_SubfigOpt(sfig, "PlotOptions",    0)
            # Plot types
            ctyp = opts.get_SubfigOpt(sfig, "ContourType")
            ltyp = opts.get_SubfigOpt(sfig, "LineType")
            # Other options
            axeq = opts.get_SubfigOpt(sfig, "AxisEqual")
            cbar = opts.get_SubfigOpt(sfig, "ColorBar")
            cmpo = opts.get_SubfigOpt(sfig, "ColorMap")
            # Draw the plot.
            h = DBc.PlotContour(
                coeff, J, x=xk, y=yk,
                ContourType=ctyp, ContourOptions=kw_c,
                LineType=ltyp, PlotOptions=kw_p,
                Label=lbl, ColorMap=cmpo,
                AxisEqual=axeq, ColorBar=cbar,
                FigureWidth=figw, FigureHeight=figh)
           # ----------
           # Formatting
           # ----------
            # Additional formatting
            self.SubfigFormatAxes(sfig, h['ax'])
            # Change back to report folder.
            os.chdir(fpwd)
            # Save the figure
            fimg = self.save_figure(sfig, h)
            # Include the graphics.
            lines.append(
                '\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
                % (fswp, frun, fimg))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- Residuals ---
    # Function to create coefficient plot and write figure
    def SubfigPlotL1(self, sfig, i, q):
        """Create plot for L1 residual

        :Call:
            >>> lines = R.SubfigPlotL1(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2014-03-09 ``@ddalle``: v1.0
            * 2015-11-25 ``@ddalle``: Moved contents to :func:`SubfigPlotResid`
        """
        return self.SubfigPlotResid(sfig, i, q, c='L1')

    # Function to create L2 plot and write figure
    def SubfigPlotL2(self, sfig, i, q):
        """Create plot for L2 residual

        :Call:
            >>> lines = R.SubfigPlotL2(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-11-25 ``@ddalle``: v1.0
        """
        return self.SubfigPlotResid(sfig, i, q, c='L2')

    # Function to create L-infinity plot and write figure
    def SubfigPlotLInf(self, sfig, i, q):
        """Create plot for L2 residual

        :Call:
            >>> lines = R.SubfigPlotL2(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-11-25 ``@ddalle``: v1.0
        """
        return self.SubfigPlotResid(sfig, i, q, c='LInf')

    # Function to create L2 plot and write figure
    def SubfigPlotTurbResid(self, sfig, i, q):
        """Create plot for turbulence residual

        :Call:
            >>> lines = R.SubfigPlotTurbResid(sfig, i, q)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-11-25 ``@ddalle``: v1.0
        """
        return self.SubfigPlotResid(sfig, i, q, c='TurbResid')

    # Function to create coefficient plot and write figure
    def SubfigPlotResid(self, sfig, i, q, c=None):
        """Create plot for named residual

        :Call:
            >>> lines = R.SubfigPlotResid(sfig, i, c=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
            *c*: :class:`str`
                Name of residual to plot (defaults to option from JSON)
        :Versions:
            * 2014-03-09 ``@ddalle``: v1.0
            * 2015-11-25 ``@ddalle``: Forked from :func:`SubfigPlotL1`
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Numbers of iterations for plots
        nPlotIter  = opts.get_SubfigOpt(sfig, "NPlotIters")
        nPlotFirst = opts.get_SubfigOpt(sfig, "NPlotFirst")
        nPlotLast  = opts.get_SubfigOpt(sfig, "NPlotLast")
        # Get caption.
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # First lines.
        lines = self.SubfigInit(sfig)
        # Check for image update
        if not q:
            # File name to check for
            fpdf = '%s.pdf' % sfig
            # Check for the file
            if os.path.isfile(fpdf):
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fpdf))
            # Set the caption.
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
        # Create plot if possible
        if nIter >= 2:
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Get figure width
            figw = opts.get_SubfigOpt(sfig, "FigureWidth")
            figh = opts.get_SubfigOpt(sfig, "FigureHeight")
            # Read the Aero history.
            H = self.ReadCaseResid(sfig)
            # Options dictionary
            kw_n = {
                "nFirst": nPlotFirst, "nLast": nPlotLast,
                "FigureWidth": figw, "FigureHeight": figh}

            # Plot options
            kw_l = opts.get_SubfigOpt(sfig, "PlotOptions", 0)
            kw_l0 = opts.get_SubfigOpt(sfig, "PlotOptions0", 0)
            kw_p = dict(kw_n, PlotOptions=kw_l, PlotOptions0=kw_l0)
            # Check for any iterations to report
            if len(H["i"]) > 0:
                # Determine which function to call
                if c == "L1":
                    # Draw the "L1" plot
                    h = H.PlotL1(n=nPlotIter, **kw_p)
                elif c == "L2":
                    # Plot global L2 residual
                    h = H.PlotL2(n=nPlotIter, **kw_p)
                elif c == "LInf":
                    # Plot global L-infinity residual
                    h = H.PlotLInf(n=nPlotIter, **kw_p)
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
                    if isinstance(cr, (list, tuple)):
                        # Plot multiple
                        for j, crj in enumerate(cr):
                            # Get options for this figure
                            kw_l = opts.get_SubfigOpt(
                                sfig, "PlotOptions", j)
                            # Merge options
                            kw_p = dict(kw_n, PlotOptions=kw_l)
                            # Plot
                            h = H.PlotResid(c=crj, n=nPlotIter, **kw_p)
                    else:
                        # Plot
                        h = H.PlotResid(c=cr, n=nPlotIter, **kw_p)
                # Additional formatting
                self.SubfigFormatAxes(sfig, h['ax'])
                # Change back to report folder.
                os.chdir(fpwd)
                # Save the figure
                fimg = self.save_figure(sfig, h)
                # Include the graphics.
                lines.append(
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fimg))
        # Set the caption.
        if fcpt:
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Ensure original directory
        os.chdir(fpwd)
        # Output
        return lines

   # --- Picture ---
    # Function to create coefficient plot and write figure
    def SubfigImage(self, sfig, i, q):
        r"""Create image based on a file present in the case folder

        :Call:
            >>> lines = R.SubfigImage(sfig, i, q)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: :class:`bool`
                Unused option whether or not to do work
        :Versions:
            * 2017-04-05 ``@ddalle``: v1.0
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
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
            # Layout file name
            fimg = opts.get_SubfigOpt(sfig, "ImageFile")
            # Check if the file is present
            if os.path.isfile(fimg):
                # Copy the file to the report folder
                shutil.copy(fimg, fpwd)
            # Check for file
            if os.path.isfile(os.path.join(fpwd, fimg)):
                # Form the line to include the image in LaTeX
                line = (
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fimg))
                # Include the graphics.
                lines.append(line)
        # Go to the report case folder
        os.chdir(fpwd)
        # Set the caption.
        if fcpt:
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- Paraview ---
    # Function to create coefficient plot and write figure
    def SubfigParaviewLayout(self, sfig, i, q):
        r"""Create image based on a Paraview Python script

        :Call:
            >>> lines = R.SubfigParaviewLayout(sfig, i, q)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
            *q*: ``True`` | ``False``
                Whether or not to update images
        :Versions:
            * 2015-11-22 ``@ddalle``: v1.0
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
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
            self.LinkVizFiles(sfig=sfig, i=i)
            # Layout file
            flay = opts.get_SubfigOpt(sfig, "Layout")
            # Full path to layout file
            fsrc = os.path.join(self.cntl.RootDir, flay)
            # Get just the file name
            flay = os.path.split(flay)[-1]
            flay = '%s.py' % sfig
            # Figure extension
            fext = opts.get_SubfigOpt(sfig, "Format")
            # Figure file name.
            fname = "%s.%s" % (sfig, fext)
            # Layout file name
            fout = opts.get_SubfigOpt(sfig, "ImageFile")
            # Name of executable
            fcmd = opts.get_SubfigOpt(sfig, "Command")
            # Check for update
            if q:
                # Run Paraview
                try:
                    # Copy the file into the current folder.
                    shutil.copy(fsrc, flay)
                    # Check executable
                    if fcmd == "pvpython":
                        # Run the layout, force offscreen rendering.
                        pvpython(flay, "--force-offscreen-rendering", cmd=fcmd)
                    else:
                        # Run the layout w/o special option
                        pvpython(flay, cmd=fcmd)
                    # Move the file to the location this subfig was built in
                    os.rename(fout, os.path.join(fpwd, fname))
                except Exception:
                    pass
            # Check for file
            if os.path.isfile(os.path.join(fpwd, fname)):
                # Form the line to include the image in LaTeX
                line = (
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fname))
                # Include the graphics.
                lines.append(line)
        # Go to the report case folder
        os.chdir(fpwd)
        # Set the caption.
        if fcpt:
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

   # --- Tecplot ---
    # Function to create coefficient plot and write figure
    def SubfigTecplotLayout(self, sfig, i, q):
        """Create image based on a Tecplot layout file

        :Call:
            >>> lines = R.SubfigTecplotLayout(sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-09-06 ``@ddalle``: v1.0
            * 2016-10-05 ``@ddalle``: Added "FieldMap" option
            * 2016-10-25 ``@ddalle``: First :mod:`cape` version
        """
        # Save current folder
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Get the vertical alignment
        hv = opts.get_SubfigOpt(sfig, "Position")
        # Get subfigure width
        wsfig = opts.get_SubfigOpt(sfig, "Width")
        # First line
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header
        fhdr = opts.get_SubfigOpt(sfig, "Header")
        # Alignment
        algn = opts.get_SubfigOpt(sfig, "Alignment")
        # Set alignment
        if algn.lower() == "center":
            lines.append('\\centering\n')
        # Write the header
        if fhdr:
            # Save the line
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-6pt\n')
        # Check for an update
        if not q:
            # Figure file name.
            fname = "%s.png" % (sfig)
            # Check if the file exists
            if os.path.isfile(fname):
                # Form the line
                line = (
                    '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                    % (frun, fname))
                # Include the graphics.
                lines.append(line)
            # Set the caption.
            if fcpt:
                lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
            # Close the subfigure.
            lines.append('\\end{subfigure}\n')
            # Output
            return lines
        # Go to the case folder
        os.chdir(self.cntl.RootDir)
        # Check if the run directory exists.
        if os.path.isdir(frun):
            # Go there.
            os.chdir(frun)
            # Get the most recent PLT files
            self.LinkVizFiles(sfig=sfig, i=i)
            # Convert VTK -> PLT if appropriate
            self.convert_vtk2plt(sfig)
            # Layout file
            flay = opts.get_SubfigOpt(sfig, "Layout")
            # Full path to layout file
            fsrc = os.path.join(self.cntl.RootDir, flay)
            # Get the local file name
            flay = '%s.lay' % sfig
            # Read the Mach number option
            omach = opts.get_SubfigOpt(sfig, "Mach")
            # Read the zone numbers for each FIELDMAP command
            fmap = opts.get_SubfigOpt(sfig, "ActiveFieldMaps")
            grps = opts.get_SubfigOpt(sfig, "FieldMap")
            # Read the Tecplot layout
            tec = self.ReadTecscript(fsrc)
            # Execute  case function
            self.SubfigFunction(sfig, i)
            # Parse fieldmaps
            # If its a list, just use those values
            if isinstance(grps, list):
                fieldmaps = grps
            elif isinstance(grps, str):
                if grps == 'auto':
                    # Init fieldmap list
                    fieldmaps = []
                    # Flag for reading initial layout lines
                    varset = True
                    # Start reading at line 1
                    iline = 1
                    # Running total of nZones
                    nZones = 0
                    # Go through each layout plt
                    while varset:
                        # Read Line
                        _fplt = tec.ReadKey(iline)
                        # Relevant lay lines start with '$!VarSet'
                        if "$!VarSet" in _fplt[0]:
                            # File names only in '|LFDSFN' lines
                            if "|LFDSFN" in _fplt[0]:
                                # Get file name
                                fplt = _fplt[1].strip("'\'\"")
                                # Read the plt file to get field map
                                tecplt = pltfile.Plt(fplt)
                                # Keep running total of nZones
                                nZones = max(nZones, nZones + tecplt.nZone)
                                # Append last zone to
                                fieldmaps.append(nZones)
                            # Line++, still in relevant lines
                            iline += 1
                        else:
                            # Assume reading past relevant lay lines
                            varset = False
                            continue
            # If nothing, assume auto with no slices
            else:
                # No field map
                fieldmaps = None
            # Set the Mach number if appropriate
            if omach:
                try:
                    tec.SetMach(self.cntl.x[omach][i])
                except Exception:
                    pass
            # Set the field map
            if fieldmaps is not None:
                try:
                    tec.SetFieldMap(fieldmaps)
                except Exception:
                    print("  Warning: FieldMap update '%s' failed" % fieldmaps)
                    pass
            # Custom turning on/off fields
            if fmap is not None:
                tec.SetPar("ActiveFieldMaps", f" = [{fmap}]", 0)
            # Layout
            self.PrepTecplotLayoutVars(tec, sfig, i)
            self.PrepTecplotSlicePosition(tec, sfig, i)
            # Color maps
            self.PrepTecplotContourLevels(tec, sfig, i)
            self.PrepTecplotColorMaps(tec, sfig, i)
            # General layout
            self.PrepTecplotLayoutKeys(tec, sfig, i)
            # Figure width in pixels (can be ``None``).
            wfig = opts.get_SubfigOpt(sfig, "FigureWidth")
            # Figure file name.
            fname = "%s.png" % (sfig)
            # Run Tecplot
            try:
                # Copy the file into the current folder.
                tec.Write(flay)
                # Run the layout.
                ExportLayout(flay, fname=fname, w=wfig)
                # Move the file.
                os.rename(fname, os.path.join(fpwd, fname))
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
            lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines

    # Evaluate a variable, expanding trajectory values
    def EvalVar(self, v, i: int) -> str:
        r"""Eval expression, ``$mach`` to ``x.mach[i]``, etc.

        :Call:
            >>> txt = R.EvalVar(raw, i)
            >>> txt = R.EvalVar(v, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *raw*: :class:`str`
                String, with ``$`` as sigil for variables to expand
            *v*: :class:`object`
                Any other format returned as ``str(v)``
            *i*: :class:`int`
                Case index
        :Outputs:
            *txt*: :class:`str`
                Input string with sigils expanded and evaluated
        :Versions:
            * 2016-10-31 ``@ddalle``: v1.0
        """
        # Check numeric
        if v is None or isinstance(v, (float, int)):
            # Do not convert
            return str(v)
        elif isinstance(v, dict):
            # Loop through dictionary
            V = {}
            for k in v:
                # Recurse
                V[k] = eval(self.EvalVar(v[k], i))
            # Return string
            return str(V)
        # Check for trajectory key
        if v in self.cntl.x.cols:
            # Get the value from the trajectory
            return str(self.cntl.x[v][i])
        # Get all sigils
        flgs = re.findall(r'\$[\w]+', v)
        # Loop through matches
        for fi in flgs:
            # Apparent name of key (remove sigil)
            ki = fi.lstrip('$')
            # Replace $mach --> x.mach[i] (for example)
            if ki in self.cntl.x.cols:
                # Find the value of the trajectory key
                vi = str(self.cntl.x[ki][i])
                # Do the string replacement
                v = v.replace(fi, vi)
        # Attempt to evaluate
        try:
            # Evaluate
            ve = eval(v)
            # Check for NumPy types
            if isinstance(ve, np.floating):
                ve = float(ve)
            elif isinstance(ve, np.integer):
                ve = int(ve)
            # Convert back to string
            return repr(ve)
        except Exception:
            # raw output
            return v

    # Function to prepare variables in Tecplot layout
    def PrepTecplotLayoutVars(self, tec: Tecscript, sfig: str, i: int):
        r"""Set any variables for Tecplot layout

        :Call:
            >>> R.PrepTecplotLayoutVars(tec, sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *tec*: :class:`cape.filecntl.tecfile.Tecscript`
                Tecplot layout interface (modified in place)
            *sfig*: :class:`str`
                Name of subfigure for accessing options
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-10-31 ``@ddalle``: v1.0
        """
        # Get list of variables
        setv = self.cntl.opts.get_SubfigOpt(sfig, "VarSet")
        # Loop through variables to set
        for k in setv:
            # Perform replacement while expanding trajectory vals
            v = self.EvalVar(setv[k], i)
            # Set the variable value
            tec.SetVar(k, v)

    # Function to prepare other keys
    def PrepTecplotLayoutKeys(self, tec: Tecscript, sfig: str, i: int):
        r"""Set any parameters for Tecplot layout

        :Call:
            >>> R.PrepTecplotLayoutKeys(tec, sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *tec*: :class:`cape.filecntl.tecfile.Tecscript`
                Tecplot layout interface (modified in place)
            *sfig*: :class:`str`
                Name of subfigure for accessing options
            *i*: :class:`int`
                Case index
        :Versions:
            * 2017-03-23 ``@ddalle``: v1.0
        """
        # Get list of options
        kopts = self.cntl.opts.get_SubfigOpt(sfig, "Keys")
        # Exit if nothing to do
        if kopts is None:
            return
        # Loop through the variables to set; each is a command
        for cmd, copts in kopts.items():
            # Loop through keys
            for key, o in copts.items():
                # Check type
                if isinstance(o, dict) and "Value" in o:
                    # Read value and target specifiers from dictionary
                    val = o.get("Value")
                    k = o.get("TargetKey")
                    v = o.get("TargetValue")
                    n = o.get("Number", 0)
                    p = o.get("Parameter")
                else:
                    # Single value; no target information
                    val = o
                    k = None
                    v = None
                    n = 0
                    p = None
                # Perform replacement while expanding trajectory vals
                if val is not None:
                    val = eval(self.EvalVar(val, i))
                if v is not None:
                    v = eval(self.EvalVar(v, i))
                if n is not None:
                    n = eval(self.EvalVar(n, i))
                if p is not None:
                    p = eval(self.EvalVar(p, i))
                # Set the variable value
                tec.SetKey(cmd, key, val, n=n, par=p, k=k, v=v)

    # Function to prepare slice locations
    def PrepTecplotSlicePosition(self, tec, sfig, i):
        """Set slice position for Tecplot layout

        :Call:
            >>> R.PrepTecplotSlicePosition(tec, sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *tec*: :class:`cape.filecntl.tecfile.Tecscript`
                Tecplot layout interface (modified in place)
            *sfig*: :class:`str`
                Name of subfigure for accessing options
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-10-31 ``@ddalle``: v1.0
        """
        # Get slice definition
        sopts = self.cntl.opts.get_SubfigOpt(sfig, "SlicePosition")
        # Exit if no slice instructions
        if not isinstance(sopts, dict):
            return
        # Initialize slice inputs
        pos = {}
        # Loop through variables to set
        for k in sopts:
            # Evaluate variable; expand trajectory keys, etc
            v = self.EvalVar(sopts[k], i)
            # Set value
            pos[k.lower()] = v
        # Default slice number
        pos.setdefault("n", 1)
        # Set the variable value
        tec.SetSliceLocation(**pos)

    # Function to prepare color maps and contours in Tecplot layout
    def PrepTecplotContourLevels(self, tec, sfig, i):
        r"""Customize contour levels for a Tecplot layout

        :Call:
            >>> R.PrepTecplotContourLevels(tec, sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *tec*: :class:`cape.filecntl.tecfile.Tecscript`
                Tecplot layout interface (modified in place)
            *sfig*: :class:`str`
                Name of subfigure for accessing options
            *i*: :class:`int`
                Case index
        :Versions:
            * 2017-01-05 ``@ddalle``: v1.0
        """
        # Get list of contour levels to alter
        clev = self.cntl.opts.get_SubfigOpt(sfig, "ContourLevels")
        # Exit if None
        if clev is None:
            return
        # Check if dictionary
        if isinstance(clev, dict):
            # Create a singleton list
            clev = [clev]
        # Loop through color maps
        for k, cl in enumerate(clev):
            # Get the options
            cl = clev[k]
            # Check type
            if not isinstance(cl, dict):
                raise TypeError(
                    "ContourLevels specification must be dict\n" +
                    "Problematic specification:\n" +
                    ("%s" % cl))
            # Get constraints
            cons = cl.get("Constraints")
            icmp = cl.get("Indices")
            fltr = cl.get("Filter")
            regx = cl.get("RegEx")
            fglb = cl.get("Glob")
            # Find indices for which these instructions should apply
            I = self.cntl.x.GetIndices(
                cons=cons, I=icmp,
                filter=fltr, re=regx, glob=fglb)
            # Check if this instruction is supposed to apply to this case
            if i not in I:
                continue
            # Get the number
            ncontour = cl.get("NContour", k)
            # Get the min/max values
            vmin = cl.get("MinLevel", 0.0)
            vmax = cl.get("MaxLevel", 1.0)
            # Evaluate the variables
            vmin = ast.literal_eval(self.EvalVar(vmin, i))
            vmax = ast.literal_eval(self.EvalVar(vmax, i))
            # Get the interval
            dv = cl.get("Delta")
            nv = cl.get("NLevel", 11)
            # Form the list
            if dv is not None:
                # Make sure min/max are divisible by *dv*
                vmin = dv * np.sign(vmin) * np.ceil(np.abs(vmin)/dv)
                vmax = dv * np.sign(vmax) * np.ceil(np.abs(vmax)/dv)
                # Override the number of levels using delta
                nv = int(np.ceil((vmax-vmin)/dv)) + 1
            # Use the number of levels
            V = np.linspace(vmin, vmax, nv)
            # Apply the change
            tec.SetContourLevels(ncontour, V)

    # Function to prepare color maps and contours in Tecplot layout
    def PrepTecplotColorMaps(self, tec, sfig, i):
        """Customize color maps for a Tecplot layout

        :Call:
            >>> R.PrepTecplotColorMaps(tec, sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *tec*: :class:`cape.filecntl.tecfile.Tecscript`
                Tecplot layout interface (modified in place)
            *sfig*: :class:`str`
                Name of subfigure for accessing options
            *i*: :class:`int`
                Case index
        :Versions:
            * 2017-01-05 ``@ddalle``: v1.0
        """
        # Get list of color maps to alter
        cmaps = self.cntl.opts.get_SubfigOpt(sfig, "ColorMaps")
        # Return if nothing to do
        if cmaps is None:
            return
        # Check if dictionary
        if type(cmaps).__name__ == "dict":
            # Create a singleton list
            cmaps = [cmaps]
        # Loop through color maps
        for k in range(len(cmaps)):
            # Get the options
            cm = cmaps[k]
            # Check type
            if not isinstance(cm, dict):
                raise TypeError(
                    "ColorMaps specification must be dict\n" +
                    "Problematic specification:\n" +
                    ("%s" % cm))
            # Get the number
            ncont = cm.get("NContour", k)
            ncmap = cm.get("NColorMap")
            # Get the name
            cname = cm.get("Name")
            # Get constraints
            cons = cm.get("Constraints")
            icmp = cm.get("Indices")
            fltr = cm.get("Filter")
            regx = cm.get("RegEx")
            fglb = cm.get("Glob")
            # Find indices for which these instructions should apply
            I = self.cntl.x.GetIndices(
                cons=cons, I=icmp,
                filter=fltr, re=regx, glob=fglb)
            # Check if this instruction is supposed to apply to this case
            if i not in I:
                continue
            # Color map
            cmk = cm.get("ColorMap")
            # Skip if no map
            if type(cmk).__name__ != "dict":
                print(
                    "  WARNING: Color map edit specification with no " +
                    '"ColorMap" key')
            # Initialize evaluated map (e.g. "2*$mach" -> 1.9)
            cme = {}
            # Evaluate the keys
            for lk in cmk:
                # Get the value (i.e. color)
                col = cmk[lk]
                # Evaluate the level
                le = eval(self.EvalVar(lk, i))
                # Save
                cme[le] = col
            # Edit the color map
            tec.EditColorMap(cname, cme, nContour=ncont, nColorMap=ncmap)

  # === Data Loaders ===
    # Read iterative history
    def ReadCaseFM(self, comp):
        r"""Read iterative history for a component

        This function needs to be customized for each solver

        :Call:
            >>> FM = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *FM*: ``None`` | :class:`cape.cfdx.databook.CaseFM`
                Case iterative force & moment history for one component
        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0
        """
        return None

    # Read residual history
    def ReadCaseResid(self, sfig=None):
        r"""Read iterative residual history for a component

        This function needs to be customized for each solver

        :Call:
            >>> hist = R.ReadCaseResid()
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
        :Outputs:
            *hist*: ``None`` | :class:`cape.cfdx.databook.CaseResid`
                Case iterative residual history for one case
        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0
        """
        return None

    # Function to read generic data book component
    def ReadFM(self, comp, targ=None):
        r"""Read a data book component and return it

        :Call:
            >>> DBc = R.ReadFM(comp, targ=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of data book component
            *targ*: {``None``} | :class:`str`
                Name of target, if any
        :Outputs:
            *DBc*: ``None`` | :class:`cape.cfdx.databook.DataBookComp`
                Individual component data book or ``None`` if not found
        :Versions:
            * 2017-04-20 ``@ddalle``: v1.0
        """
        # Make sure the data book is present
        self.ReadDataBook()
        # Check for patch delimiter
        if "/" in comp:
            # Format: MPCV_Camera_patch/front
            compo, patch = comp.split("/")
        elif "." in comp:
            # Format: MPCV_Camera_patch.front
            compo, patch = comp.split(".")
        else:
            # Only comp given; use total of patches
            compo = comp
            patch = None
        # Get the component type
        tcomp = self.cntl.opts.get_DataBookType(compo)
        # Read the target if any
        if targ is not None:
            # Read the target if necessary
            self.cntl.DataBook.ReadTarget(targ)
            # Get the target type
            ttype = self.cntl.opts.get_TargetDataBookType(targ)
            # Check if this is a duplicate type
            qdup = (ttype in ['cape', 'duplicate']) or ttype.startswith("py")
            # Check for duplicate
            if not qdup:
                # Just get the target
                return self.cntl.DataBook.Targets[targ]
            # Otherwise, get handle to the data book target
            DB = self.cntl.DataBook.Targets[targ]
        else:
            # Use the handle to the master data book
            DB = self.cntl.DataBook
        # Filter on the type
        try:
            if tcomp in ["Force", "FM", "Moment", "DataFM"]:
                # Read if necessary
                if comp not in DB:
                    DB.ReadFM(comp)
                # Output component
                return DB[comp]
            elif tcomp in ["LineLoad"]:
                # Read the line load
                DB.ReadLineLoad(comp, targ=targ)
                # Check for target component name
                if targ is None:
                    # Use the original component name
                    tcomp = comp
                else:
                    # Prepend target name
                    tcomp = '%s\\%s' % (targ, comp)
                # Return it
                return DB.LineLoads[tcomp]
            elif tcomp in ["TriqFM"]:
                # Read the component
                DB.ReadTriqFM(compo)
                # Output
                return DB.TriqFM[compo][patch]
            elif tcomp in ["TriqPoint"]:
                # Read the point group
                DB.ReadTriqPoint(compo, pt=patch)
                # Output
                return DB.TriqPoint[compo][patch]
        except Exception:
            # Error if reading from data book
            if targ is None:
                raise ValueError(
                    ("Could not read data book component '%s'" % comp) +
                    (", type '%s'" % tcomp))
            else:
                # Empty data book
                return None

    # Function to read the data book and reread it if necessary
    def ReadDataBook(self, fsrc="data"):
        r"""Read the data book if necessary for a specific sweep

        :Call:
            >>> R.ReadDataBook(fsrc="data")
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fsrc*: {``"data"``} | ``"trajectory"`` | :class:`str`
                Data book trajectory source
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
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
            self.cntl.DataBook.MatchRunMatrix()
            # Save the data book source.
            self.cntl.DataBook.source = "trajectory"
        else:
            # Match the trajectory to the data book.
            self.cntl.DataBook.UpdateRunMatrix()
            # Save the data book source.
            self.cntl.DataBook.source = "data"

    # Read a TriqFM data book
    def ReadTriqFM(self, comp, fsrc="data", targ=None):
        """Read a TriqFM data book if necessary for a specific sweep

        :Call:
            >>> DBF = R.ReadTriqFM(comp, fsrc="data", targ=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of TriqFM component
            *fsrc*: {``"data"``} | ``"trajectory"`` | :class:`str`
                Data book trajectory source
            *targ*: {``None``} | :class:`str`
                Name of target data book, if any
        :Outputs:
            *DBF*: :class:`cape.cfdx.pointsensor.TriqFMDataBook`
                Patch loads data book
        :Versions:
            * 2017-04-05 ``@ddalle``: v1.0
        """
        # Read the data book
        self.ReadDataBook(fsrc)
        # Check for target
        if targ is None:
            # Read the data book as approrpiate
            self.cntl.DataBook.ReadTriqFM(comp)
            # Handle to data book
            DB = self.cntl.DataBook
            # Get a handle to the TriqFM data book
            DBF = DB.TriqFM[comp]
        else:
            # Read the Target
            self.cntl.DataBook.ReadTarget(targ)
            # Get target
            DB = self.cntl.DataBook.Targets[targ]
            # Read the TriqFM component
            DB.ReadTriqFM(comp)
            # Get TriqFM data book
            DBF = DB.TriqFM[comp]
        # Check the source
        try:
            # See if a source has been marked
            DBF.source
        except AttributeError:
            # Set default source
            DBF.source = "none"
        # Check the existing source
        if fsrc == DBF.source:
            # Everything is good
            return DBF
        elif DBF.source != "none":
            # Previously read data book with opposite source
            del DBF
            del DB.TriqFM[comp]
            # Reread the data book
            DB.ReadTriqFM(comp)
            # Update handle
            DBF = DB.TriqFM[comp]
            DBF.source = "none"
        # Check the requested source.
        if fsrc == "trajectory":
            # Match the data book to the trajectory
            DBF.MatchRunMatrix()
            # Save the data book source.
            DBF.source = "trajectory"
        else:
            # Save the data book source.
            DBF.source = "data"
        # Output if desired
        return DBF

    # Read a point sensor group
    def ReadTriqPoint(self, grp, pt,  targ=None):
        r"""Read a point sensor

        :Call:
            >>> DBP = R.ReadTriqPoint(grp, pt, targ=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *grp*: :class:`str`
                Name of TriqPoint group
            *pt*: :class:`str`
                Name of the TriqPoint sensor
            *targ*: {``None``} | :class:`str`
                Name of target data book, if any
        :Outputs:
            *DBF*: :class:`cape.cfdx.pointsensor.TriqFMDataBook`
                Point sensor group data book
        :Versions:
            * 2018-02-09 ``@ddalle``: v1.0
        """
        # Check for target
        if targ is None:
            # Read the data book as approrpiate
            self.cntl.DataBook.ReadTriqPoint(grp, pt)
            # Handle to data book
            DB = self.cntl.DataBook
            # Get a handle to the TriqFM data book
            DBP = DB.TriqPoint[grp][pt]
        else:
            # Read the Target
            self.cntl.DataBook.ReadTarget(targ)
            # Get target
            DB = self.cntl.DataBook.Targets[targ]
            # Get the target type
            ttype = self.cntl.opts.get_TargetDataBookType(targ)
            # Check if this is a duplicate type
            qdup = (ttype in ['cape', 'duplicate']) or ttype.startswith("py")
            # Check for duplicate
            if not qdup:
                # Just get the target
                return DB
            # Read the TriqFM component
            DB.ReadTriqPoint(grp, pt)
            # Get TriqFM data book
            DBP = DB.TriqPoint[grp][pt]
        # Output if desired
        return DBP

    # Read line loads
    def ReadLineLoad(self, comp, i, targ=None, update=False):
        r"""Read line load for a case

        :Call:
            >>> LL = R.ReadLineLoad(comp, i, targ=None, update=False)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of line load component
            *i*: :class:`int`
                Case number
            *targ*: {``None``} | :class:`str`
                Name of target data book to read, if not ``None``
            *update*: ``True`` | {``False``}
                Whether to attempt an update if case not in data book
        :Outputs:
            *LL*: :class:`cape.cfdx.lineload.CaseLL`
                Individual case line load interface
        :Versions:
            * 2016-06-10 ``@ddalle``: v1.0
            * 2017-04-05 ``@ddalle``: v1.1; ``cfdx`` version
        """
        # Ensure configuration is present
        self.cntl.ReadConfig()
        # Read the data book and line load data book
        self.cntl.ReadDataBook()
        # Get data book handle
        if targ is None:
            # General data book
            DB = self.cntl.DataBook
            # Read the line load data book
            DB.ReadLineLoad(comp, conf=self.cntl.config)
            DBL = DB.LineLoads[comp]
            # Use the index directly
            j = i
        else:
            # Pointer to the data book
            DB = self.cntl.DataBook
            # Read the target as appropriate
            DB.ReadTarget(targ)
            # Target data book
            DBT = DB.Targets[targ]
            # Get target options
            topts = self.cntl.opts.get_TargetDataBookByName(targ)
            # Read Line load
            DBT.ReadLineLoad(comp, targ=targ, conf=self.cntl.config)
            # Get title
            ttl = "%s\\%s" % (targ, comp)
            # Copy the line load
            DBT.LineLoads[comp] = DBT.LineLoads[ttl]
            # Extract it
            DBL = DBT.LineLoads[comp]
            # Update the trajectory
            DBL.UpdateRunMatrix()
            # Find a match
            J = DBL.FindTargetMatch(DB, i, topts, keylist='tol')
            # Check for a match
            if len(J) == 0:
                print(
                    ("  Warning: Could not find target line load ") +
                    ("for target '%s', comp '%s', index %s" % (targ, comp, i)))
                return None
            # Get the first match
            j = J[0]
            # Move the handle.
            DB = DBT
        # Check auto-update flag
        if update:
            # Update the case (even if up-to-date)
            DBL.UpdateCase(j, seam=True)
        # Read the case
        DBL.ReadCase(j)
        # Output the case line load
        return DBL.get(j)

    # Read a Tecplot script
    def ReadTecscript(self, fsrc):
        r"""Read a Tecplot script interface

        :Call:
            >>> R.ReadTecscript(fsrc)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fscr*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0
        """
        return Tecscript(fsrc)

  # === Status Tools ===
    # Read the ``report.json`` file
    def ReadCaseJSON(self):
        r"""Read the JSON file which contains the current statuses

        :Call:
            >>> rc = R.ReadCaseJSON()
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
        :Outputs:
            *rc*: :class:`dict`
                Dictionary of subfigure definitions and status
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0
        """
        # Check for the file
        if not os.path.isfile('report.json'):
            # Default output
            return {
                "Status": {},
                "Subfigures": {}
            }
        # Open the file
        f = open('report.json')
        # Read the settings.
        try:
            # Read from file
            rc = json.load(f)
        except Exception:
            # No settings read
            rc = {}
        # Close the file.
        f.close()
        # Ensure the existence of main sections
        rc.setdefault("Status", {})
        rc.setdefault("Subfigures", {})
        # Return the settings
        return rc

    # Write all settings
    def WriteCaseJSON(self, rc):
        """Write the current status to ``report.json``

        :Call:
            >>> R.WriteCaseJSON(rc)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *rc*: :class:`dict`
                Dictionary of subfigure definitions and status
        :Versions:
            * 2016-10-25 ``@ddalle``: v1.0
        """
        # Open the file
        f = open('report.json', 'w')
        # Write the contents.
        json.dump(rc, f, indent=1)
        # Close the file.
        f.close()

  # === Sweep Indices ===
    # Function to get update sweeps
    def GetSweepIndices(self, fswp, I=None, cons=[], comp=None):
        r"""Divide cases into individual sweeps

        :Call:
            >>> J = R.GetSweepIndices(fswp, I=None, cons=[], comp=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints to define what cases to update
            *comp*: {``None``} | :class:`str`
                Optional name of DataBook component (useful for TriqFM
                components, which may have a different trajectory)
        :Outputs:
            *J*: :class:`list` (:class:`numpy.ndarray`\ [:class:`int`])
                List of sweep index lists
        :Versions:
            * 2015-05-29 ``@ddalle``: v1.0
        """
        # Extract options
        opts = self.cntl.opts
        # Check if only restricting to point currently in the trajectory.
        if opts.get_SweepOpt(fswp, 'RunMatrixOnly'):
            # Read the data book with the trajectory as the source
            fsrc = "trajectory"
            self.ReadDataBook(fsrc)
        else:
            # Read the data book with the data book as the source
            fsrc = "data"
            self.ReadDataBook(fsrc)
        # Get component
        if comp is None:
            # Extract from global data book
            x = self.cntl.DataBook.x
        elif opts.get_DataBookType(comp) in ["TriqFM"]:
            # Read the TriqFM data book
            self.ReadTriqFM(comp, fsrc)
            # Use that DataBook
            x = self.cntl.DataBook.TriqFM[comp].x
        else:
            # Component given, but still use global *x*
            x = self.cntl.DataBook.x
        # Do not restrict indexes if using *data*
        if fsrc == "data":
            I = np.arange(x.nCase)
        # Sort variable
        xk = opts.get_SweepOpt(fswp, 'XAxis')
        # Sweep constraints
        EqCons = opts.get_SweepOpt(fswp, 'EqCons')
        TolCons = opts.get_SweepOpt(fswp, 'TolCons')
        IndexTol = opts.get_SweepOpt(fswp, 'IndexTol')
        # Global constraints
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        Indices  = opts.get_SweepOpt(fswp, 'Indices')
        # Turn command-line constraints into indices.
        I0 = x.GetIndices(cons=cons, I=I)
        # Turn report sweep definition into indices.
        I1 = x.GetIndices(cons=GlobCons, I=Indices)
        # Restrict Indices
        I = np.intersect1d(I0, I1)
        # Divide the cases into individual sweeps.
        J = x.GetSweeps(
            I=I, SortVar=xk,
            EqCons=EqCons, TolCons=TolCons, IndexTol=IndexTol)
        # Output
        return J

    # Function to get subset of target catches matching a sweep
    def GetTargetSweepIndices(self, fswp, i0, targ, cons=[]):
        r"""Get indices of target matching a local sweep's constraint

        :Call:
            >>> I = R.GetTargetSweepIndices(fswp, i0, targ)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *i0*: :class:`int`
                Index of point in *R.cntl.DataBook.x* to use as reference
            *targ*: :class:`int`
                Name of the target in data book to use
        :Outputs:
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of target data indices
        :Versions:
            * 2015-06-03 ``@ddalle``: v1.0
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
        # Carpet constraints
        CEqCons = opts.get_SweepOpt(fswp, 'CarpetEqCons')
        CTolCons = opts.get_SweepOpt(fswp, 'CarpetTolCons')
        # Append constraints
        EqCons += CEqCons
        for k in CTolCons:
            TolCons[k] = CTolCons[k]
        # Global constraints
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        # Turn command-line constraints into indices.
        I0 = DBT.x.GetIndices(cons=cons)
        # Turn report sweep definition into indices.
        I1 = DBT.x.GetIndices(cons=GlobCons)
        # Restrict Indices
        I = np.intersect1d(I0, I1)
        # Get the matching sweep.
        I = DBT.x.GetCoSweep(
            self.cntl.DataBook.x, i0,
            SortVar=xk, EqCons=EqCons, TolCons=TolCons, I=I)
        # Output
        return I

    # Function to get subset of target catches matching a sweep
    def GetCoSweepIndices(self, fswp, i0, comp, cons=[], targ=None):
        r"""Get indices of a target matching local sweep constraints

        :Call:
            >>> I = R.GetTargetSweepIndices(fswp, i0, comp, cons, targ)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *fswp*: :class:`str`
                Name of sweep to update
            *i0*: :class:`int`
                Index of ref point in *R.cntl.DataBook.x*
            *targ*: {``None``} | :class:`str`
                Name of the target in data book to use (if any)
        :Outputs:
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of target data indices
        :Versions:
            * 2015-06-03 ``@ddalle``: v1.0
        """
        # Extract options
        opts = self.cntl.opts
        # Get data book type
        ctyp = opts.get_DataBookType(comp)
        # Check type
        if ctyp != "TriqFM":
            raise ValueError(
                ("Invalid co-sweep request for component '%s'" % comp) +
                (" of type '%s'" % ctyp))
        # Read the data book
        DBT = self.ReadTriqFM(comp, targ=targ)
        # Sort variable
        xk = opts.get_SweepOpt(fswp, 'XAxis')
        # Sweep constraints
        EqCons = opts.get_SweepOpt(fswp, 'EqCons')
        TolCons = opts.get_SweepOpt(fswp, 'TolCons')
        # Carpet constraints
        CEqCons = opts.get_SweepOpt(fswp, 'CarpetEqCons')
        CTolCons = opts.get_SweepOpt(fswp, 'CarpetTolCons')
        # Append constraints
        EqCons += CEqCons
        for k in CTolCons:
            TolCons[k] = CTolCons[k]
        # Global constraints
        GlobCons = opts.get_SweepOpt(fswp, 'GlobalCons')
        # Turn command-line constraints into indices.
        I0 = DBT.x.GetIndices(cons=cons)
        # Turn report sweep definition into indices.
        I1 = DBT.x.GetIndices(cons=GlobCons)
        # Restrict Indices
        I = np.intersect1d(I0, I1)
        # Get the matching sweep.
        I = DBT.x.GetCoSweep(
            self.cntl.DataBook.x, i0,
            SortVar=xk, EqCons=EqCons, TolCons=TolCons, I=I)
        # Output
        return I

  # === Run Folder Tools ===
    # Function to link appropriate visualization files
    def LinkVizFiles(self, sfig=None, i=None):
        r"""Create links to appropriate visualization files

        :Call:
            >>> R.LinkVizFiles(sfig, i)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-01-07 ``@ddalle``: v1.1; add *sfig* and *i* inputs
        """
        pass

    # Convert VTK -> PLT
    def convert_vtk2plt(self, sfig: str):
        r"""Convert VTK file(s) to Tecplot(R) PLT format if requested

        :Call:
            >>> rep.convert_vtk2plt(sfig, i)
        :Inputs:
            *rep*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-07-17 ``@ddalle``: v1.0
        """
        # Get list of VTK files
        vtkfiles = self.cntl.opts.get_SubfigOpt(sfig, "VTKFiles")
        # Check for empty
        if (vtkfiles is None) or len(vtkfiles) == 0:
            return
        # Loop through
        for vtkfile in vtkfiles:
            # Check for existence
            if not os.path.isfile(vtkfile):
                print(f"   Missing VTK file '{vtkfile}'")
                continue
            # Check for link
            if os.path.islink(vtkfile):
                # Get the real name
                vtkrealfile = os.path.realpath(vtkfile)
                # Path to folder containing link
                vtkdir = os.path.join(os.getcwd(), os.path.dirname(vtkfile))
                # Relative path for link
                vtkrelfile = os.path.relpath(vtkrealfile, vtkdir)
                # Replace VTK link with original file
                vtkfile = os.path.relpath(vtkrealfile, os.getcwd())
                # Relative path for PLT link
                pltrelfile = vtkrelfile.rsplit('.', 1)[0] + ".plt"
                # Path to PLT link
                pltlinkfile = vtkfile.rsplit('.', 1)[0] + ".plt"
                # Pre-create viz link for .plt file
                if os.path.isfile(pltlinkfile):
                    os.remove(pltlinkfile)
                os.symlink(pltrelfile, pltlinkfile)
            # Convert it
            convert_vtk(vtkfile)

  # === Image I/O ===
    # Function to save images in various formats
    def save_figure(self, sfig=None, h=None):
        """Write out image files in varous formats

        :Call:
            >>> R.save_figure(sfig, sfig, h)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure
            *h*: :class:`matplotlib.figure`
                Matplotlib figure handle
        :Output:
            *fimg*: :class:`str`
                Figure name
        :Versions:
            * 2023-06-06 ``@jmeeroff``: First version
        """
        # Extract options
        opts = self.cntl.opts
        # Get the file formatting
        fmt = opts.get_SubfigOpt(sfig, "Format")
        dpi = opts.get_SubfigOpt(sfig, "DPI")
        # Figure name
        fimg = '%s.%s' % (sfig, fmt)
        fpdf = '%s.pdf' % sfig
        # Save the figure.
        try:
            if fmt.lower() in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt.lower() in ['svg']:
                # Save as PDF and SVG
                h['fig'].savefig(fimg)
                h['fig'].savefig(fpdf)
                # Use PDF in LaTex
                fimg = fpdf
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
                h['fig'].savefig(fpdf)
            # Close the figure.
            h['fig'].clf()
            # Return the image name
            return fimg
        except Exception:
            print("    Plotting failed, probably due to a NaN.")
