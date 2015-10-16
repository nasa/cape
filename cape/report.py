"""
Basis Module for Automated Report Interface: :mod:`cape.report`
===============================================================

"""

# File system inreface
import os, json, shutil, glob
# Numeris
import numpy as np

# Local modules needed
import tex, tar


# Class to interface with report generation and updating.
class Report(object):
    """Interface for automated report generation
    
    :Call:
        >>> R = cape.report.Report(cntl, rep)
    :Inputs:
        *cntl*: :class:`cape.cntl.Cntl`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyCart.report.Report
            Automated report interface
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
            *R*: :class:`pyCart.report.Report`
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
        
        # Set the title and author.
        f.write('\\title{%s}\n' % self.cntl.opts.get_ReportTitle(self.rep))
        f.write('\\author{%s}\n' % self.cntl.opts.get_ReportAuthor(self.rep))
        
        # Format the header and footer
        f.write('\n\\fancypagestyle{pycart}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0.4pt}%\n')
        f.write(' \\renewcommand{\\footrulewidth}{0.4pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write(' \\fancyfoot[C]{\\textbf{\\textsf{%s}}}%%\n'
            % self.cntl.opts.get_ReportRestriction(self.rep))
        f.write(' \\fancyfoot[R]{\\thepage}%\n')
        # Check for a logo.
        if len(self.opts.get('Logo','')) > 0:
            f.write(' \\fancyfoot[L]{\\raisebox{-0.32in}{%\n')
            f.write('  \\includegraphics[height=0.45in]{%s}}}%%\n'
                % self.opts['Logo'])
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
        f.write('\\pagestyle{plain}\n')
        f.write('\\maketitle\n\n')
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
        f.write('\\newpage\n')
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
        f.write('\n\\newpage\n')
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
    
    
    
    
    
    
    # Function to write sweep conditions table
    def SubfigSweepCases(self, sfig, fswp, I):
        """Create lines for a "SweepConditions" subfigure
        
        :Call:
            >>> lines = R.SubfigSweepCases(sfig, fswp, I)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
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
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-09 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Case folder
        frun = self.cart3d.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cart3d.opts
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
        nIter  = self.cart3d.CheckCase(i)
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
            os.chdir(self.cart3d.RootDir)
            os.chdir(frun)
            # Read the Aero history.
            FM = Aero([comp])[comp]
            # Loop through the transformations.
            for topts in opts.get_DataBookTransformations(comp):
                # Apply the transformation.
                FM.TransformFM(topts, self.cart3d.x, i)
            # Get the statistics.
            s = FM.GetStats(nStats=nStats, nMax=nMax, nLast=nPlotLast)
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
            # Draw the plot.
            h = FM.PlotCoeff(coeff, n=nPlotIter, nAvg=s['nStats'],
                nFirst=nPlotFirst, nLast=nPlotLast,
                LineOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
                ShowMu=sh_m, ShowDelta=sh_d,
                ShowSigma=sh_s, ShowEspsilon=sh_e,
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
            if fmt in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            elif fmt in ['svg']:
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
            *R*: :class:`pyCart.report.Report`
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
            figw = opts.get_SubfigOpt(sfig, "FigureWidth", k)
            figh = opts.get_SubfigOpt(sfig, "FigureHeight", k)
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
        # Save the figure.
        if fmt in ['pdf', 'svg']:
            # Save as vector-based image.
            h['fig'].savefig(fimg)
        else:
            # Save with resolution.
            h['fig'].savefig(fimg, dpi=dpi)
        # Close the figure.
        h['fig'].clf()
        # Include the graphics.
        lines.append('\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fimg))
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
        # Create plot if possible
        if nIter >= 2:
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Get figure width
            figw = opts.get_SubfigOpt(sfig, "FigureWidth")
            figh = opts.get_SubfigOpt(sfig, "FigureHeight")
            # Read the Aero history.
            hist = CaseResid()
            # Draw the plot.
            h = hist.PlotL1(n=nPlotIter, 
                nFirst=nPlotFirst, nLast=nPlotLast,
                FigWidth=figw, FigHeight=figh)
            # Change back to report folder.
            os.chdir(fpwd)
            # Get the file formatting
            fmt = opts.get_SubfigOpt(sfig, "Format")
            dpi = opts.get_SubfigOpt(sfig, "DPI")
            # Figure name
            fimg = '%s.%s' % (sfig, fmt)
            # Save the figure.
            if fmt in ['pdf']:
                # Save as vector-based image.
                h['fig'].savefig(fimg)
            else:
                # Save with resolution.
                h['fig'].savefig(fimg, dpi=dpi)
            # Close the figure.
            h['fig'].clf()
            # Include the graphics.
            lines.append('\\includegraphics[width=\\textwidth]{%s/%s}\n'
                % (frun, fimg))
        # Set the caption.
        if fcpt: lines.append('\\caption*{\scriptsize %s}\n' % fcpt)
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
            *R*: :class:`pyCart.report.Report`
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
        # Get statistics if possible.
        if nCur >= max(1, nMin+nStats):
            # Don't use iterations before *nMin*
            nMax = min(nMax, nCur-nMin)
            # Go to the run directory.
            os.chdir(self.cart3d.RootDir)
            os.chdir(self.cart3d.x.GetFullFolderNames(i))
            # Read the Aero history
            A = Aero(comps)
            # Transform each comparison.
            for comp in comps:
                # Loop through the transformations.
                for topts in opts.get_DataBookTransformations(comp):
                    # Apply the transformation.
                    A[comp].TransformFM(topts, self.cart3d.x, i)
            # Get the statistics.
            S = A.GetStats(nStats=nStats, nMax=nMax, nLast=nCur)
        else:
            # No stats.
            S = {}
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
        for c in self.cart3d.opts.get_SubfigOpt(sfig, "Coefficients"):
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
            for fs in self.cart3d.opts.get_SubfigOpt(sfig, c):
                # Write the description
                if fs == 'mu':
                    # Mean
                    lines.append('\\textit{%s} mean, $\\mu(%s)$\n' % (c, fc))
                elif fs == 'std':
                    # Standard deviation
                    lines.append(
                        '\\textit{%s} standard deviation, $\\sigma(%s)$\n'
                        % (c, fc))
                elif fs == 'err':
                    # Uncertainty
                    lines.append('\\textit{%s} iterative uncertainty, ' % c 
                        + '$\\varepsilon(%s)$\n' % fc)
                elif fs == 'min':
                    # Min value
                    lines.append(
                        '\\textit{%s} minimum, $\\min(%s)$\n' % (c, fc))
                elif fs == 'max':
                    # Min value
                    lines.append(
                        '\\textit{%s} maximum, $\\max(%s)$\n' % (c, fc))
                elif fs == "t":
                    # Target value
                    lines.append(
                        '\\textit{%s} target, $t(%s)$\n' % (c, fc))
                # Initialize line
                line = ''
                # Loop through components.
                for comp in comps:
                    # Check for iterations.
                    if nCur <= 0 or comp not in S:
                        # No iterations
                        line += '& $-$ '
                    elif fs == 'mu':
                        # Process value.
                        line += ('& $%.4f$ ' % S[comp][c])
                    elif (fs in ['min', 'max']) or (S[comp]['nStats'] > 1):
                        # Present?
                        if (c+'_'+fs) in S[comp]:
                            # Process min/max or statistical value
                            line += ('& $%.4f$ ' % S[comp][c+'_'+fs])
                        else:
                            # Missing
                            line += '& $-$ '
                    else:
                        # No statistics
                        line += '& $-$ '
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
        DBc = self.cntl.DataBook[self.cart3d.DataBook.Components[0]]
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
            fitr = '-/%s' % self.cntl.opts.get_IterSeq(-1)
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
                Index of point in *R.cart3d.DataBook.x* to use as reference
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


