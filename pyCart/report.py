"""Function for Automated Report Interface"""

# File system interface
import os, json, shutil, glob

# Local modules needed
from . import tex, tar
# Data book and plotting
from .dataBook import Aero, CaseFM, CaseResid
# Folder and Tecplot management
from .case    import LinkPLT
from .tecplot import ExportLayout
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
class Report(object):
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
        # Check for this report.
        if rep not in cart3d.opts.get_ReportList():
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Go to home folder.
        fpwd = os.getcwd()
        os.chdir(cart3d.RootDir)
        # Create the report folder if necessary.
        if not os.path.isdir('report'): os.mkdir('report', dmask)
        # Go into the report folder.
        os.chdir('report')
        # Get the options and save them.
        self.rep = rep
        self.opts = cart3d.opts['Report'][rep]
        # Initialize a dictionary of handles to case LaTeX files
        self.cases = {}
        # Read the file if applicable
        self.OpenMain()
        # Return
        os.chdir(fpwd)
        
        
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
        os.chdir(self.cart3d.RootDir)
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
        """
        # Check for use of constraints instead of direct list.
        if I is None:
            # Apply constraints
            I = self.x.Filter(kw.get('cons', []))
        # Clear out the lines.
        del self.tex.Section['Cases'][1:-1]
        # Loop through those cases.
        for i in I:
            # Update the case
            self.UpdateCase(i)
        # Update the text.
        self.tex._updated_sections = True
        self.tex.UpdateLines()
        # Master file location\
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
        os.chdir('report')
        # Write the file.
        self.tex.Write()
        # Compmile it.
        print("Compiling...")
        self.tex.Compile()
        # Need to compile twice for links
        print("Compiling...")
        self.tex.Compile()
        # Clean up
        print("Cleaning up")
        # Get other 'report-*.*' files.
        fglob = glob.glob('%s*' % self.fname[:-3])
        # Delete most of them.
        for f in fglob:
            # Check for the two good ones.
            if f[-3:] in ['tex', 'pdf']: continue
            # Else remove it.
            os.remove(f)
        
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
        # Status update
        print('%s/%s' % (fgrp, fdir))
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
        # Add the line to the master LaTeX file
        self.tex.Section['Cases'].insert(-1,
            '\\input{%s/%s/%s}\n' % (fgrp, fdir, self.fname))
        # Read the status file.
        nr, stsr = self.ReadCaseJSONIter()
        # Get the actual iteration number.
        n = self.cart3d.CheckCase(i)
        sts = self.cart3d.CheckCaseStatus(i)
        # Check if there's anything to do.
        if not ((nr is None) or (nr < n) or (stsr != sts)):
            # Go home and quit.
            os.chdir(fpwd)
            return
        # -------------
        # Initial setup
        # -------------
        # Status update
        if nr == n:
            # Changing status
            print("  Updating status %s --> %s" % (stsr, sts))
        elif nr:
            # More iterations
            print("  Updating from iteration %s --> %s" % (nr, n))
        else:
            # New case
            print("  New report at iteration %s" % n)
        # Check for the file.
        if not os.path.isfile(self.fname): self.WriteCaseSkeleton(i)
        # Open it.
        self.cases[i] = tex.Tex(self.fname)
        # Set the iteration number and status header.
        self.SetHeaderStatus(i)
        # -------
        # Figures
        # -------
        # Check for alternate statuses.
        if sts == "ERROR":
            # Get the figures for FAILed cases
            figs = self.cart3d.opts.get_ReportErrorFigList(self.rep)
        elif n:
            # Nominal case with some results
            figs = self.cart3d.opts.get_ReportFigList(self.rep)
        else:
            # No FAIL file, but no iterations
            figs = self.cart3d.opts.get_ReportZeroFigList(self.rep)
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
    def UpdateFigure(self, fig, i):
        """Write the figure and update the contents as necessary for *fig*
        
        :Call:
            >>> R.UpdateFigure(fig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fig*: :class:`str`
                Name of figure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # -----
        # Setup
        # -----
        # Handle for the case file.
        tx = self.cases[i]
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
        # Do not indent anything.
        lines.append('\\noindent\n')
        # Get the optional header
        fhdr = self.cart3d.opts.get_FigHeader(fig)
        if fhdr:
            # Add it.
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-10pt\n')
        # Start the figure.
        lines.append('\\begin{figure}[!h]\n')
        # Get figure alignment
        falgn = self.cart3d.opts.get_FigAlignment(fig)
        if falgn.lower() == "center":
            # Centering
            lines.append('\\centering\n')
        # -------
        # Subfigs
        # -------
        # Get list of subfigures.
        sfigs = self.cart3d.opts.get_FigSubfigList(fig)
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
        # -------
        # Cleanup
        # -------
        # End the figure for LaTeX
        lines.append('\\end{figure}\n')
        # pyCart report end figure marker
        lines.append('%>\n')
        # Add the lines to the section.
        for line in lines:
            tf.insert(ifig, line)
            ifig += 1
        # Update the section
        tx.Section['Figures'] = tf
        tx._updated_sections = True
        # Synchronize the document
        tx.UpdateLines()
        
    # Function to write conditions table
    def SubfigConditions(self, sfig, i):
        """Create lines for a "Conditions" subfigure
        
        :Call:
            >>> lines = R.SubfigConditions(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Extract the trajectory
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
        # Loop through the figures.
        for k in x.keys:
            # Check if it's a skip variable
            if k in skvs: continue
            # Write the variable name.
            line = "{\\small\\textsf{%s}}" % k.replace('_', '\_')
            # Append the abbreviation.
            line += (" & {\\small\\textsf{%s}} & " % 
                x.defns[k]['Abbreviation'].replace('_', '\_'))
            # Append the value.
            if x.defns[k]['Value'] in ['str', 'unicode']:
                # Put the value in sans serif
                line += "{\\small\\textsf{%s}} \\\\\n" % getattr(x,k)[i]
            elif x.defns[k]['Value'] in ['float', 'int']:
                # Put the value as a number
                line += "$%s$ \\\\\n" % getattr(x,k)[i]
            else:
                # Just put a string
                line += "%s \\\\\n" % getattr(x,k)[i]
            # Add the line to the table.
            lines.append(line)
        
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
                Name of sfigure to update
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
        # Current status
        nIter  = self.cart3d.CheckCase(i)
        # Numbers of iterations for statistics
        nStats = opts.get_SubfigOpt(sfig, "nStats")
        nMin   = opts.get_SubfigOpt(sfig, "nMinStats")
        nMax   = opts.get_SubfigOpt(sfig, "nMaxStats")
        # Get the status and data book options
        if nStats is None: nStats = opts.get_nStats()
        if nMin   is None: nMin   = opts.get_nMin()
        if nMax   is None: nMax   = opts.get_nMaxStats()
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
        # Process default caption. 
        if fcpt is None: fcpt = "%s/%s" % (comp.replace('_','\_'), coeff)
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
            # Don't use iterations before *nMin*
            nMax = min(nMax, nIter-nMin)
            # Go to the run directory.
            os.chdir(self.cart3d.RootDir)
            os.chdir(frun)
            # Read the Aero history.
            FM = Aero([comp])[comp]
            # Get the statistics.
            s = FM.GetStats(nStats=nStats, nMax=nMax, nLast=nPlotLast)
            # Get the manual range to show
            dc = opts.get_SubfigOpt(sfig, "Delta")
            # Get the multiple of standard deviation to show
            ksig = opts.get_SubfigOpt(sfig, "StandardDeviation")
            # Get the multiple of iterative error to show
            uerr = opts.get_SubfigOpt(sfig, "IterativeError")
            # Get figure dimensions.
            figw = opts.get_SubfigOpt(sfig, "FigureWidth")
            figh = opts.get_SubfigOpt(sfig, "FigureHeight")
            # Plot options
            kw_p = opts.get_SubfigOpt(sfig, "LineOptions")
            kw_m = opts.get_SubfigOpt(sfig, "MeanOptions")
            kw_s = opts.get_SubfigOpt(sfig, "StDevOptions")
            kw_u = opts.get_SubfigOpt(sfig, "ErrPltOptions")
            kw_d = opts.get_SubfigOpt(sfig, "DeltaOptions")
            # Draw the plot.
            h = FM.PlotCoeff(coeff, n=nPlotIter, nAvg=s['nStats'],
                nFirst=nPlotFirst, nLast=nPlotLast,
                LineOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
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
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure.
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Function to create coefficient plot and write figure
    def SubfigPlotL1(self, sfig, i):
        """Create plot for L1 residual
        
        :Call:
            >>> lines = R.SubfigPlotL1(sfig, i)
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
        # Case folder
        frun = self.cart3d.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cart3d.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Current status
        nIter  = self.cart3d.CheckCase(i)
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
            os.chdir(self.cart3d.RootDir)
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
            # Figure width in pixels (can be ``None``).
            wfig = opts.get_SubfigOpt(sfig, "FigWidth")
            # Layout file without extension
            fname = ".".join(flay.split(".")[:-1])
            # Figure file name.
            fname = "%s-%s.png" % (sfig, fname)
            # Check for the files.
            if (os.path.isfile('Components.i.plt') and 
                    os.path.isfile('cutPlanes.plt')):
                # Run Tecplot
                try:
                    # Copy the file into the current folder.
                    shutil.copy(fsrc, '.')
                    # Run the layout.
                    ExportLayout(flay, fname=fname, w=wfig)
                    # Move the file.
                    os.rename(fname, os.path.join(fpwd,fname))
                    # Include the graphics.
                    lines.append(
                        '\\includegraphics[width=\\textwidth]{%s/%s}\n'
                        % (frun, fname))
                    # Remove the layout file.
                    os.remove(flay)
                except Exception:
                    pass
        # Go to the report case folder
        os.chdir(fpwd)
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
        opts = self.cart3d.opts
        # Current status
        nIter  = self.cart3d.CheckCase(i)
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
            # Get the statistics.
            S = A.GetStats(nStats=nStats, nMax=nMax, nLast=nCur)
        else:
            # No stats.
            S = {}
        # Go back to original folder.
        os.chdir(fpwd)
        # Get the vertical alignment.
        hv = self.cart3d.opts.get_SubfigOpt(sfig, 'Position')
        # Get subfigure width
        wsfig = self.cart3d.opts.get_SubfigOpt(sfig, 'Width')
        # First line.
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header.
        fhdr = self.cart3d.opts.get_SubfigOpt(sfig, 'Header')
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
                # Initialize line
                line = ''
                # Loop through components.
                for comp in comps:
                    # Check for iterations.
                    if nCur <= 0:
                        # No iterations
                        line += '& $-$ '
                    elif fs == 'mu':
                        # Process value.
                        line += ('& $%.4f$ ' % S[comp][c])
                    elif (fs in ['min', 'max']) or (S[comp]['nStats'] > 1):
                        # Process min/max or statistical value
                        line += ('& $%.4f$ ' % S[comp][c+'_'+fs])
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
        """Read JSON file to determine what the current iteration for the report is
        
        The status is read from the present working directory
        
        :Call:
            >>> n, sts = R.ReadCaseJSONIter()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
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
        
    # Set the iteration of the JSON file.
    def SetCaseJSONIter(self, n, sts):
        """Mark the JSON file to say that the figure is up to date
        
        :Call:
            >>> R.SetCaseJSONIter(n, sts)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
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
        
    # Function to create the skeleton for a master LaTeX file
    def WriteSkeleton(self):
        """Create and write preamble for master LaTeX file for report
        
        :Call:
            >>> R.WriteSkeleton()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Go to the report folder.
        fpwd = os.getcwd()
        os.chdir(self.cart3d.RootDir)
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
        f.write('\\usepackage[usenames]{xcolor}\n')
        f.write('\\usepackage[T1]{fontenc}\n')
        f.write('\\usepackage[scaled]{beramono}\n\n')
        
        # Set the title and author.
        f.write('\\title{%s}\n' % self.cart3d.opts.get_ReportTitle(self.rep))
        f.write('\\author{%s}\n' % self.cart3d.opts.get_ReportAuthor(self.rep))
        
        # Format the header and footer
        f.write('\n\\fancypagestyle{pycart}{%\n')
        f.write(' \\renewcommand{\\headrulewidth}{0.4pt}%\n')
        f.write(' \\renewcommand{\\footrulewidth}{0.4pt}%\n')
        f.write(' \\fancyhf{}%\n')
        f.write(' \\fancyfoot[C]{\\textbf{\\textsf{%s}}}%%\n'
            % self.cart3d.opts.get_ReportRestriction(self.rep))
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
        f.write('\\newcommand{\\setcase}[1]{\\renewcommand{\\thecase}{#1}}\n')
        
        # Actual document
        f.write('\n%$__Begin\n')
        f.write('\\begin{document}\n')
        f.write('\\pagestyle{plain}\n')
        f.write('\\maketitle\n\n')
        f.write('\\pagestyle{pycart}\n\n')
        
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
            >>> R.WriteCaseSkeleton(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Get the name of the case
        frun = self.cart3d.x.GetFullFolderNames(i)
        
        # Create the file (delete if necessary)
        f = open(self.fname, 'w')
        
        # Write the header.
        f.write('\n\\newpage\n')
        f.write('\\setcase{%s}\n' % frun.replace('_', '\\_'))
        f.write('\\phantomsection\n')
        f.write('\\addcontentsline{toc}{section}{\\texttt{\\thecase}}\n')
        f.write('\\fancyhead[L]{\\texttt{\\thecase}}\n\n')
        
        # Empty section for the figures
        f.write('%$__Figures\n')
        f.write('\n')
        
        # Close the file.
        f.close()
        
    # Function to set the upper-right header
    def SetHeaderStatus(self, i):
        """Set header to state iteration progress and summary status
        
        :Call:
            >>> R.WriteCaseSkeleton(i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *i*: :class:`int`
                Case index
        :Versions:
            * 2014-03-08 ``@ddalle``: First version
        """
        # Get case current iteration
        n = self.cart3d.CheckCase(i)
        # Get case number of required iterations
        nMax = self.cart3d.GetLastIter(i)
        # Get status
        sts = self.cart3d.CheckCaseStatus(i)
        # Form iteration string
        if n is None:
            # Unknown.
            fitr = '-/%s' % self.cart3d.opts.get_IterSeq(-1)
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
            
        
    
        
    # Function to go into a folder, respecting archive option
    def cd(self, fdir):
        """Interface to :func:`os.chdir`, respecting "Archive" option
        
        This function can only change one directory at a time.
        
        :Call:
            >>> R.cd(fdir)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *fdir*: :class:`str`
                Name of directory to change to
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get archive option.
        q = self.cart3d.opts.get_ReportArchive()
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
            # Check archive option.
            if q:
                # Untar if necessary
                tar.chdir_in(fdir)
            else:
                # Go into the folder
                os.chdir(fdir)
        
        
