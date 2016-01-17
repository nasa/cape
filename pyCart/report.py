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
from .dataBook import CaseFM, CaseResid
# Folder and Tecplot management
from .case    import LinkPLT
from .tecplot import ExportLayout, Tecscript
# Configuration and surface tri
from .tri    import Tri
from .config import Config

# Dedicated function to load pointSensor only when needed.
def ImportPointSensor():
    """Import :mod:`pyCart.pointSensor` if not loaded
    
    :Call:
        >>> pyCart.report.ImportPointSensor()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Make global variables
    global pointSensor
    # Check for PyPlot.
    try:
        pointSensor
    except Exception:
        # Load the modules.
        import pointSensor


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
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return '<pyCart.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
    
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
        return CaseFM(comp)
        
    # Read residual history
    def ReadCaseResid(self):
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
        return CaseResid()
        
    # Read point sensor history]
    def ReadPointSensor(self):
        """Read iterative history for a case
        
        :Call:
            >>> P = R.ReadPointSensor()
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
        :Outputs:
            *P*: :class:`pyCart.pointSensor.CasePointSensor`
                Iterative history of point sensors
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Make sure the modules are present.
        ImportPointSensor()
        # Read point sensors history
        return pointSensor.CasePointSensor()
        
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
            elif btyp == 'PointSensorTable':
                # Get the point sensor table summary
                lines += self.SubfigPointSensorTable(sfig, i)
            elif btyp == 'PlotCoeff':
                # Get the force or moment history plot
                lines += self.SubfigPlotCoeff(sfig, i)
            elif btyp == 'PlotPoint':
                # Get the poitn sensor history plot
                lines += self.SubfigPlotPoint(sfig, i)
            elif btyp == 'PlotL1':
                # Get the residual plot
                lines += self.SubfigPlotL1(sfig, i)
            elif btyp == 'PlotResid':
                # Plot generic residual
                lines += self.SubfigPlotResid(sfig, i)
            elif btyp == 'Tecplot3View':
                # Get the Tecplot component view
                lines += self.SubfigTecplot3View(sfig, i)
            elif btyp == 'Tecplot':
                # Get the Tecplot layout view
                lines += self.SubfigTecplotLayout(sfig, i)
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
            elif btyp == 'SweepPointHist':
                # Plot a point sensor histogram
                lines += self.SubfigSweepPointHist(sfig, fswp, I)
        # Output
        return lines
        
        
    # Function to create coefficient plot and write figure
    def SubfigTecplot3View(self, sfig, i):
        """Create image of surface for one component using Tecplot
        
        :Call:
            >>> lines = R.SubfigTecplot3View(sfig, i)
        :Inputs:
            *R*: :class:`cape.report.Report`
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
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Get the component.
        comp = opts.get_SubfigOpt(sfig, "Component")
        # Go to the Cart3D folder
        os.chdir(self.cntl.RootDir)
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
            self.cntl.ReadTri()
            # Make sure to start from initial position.
            self.cntl.tri = self.cart3d.tri0.Copy()
            # Rotate for the appropriate case.
            self.cntl.PrepareTri(i)
            # Extract the triangulation
            tri = self.cntl.tri
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
        
    # Function to plot point sensor history
    def SubfigPlotPoint(self, sfig, i):
        """Plot iterative history of a point sensor state
        
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
            * 2015-12-07 ``@ddalle``: First version
        """
        # Save current folder
        fpwd = os.getcwd()
        # Case folder
        frun = self.cntl.x.GetFullFolderNames(i)
        # Extract options
        opts = self.cntl.opts
        # Extract the point
        pt = opts.get_SubfigOpt(sfig, "Point")
        # Get the group
        grp = opts.get_SubfigOpt(sfig, "Group")
        # Get the state
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # List of coefficients
        if type(coeff).__name__ in ['list', 'ndarray']:
            # List of coefficients
            nCoeff = len(coeff)
        else:
            # One entry
            nCoeff = 1
        # Check for list of points
        if type(pt).__name__ in ['list', 'ndarray']:
            # List of components
            nCoeff = max(nCoeff, len(pt))
        # Current status
        nIter  = self.cntl.CheckCase(i)
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption
        if fcpt is None:
            # Check for a list
            if type(pt).__name__ in ['list']:
                # Join them, e.g. "[P1,P2]/Cp"
                fcpt = "[" + ",",join(pt) + "]"
            else:
                # Use the point name
                fcpt = pt
            # Add the coefficient title
            fcpt = "%s/%s" % (fcpt, coeff)
            # Ensure there are not underscores
            fcpt = fcpt.replace('_', '\_')
        # Get the vertical alignment.
        hv = opts.get_SubfigOpt(sfig, "Position")
        # Get subfigure width
        wsfig = opts.get_SubfigOpt(sfig, "Width")
        # First line
        lines = ['\\begin{subfigure}[%s]{%.2f\\textwidth}\n' % (hv, wsfig)]
        # Check for a header
        fhdr = opts.get_SubfigOpt(sfig, "Header")
        # Alginment
        algn = opts.get_SubfigOpt(sfig, "Alignment")
        # Set alignment
        if algn.lower() == "center":
            lines.append('\\centering\n')
        # Write the header
        if fhdr:
            lines.append('\\textbf{\\textit{%s}}\\par\n' % fhdr)
            lines.append('\\vskip-6pt\n')
        # Go to the run directory.
        os.chdir(self.cntl.RootDir)
        os.chdir(frun)
        # Read the Aero history.
        P = self.ReadPointSensor()
        # Loop through plots
        for k in range(nCoeff):
            # Get the point and coefficient
            pt    = opts.get_SubfigOpt(sfig, "Point", k)
            coeff = opts.get_SubfigOpt(sfig, "Coefficient", k)
            # Numbers of iterations
            nStats = opts.get_SubfigOpt(sfig, "nStats",     k)
            nMin   = opts.get_SubfigOpt(sfig, "nMinStats",  k)
            nMax   = opts.get_SubfigOpt(sfig, "nMaxStats",  k)
            nLast  = opts.get_SubfigOpt(sfig, "nLastStats", k)
            # Default to databook options
            if nStats is None: nStats = opts.get_nStats(grp)
            if nMin   is None: nMin   = opts.get_nMin(grp)
            if nMax   is None: nMax   = opts.get_nMaxStats(grp)
            if nLast  is None: nLast  = opts.get_nLastStats(grp)
            # Numbers of iterations for plots
            nPlotIter  = opts.get_SubfigOpt(sfig, "nPlot",      k)
            nPlotFirst = opts.get_SubfigOpt(sfig, "nPlotFirst", k)
            nPlotLast  = opts.get_SubfigOpt(sfig, "nPlotLast",  k)
            
            # Check if there are iterations.
            if nIter < 2: continue
            # Don't use iterations before *nMin*
            nMax = min(nMax, nIter-nMin)
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
            # Label formatting
            f_s = opts.get_SubfigOpt(sfig, "SigmaFormat", k)
            f_e = opts.get_SubfigOpt(sfig, "EpsilonFormat", k)
            # Draw the plot.
            h = P.PlotState(coeff, pt, n=nPlotIter, nAvg=nStats,
                nFirst=nPlotFirst, nLast=nPlotLast,
                LineOptions=kw_p, MeanOptions=kw_m,
                d=dc, DeltaOptions=kw_d,
                k=ksig, StDevOptions=kw_s,
                u=uerr, ErrPltOptions=kw_u,
                ShowMu=sh_m, ShowDelta=sh_d,
                ShowSigma=sh_s, SigmaFormat=f_s,
                ShowEspsilon=sh_e, EpsilonFormat=f_e,
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
            LinkPLT()
            # Layout file
            flay = opts.get_SubfigOpt(sfig, "Layout")
            # Full path to layout file
            fsrc = os.path.join(self.cntl.RootDir, flay)
            # Get just the file name
            flay = os.path.split(flay)[-1]
            # Read the Mach number option
            omach = opts.get_SubfigOpt(sfig, "Mach")
            # Read the Tecplot layout
            tec = Tecscript(fsrc)
            # Set the Mach number
            try:
                tec.SetMach(getattr(self.cntl.x,omach)[i])
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
    # Function to write pressure coefficient table
    def SubfigPointSensorTable(self, sfig, i):
        """Create lines for a "PointSensorTable" subfigure
        
        :Call:
            >>> lines = R.SubfigPointTable(sfig, i)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-12-07 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options
        opts = self.cntl.opts
        # Current statuts
        nIter = self.cntl.CheckCase(i)
        # Coefficients to plot
        coeffs = opts.get_SubfigOpt(sfig, "Coefficients")
        # Get the targets
        targs = opts.get_SubfigOpt(sfig, "Targets")
        # Target data and target options
        DBT = {}
        topts = {}
        # Read those targets into the data book, if necessary
        for targ in targs:
            # Read the target data book
            self.cntl.ReadDataBook()
            self.cntl.DataBook.ReadTarget(targ)
            # Append it to the list
            DBT[targ] = self.cntl.DataBook.GetTargetByName(targ)
            # Check for coefficient lists
            tcoeffs = opts.get_SubfigOpt(sfig, "%s.Coefficients"%targ)
            # Default to main list
            if tcoeffs is None: tcoeffs = coeffs
            # Initialize the options for this target
            topts[targ] = {"Coefficients": tcoeffs}
            # Get index of first matching point
            J = DBT[targ].FindMatch(self.cntl.x, i)
            # Check for matches
            if len(J) == 0:
                # No match
                topts[targ]["Index"] = None
            else:
                # Take the first match
                topts[targ]["Index"] = J
            # Loop through the coefficients
            for coeff in tcoeffs:
                # Get the list of statistical fields for DB and target
                fsd = opts.get_SubfigOpt(sfig, coeff)
                fst = opts.get_SubfigOpt(sfig, '%s.%s' % (targ,coeff))
                # Save the settings
                if fst is None:
                    # Use defaults
                    topts[targ][coeff] = fsd
                else:
                    # Use specific settings
                    topts[targ][coeff] = fst
        # Get the points.
        grp = opts.get_SubfigOpt(sfig, "Group")
        pts = opts.get_SubfigOpt(sfig, "Points")
        # Process point list if a group is specified
        if grp: pts = opts.get_DBGroupPoints(grp)
        # Numbers of iterations for statistics
        nStats = opts.get_SubfigOpt(sfig, "nStats")
        nMin   = opts.get_SubfigOpt(sfig, "nMinStats")
        # Get the status and data book options
        if nStats is None: nStats = opts.get_nStats(grp)
        if nMin   is None: nMin   = opts.get_nMin(grp)
        # Iteration at which to build table
        nOpt = opts.get_SubfigOpt(sfig, "Iteration")
        # Make sure current progress is a number
        if nIter is None: nIter = 0
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
            # Go to the run directory.
            os.chdir(self.cntl.RootDir)
            os.chdir(self.cntl.x.GetFullFolderNames(i))
            # Read the point sensor history
            P = self.ReadPointSensor()
            # Loop through components
            for pt in pts:
                # Process point index
                if type(pt).__name__.startswith('int'):
                    # Integer specification (dangerous)
                    kpt = pt
                else:
                    # Specified by name
                    kpt = P.GetPointSensorIndex(pt)
                # Check for adequate iterations
                if np.sum(P.i>=nMin) < nStats: continue
                # Get the statistics.
                S[pt] = P.GetStats(kpt, nStats=nStats, nLast=nCur)
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
            line += (', {\\small\\texttt{nStats=%i}}' % nStats)
        # Close parentheses
        line += ')\\par\n'
        # Write the header.
        lines.append('\\noindent\n')
        lines.append(line)
        # Begin the table with the right amount of columns.
        line = '\\begin{tabular}{l'
        for coeff in coeffs:
            # Get the statsitical values to print
            fs = opts.get_SubfigOpt(sfig, coeff)
            # Column count
            nf = len(fs)
        # Loop through targets
        for targ in targs:
            # List of coefficients
            for coeff in topts[targ]["Coefficients"]:
                # Get list of target statistical fields for this coeff
                nf += len(topts[targ][coeff])
        # Append that many centered columns.
        line += ('|c'*nf)
        # Write the line to begin the table
        lines.append(line + '}\n')
        # Write headers.
        lines.append('\\hline \\hline\n')
        lines.append('\\textbf{\\textsf{Point}}\n')
        # Write headers
        for coeff in coeffs:
            # Loop through suffixes for this coefficient
            for fs in opts.get_SubfigOpt(sfig, coeff):
                # Get the symbol
                sym = self.GetStateSymbol(coeff, fs)
                # Append line line
                lines.append(' & ' + sym + ' \n')
        # Write headers for each target
        for targ in targs:
            # Get name of target with underscores removed
            ltarg = targ.replace('_', '\_')
            # Check coefficients for this target
            for coeff in topts[targ]["Coefficients"]:
                # Loop through suffixes for this coefficient
                for fs in topts[targ][coeff]:
                    # Get the symbol
                    sym = self.GetStateSymbol(coeff, fs)
                    # Append line line
                    lines.append(' & ' + ltarg + '/' + sym + ' \n')
        # End header
        lines.append('\\\\\n')
        lines.append('\hline\n')
        # Loop through points.
        for pt in pts:
            # Write point name.
            lines.append('\\texttt{%s}\n' % pt.replace('_', '\_'))
            # Initialize line
            line = ''
            # Loop through the coefficients.
            for coeff in coeffs:
                # Loop through statistics
                for fs in opts.get_SubfigOpt(sfig, coeff):
                    # Process the statistic type
                    if nCur <= 0 or pt not in S:
                        # No iterations
                        line += '& $-$ '
                    elif fs == 'mu':
                        # Mean value
                        line += ('& $%.4f$ ' % S[pt][coeff])
                    elif fs == 'std':
                        # Standard deviation
                        line += ('& %.2e ' % S[pt][coeff+'_'+fs])
                    elif (coeff+'_'+fs) in S[pt]:
                        # Other statistic
                        line += ('& $%.4f$ ' % S[pt][coeff+'_'+fs])
                    else:
                        # Missing
                        line += '& $-$ '
            # Loop through targets
            for targ in targs:
                # Loop through the coefficients.
                for coeff in topts[targ]["Coefficients"]:
                    # Loop through statistics
                    for fs in topts[targ][coeff]:
                        # Name of field
                        c = ('%s.%s_%s' % (pt, coeff, fs)).rstrip('_mu')
                        # Column names
                        ckeys = DBT[targ].ckeys
                        # Name of column
                        if grp not in ckeys or c not in ckeys[grp]:
                            # No match for this coefficient
                            line += '& $-$ '
                            continue
                        else:
                            # Get the column
                            col = ckeys[grp][c]
                        # index
                        J = topts[targ]["Index"]
                        # Process the statistic type
                        if J is None or col not in DBT[targ]:
                            # No iterations
                            line += '& $-$ '
                        elif fs == 'mu':
                            # Mean value
                            val = np.mean(DBT[targ][col][J])
                            line += ('& $%.4f$ ' % val)
                        elif fs == 'std':
                            # Standard deviation
                            val = np.mean(DBT[targ][col][J])
                            line += ('& %.2e ' % val)
                        else:
                            # Other statistic
                            val = np.mean(DBT[targ][col][J])
                            line += ('& $%.4f$ ' % val)
            # Finish the line and append it.
            line += '\\\\\n'
            lines.append(line)
        # Finish table and subfigure
        lines.append('\\hline \\hline\n')
        lines.append('\\end{tabular}\n')
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Function to plot mean coefficient for a sweep
    def SubfigSweepPointHist(self, sfig, fswp, I):
        """Plot a histogram of a point sensor coefficient over several cases
        
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
            * 2016-01-16 ``@ddalle``: First version
        """
        # Save current folder.
        fpwd = os.getcwd()
        # Extract options and trajectory
        x = self.cntl.DataBook.x
        opts = self.cntl.opts
        # Case folder
        frun = x.GetFullFolderNames(I[0])
        # Get the component.
        grp = opts.get_SubfigOpt(sfig, "Group")
        # Get the point
        pt = opts.get_SubfigOpt(sfig, "Point")
        # Get the coefficient
        coeff = opts.get_SubfigOpt(sfig, "Coefficient")
        # Get list of targets
        targs = self.SubfigTargets(sfig)
        # Read the point sensor group data book
        self.cntl.DataBook.ReadPointSensor(grp)
        # Get the point sensor
        DBP = self.cntl.DataBook.PointSensors[grp][pt]
        # Get the targets
        targs = self.SubfigTargets(sfig)
        # Target labels
        ltarg = opts.get_SubfigOpt(sfig, "TargetLabel")
        # Ensure list
        if type(ltarg).__name__ != 'list':
            ltarg = [ltarg]
        # Initialize target values
        vtarg = []
        # Number of targets
        ntarg = len(targs)
        # Loop through targets.
        for i in range(ntarg):
            # Select the target
            targ = targs[i]
            # Read the target if not present.
            self.cntl.DataBook.ReadTarget(targ)
            # Get the data book target
            DBT = self.cntl.DataBook.GetTargetByName(targ)
            # Get the target co-sweep
            jt = self.GetTargetSweepIndices(fswp, I[0], targ)
            # Get the proper coefficient
            ckey = '%s.%s' % (pt, coeff)
            # Check for a match in the target
            if grp in DBT.ckeys and ckey in DBT.ckeys[grp]:
                # Get the name of the column in the target
                col = DBT.ckeys[grp][ckey]
                # Save the value.
                vtarg.append(np.mean(DBT[col][I]))
            else:
                # No value!
                vtarg.append(None)
            # Default label
            if len(ltarg) < i or ltarg[i] in [None,False]:
                ltarg[i] = targ
        # Distribution variable
        xk = opts.get_SweepOpt(fswp, "XAxis")
        # Get caption
        fcpt = opts.get_SubfigOpt(sfig, "Caption")
        # Process default caption
        if fcpt is None:
            # Use the point name and the coefficient
            fcpt = "%s/%s" % (pt, coeff)
        # Ensure that there are not underscores.
        fcpt = fcpt.replace("_", "\_")
        # Initialize subfigure
        lines = self.SubfigInit(sfig)
        # Form options for point sensor histogram
        kw_h = {
            # Reference values
            "StDev":          opts.get_SubfigOpt(sfig, "StandardDeviation"),
            "Delta":          opts.get_SubfigOpt(sfig, "Delta"),
            "PlotMu":         opts.get_SubfigOpt(sfig, "PlotMu"),
            # Target information
            "TargetValue":    vtarg,
            "TargetLabel":    ltarg,
            # Figure dimensions
            "FigureWidth":    opts.get_SubfigOpt(sfig, "FigureWidth"),
            "FigureHeight":   opts.get_SubfigOpt(sfig, "FigureHeight"),
            # Text labels of reference values
            "ShowMu":         opts.get_SubfigOpt(sfig, "ShowMu"),
            "ShowSigma":      opts.get_SubfigOpt(sfig, "ShowSigma"),
            "ShowDelta":      opts.get_SubfigOpt(sfig, "ShowDelta"),
            "ShowTarget":     opts.get_SubfigOpt(sfig, "ShowTarget"),
            # Format flags
            "MuFormat":       opts.get_SubfigOpt(sfig, "MuFormat"),
            "SigmaFormat":    opts.get_SubfigOpt(sfig, "SigmaFormat"),
            "DeltaFormat":    opts.get_SubfigOpt(sfig, "DeltaFormat"),
            "TargetFormat":   opts.get_SubfigOpt(sfig, "TargetFormat"),
            # Plot options
            "HistOptions":    opts.get_SubfigOpt(sfig, "HistOptions"),
            "MeanOptions":    opts.get_SubfigOpt(sfig, "MeanOptions"),
            "StDevOptions":   opts.get_SubfigOpt(sfig, "StDevOptions"),
            "DeltaOptions":   opts.get_SubfigOpt(sfig, "DeltaOptions"),
            "TargetOptions":  opts.get_SubfigOpt(sfig, "TargetOptions")
        }
        # Plot the histogram with labels
        h = DBP.PlotValueHist(coeff, I, **kw_h)
        # Change back to report folder
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
            # Save with resolution
            h['fig'].savefig(fimg, dpi=dpi)
            h['fig'].savefig(fpdf)
        # Close the figure
        h['fig'].clf()
        # Include the graphics
        lines.append('\\includegraphics[width=\\textwidth]{sweep-%s/%s/%s}\n'
            % (fswp, frun, fpdf))
        # Set the caption.
        lines.append('\\caption*{\\scriptsize %s}\n' % fcpt)
        # Close the subfigure
        lines.append('\\end{subfigure}\n')
        # Output
        return lines
        
    # Process state symbol
    def GetStateSymbol(self, coeff, fs):
        """Get a TeX symbol for a coefficient and statistical field
        
        :Call:
            >>> sym = R.GetStateSymbol(coeff, fs)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report instance
            *coeff*: Cp | dp | rho | M | T | p
                Name of coefficient
            *fs*: mu | std | err | min | max | u
                Statistical quantity
        :Outputs:
            *sym*: :class:`str`
                TeX symbol (including ``$`` chars) for this field
        :Versions:
            * 2015-12-18 ``@ddalle``: First version
        """
        # Process label
        if coeff == 'Cp':
            # Pressure coefficient
            lbl = "C_p"
        elif coeff == 'dp':
            # Delta pressure
            lbl = "(p-p_\infty)/p_\infty"
        elif coeff == 'rho':
            # Static density
            lbl = '\rho/\rho_\infty'
        elif coeff == 'U':
            # x-velocity
            lbl = 'u/a_\infty'
        elif coeff == 'V':
            # y-velocity
            lbl = 'v/a_\infty'
        elif coeff == 'W':
            # z-velocity
            lbl = 'w/a_\infty'
        elif coeff == 'P':
            # weird pressure
            lbl = 'p/\gamma p_\infty'
        else:
            # Something else?
            lbl = coeff
        # Check suffix type
        if fs in ['std', 'sigma']:
            # Standard deviation
            sym = '$\sigma(%s)$' % lbl
        elif fs in ['err', 'eps', 'epsilon']:
            # Sampling error
            sym = '$\varepsilon(%s)$' % lbl
        elif fs == 'max':
            # Maximum value
            sym = 'max$(%s)$' % lbl
        elif fs == 'min':
            # Minimum value
            sym = 'min$(%s)$' % lbl
        elif fs == 'u':
            # Uncertainty
            sym = '$U(%s)$' % lbl
        else:
            # Mean
            sym = '$%s$' % lbl
        # Output
        return sym
# class Report

        
