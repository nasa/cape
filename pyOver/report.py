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
from .case    import LinkQ, LinkX
from .tecplot import ExportLayout, Tecscript


# Class to interface with report generation and updating.
class Report(cape.report.Report):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyOver.report.Report(oflow, rep)
    :Inputs:
        *oflow*: :class:`pyOver.overflow.Overflow`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyOver.report.Report`
            Automated report interface
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        return '<pyOver.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
    
    # Read iterative history
    def ReadCaseFM(self, comp):
        """Read iterative history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> FM = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *FM*: ``None`` or :class:`cape.dataBook.CaseFM` derivative
                Case iterative force & moment history for one component
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetPrefix()
        # Read the history for that component
        return CaseFM(proj, comp)
            
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
            elif btyp == 'PlotLineLoad':
                # Get the sectional loads plot
                lines += self.SubfigPlotLineLoad(sfig, i)
            elif btyp == 'PlotPoint':
                # Get the point sensor history plot
                lines += self.SubfigPlotPoint(sfig, i)
            elif btyp == 'PlotL2':
                # Get the residual plot
                lines += self.SubfigPlotL2(sfig, i)
            elif btyp == 'PlotInf':
                # Get the residual plot
                lines += self.SubfigPlotLInf(sfig, i)
            elif btyp == 'PlotResid':
                # Plot generic residual
                lines += self.SubfigPlotResid(sfig, i)
            elif btyp == 'Tecplot':
                # Get the Tecplot layout view
                lines += self.SubfigTecplotLayout(sfig, i)
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
            * 2016-09-06 ``@ddalle``: First version
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
            LinkX()
            LinkQ()
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
            if (os.path.isfile('x.pyover.p3d') and 
                    os.path.isfile('q.pyover.p3d')):
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
                    os.remove(flay)
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
        
    # Read residual history
    def ReadCaseResid(self, sfig=None):
        """Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid()
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
        :Outputs:
            *hist*: ``None`` or :class:`cape.dataBook.CaseResid` derivative
                Case iterative residual history for one case
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetPrefix()
        # Initialize the residual history
        R = CaseResid(proj)
        # Check type
        t = self.cntl.opts.get_SubfigBaseType(sfig)
        # Check the residual to read
        resid = self.cntl.opts.get_SubfigOpt(sfig, "Residual")
        # Check the component to read
        comp = self.cntl.opts.get_SubfigOpt(sfig, "Component")
        # Check type of residual to read
        if t in ['PlotL2']:
            # Read spatial L2-norm
            if comp is None:
                # Read global residual
                R.ReadGlobalL2()
            else:
                # Read individual residual
                R.ReadL2Grid(comp)
        elif t in ['PlotLInf']:
            # Read spatial L-infinity norm
            if comp is None:
                # Read global residual
                R.ReadGlobalLInf()
            else:
                # Read individual residual
                R.ReadLInfGrid(comp)
        elif t in ['PlotResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadGlobalL2()
                else:
                    # Not implemented
                    R.ReadL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadGlobalLInf()
                else:
                    # Read for a specific grid
                    R.ReadLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotTurbResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadTurbResidL2()
                else:
                    # Not implemented
                    R.ReadTurbL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadTurbResidLInf()
                else:
                    # Read for a specific grid
                    R.ReadTurbLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotSpeciesResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadSpeciesResidL2()
                else:
                    # Not implemented
                    R.ReadSpeciesL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadSpeciesResidLInf()
                else:
                    # Read for a specific grid
                    R.ReadSpeciesLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        # Output
        return R
        
        
# class Report

