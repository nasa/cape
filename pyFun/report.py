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


# Class to interface with report generation and updating.
class Report(cape.report.Report):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyFun.report.Report(fun3d, rep)
    :Inputs:
        *fun3d*: :class:`pyFun.fun3d.Fun3d`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyFun.report.Report`
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
        return '<pyFun.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
            
    # Update subfig for case
    def SubfigSwitch(self, sfig, i, lines):
        """Switch function to find the correct subfigure function
        
        This function may need to be defined for each CFD solver
        
        :Call:
            >>> lines = R.SubfigSwitch(sfig, i, lines)
        :Inputs:
            *R*: :class:`pyCart.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *lines*: :class:`list` (:class:`str`)
                List of lines already in LaTeX file
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                Updated list of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
            * 2016-10-25 ``@ddalle``: *UpdateCaseSubfigs* -> *SubfigSwitch*
        """
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
        else:
            print("  %s: No function for subfigure type '%s'" % (sfig, btyp))
        # Output
        return lines
        
    # Function to create coefficient plot and write figure
    def SubfigTecplotLayout(self, sfig, i):
        """Create image based on a Tecplot layout file
        
        :Call:
            >>> lines = R.SubfigTecplotLayout(sfig, i)
        :Inputs:
            *R*: :class:`pyFun.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of sfigure to update
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-09-06 ``@ddalle``: First version
            * 2016-10-05 ``@ddalle``: Added "FieldMap" option
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
            # Read the zone numbers for each FIELDMAP command
            grps = opts.get_SubfigOpt(sfig, "FieldMap")
            # Read the Tecplot layout
            tec = Tecscript(fsrc)
            # Set the field map
            if grps is not None:
                try:
                    tec.SetFieldMap(grps)
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
        # Project rootname
        proj = self.cntl.GetProjectRootName(None)
        # Read the history for that component
        return CaseFM(proj, comp)
        
    # Read residual history
    def ReadCaseResid(self, sfig=None):
        """Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid(sfig=None)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *sfig*: :class:`str` | ``None``
                Name of subfigure to process
        :Outputs:
            *hist*: ``None`` or :class:`cape.dataBook.CaseResid` derivative
                Case iterative residual history for one case
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2016-02-04 ``@ddalle``: Added argument
        """
        # Project rootname
        proj = self.cntl.GetProjectRootName(None)
        # Read the residual history
        return CaseResid(proj)
        
        
# class Report

