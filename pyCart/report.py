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
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return '<pyCart.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
    
        
        
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
# class Report

        
