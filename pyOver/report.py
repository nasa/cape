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
    def SubfigSwitch(self, sfig, i, lines, q):
        """Switch function to find the correct subfigure function
        
        This function may need to be defined for each CFD solver
        
        :Call:
            >>> lines = R.SubfigSwitch(sfig, i, lines, q)
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *lines*: :class:`list` (:class:`str`)
                List of lines already in LaTeX file
            *q*: ``True`` | ``False``
                Whether or not to redraw images
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
            lines += self.SubfigConditions(sfig, i, q)
        elif btyp == 'Summary':
            # Get the force and/or moment summary
            lines += self.SubfigSummary(sfig, i, q)
        elif btyp == 'PointSensorTable':
            # Get the point sensor table summary
            lines += self.SubfigPointSensorTable(sfig, i, q)
        elif btyp == 'PlotCoeff':
            # Get the force or moment history plot
            lines += self.SubfigPlotCoeff(sfig, i, q)
        elif btyp == 'PlotLineLoad':
            # Get the sectional loads plot
            lines += self.SubfigPlotLineLoad(sfig, i, q)
        elif btyp == 'PlotPoint':
            # Get the point sensor history plot
            lines += self.SubfigPlotPoint(sfig, i, q)
        elif btyp == 'PlotL2':
            # Get the residual plot
            lines += self.SubfigPlotL2(sfig, i, q)
        elif btyp == 'PlotInf':
            # Get the residual plot
            lines += self.SubfigPlotLInf(sfig, i, q)
        elif btyp == 'PlotResid':
            # Plot generic residual
            lines += self.SubfigPlotResid(sfig, i, q)
        elif btyp == 'Tecplot':
            # Get the Tecplot layout view
            lines += self.SubfigTecplotLayout(sfig, i, q)
        else:
            # No type found
            print("  %s: No function found for base type '%s'" % (sfig, btyp))
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
        
    # Read a Tecplot script
    def ReadTecscript(self, fsrc):
        """Read a Tecplot script interface
        
        :Call:
            >>> R.ReadTecscript(fsrc)
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
            *fscr*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-10-25 ``@ddalle``: First version
        """
        return Tecscript(fsrc)
            
    # Function to link appropriate visualization files
    def LinkVizFiles(self):
        """Create links to appropriate visualization files
        
        Specifically, ``Components.i.plt`` and ``cutPlanes.plt`` or
        ``Components.i.dat`` and ``cutPlanes.dat`` are created.
        
        :Call:
            >>> R.LinkVizFiles()
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
        :See Also:
            :func:`pyCart.case.LinkPLT`
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Defer to function from :func:`pyOver.case`
        LinkX()
        LinkQ()
        
# class Report

