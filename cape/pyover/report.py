"""
:mod:`pyOver.report`: Automated report interface
=================================================

The pyOver module for generating automated results reports using PDFLaTeX
provides a single class :class:`pyOver.report.Report`, which is based off the
CAPE version :class:`cape.report.Report`. The :class:`cape.report.Report` class
is a sort of dual-purpose object that contains a file interface using
:class:`cape.tex.Tex` combined with a capability to create figures for each
case or sweep of cases mostly based on :mod:`cape.dataBook`.

An automated report is a multi-page PDF generated using PDFLaTeX. Usually, each
CFD case has one or more pages dedicated to results for that case. The user
defines a list of figures, each with its own list of subfigures, and these are
generated for each case in the run matrix (subject to any command-line
constraints the user may specify). Types of subfigures include

    * Table of the values of the input variables for this case
    * Table of force and moment values and statistics
    * Iterative histories of forces or moments (for one or more coefficients)
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more
    
In addition, the user may also define "sweeps," which analyze groups of cases
defined by user-specified constraints. For example, a sweep may be used to plot
results as a function of Mach number for sets of cases having matching angle of
attack and sideslip angle. When using a sweep, the report contains one or more
pages for each sweep (rather than one or more pages for each case).

Reports are usually created using system commands of the following format.

    .. code-block: console
    
        $ pyover --report

The class has an immense number of methods, which can be somewhat grouped into
bookkeeping methods and plotting methods.  The main user-facing methods are
:func:`cape.report.Report.UpdateCases` and
:func:`cape.report.Report.UpdateSweep`.  Each 
:ref:`type of subfigure <cape-json-ReportSubfigure>` has its own method, for
example :func:`cape.report.Report.SubfigPlotCoeff` for ``"PlotCoeff"``  or
:func:`cape.report.Report.SubfigPlotL2` for ``"PlotL2"``.

:See also:
    * :mod:`cape.report`
    * :mod:`pyOver.options.Report`
    * :mod:`cape.options.Report`
    * :class:`cape.dataBook.DBComp`
    * :class:`cape.dataBook.CaseFM`
    * :class:`cape.lineLoad.DBLineLoad`
    
"""

# File system interface
import os, json, shutil, glob
# Numerics
import numpy as np

# Import basis module
import cape.report

# Local modules needed
from cape import tex, tar
# Folder/run management
from . import case
# DataBook and Tecplot management
from .dataBook import CaseFM, CaseResid
from .tecplot  import ExportLayout, Tecscript


# Class to interface with report generation and updating.
class Report(cape.report.Report):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyOver.report.Report(oflow, rep)
    :Inputs:
        *oflow*: :class:`cape.pyover.cntl.Cntl`
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
            * 2017-03-27 ``@ddalle``: Added *CompID* option
        """
        # Project rootname
        proj = self.cntl.GetPrefix()
        # Get component (note this automatically defaults to *comp*)
        compID = self.cntl.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if type(compID).__name__ in ['list', 'ndarray']:
            # Read the first component
            FM = CaseFM(proj, compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-1'):
                    # Subtract the component
                    FM -= CaseFM(proj, compi.lstrip('-'))
                else:
                    # Add in the component
                    FM += CaseFM(proj, compi)
        else:
            # Read the iterative history for single component
            FM = CaseFM(proj, compID)
        # Read the history for that component
        return FM
            
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
        elif btyp == 'Image':
            # Coy an image
            lines += self.SubfigImage(sfig, i, q)
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
        grid = self.cntl.opts.get_SubfigOpt(sfig, "Grid")
        # Check type of residual to read
        if t in ['PlotL2']:
            # Read spatial L2-norm
            R.ReadGlobalL2(grid)
        elif t in ['PlotLInf']:
            # Read spatial L-infinity norm
            # Read global residual
            R.ReadGlobalLInf(grid)
        elif t in ['PlotResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Read spatial L2-norm
                R.ReadGlobalL2(grid)
            elif resid in ['LInf']:
                # Read global L-infinity residual
                R.ReadGlobalLInf(grid)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotTurbResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Read spatial
                R.ReadTurbResidL2(grid)
            elif resid in ['LInf']:
                # Read global L-infinity residual
                R.ReadTurbResidLInf(grid)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotSpeciesResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Read global residual
                R.ReadSpeciesResidL2(grid)
            elif resid in ['LInf']:
                # Read global L-infinity residual
                R.ReadSpeciesResidLInf(grid)
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
    def LinkVizFiles(self, sfig=None, i=None):
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
        case.LinkX()
        case.LinkQ()
        # Extract Options
        opts = self.cntl.opts
        # Check for splitmq options
        fsplitmq = opts.get_SubfigOpt(sfig, "SplitmqI")
        # Exit if None
        if fsplitmq is None:
            # Try another name
            fsplitmq = opts.get_SubfigOpt(sfig, "splitmq")
        # Exit if none
        if fsplitmq is None:
            return
        # Path to the file
        frun = self.cntl.x.GetFullFolderNames(i)
        # Assemble absolute path
        if os.path.isabs(fsplitmq):
            # Already absolute path
            fabs = fsplitmq
        else:
            # Append root directory
            fabs = os.path.join(self.cntl.RootDir, fsplitmq)
        # Get the file name alone
        fname = os.path.split(fabs)[-1]
        # Check if the file exists
        if not os.path.isfile(fabs): return
        # Check which ``q`` and ``x`` files to use as input and output
        fqi = opts.get_SubfigOpt(sfig, "QIn")
        fqo = opts.get_SubfigOpt(sfig, "QOut")
        fxi = opts.get_SubfigOpt(sfig, "Xin")
        fxo = opts.get_SubfigOpt(sfig, "XOut")
        # Defaults
        if fqi is None: fqi = "q.pyover.p3d"
        if fqo is None: fqo = "q.pyover.srf"
        if fxi is None: fxi = "x.pyover.p3d"
        if fxo is None: fxo = "x.pyover.srf"
        # Check if the surf file exists
        if not os.path.isfile(fqi):
            # No input file to use
            qq = False
        elif os.path.isfile(fqo):
            # If both files exist, check iteration nubers
            nqi = case.checkqt(fqi)
            try:
                nqo = case.checkqt(fqo)
            except Exception:
                nqo = 0
            # Check if surface file is already up-to-date
            if nqi <= nqo:
                # Up-to-date
                qq = False
            else:
                # Old q.srf file
                os.remove(fqo)
                qq = True
        else:
            # No q.srf file
            qq = True
        # Check if the surf grid file exists
        if not os.path.isfile(fxi):
            # No input file
            qx = False
        elif os.path.isfile(fxo):
            # Update *x.srf* only if updating *q.srf*
            if qq:
                # Delete the link
                os.remove(fxo)
                qx = True
            else:
                # Up-to-date files
                qx = False
        else:
            # No x.srf file
            qx = True
        # Exit if nothing to do
        if not (qx or qq): return
        # Name for the file here
        fspq = 'splitmq.%s.i' % sfig
        fspx = 'splitmx.%s.i' % sfig
        fspqo = 'splitmq.%s.o' % sfig
        fspxo = 'splitmx.%s.o' % sfig
        # Copy the file to this directory
        shutil.copy(fabs, fname)
        # Update or create q.srf if necessary
        if qq:
            # Edit the splitmq files
            case.EditSplitmqI(fname, fspq, fqi, fqo)
            # Split the solution
            cmd = 'splitmq < %s > %s' % (fspq, fspqo)
            print("    %s" % cmd)
            ierr = os.system(cmd)
            # Check for errors
            if ierr: return
            # Delete files
            os.remove(fspq)
            os.remove(fspqo)
        # Update or create x.srf if necessary
        if qx:
            # Edit the splitmx file
            case.EditSplitmqI(fname, fspx, fxi, fxo)
            # Split the surface grid
            cmd = 'splitmx < %s > %s' % (fspx, fspxo)
            print("    %s" % cmd)
            ierr = os.system(cmd)
            # Check for errors
            if ierr: return
            # Delete files
            os.remove(fspx)
            os.remove(fspxo)
        # Delete the template
        os.remove(fname)
        
# class Report

