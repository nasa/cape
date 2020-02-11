r"""
:mod:`cape.pyfun.report`: Automated report interface
======================================================

The pyFun module for generating automated results reports using 
PDFLaTeX provides a single class :class:`pyFun.report.Report`, which is
based off the CAPE version :class:`cape.cfdx.report.Report`. The :class:`cape.cfdx.report.Report` class is a sort of dual-purpose object
that contains a file interface using :class:`cape.tex.Tex` combined 
with a capability to create figures for each case or sweep of cases 
mostly based on :mod:`cape.cfdx.dataBook`.

An automated report is a multi-page PDF generated using PDFLaTeX. 
Usually, each CFD case has one or more pages dedicated to results for 
that case. The user defines a list of figures, each with its own list
of subfigures, and these are generated for each case in the run matrix
(subject to any command-line constraints the user may specify). Types 
of subfigures include

    * Table of the values of the input variables for this case
    * Table of force and moment values and statistics
    * Iterative histories of forces or moments (for one or more 
      coefficients)
    * Iterative histories of residuals
    * Images using a Tecplot layout
    * Many more
    
In addition, the user may also define "sweeps," which analyze groups of
cases defined by user-specified constraints. For example, a sweep may 
be used to plot results as a function of Mach number for sets of cases
having matching angle of attack and sideslip angle. When using a sweep,
the report contains one or more pages for each sweep (rather than one 
or more pages for each case).

Reports are usually created using system commands of the following 
format.

    .. code-block: console
    
        $ pyfun --report

The class has an immense number of methods, which can be somewhat 
grouped into bookkeeping methods and plotting methods.  The main 
user-facing methods are
:func:`cape.cfdx.report.Report.UpdateCases` and
:func:`cape.cfdx.report.Report.UpdateSweep`.  Each 
:ref:`type of subfigure <cape-json-ReportSubfigure>` has its own 
method, for example
:func:`cape.cfdx.report.Report.SubfigPlotCoeff` for ``"PlotCoeff"`` or
:func:`cape.cfdx.report.Report.SubfigPlotL2` for ``"PlotL2"``.

:See also:
    * :mod:`cape.cfdx.report`
    * :mod:`cape.pyfun.options.Report`
    * :mod:`cape.options.Report`
    * :class:`cape.cfdx.dataBook.DBComp`
    * :class:`cape.cfdx.dataBook.CaseFM`
    * :class:`cape.cfdx.lineLoad.DBLineLoad`
    
"""

# Standard library
import os
import json
import shutil
import glob

# Numerics
import numpy as np

# CAPE modules
import cape.cfdx.report
from cape.filecntl import tex
from cape          import tar

# Local modules
from .dataBook import CaseFM, CaseResid
from .case     import LinkPLT
from .tecplot  import ExportLayout, Tecscript


# Class to interface with report generation and updating.
class Report(cape.cfdx.report.Report):
    r"""Interface for automated report generation
    
    :Call:
        >>> R = pyFun.report.Report(fun3d, rep)
    :Inputs:
        *fun3d*: :class:`cape.pyfun.cntl.Cntl`
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
    def SubfigSwitch(self, sfig, i, lines, q):
        r"""Switch function to find the correct subfigure function
        
        This function may need to be defined for each CFD solver
        
        :Call:
            >>> lines = R.SubfigSwitch(sfig, i, lines)
        :Inputs:
            *R*: :class:`pyFun.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of subfigure to update
            *i*: :class:`int`
                Case index
            *lines*: :class:`list` (:class:`str`)
                List of lines already in LaTeX file
            *q*: ``True`` | ``False``
                Whether or not to regenerate subfigure
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                Updated list of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: First version
            * 2016-10-25 ``@ddalle``: *UpdateCaseSubfigs* -> 
                                      *SubfigSwitch*
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
        elif btyp == 'Paraview':
            # Get the Paraview layout view
            lines += self.SubfigParaviewLayout(sfig, i, q)
        elif btyp == 'Tecplot':
            # Get the Tecplot layout view
            lines += self.SubfigTecplotLayout(sfig, i, q)
        elif btyp == 'Image':
            # Coy an image
            lines += self.SubfigImage(sfig, i, q)
        else:
            print("  %s: No function for subfigure type '%s'" % (sfig, btyp))
        # Output
        return lines
        
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
            *FM*: ``None`` or :class:`cape.cfdx.dataBook.CaseFM` 
                derivative
                Case iterative force & moment history for one component
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2017-03-27 ``@ddalle``: Added *CompID* option
        """
        # Project rootname
        proj = self.cntl.GetProjectRootName(None)
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
        
    # Read residual history
    def ReadCaseResid(self, sfig=None):
        r"""Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid(sfig=None)
        :Inputs:
            *R*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *sfig*: :class:`str` | ``None``
                Name of subfigure to process
        :Outputs:
            *hist*: ``None`` or :class:`cape.cfdx.dataBook.CaseResid` 
                derivative
                Case iterative residual history for one case
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2016-02-04 ``@ddalle``: Added argument
        """
        # Project rootname
        proj = self.cntl.GetProjectRootName(None)
        # Read the residual history
        return CaseResid(proj)
        
    # Read a Tecplot script
    def ReadTecscript(self, fsrc):
        r"""Read a Tecplot script interface
        
        :Call:
            >>> R.ReadTecscript(fsrc)
        :Inputs:
            *R*: :class:`pyFun.report.Report`
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
            >>> R.LinkVizFiles(sfig, i)
        :Inputs:
            *R*: :class:`pyFun.report.Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of the subfigure
            *i*: :class:`int`
                Case index
        :See Also:
            :func:`pyFun.case.LinkPLT`
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
        """
        # Defer to function from :func:`pyFun.case`
        LinkPLT()
        
        
# class Report

