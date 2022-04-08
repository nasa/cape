r"""
:mod:`cape.pykes.report`: Automated report interface
======================================================

The pyFun module for generating automated results reports using 
PDFLaTeX provides a single class :class:`pyFun.report.Report`, which is
based off the CAPE version :class:`cape.cfdx.report.Report`. The
:class:`cape.cfdx.report.Report` class is a sort of dual-purpose object
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
    
        $ pykes --report

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
    * :mod:`cape.pykes.options.Report`
    * :mod:`cape.options.Report`
    * :class:`cape.cfdx.dataBook.DBComp`
    * :class:`cape.cfdx.dataBook.CaseFM`
    * :class:`cape.cfdx.lineLoad.DBLineLoad`
    
"""

# Standard library

# Third-party
import numpy as np

# Local import
from .case import link_plt
from .dataBook import CaseFM, CaseProp, CaseResid
from ..cfdx import report as capereport


# Class to interface with report generation and updating.
class Report(capereport.Report):
    r"""Interface for automated report generation
    
    :Call:
        >>> R = Report(cntl, rep)
    :Inputs:
        *cntl*: :class:`cape.pykes.cntl.Cntl`
            CAPE main control instance
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyFun.report.Report`
            Automated report interface
    :Versions:
        * 2021-12-01 ``@ddalle``: Version 1.0
    """
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2015-10-16 ``@ddalle``: Version 1.0
        """
        return "<pykes.Report('%s')>" % self.rep
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
            *lines*: :class:`list`\ [:class:`str`]
                List of lines already in LaTeX file
            *q*: ``True`` | ``False``
                Whether or not to regenerate subfigure
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Updated list of lines for LaTeX file
        :Versions:
            * 2015-05-29 ``@ddalle``: Version 1.0
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
            * 2015-10-16 ``@ddalle``: Version 1.0
            * 2017-03-27 ``@ddalle``: Added *CompID* option
        """
        # Get component (note this automatically defaults to *comp*)
        compID = self.cntl.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if isinstance(compID, (list, np.ndarray)):
            # Read the first component
            FM = _read_case_prop(compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-'):
                    # Subtract the component
                    FM -= _read_case_prop(compi.lstrip('-'))
                else:
                    # Add in the component
                    FM += _read_case_prop(compi)
        else:
            # Read the iterative history for single component
            FM = _read_case_prop(compID)
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
            * 2021-12-01 ``@ddalle``: Version 1.0
        """
        # Read the residual history
        return CaseResid()
            
    # Function to link appropriate visualization files
    def LinkVizFiles(self, sfig=None, i=None):
        r"""Create links to appropriate visualization files
        
        :Call:
            >>> R.LinkVizFiles(sfig, i)
        :Inputs:
            *R*: :class:`Report`
                Automated report interface
            *sfig*: :class:`str`
                Name of the subfigure
            *i*: :class:`int`
                Case index
        :See Also:
            :func:`cape.pyfun.case.LinkPLT`
        :Versions:
            * 2016-02-06 ``@ddalle``: Version 1.0
        """
        # Defer to function from :mod:`pykes.case`
        link_plt()


def _read_case_prop(comp):
    # Check if it's a force & moment or not
    if "/" in comp:
        # Read as an "other-properties" object
        return CaseProp(comp)
    else:
        # Read as an iterative force & moment history
        return CaseFM(comp)

