r"""
:mod:`cape.pylch.report`: Automated report interface
======================================================

This module provides customizations for Loci/CHEM on the base
:mod:`cape.cfdx.report` module.

"""

# Standard library

# Third-party

# Local import
from .databook import CaseFM, CaseResid
from ..cfdx import report as capereport
from ..filecntl.tecfile import Tecscript


# Class to interface with report generation and updating.
class Report(capereport.Report):
    r"""Interface for automated report generation

    :Call:
        >>> R = pyFun.report.Report(fun3d, rep)
    :Inputs:
        *fun3d*: :class:`cape.pyfun.cntl.Cntl`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`Report`
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
            *R*: :class:`Report`
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
            *FM*: ``None`` or :class:`cape.cfdx.databook.CaseFM`
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
            *hist*: ``None`` or :class:`cape.cfdx.databook.CaseResid`
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
            *R*: :class:`Report`
                Automated report interface
            *fscr*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-10-25 ``@ddalle``: First version
        """
        return Tecscript(fsrc)

