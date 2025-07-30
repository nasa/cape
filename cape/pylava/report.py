r"""
:mod:`cape.pylava.report`: Automated report interface
======================================================

This customization of the :mod:`cape.cfdx.report` class for LAVA mostly
just points to the correct ``databook`` classes that are specific to
LAVA.

"""

# Standard library
from typing import Optional

# Third-party
import numpy as np

# Local import
from .casecntl import LinkViz
from .databook import CaseFM, CaseResid
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
    def ReadCaseFM(self, comp: str):
        r"""Read iterative history for a component

        This function needs to be customized for each solver

        :Call:
            >>> fm = r.ReadCaseFM(comp)
        :Inputs:
            *r*: :class:`cape.cfdx.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *fm*: ``None`` | :class:`CaseFM`
                Case iterative force & moment history for one component
        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        # Use appropriate reader
        cls = CaseFM
        # Get component (note this automatically defaults to *comp*)
        compID = self.cntl.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if isinstance(compID, (list, np.ndarray)):
            # Read the first component
            fm = cls(compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-'):
                    # Subtract the component
                    fm -= cls(compi.lstrip('-'))
                else:
                    # Add in the component
                    fm += cls(compi)
        else:
            # Read the iterative history for single component
            fm = cls(compID)
        # Output
        return fm

    # Read residual history
    def ReadCaseResid(self, sfig: Optional[str] = None):
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
            * 2024-10-11 ``@ddalle``: v1.0
        """
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
            :func:`cape.pylava.casecntl.LinkViz`
        :Versions:
            * 2025-07-28 ``@jmeeroff``: First version
        """
        # Defer to function from :func:`pylava.casecntl`
        LinkViz()

