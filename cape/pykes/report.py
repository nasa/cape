r"""
:mod:`cape.pykes.report`: Automated report interface
======================================================

The pyFun module for generating automated results reports using
PDFLaTeX provides a single class :class:`pyFun.report.Report`, which is
based off the CAPE version :class:`cape.cfdx.report.Report`. The
:class:`cape.cfdx.report.Report` class is a sort of dual-purpose object
that contains a file interface using :class:`cape.texfile.Tex` combined
with a capability to create figures for each case or sweep of cases
mostly based on :mod:`cape.cfdx.dataBook`.

An automated report is a multi-page PDF generated using PDFLaTeX.
Usually, each CFD case has one or more pages dedicated to results for
that casecntl. The user defines a list of figures, each with its own list
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
    * :class:`cape.cfdx.databook.DBComp`
    * :class:`cape.cfdx.databook.CaseFM`
    * :class:`cape.cfdx.lineload.LineLoadDataBook`

"""

# Standard library

# Third-party

# Local import
from .casecntl import link_plt
from .databook import CaseFM, CaseProp, CaseResid, CaseTurbResid
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
    # Read residual history
    def _ReadCaseResid(self, sfig=None):
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
            * 2021-12-01 ``@ddalle``: Version 1.0
        """
        # Get option for residual type
        rtype = self.cntl.opts.get_SubfigOpt(sfig, "ResidualType")
        # Read the residual history
        if rtype in ("turb", "turbulence"):
            # Read turbulence residual
            return CaseTurbResid()
        else:
            # Residuals for other states
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
            :func:`cape.pyfun.casecntl.LinkPLT`
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

