r"""
:mod:`cape.pyfun.report`: Automated report interface
======================================================

The pyFun module for generating automated results reports using
PDFLaTeX provides a single class :class:`Report`, which is
based off the CAPE version :class:`cape.cfdx.report.Report`. The
:class:`cape.cfdx.report.Report` class is a sort of dual-purpose object
that contains a file interface using :class:`cape.texfile.Tex` combined
with a capability to create figures for each case or sweep of cases
mostly based on :mod:`cape.cfdx.databook`.

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
    * :class:`cape.cfdx.databook.DBComp`
    * :class:`cape.cfdx.databook.CaseFM`
    * :class:`cape.cfdx.lineload.LineLoadDataBook`

"""

# Standard library

# Third-party

# Local import
from .casecntl import LinkPLT
from ..cfdx import report
from ..filecntl.tecfile import Tecscript


# Class to interface with report generation and updating.
class Report(report.Report):
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
            * 2016-02-06 ``@ddalle``: First version
        """
        # Defer to function from :func:`pyFun.case`
        LinkPLT()

