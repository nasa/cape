r"""
:mod:`cape.pylava.report`: Automated report interface
======================================================

This customization of the :mod:`cape.cfdx.report` class for LAVA mostly
just points to the correct ``databook`` classes that are specific to
LAVA.

"""

# Standard library

# Third-party

# Local import
from .casecntl import LinkViz
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

