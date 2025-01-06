r"""
:mod:`cape.pylch.report`: Automated report interface
======================================================

This module provides customizations for Loci/CHEM on the base
:mod:`cape.cfdx.report` module.

"""

# Standard library

# Third-party

# Local import
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
    """
    pass

