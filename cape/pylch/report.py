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
from ..cfdx import report


# Class to interface with report generation and updating.
class Report(report.Report):
    # Read case FM history and normalize
    def ReadCaseFM(self, comp: str) -> CaseFM:
        r"""Read iterative history for a component

        :Call:
            >>> fm = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *fm*: :class:`CaseFM`
                Iterative force & moment history
        :Versions:
            * 2025-05-23 ``@ddalle``: v1.0
        """
        # Get component (note this automatically defaults to *comp*)
        compID = self.cntl.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if isinstance(compID, list):
            # Read the first component
            fm = CaseFM(compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-'):
                    # Subtract the component
                    fm -= CaseFM(compi.lstrip('-'))
                else:
                    # Add in the component
                    fm += CaseFM(compi)
        else:
            # Read the iterative history for single component
            fm = CaseFM(comp)
        # Normalize it
        fm.normalize_by_cntl(self.cntl, self.i)
        # Output
        return fm

    # Read residual history
    def ReadCaseResid(self, sfig=None):
        r"""Read iterative residual history for a component

        This function needs to be customized for each solver

        :Call:
            >>> h = R.ReadCaseResid(sfig=None)
        :Inputs:
            *R*: :class:`Report`
                Automated report interface
            *sfig*: :class:`str` | ``None``
                Name of subfigure to process
        :Outputs:
            *hist*: `:class:`CaseResid`
                Iterative residual history for one case
        :Versions:
            * 2025-05-23 ``@ddalle``: v1.0
        """
        # Read the residual history
        return CaseResid()

