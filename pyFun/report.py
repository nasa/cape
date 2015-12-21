"""Function for Automated Report Interface"""

# File system interface
import os, json, shutil, glob
# Numerics
import numpy as np

# Import basis module
import cape.report

# Local modules needed
from cape import tex, tar
# Data book and plotting
from .dataBook import CaseFM, CaseResid
# Folder and Tecplot management
#from .case    import LinkPLT


# Class to interface with report generation and updating.
class Report(cape.report.Report):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyFun.report.Report(fun3d, rep)
    :Inputs:
        *fun3d*: :class:`pyFun.fun3d.Fun3d`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyFun.report.Report
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
    
    # Read iterative history
    def ReadCaseFM(self, comp):
        """Read iterative history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> FM = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *FM*: ``None`` or :class:`cape.dataBook.CaseFM` derivative
                Case iterative force & moment history for one component
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetProjectRootName()
        # Read the history for that component
        return CaseFM(proj, comp)
        
    # Read residual history
    def ReadCaseResid(self):
        """Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid()
        :Inputs:
            *R*: :class:`cape.report.Report`
                Automated report interface
        :Outputs:
            *hist*: ``None`` or :class:`cape.dataBook.CaseResid` derivative
                Case iterative residual history for one case
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetProjectRootName()
        # Read the residual history
        return CaseResid(proj)
        
        
# class Report

        
