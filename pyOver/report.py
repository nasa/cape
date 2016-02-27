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
        >>> R = pyOver.report.Report(oflow, rep)
    :Inputs:
        *oflow*: :class:`pyOver.overflow.Overflow`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyOver.report.Report`
            Automated report interface
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    
    # String conversion
    def __repr__(self):
        """String/representation method
        
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        return '<pyOver.Report("%s")>' % self.rep
    # Copy the function
    __str__ = __repr__
    
    # Read iterative history
    def ReadCaseFM(self, comp):
        """Read iterative history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> FM = R.ReadCaseFM(comp)
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
            *comp*: :class:`str`
                Name of component to read
        :Outputs:
            *FM*: ``None`` or :class:`cape.dataBook.CaseFM` derivative
                Case iterative force & moment history for one component
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetPrefix()
        # Read the history for that component
        return CaseFM(proj, comp)
        
    # Read residual history
    def ReadCaseResid(self, sfig=None):
        """Read iterative residual history for a component
        
        This function needs to be customized for each solver
        
        :Call:
            >>> hist = R.ReadCaseResid()
        :Inputs:
            *R*: :class:`pyOver.report.Report`
                Automated report interface
        :Outputs:
            *hist*: ``None`` or :class:`cape.dataBook.CaseResid` derivative
                Case iterative residual history for one case
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Project rootname
        proj = self.cntl.GetPrefix()
        # Initialize the residual history
        R = CaseResid(proj)
        # Check type
        t = self.cntl.opts.get_SubfigBaseType(sfig)
        # Check the residual to read
        resid = self.cntl.opts.get_SubfigOpt(sfig, "Residual")
        # Check the component to read
        comp = self.cntl.opts.get_SubfigOpt(sfig, "Component")
        # Check type of residual to read
        if t in ['PlotL2']:
            # Read spatial L2-norm
            if comp is None:
                # Read global residual
                R.ReadGlobalL2()
            else:
                # Read individual residual
                R.ReadL2Grid(comp)
        elif t in ['PlotLInf']:
            # Read spatial L-infinity norm
            if comp is None:
                # Read global residual
                R.ReadGlobalLInf()
            else:
                # Read individual residual
                R.ReadLInfGrid(comp)
        elif t in ['PlotResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadGlobalL2()
                else:
                    # Not implemented
                    R.ReadL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadGlobalLInf()
                else:
                    # Read for a specific grid
                    R.ReadLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotTurbResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadTurbResidL2()
                else:
                    # Not implemented
                    R.ReadTurbL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadTurbResidLInf()
                else:
                    # Read for a specific grid
                    R.ReadTurbLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        elif t in ['PlotSpeciesResid']:
            # Read spatial residual, but check which to read
            if resid in ['L2']:
                # Check component
                if comp is None:
                    # Read global residual
                    R.ReadSpeciesResidL2()
                else:
                    # Not implemented
                    R.ReadSpeciesL2Grid(comp)
            elif resid in ['LInf']:
                # Check component
                if comp is None:
                    # Read global L-infinity residual
                    R.ReadSpeciesResidLInf()
                else:
                    # Read for a specific grid
                    R.ReadSpeciesLInfGrid(comp)
            else:
                # Unrecognized type
                raise ValueError(
                    ("Residual '%s' for subfigure '%s' is unrecognized\n"
                    % (resid, sfig)) +
                    "Available options are 'L2' and 'LInf'")
        # Output
        return R
        
        
# class Report

