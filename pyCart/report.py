"""Function for Automated Report Interface"""

# Local modules needed
from . import tex, tar
# Data book and plotting
from .dataBook import Aero, CaseFM, CaseResid

# Class to interface with report generation and updating.
class Report(object):
    """Interface for automated report generation
    
    :Call:
        >>> R = pyCart.report.Report(cart3d, rep)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master Cart3D settings interface
        *rep*: :class:`str`
            Name of report to update
    :Outputs:
        *R*: :class:`pyCart.report.Report
            Automated report interface
    :Versions:
        * 2015-03-07 ``@dalle``: Started
    """
    # Initialization method
    def __init__(self, cart3d, rep):
        """Initialization method"""
        # Save the interface
        self.cart3d = cart3d
        # Check for this report.
        if rep not in cart3d.opts.get('Report', {}):
            # Raise an exception
            raise KeyError("No report named '%s'" % rep)
        # Get the options.
        self.opts = cart3d.opts['Report'][rep]
        
        
        
