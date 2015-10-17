"""Interface for automated report generation: :mod:`pyFun.options.Report`"""


# Import options-specific utilities
from .util import rc0, getel, isArray

# Import template module
import cape.options.Report

# Class for Report settings
class Report(cape.options.Report):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    pass
# class Report


