"""Interface for automated report generation: :mod:`pyOver.options.Report`"""


# Import options-specific utilities
from .util import rc0, getel

# Import template module
import cape.options.Report

# Class for Report settings
class Report(cape.options.Report):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, **kw):
        """Initialization method
        
        :Call:
            >>> opts = Report(**kw)
        :Inputs:
            *kw*: :class:`dict` | :class:`cape.options.util.odict`
                Dictionary that is converted to this class
        :Outputs:
            *opts*: :class:`cape.options.Report.Report`
                Options interface
        :Versions:
            * 2016-02-04 ``@ddalle``: First version (not using :class:`dict`)
        """
        # Initialize
        for k in kw:
            self[k] = kw[k]
        # Initialize subfigure defaults
        self.SetSubfigDefaults()
        self.ModSubfigDefaults()
        
    # Modify defaults
    def ModSubfigDefaults(self):
        """Modify subfigure defaults for OVERFLOW
        
        :Call:
            >>> opts.SetSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Effects:
            *opts.defns*: :class:`dict`
                Some subfigure default options sets are reset
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Plot L2 residual
        self.defs['PlotL2'] = {
            "Header": "",
            "Residual": "L2Resid",
            "Component": None,
            "Position": "b",
            "Alignment": "center",
            "YLabel": "L2 residual",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
        # Plot general residual
        self.defs["PlotResid"] = {
            "Residual": "L2Resid",
            "Component": None,
            "YLabel": "Residual",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
# class Report


