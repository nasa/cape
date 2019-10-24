"""
:mod:`cape.pycart.options.Report`: pyCart Report Options
=========================================================

This module provides options for creating automated reports for pyCart results.
Although there are some subfigure types that are unique to pyCart, all of the
methods in the :class:`pyCart.options.Report.Report` class are inherited from
:mod:`cape` version.

See the :ref:`JSON Report <cape-json-Report>` section (and its
:ref:`pyCart page <pyCart-json-Report>`). for the list of available figures and
subfigures (along with other options).

:See Also:
    * :mod:`cape.cfdx.options.Report`
    * :mod:`cape.cfdx.report`
    * :mod:`cape.pycart.report`
"""


# Import options-specific utilities
from .util import rc0, getel, isArray

# Import template module
import cape.cfdx.options.Report

# Class for Report settings
class Report(cape.cfdx.options.Report):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = Report(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    pass

# class Report


