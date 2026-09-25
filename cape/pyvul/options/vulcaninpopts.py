"""
:mod:`cape.pyvul.options.vulcaninpopts`: VULCAN input file options
=================================================================

This module provides a class to interpret JSON options that are
converted to VULCAN-CFD input file options.
"""

# Local imports
from ...optdict import OptionsDict


# Class for namelist settings
class VulcanInpOpts(OptionsDict):
    r"""Dictionary-based interface for VULCAN ``.inp`` input files"""

    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "options for VULCAN-CFD ``.inp`` input files"
