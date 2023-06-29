r"""
:mod:`cape.pyfun.options.refineopts`: Refine[3] mesh adaptation options
=======================================================================

This module provides a class to access command-line options to the
refine mesh adapation software package.
"""

# Local imports
from ...optdict import BOOL_TYPES, FLOAT_TYPES, INT_TYPES
from .util import ExecOpts


# Class for ref cli options
class RefineOpts(ExecOpts):
    r"""Class for refine command kine settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
    }

    # Types
    _opttypes = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
    }


# Add properties
RefineOpts.add_properties(RefineOpts._optlist, prefix="refine_")


# Class for refmpi cli options
class RefineMPIOpts(ExecOpts):
    r"""Class for refine mpi command line settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineMPIOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
    }

    # Types
    _opttypes = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions
    _rst_descriptions = {
    }


# Add properties
RefineMPIOpts.add_properties(RefineMPIOpts._optlist, prefix="refinempi_")