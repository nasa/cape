r"""
:mod:`cape.pylch.cntl`: Main Loci/CHEM run matrix controller
===============================================================

This module provides the :class:`Cntl` class that is specific to
``pylch``, the CAPE interface to Loci/CHEM.

"""

# Local imports
from . import options
from ..cntl import cntl as ccntl


# Primary class
class Cntl(ccntl.Cntl):
  # =================
  # Class attributes
  # =================
  # <
    # Hooks to py{x} specific modules
    # Hooks to py{x} specific classes
    _opts_cls = options.Options
    # Other settings
    _fjson_default = "pyLCH.json"
  # >

