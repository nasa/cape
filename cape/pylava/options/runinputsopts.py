"""
:mod:`cape.pylava.options.runinputsopts`: LAVA-Cart input file options
=======================================================================

This module provides the class :class:`RunInputsOpts`, which governs the
section ``"RunInputs"`` in the CAPE-LAVA JSON file.
"""

# Local imports
from ...optdict import OptionsDict


# Class for namelist settings
class RunInputsOpts(OptionsDict):
    r"""Dictionary-based interface for LAVA-Cart ``run.inputs`` file"""

    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "options for LAVA-Cart ``run.inputs`` file"
