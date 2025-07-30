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

    # Reduce to a single run sequence
    def select_runinputs_phase(self, j: int = 0, **kw):
        r"""Sample namelist at particular conditions

        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase index
        :Outputs:
            *d*: :class:`dict`
                Namelist sampled for phase and case indices
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0; `pyfun` > `select_namelist`
        """
        # Sample list -> scalar, evaluate @expr, etc.
        return self.sample_dict(self, j=j, **kw)
