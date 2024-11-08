"""
:mod:`cape.pylch.options.varsopts`: Loci/CHEM ``.vars`` options
=================================================================

This module provides a class to interpret JSON options that are
converted to Loci/CHEM ``.vars`` file format for ``pylch``. The module
provides a class, :class:`VarsOpts`, which interprets the settings of
the ``"Vars"`` section of the master JSON file.
"""

# Standard library
from typing import Any, Optional

# Local imports
from ...optdict import OptionsDict
from ...optdict.optitem import setel


# Class for namelist settings
class VarsOpts(OptionsDict):
    r"""Dictionary-based interface for Loci/CHEM ``.vars`` files"""

    # Reduce to a single phase
    def select_vars_phase(self, j: int = 0, **kw) -> dict:
        r"""Sample namelist at particular conditions

        :Call:
            >>> d = opts.select_namelist(i)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *d*: :class:`dict`
                Namelist sampled for phase and case indices
        :Versions:
            * 2024-11-08 ``@ddalle``: v1.0
        """
        # Sample list -> scalar, evaluate @expr, etc.
        return self.sample_dict(self, j=j, **kw)

    # Get value by name
    def get_vars_var(self, key: str, j: Optional[int] = None, **kw) -> Any:
        r"""Select a ``.vars`` file option

        Roughly, this returns ``opts[key]``.

        :Call:
            >>> val = opts.get_vars_var(key, j=None, **kw)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *key*: :class:`str`
                Variable name
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *val*: :class:`object`
                Value from JSON options
        :Versions:
            * 2024-11-08 ``@ddalle``: v1.0
        """
        # Set sample to false
        kw.setdefault("sample", False)
        # Return subsection options
        return self.get_opt(key, j=j, **kw)

    # Set value by name
    def set_vars_var(self, key: str, val: Any, j: Optional[int] = None):
        r"""Set a ``.vars`` file key for a specified phase or phases

        Roughly, this sets ``opts["Vars"][key]`` or
        ``opts["Vars"][key][j]`` equal to *val*

        :Call:
            >>> opts.set_vars_var(key, val, j=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *key*: :class:`str`
                Variable name
            *val*: :class:`object`
                Value from JSON options
            *j*: {``None``} | :class:`int`
                Phase index
        :Versions:
            * 2024-11-08 ``@ddalle``: v1.0
        """
        # Initialize key
        v0 = self.setdefault(key, None)
        # Set value
        self[key] = setel(v0, val, j=j)

