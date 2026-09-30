r"""
:mod:`cape.cfdx.options.waitopts`: Options for ``cape --wait``
==============================================================

This module contains options for controlling the CAPE polling function,
``cape-wait``, which waits for a certain number of cases to require
the user or agent to take action. It governs the ``"Wait"`` section of
the main CAPE JSON file.

Key options include:

    * ``"Interval"``: time to wait between polls
    * ``"NCase"``: number of cases requiring action
"""

# Local imports
from ...optdict import OptionsDict, FLOAT_TYPES, INT_TYPES


# Options class
class WaitOpts(OptionsDict):
    # No attributes
    __slots__ = ()


