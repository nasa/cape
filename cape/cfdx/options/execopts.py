r"""
:mod:`cape.cfdx.options.execopts`: ``OptionsDict`` derivative for CLI
----------------------------------------------------------------------

This module provides :class:`ExecOpts`, which is a slight modification
of the default :class:`OptionsDict` class. If initialize with a single
boolean argument (e.g. ``ExecOpts(True)``), it will take that boolean to
be the value for the *run* option.

.. code-block:: pycon

    >>> ExecOpts(True)
    {'run': True}

"""

# Local imports
from ...optdict import OptionsDict, BOOL_TYPES


# Special class for CLI options
class ExecOpts(OptionsDict):
    r"""Special options class for CLI arguments to an executable

    This allows a JSON setting like

    .. code-block:: javascript

        "aflr3": false

    to be interpreted as

    .. code-block:: javascript

        "aflr3": {
            "run": false
        }

    :Call:
        >>> opts = ExecOpts(q, **kw)
        >>> opts = ExecOpts(d, **kw)
        >>> opts = ExecOpts(fjson, **kw)
    :Ipnuts:
        *q*: ``True`` | ``False``
            Positional input interpreted as ``run=q``
        *d*: :class:`dict`
            Dictionary of other CLI options
        *fjson*: :class:`str`
            Name of JSON file to read
        *kw*: :class:`dict`
            Dictionary of other CLI options
    :Outputs:
        *opts*: :class:`ExecOpts`
            Options interface for CLI options
    :Versions:
        * 2022-10-28 ``@ddalle``: Version 1.0
    """
    # Class attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "run",
    }

    # Types
    _opttypes = {
        "run": BOOL_TYPES,
    }

    # Descriptions
    _rst_descriptions = {
        "run": "whether to execute program",
    }

    # Initialization method
    def __init__(self, *args, **kw):
        # Test for input like ExecOpts(False)
        if len(args) > 0:
            # Get first argument
            a = args[0]
            # Test if it's false-like
            if isinstance(a, BOOL_TYPES):
                # Reset ExecOpts(False) -> ExecOpts(run=False)
                args = {"run": a},
        # Pass to parent initializer
        OptionsDict.__init__(self, *args, **kw)

    # Set default *run* option accordingly
    def init_post(self):
        r"""Post-init hook for AFLR3Opts

        This hook sets a case-dependent default value for *run*

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`AFLR3Opts`
                Options interface
        :Versions:
            * 2022-10-14 ``@ddalle``: Version 1.0
        """
        # Check for any options
        if self:
            self.setdefault("run", True)
        else:
            # For empty options, no entries -> run=False
            self.setdefault("run", False)
