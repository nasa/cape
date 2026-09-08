r"""
:mod:`cape.cfdx.options.actionopts`: Customize case-disposition actions
------------------------------------------------------------------------

This provides a list of what CAPE should normally do in any of a variety
of case disposition actions. The most common example is how to handle
"approved" cases. This would normally start with

.. code-block:: console

    $ cape approve -I {CASES}
    $ cape dex -I {CASES}

but may also include custom actions (e.g. ``./tools/writesurfcp.py``) or
divide the commands into groups.

.. code-block:: console

    $ cape --PASS -I {CASES}
    $ cape --dex "[A-L]*" -I {CASES}
    $ cape --dex "[M-Z]*" -I {CASES}
    $ ./tools/writesurfcp.py -I {CASES}

This would look like the following in the ``"Actions"`` section of the
CAPE JSON file:

.. code-block:: javascript

    "Actions": {
        "approve": [
            {
                "type": "cntl",
                "function": "MarkPASS"
            },
            "./tools/writesurfcp.py",
            {
                "type": "cli",
                "function": "cape_extract_dex",
                "index": 2,
                "kwargs": {
                    "dex": "[A-L]*"
                }
            },
            {
                "type": "cli",
                "function": "cape_extract_dex",
                "index": 2,
                "kwargs": {
                    "dex": "[M-Z]*"
                }
            }
        ]
    }

This means that when a user calls

    .. code-block:: console

        $ cape perform approve -I {CASES}

The following commands will happen:

    .. code-block:: console

        $ cape approve -I {CASES}
        $ ./tools/writesurfcp.py -I {CASES}
        $ cape dex --dex "[A-L]*" -I {CASES}
        $ cape dex --dex "[M-Z]*" -I {CASES}

However, the last two will happen simultaneously because the have the
same value for ``"index"``.
"""

# Local imports
from ...optdict import OptionsDict


# Options for a single action
class ActionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Options
    _optlist = (
        "args",
        "function",
        "index",
        "kwargs",
        "type",
    )

    # Aliases
    _optmap = {
        "command": "function",
    }

    # Types
    _opttypes = {
        "type": str,
        "function": str,
        "index": int,
        "kwargs": dict,
        "args": list,
    }

    # Allowed values
    _optvals = {
        "type": ("shell", "cntl", "cli"),
    }

    # Defaults
    _rc = {
        "type": "shell",
    }

    # Descriptions
    _rst_descriptions = {
        "type": "Action method, ``Cntl`` method, ``cli`` function, or shell",
        "index": "Action index; enables simultaneous actions",
        "function": "Name of function to call",
        "args": "Positional args to give to `cli` | `cntl` function",
        "kwargs": "Keyword args to give to `cli` | `cntl` function",
    }

    # Initialization method
    def __init__(self, *args, **kw):
        # Test for input like ExecOpts(False)
        if len(args) > 0:
            # Get first argument
            a = args[0]
            # Test if it's false-like
            if isinstance(a, str):
                # Reset "my_cmd" -> {"function": "my_cmd"}
                args = {"function": a}
        # Pass to parent initializer
        OptionsDict.__init__(self, *args, **kw)


# Class for overall options
class ActionsOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Accepted values
    _optlist = (
        "UserTools",
        "approve",
        "defail",
        "dezombie",
        "extend",
        "extend2",
    )

    # Additional options
    _xoptkey = "UserTools"

    # Types
    _opttypes = {
        "UserTools": str,
        "_default_": ActionOpts,
    }
