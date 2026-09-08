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

However, the last two will happen simultaneously because they have the
same value for ``"index"``. STDOUT and STDERR are suppressed while
simultaneous actions are running.

User-defined tools for ``cape review`` and ``cape dispatch`` are also
defined in this section: the ``"UserTools"`` option is a list of
names, and each name is defined as an action just like ``"approve"``
in the example above.

.. code-block:: javascript

    "Actions": {
        "UserTools": ["writecp"],
        "writecp": "./tools/writesurfcp.py {I}"
    }

The names ``"approve"``, ``"defail"``, ``"dezombie"``, ``"extend"``,
and ``"extend2"`` have default definitions (see *DEFAULT_ACTIONS*)
that reproduce the behavior of ``cape approve``, ``cape defail``,
etc., unless redefined in this section.

For shell commands, a ``{I}`` placeholder is replaced by the case
indices being processed; shell commands without one get
``-I {CASES}`` appended automatically. Shell commands from the legacy
top-level *UserTools* option (a dict of names and commands), however,
must contain a ``{I}`` placeholder.
"""

# Local imports
from ...optdict import OptionsDict


# Default actions for built-in names; defined here as immutable module
# data rather than in ``ActionsOpts._rc`` b/c ``_rc`` values can be
# returned to callers by reference, making mutable defaults fragile
DEFAULT_ACTIONS = {
    "approve": (
        {"type": "cntl", "function": "MarkPASS"},
    ),
    "defail": (
        {"type": "cntl", "function": "Defail"},
    ),
    "dezombie": (
        {"type": "cntl", "function": "Dezombie"},
    ),
    "extend": (
        {"type": "cntl", "function": "ExtendCases"},
    ),
    "extend2": (
        {"type": "cntl", "function": "ExtendCases",
         "kwargs": {"extend": 2}},
    ),
}


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
        # Test for input like ActionOpts("mycmd {I}")
        if len(args) > 0:
            # Get first argument
            a = args[0]
            # Test if it's a shell command
            if isinstance(a, str):
                # Reset "my_cmd" -> {"function": "my_cmd"}
                args = ({"function": a},) + args[1:]
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

    # Additional options; each name listed in *UserTools* becomes an
    # accepted option whose definition is an *ActionOpts*
    _xoptkey = "UserTools"

    # Types
    _opttypes = {
        "UserTools": str,
        "_default_": ActionOpts,
    }

    # List depth: *UserTools* is a list of action names
    _optlistdepth = {
        "UserTools": 1,
    }

    # Descriptions
    _rst_descriptions = {
        "UserTools": "names of user tools for ``cape review``/``dispatch``",
    }

   # --- Type checks ---
    # Any name is a valid option b/c users can define new action names
    def check_optname(self, opt, mode=None):
        return True

    # Convert strings and lists thereof into actions before checking
    def check_opttype(self, opt, val, mode=None):
        # Apply alias
        opt = self.apply_optmap(opt)
        # Only action-definition options need conversion
        if opt != "UserTools":
            # Convert "cmd" -> ActionOpts("cmd")
            if isinstance(val, (str, dict)):
                val = ActionOpts(val)
            elif isinstance(val, (list, tuple)):
                # Convert each action in the list
                val = [ActionOpts(v) for v in val]
        # Pass to parent checker
        return OptionsDict.check_opttype(self, opt, val, mode)

   # --- Actions ---
    # Get list of actions for one action name
    def get_Action(self, name) -> list:
        r"""Get list of action definitions for an action name

        :Call:
            >>> actlist = opts.get_Action(name)
        :Inputs:
            *opts*: :class:`ActionsOpts`
                Actions options interface
            *name*: :class:`str`
                Name of action, e.g. ``"approve"`` or a *UserTools* name
        :Outputs:
            *actlist*: ``None`` | :class:`list`\ [:class:`ActionOpts`]
                List of actions to perform, or ``None`` if *name* is
                neither defined locally nor a default action name
        """
        # Get user-defined action list
        v = self.get(name)
        # Fall back to defaults
        if v is None:
            v = DEFAULT_ACTIONS.get(name)
            # No action found
            if v is None:
                return None
        # Convert to a list if only one action given
        if isinstance(v, (str, dict)):
            v = [v]
        # Normalize fresh *ActionOpts* for each action
        return [ActionOpts(vj) for vj in v]

    # Get names of user-defined tools
    def get_UserToolsNames(self) -> list:
        r"""Get names of user-defined tools

        :Call:
            >>> names = opts.get_UserToolsNames()
        :Inputs:
            *opts*: :class:`ActionsOpts`
                Actions options interface
        :Outputs:
            *names*: :class:`list`\ [:class:`str`]
                Names of user tools for ``cape review``/``dispatch``
        """
        # Read option; convert str to list
        names = self.get("UserTools", [])
        # Ensure list
        if isinstance(names, str):
            names = [names]
        # Output
        return names
