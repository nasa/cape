r"""
:mod:`cape.agent.skills.checkdbtools`: Check databook components
================================================================

This module defines the built-in agent skill ``"check-db"``. It
combines the closely related ``cape check-db``, ``check-fm``,
``check-ll``, and ``check-triqfm`` commands into one compact tool
schema.
"""

# Local imports
from ..tools import toolutils
from ...cfdx import cli


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "component": {
        "description": (
            "Databook component type to check: 'all', 'fm', 'll', or "
            "'triqfm'."
        ),
        "type": "string",
        "enum": ["all", "fm", "ll", "triqfm"],
    },
    "f": {
        "description": (
            "Name of CAPE JSON file to use. If empty, CAPE finds the "
            "most appropriate file."
        ),
        "type": ["string", "null"],
    },
    "I": {
        "description": (
            "Case indices using Python slice syntax, for example '8', "
            "'5:11', or '14,17:20'."
        ),
        "type": ["string", "null"],
    },
}


# Map compact component names to their CAPE CLI implementations
CHECK_FUNCS = {
    "all": cli.cape_check_db,
    "fm": cli.cape_check_fm,
    "ll": cli.cape_check_ll,
    "triqfm": cli.cape_check_triqfm,
}


def cape_check_db(component: str, *a, **kw) -> dict:
    r"""Check databook completion for selected component types

    :Call:
        >>> result = cape_check_db(component, *a, **kw)
    :Inputs:
        *component*: {``"all"``, ``"fm"``, ``"ll"``, ``"triqfm"``}
            Databook component type to check
        *a*: :class:`tuple`
            Positional arguments passed to the CAPE CLI function
        *kw*: :class:`dict`
            Keyword arguments passed to the CAPE CLI function
    :Outputs:
        *result*: :class:`dict`
            Wrapped CLI result
    """
    func = CHECK_FUNCS.get(component)
    if func is None:
        return {
            "success": False,
            "error": f"Unknown databook component type: {component!r}",
            "allowed_components": list(CHECK_FUNCS),
        }
    return toolutils.wrap_cli(func, *a, **kw)


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# check-db: checking databook component completion

Use `cape_check_db` to update/check databook completion status for one
or more cases. Select the scope with `component`:

* `all` runs all three checks (force and moment, line load, and TriqFM).
* `fm` checks force-and-moment components only.
* `ll` checks line-load components only.
* `triqfm` checks TriqFM components only.

Pass case indices in `I` using Python slice syntax. For example, use
`I="5:11"` for cases 5 through 10. Continue to pass `f` if the user
selected a specific CAPE JSON file earlier in the conversation.
"""


# Simplified skill definition
SKILL_DICT = {
    "check-db": {
        "description": (
            "Check completion of force-and-moment, line-load, and "
            "TriqFM databook components for selected cases."
        ),
        "content": SKILL_CONTENT,
        "tools": ["cape_check_db"],
    },
}


# Simplified tool definition not in OpenAPI format
TOOL_DICT = {
    "cape_check_db": {
        "description": (
            "Check databook component completion. Call "
            "use_skill('check-db') for usage guidance first."
        ),
        "parameters": ["component", "f", "I"],
        "required": ["component", "I"],
    },
}


# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)
