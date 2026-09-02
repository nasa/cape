r"""
:mod:`cape.agent.skills.skilltools`: Skill activation tool
==========================================================

This module provides the ``use_skill`` tool, which the CAPE agent uses
to read the full instructions of an available agent skill. Skills are
listed by name and one-line description in the system prompt; the model
calls ``use_skill`` to load a skill's full Markdown instructions before
applying it.
"""

# Local imports
from . import skillbase
from ..tools import toolutils


# Parameter definitions for the tool schema
SKILLTOOL_PARAMS = {
    "name": {
        "description": (
            "Name of the skill to load, from the list of available "
            "skills shown in the system prompt."
        ),
        "type": "string",
    },
}


# Load the full instructions for an active skill
def use_skill(name: str) -> dict:
    r"""Read the full instructions of an active agent skill

    :Call:
        >>> result = use_skill(name)
    :Inputs:
        *name*: :class:`str`
            Name of an active skill
    :Outputs:
        *result*: :class:`dict`
            Keys include *success* and, if skill found, *content*
    """
    # Look up skill in active registry
    skill = skillbase.ACTIVE_SKILLS.get(name)
    # Check for unknown skill
    if skill is None:
        return {
            "success": False,
            "error": f"Unknown skill: '{name}'",
            "available_skills": sorted(skillbase.ACTIVE_SKILLS),
        }
    # Return the full instructions
    return {
        "success": True,
        "name": skill.name,
        "description": skill.description,
        "content": skill.content,
    }


# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "use_skill": {
        "description": (
            "Load the full instructions for an agent skill by name. "
            "Skills describe how and when to use certain tools and how "
            "to chain tool calls. Call this before applying a skill."
        ),
        "parameters": ["name"],
        "required": ["name"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILLTOOL_PARAMS)
