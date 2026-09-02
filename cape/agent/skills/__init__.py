r"""
This package collects the *agent skills* available to the CAPE agent.
A skill is a documented workflow that tells the agent how and when to
use certain tools and how to chain tool calls together. Skills come
from two sources:

*Built-in skills* are defined in this package, alongside any tools they
provide (see :mod:`cape.agent.skills.cntlrunner`).

*User skills* are discovered from ``.agents/skills/<NAME>/SKILL.md``
files in the folder in which the user launches the agent (see
:func:`cape.agent.skills.skillbase.discover_user_skills`).

Skills are gated by the *SkillSet* option of the model in use (see
:mod:`cape.agent.options.modelopts`), and activated by the agent using
the ``use_skill`` tool (see :mod:`cape.agent.skills.skilltools`).
"""

# Local imports
from . import cntlrunner
from . import skillbase
from . import skilltools
from . import usertools

# Re-exports
from .skillbase import Skill, discover_user_skills, read_skillfile


# Public interface
__all__ = (
    "BUILTIN_SKILLS",
    "SKILL_SETS",
    "SKILL_TOOL_MODULES",
    "Skill",
    "cntlrunner",
    "discover_user_skills",
    "read_skillfile",
    "skillbase",
    "skilltools",
    "usertools",
)


# Built-in skills, merged from skill modules
BUILTIN_SKILLS = {
    **{name: Skill.from_defn(name, defn)
       for name, defn in cntlrunner.SKILL_DICT.items()},
    **{name: Skill.from_defn(name, defn)
       for name, defn in usertools.SKILL_DICT.items()},
}

# Map of built-in skill names to the modules providing their tools
SKILL_TOOL_MODULES = {
    "cntl-runner": cntlrunner,
    "user-tools": usertools,
}

# Skills per capability level
SKILL_SETS = {
    "none": [],
    "low": [],
    "medium": [
        "cntl-runner",
        "user-tools",
    ],
    "full": list(BUILTIN_SKILLS),
}
