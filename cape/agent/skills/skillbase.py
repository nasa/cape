r"""
:mod:`cape.agent.skills.skillbase`: Core skill definition and discovery
========================================================================

This module provides the :class:`Skill` class, which defines an *agent
skill*: a documented workflow that tells the CAPE agent harness how and
when to use certain tools, and how to chain tool calls together.

Skills can be defined in the code base (see
:mod:`cape.agent.skills.cntlrunner`) or discovered from the folder in
which the user launches the CAPE agent. User skills are read from
``.agents/skills/<NAME>/SKILL.md`` files, which consist of a YAML
front matter section and a Markdown body:

.. code-block:: markdown

    ---
    name: my-skill
    description: One-line summary shown in the skill listing
    ---

    # My skill

    Full instructions for the agent go here.

The *name* and *description* fields are used to list available skills in
the system prompt; the full *content* is only shown to the model when it
asks for it using the ``use_skill`` tool.
"""

# Standard library
import os
import glob

# Third-party
import yaml


# Name of the main file in a skill folder
SKILL_FILE_NAME = "SKILL.md"

# Metadata keys required in a skill file
REQUIRED_KEYS = (
    "name",
    "description",
)

# Registry of skills that are currently active (for this agent session)
ACTIVE_SKILLS: dict = {}


# Class definition
class Skill:
    r"""Definition of an agent skill, from code or a ``SKILL.md`` file

    :Call:
        >>> skill = Skill(name, description, content, **kw)
    :Inputs:
        *name*: :class:`str`
            Short name of the skill, e.g. ``"cntl-runner"``
        *description*: :class:`str`
            One-line summary shown in the skill listing
        *content*: :class:`str`
            Full Markdown instructions for the agent
        *tools*: {``None``} | :class:`list`\ [:class:`str`]
            Names of tools this skill provides, if any
        *fname*: {``None``} | :class:`str`
            Name of file skill was read from, if any
    :Outputs:
        *skill*: :class:`Skill`
            Definition of one agent skill
    """
    # Attributes
    __slots__ = (
        "content",
        "description",
        "fname",
        "name",
        "tools",
    )

    # Initialize
    def __init__(
            self,
            name: str,
            description: str,
            content: str,
            tools: list | None = None,
            fname: str | None = None):
        #: :class:`str`
        #: Short name of the skill
        self.name = name
        #: :class:`str`
        #: One-line summary shown in the skill listing
        self.description = description
        #: :class:`str`
        #: Full Markdown instructions for the agent
        self.content = content
        #: :class:`list`\ [:class:`str`]
        #: Names of tools this skill provides, if any
        self.tools = [] if tools is None else list(tools)
        #: :class:`str` | ``None``
        #: Name of file skill was read from, if any
        self.fname = fname

    # Create a skill from a dictionary definition (for built-in skills)
    @classmethod
    def from_defn(cls, name: str, defn: dict):
        r"""Create a :class:`Skill` from a dictionary definition

        :Call:
            >>> skill = Skill.from_defn(name, defn)
        :Inputs:
            *name*: :class:`str`
                Short name of the skill
            *defn*: :class:`dict`
                Dictionary with at least *description* and *content*
        :Outputs:
            *skill*: :class:`Skill`
                Definition of one agent skill
        """
        return cls(
            name,
            defn["description"],
            defn["content"],
            tools=defn.get("tools"),
        )


# Read a skill from a SKILL.md file
def read_skillfile(fname: str) -> Skill:
    r"""Read an agent skill definition from a ``SKILL.md`` file

    The file must start with a YAML front matter section delimited by
    ``---`` lines, containing at least *name* and *description*. The
    remainder of the file is Markdown instructions for the agent.

    :Call:
        >>> skill = read_skillfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of ``SKILL.md`` file to read
    :Outputs:
        *skill*: :class:`Skill`
            Definition of one agent skill
    :Raises:
        :class:`ValueError`
            If *fname* has no valid front matter or is missing a
            required key
    :Versions:
        * 2026-09-01 ``@ddalle``: v1.0
    """
    # Read the file
    with open(fname) as fp:
        text = fp.read()
    # Check for front matter delimiter on first line
    if not text.startswith("---"):
        raise ValueError(
            f"Skill file '{fname}' has no YAML front matter")
    # Split into front matter and body
    parts = text.split("\n---", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Skill file '{fname}' has no closing '---' for front matter")
    # Remove leading "---" from first part
    front = parts[0].lstrip("-").strip()
    body = parts[1].lstrip("-").lstrip()
    # Parse the front matter
    meta = yaml.safe_load(front)
    # Check type
    if not isinstance(meta, dict):
        raise ValueError(
            f"Skill file '{fname}' front matter is not a mapping")
    # Check for required keys
    for key in REQUIRED_KEYS:
        if key not in meta:
            raise ValueError(
                f"Skill file '{fname}' is missing required key '{key}'")
    # Create the skill
    return Skill(
        meta["name"],
        meta["description"],
        body,
        tools=meta.get("tools"),
        fname=fname,
    )


# Discover user skills from a root folder
def discover_user_skills(rootdir: str) -> dict:
    r"""Discover user skills from an ``.agents/skills/`` folder

    Scans ``<rootdir>/.agents/skills/<NAME>/SKILL.md`` and reads every
    valid skill file. Malformed files are skipped with a warning.

    :Call:
        >>> user_skills = discover_user_skills(rootdir)
    :Inputs:
        *rootdir*: :class:`str`
            Folder in which the agent was launched
    :Outputs:
        *user_skills*: :class:`dict`\ [:class:`Skill`]
            Map of skill names to skill definitions
    :Versions:
        * 2026-09-01 ``@ddalle``: v1.0
    """
    # Folder in which user skills are stored
    skilldir = os.path.join(rootdir, ".agents", "skills")
    # Initialize output
    user_skills = {}
    # Check for folder
    if not os.path.isdir(skilldir):
        return user_skills
    # Loop through candidate skill files
    fskills = sorted(glob.glob(os.path.join(skilldir, "*", SKILL_FILE_NAME)))
    for fskill in fskills:
        # Read the skill
        try:
            skill = read_skillfile(fskill)
        except Exception as e:
            # Skip malformed skill files with a warning
            print(f"Warning: skipping skill file '{fskill}': {e}")
            continue
        # Add to registry (later skills override same-named earlier ones)
        user_skills[skill.name] = skill
    # Output
    return user_skills
