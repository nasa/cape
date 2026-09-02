
# Standard library
import os

# Third-party
import pytest
import testutils

# Local imports
from cape.agent.skills import skillbase


# Sample skill file text
SKILL_TEXT = """\
---
name: demo
description: A demo user skill
---

# Demo

Full instructions here.
"""

# Text with no front matter
NO_FRONT_TEXT = "# No front matter here\n"

# Text missing a required key
NO_DESC_TEXT = """\
---
name: nodefskill
---

# This one has no description
"""


# Test reading a valid SKILL.md file
@testutils.run_sandbox(__file__)
def test_01_read_skillfile():
    # Write a valid skill file
    with open("SKILL.md", "w") as fp:
        fp.write(SKILL_TEXT)
    # Read it
    skill = skillbase.read_skillfile("SKILL.md")
    # Check attributes
    assert skill.name == "demo"
    assert skill.description == "A demo user skill"
    assert skill.content.startswith("# Demo")
    assert "Full instructions here." in skill.content
    assert skill.fname == "SKILL.md"
    assert skill.tools == []


# Test error on missing front matter
@testutils.run_sandbox(__file__)
def test_02_read_skillfile_no_front():
    # Write a skill file with no front matter
    with open("SKILL.md", "w") as fp:
        fp.write(NO_FRONT_TEXT)
    # Attempt to read
    with pytest.raises(ValueError):
        skillbase.read_skillfile("SKILL.md")


# Test error on missing required key
@testutils.run_sandbox(__file__)
def test_03_read_skillfile_missing_key():
    # Write a skill file missing *description*
    with open("SKILL.md", "w") as fp:
        fp.write(NO_DESC_TEXT)
    # Attempt to read
    with pytest.raises(ValueError):
        skillbase.read_skillfile("SKILL.md")


# Test discovering user skills from .agents/skills
@testutils.run_sandbox(__file__)
def test_04_discover_user_skills():
    # Create skill folder
    os.makedirs(os.path.join(".agents", "skills", "demo"))
    # Write a skill file
    fskill = os.path.join(".agents", "skills", "demo", "SKILL.md")
    with open(fskill, "w") as fp:
        fp.write(SKILL_TEXT)
    # Discover skills
    found = skillbase.discover_user_skills(".")
    # Check results
    assert sorted(found) == ["demo"]
    assert found["demo"].description == "A demo user skill"


# Test discovery with malformed and missing folders
@testutils.run_sandbox(__file__)
def test_05_discover_user_skills_skip_bad():
    # No .agents folder at all -> empty
    assert skillbase.discover_user_skills(".") == {}
    # Create one good and one malformed skill
    os.makedirs(os.path.join(".agents", "skills", "demo"))
    os.makedirs(os.path.join(".agents", "skills", "broken"))
    with open(os.path.join(".agents", "skills", "demo", "SKILL.md"), "w") as fp:
        fp.write(SKILL_TEXT)
    with open(os.path.join(".agents", "skills", "broken", "SKILL.md"), "w") as fp:
        fp.write(NO_FRONT_TEXT)
    # Discover; malformed file skipped
    found = skillbase.discover_user_skills(".")
    assert sorted(found) == ["demo"]
