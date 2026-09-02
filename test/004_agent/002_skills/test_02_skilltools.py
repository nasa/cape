
# Third-party
import pytest

# Local imports
from cape.agent.skills import skillbase, skilltools
from cape.agent.skills.skillbase import Skill


# Save and restore ACTIVE_SKILLS around each test
@pytest.fixture(autouse=True)
def active_skills():
    # Save current registry
    saved = dict(skillbase.ACTIVE_SKILLS)
    # Reset for test
    skillbase.ACTIVE_SKILLS.clear()
    yield
    # Restore
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(saved)


# Test use_skill with an active skill
def test_01_use_skill():
    # Seed registry
    skill = Skill("demo", "A demo skill", "# Demo\n\nInstructions.")
    skillbase.ACTIVE_SKILLS["demo"] = skill
    # Load the skill
    result = skilltools.use_skill("demo")
    # Check results
    assert result["success"] is True
    assert result["name"] == "demo"
    assert "Instructions." in result["content"]


# Test use_skill with an unknown skill
def test_02_use_skill_unknown():
    # Seed registry with one skill
    skill = Skill("demo", "A demo skill", "# Demo\n\nInstructions.")
    skillbase.ACTIVE_SKILLS["demo"] = skill
    # Attempt to load a different skill
    result = skilltools.use_skill("nope")
    # Check results
    assert result["success"] is False
    assert result["available_skills"] == ["demo"]
