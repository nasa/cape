
# Third-party
import pytest
import testutils

# Local imports
from cape.agent import skills
from cape.agent.skills import skillbase, skilltools, usertools


# Save and restore usertools module state around each test
@pytest.fixture(autouse=True)
def usertools_state():
    # Save current state
    rootdir = usertools.ROOT_DIR
    tooldir = usertools.TOOL_DIR_NAME
    registry = dict(usertools.TOOL_REGISTRY)
    # Reset for test: use cwd, empty discovery cache
    usertools.ROOT_DIR = None
    usertools.TOOL_DIR_NAME = "tools"
    usertools.TOOL_REGISTRY.clear()
    yield
    # Restore
    usertools.ROOT_DIR = rootdir
    usertools.TOOL_DIR_NAME = tooldir
    usertools.TOOL_REGISTRY.clear()
    usertools.TOOL_REGISTRY.update(registry)


# Save and restore ACTIVE_SKILLS around each test
@pytest.fixture(autouse=True)
def active_skills():
    # Save current registry
    saved = dict(skillbase.ACTIVE_SKILLS)
    yield
    # Restore
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(saved)


# Test the built-in skill registry
def test_01_builtin_skill():
    # Check registry and skill definition
    assert "user-tools" in skills.BUILTIN_SKILLS
    skill = skills.BUILTIN_SKILLS["user-tools"]
    assert skill.tools == ["discover_user_tools", "run_user_tool"]
    assert skill.description
    assert skill.content
    # Skill appears in "medium" and "full" skill sets
    assert "user-tools" in skills.SKILL_SETS["medium"]
    assert "user-tools" in skills.SKILL_SETS["full"]
    assert "user-tools" not in skills.SKILL_SETS["none"]
    assert "user-tools" not in skills.SKILL_SETS["low"]


# Test use_skill with the built-in "user-tools" skill
def test_02_use_skill():
    # Seed registry with the built-in skills
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(skills.BUILTIN_SKILLS)
    # Load the skill
    result = skilltools.use_skill("user-tools")
    # Check results
    assert result["success"] is True
    assert result["name"] == "user-tools"
    assert "discover_user_tools" in result["content"]


# Test discovery of the real script in the tools/ fixture folder
@testutils.run_sandbox(__file__, copydirs=["tools"])
def test_03_discover():
    # Discover scripts in sandbox's tools/ folder
    result = usertools.discover_user_tools()
    # Check results
    assert result["success"] is True
    names = [tool["name"] for tool in result["tools"]]
    assert names == ["autocommit"]
    assert result["tools"][0]["file"] == "autocommit.py"
