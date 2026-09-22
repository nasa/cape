
# Standard library
import json

# Third-party
import pytest

# Local imports
from cape.agent import agentcntl, skills
from cape.agent.skills import fileedit, fileread, fixjson, skillbase
from cape.agent.skills import skilltools


# Save and restore module state around each test
@pytest.fixture(autouse=True)
def skill_state():
    # Save current state
    rootdir = fileread.ROOT_DIR
    patterns = list(fileedit.ALLOW_PATTERNS)
    registered = list(agentcntl.EDIT_FILE_ALLOW_LIST)
    # Reset for test
    fileread.ROOT_DIR = None
    fileedit.ALLOW_PATTERNS.clear()
    agentcntl.EDIT_FILE_ALLOW_LIST.clear()
    yield
    # Restore
    fileread.ROOT_DIR = rootdir
    fileedit.ALLOW_PATTERNS.clear()
    fileedit.ALLOW_PATTERNS.extend(patterns)
    agentcntl.EDIT_FILE_ALLOW_LIST.clear()
    agentcntl.EDIT_FILE_ALLOW_LIST.extend(registered)


# Save and restore ACTIVE_SKILLS around each test
@pytest.fixture(autouse=True)
def active_skills():
    # Save current registry
    saved = dict(skillbase.ACTIVE_SKILLS)
    yield
    # Restore
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(saved)


# Create a sample repo with valid and invalid JSON files
@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Valid JSON file, with a CAPE-style '//' comment
    (tmp_path / "good.json").write_text(
        '{\n    // a comment\n    "a": 2,\n    "b": [1, 2]\n}\n')
    # Missing comma between keys
    (tmp_path / "comma.json").write_text(
        '{\n    "a": 2\n    "b": 3\n}\n')
    # Missing closing brace: parser reaches end of file
    (tmp_path / "brace.json").write_text(
        '{\n    "a": {\n        "b": 3\n    }\n')
    # Trailing comma
    (tmp_path / "trailing.json").write_text(
        '{\n    "a": 2,\n}\n')
    # A non-JSON file
    (tmp_path / "notes.txt").write_text("not json\n")
    # Point the file skills at this repo; run relative to it
    monkeypatch.setattr(fileread, "ROOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# Test the built-in skill registry
def test_01_builtin_skill():
    # Check registry and skill definition
    assert "fix-json" in skills.BUILTIN_SKILLS
    skill = skills.BUILTIN_SKILLS["fix-json"]
    assert skill.tools == ["validate_json"]
    assert skill.description
    assert skill.content
    # Skill appears only in the "full" skill set (needs file-editor)
    assert "fix-json" in skills.SKILL_SETS["full"]
    assert "fix-json" not in skills.SKILL_SETS["medium"]
    assert "fix-json" not in skills.SKILL_SETS["low"]
    assert "fix-json" not in skills.SKILL_SETS["none"]


# Test tool registration
def test_02_tools_registered():
    # Tool wired to module function
    assert fixjson.TOOLS["validate_json"] is fixjson.validate_json
    # Schema present
    names = {s["function"]["name"] for s in fixjson.TOOL_SCHEMAS}
    assert names == {"validate_json"}


# Test use_skill with the built-in "fix-json" skill
def test_03_use_skill():
    # Seed registry with the built-in skills
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(skills.BUILTIN_SKILLS)
    # Load the skill
    result = skilltools.use_skill("fix-json")
    # Check results
    assert result["success"] is True
    assert result["name"] == "fix-json"
    assert "validate_json" in result["content"]


# Test validating a valid file, including C-style comments
def test_04_valid(repo):
    # Validate the file
    result = fixjson.validate_json("good.json")
    # Check results; comments are stripped, so file is valid
    assert result["success"] is True
    assert result["valid"] is True
    assert result["file"] == "good.json"
    assert result["registered"] is True
    # The file is registered with the session allow-list
    assert agentcntl.EDIT_FILE_ALLOW_LIST == ["good.json"]
    # Validating again does not duplicate the registration
    result = fixjson.validate_json("good.json")
    assert result["registered"] is False
    assert agentcntl.EDIT_FILE_ALLOW_LIST == ["good.json"]


# Test error reporting for a missing comma mid-file
def test_05_missing_comma(repo):
    # Validate the file
    result = fixjson.validate_json("comma.json")
    # Check results
    assert result["success"] is True
    assert result["valid"] is False
    assert result["error"] == "Expecting ',' delimiter"
    assert result["lineno"] == 3
    assert result["eof"] is False
    # Context shows the surrounding lines with the error marked
    assert "--> 3:" in result["context"]
    assert '2:     "a": 2' in result["context"]


# Test error reporting for a missing closing brace
def test_06_eof_error(repo):
    # Validate the file; parser reaches end of file
    result = fixjson.validate_json("brace.json")
    # Check results
    assert result["success"] is True
    assert result["valid"] is False
    assert result["eof"] is True
    assert "missing a closing" in result["hint"]
    # Error line is clamped to the last line of the file
    assert "--> 4:" in result["context"]


# Test error reporting for a trailing comma
def test_07_trailing_comma(repo):
    # Validate the file
    result = fixjson.validate_json("trailing.json")
    # Check results
    assert result["success"] is True
    assert result["valid"] is False
    assert result["error"] == (
        "Expecting property name enclosed in double quotes")
    assert result["eof"] is False


# Test that registration enables file-editor access
def test_08_registration_enables_editing(repo):
    # No static patterns; file is initially off-limits to editing
    fileedit.ALLOW_PATTERNS.clear()
    result = fileedit.edit_file("comma.json", '"a": 2\n', '"a": 2,\n')
    assert result["success"] is False
    assert "not in the edit allow-list" in result["error"]
    # But it can always be read
    result = fileedit.read_file("comma.json")
    assert result["success"] is True
    # Validating registers the file
    result = fixjson.validate_json("comma.json")
    assert result["valid"] is False
    # Now the file-editor tools accept it
    # Fix the missing comma
    result = fileedit.edit_file("comma.json", '"a": 2\n', '"a": 2,\n')
    assert result["success"] is True
    # The file now validates
    result = fixjson.validate_json("comma.json")
    assert result["valid"] is True
    # Registration list contains just this file
    assert agentcntl.EDIT_FILE_ALLOW_LIST == ["comma.json"]


# Test tool-level failures
def test_09_failures(repo):
    # Non-JSON file names are rejected (and not registered)
    result = fixjson.validate_json("notes.txt")
    assert result["success"] is False
    assert ".json" in result["error"]
    assert agentcntl.EDIT_FILE_ALLOW_LIST == []
    # Missing JSON file
    result = fixjson.validate_json("nope.json")
    assert result["success"] is False
    assert "No such file" in result["error"]
    # File outside the root folder is rejected
    result = fixjson.validate_json("/etc/hostname.json")
    assert result["success"] is False
    assert "outside the repo root folder" in result["error"]


# Test that the full workflow fixes a file (no LLM, direct tool calls)
def test_10_workflow(repo):
    # Validate, fix, and revalidate the brace error
    result = fixjson.validate_json("brace.json")
    assert result["valid"] is False
    result = fileedit.edit_file("brace.json", "    }\n", "    }\n}\n")
    assert result["success"] is True
    result = fixjson.validate_json("brace.json")
    assert result["valid"] is True
    # The file reads as JSON and preserves its values
    with open(repo / "brace.json") as fp:
        assert json.load(fp) == {"a": {"b": 3}}
