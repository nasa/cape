
# Standard library
import os

# Third-party
import pytest
import testutils

# Local imports
from cape.agent import skills
from cape.agent.skills import fileedit, fileread, skillbase, skilltools
from cape.cfdx import cli


# Save and restore fileedit module state around each test
@pytest.fixture(autouse=True)
def fileedit_state():
    # Save current state
    rootdir = fileread.ROOT_DIR
    patterns = list(fileedit.ALLOW_PATTERNS)
    # Reset for test: use cwd, empty allow-list
    fileread.ROOT_DIR = None
    fileedit.ALLOW_PATTERNS.clear()
    yield
    # Restore
    fileread.ROOT_DIR = rootdir
    fileedit.ALLOW_PATTERNS.clear()
    fileedit.ALLOW_PATTERNS.extend(patterns)


# Save and restore ACTIVE_SKILLS around each test
@pytest.fixture(autouse=True)
def active_skills():
    # Save current registry
    saved = dict(skillbase.ACTIVE_SKILLS)
    yield
    # Restore
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(saved)


# Create a sample repo with editable and non-editable files
@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Create text files, including one with duplicate lines
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "dup.txt").write_text("same\nsame\n")
    (tmp_path / "secret.md").write_text("not editable\n")
    # Create a script in a subfolder
    fdir = tmp_path / "tools"
    fdir.mkdir()
    (fdir / "script.py").write_text("print('hi')\n")
    # Point the skills at this repo and run relative to it
    monkeypatch.setattr(fileread, "ROOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # Set the static allow-list (no CAPE JSON file here)
    fileedit.ALLOW_PATTERNS.clear()
    fileedit.ALLOW_PATTERNS.extend(["*.txt", "tools/*.py"])
    yield tmp_path


# Test the built-in skill registry
def test_01_builtin_skill():
    # Check registry and skill definition
    assert "file-editor" in skills.BUILTIN_SKILLS
    skill = skills.BUILTIN_SKILLS["file-editor"]
    assert skill.tools == ["list_editable_files", "read_file", "edit_file"]
    assert skill.description
    assert skill.content
    # Skill appears only in the "full" skill set
    assert "file-editor" in skills.SKILL_SETS["full"]
    assert "file-editor" not in skills.SKILL_SETS["medium"]
    assert "file-editor" not in skills.SKILL_SETS["low"]
    assert "file-editor" not in skills.SKILL_SETS["none"]


# Test tool registration
def test_02_tools_registered():
    # Tools wired to module functions
    assert fileedit.TOOLS["list_editable_files"] is fileedit.list_editable_files
    assert fileedit.TOOLS["read_file"] is fileedit.read_file
    assert fileedit.TOOLS["edit_file"] is fileedit.edit_file
    # Schemas present
    names = {s["function"]["name"] for s in fileedit.TOOL_SCHEMAS}
    assert names == {"list_editable_files", "read_file", "edit_file"}


# Test use_skill with the built-in "file-editor" skill
def test_03_use_skill():
    # Seed registry with the built-in skills
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(skills.BUILTIN_SKILLS)
    # Load the skill
    result = skilltools.use_skill("file-editor")
    # Check results
    assert result["success"] is True
    assert result["name"] == "file-editor"
    assert "edit_file" in result["content"]


# Test listing editable files in a sample repo
def test_04_list_editable_files(repo):
    # List the files
    result = fileedit.list_editable_files()
    # Check results
    assert result["success"] is True
    assert result["patterns"] == ["*.txt", "tools/*.py"]
    # No CAPE JSON file in this repo, so no Cntl-provided files
    assert result["cntl_files"] == []
    # Check matched files (secret.md not matched)
    assert result["files"] == [
        "dup.txt", "notes.txt", "tools/script.py"]


# Test reading an allow-listed file
def test_05_read_file(repo):
    # Read it
    result = fileedit.read_file("notes.txt")
    # Check results
    assert result["success"] is True
    assert result["file"] == "notes.txt"
    assert result["n_lines"] == 3
    assert result["truncated"] is False
    # Lines are prefixed with their numbers
    assert result["content"] == "1: alpha\n2: beta\n3: gamma"


# Test edit rejection (but not read rejection) outside the allow-list
def test_06_not_allowed(repo):
    # File exists but does not match the patterns: editing rejected
    result = fileedit.edit_file("secret.md", "not", "yes")
    assert result["success"] is False
    assert "not in the edit allow-list" in result["error"]
    assert result["allowed_patterns"] == ["*.txt", "tools/*.py"]
    # Reading the same file is fine: allow-list only governs edits
    result = fileedit.read_file("secret.md")
    assert result["success"] is True
    assert result["content"] == "1: not editable"
    # Path escaping the root folder
    result = fileedit.edit_file(
        os.path.join("..", "..", "etc", "hostname"), "x", "y")
    assert result["success"] is False
    assert "outside the repo root folder" in result["error"]
    # Absolute path outside the root folder
    result = fileedit.read_file("/etc/hostname")
    assert result["success"] is False
    assert "outside the repo root folder" in result["error"]


# Test successful and failing edits
def test_07_edit_file(repo):
    # Apply an edit
    result = fileedit.edit_file("notes.txt", "beta", "BETA")
    # Check results
    assert result["success"] is True
    assert result["file"] == "notes.txt"
    assert (repo / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"
    # Result includes a unified diff
    assert "--- a/notes.txt" in result["diff"]
    assert "+++ b/notes.txt" in result["diff"]
    assert "-beta" in result["diff"]
    assert "+BETA" in result["diff"]
    # Editing with an empty replacement deletes the match
    result = fileedit.edit_file("notes.txt", "alpha\n", "")
    assert result["success"] is True
    assert (repo / "notes.txt").read_text() == "BETA\ngamma\n"
    # Non-unique search text is rejected
    result = fileedit.edit_file("dup.txt", "same", "other")
    assert result["success"] is False
    assert "occurs 2 times" in result["error"]
    # Missing search text is rejected
    result = fileedit.edit_file("notes.txt", "delta", "zeta")
    assert result["success"] is False
    assert "not found" in result["error"]
    # Missing file is rejected
    result = fileedit.edit_file("no-such-file.txt", "x", "y")
    assert result["success"] is False
    assert "No such file" in result["error"]
    # Bad argument types are rejected
    result = fileedit.edit_file("notes.txt", "", "y")
    assert result["success"] is False
    result = fileedit.edit_file("notes.txt", "x", None)
    assert result["success"] is False
    result = fileedit.read_file(None)
    assert result["success"] is False


# Test truncation of large files
def test_08_read_file_truncated(repo):
    # Create a large file
    (repo / "big.txt").write_text(
        "".join(f"line{j + 1}\n" for j in range(3000)))
    # Read it
    result = fileedit.read_file("big.txt")
    # Check results
    assert result["success"] is True
    assert result["n_lines"] == 3000
    assert result["truncated"] is True
    content = result["content"]
    # Head, marker, and tail are all present
    assert content.startswith("1: line1\n")
    assert "[1000 lines omitted]" in content
    assert "1500: line1500" in content
    assert "1501: line1501" not in content
    assert "2501: line2501" in content
    assert content.endswith("3000: line3000")


# Test the Cntl-provided allow-list on a real CAPE JSON file
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_09_cntl_allowlist():
    # No static patterns; allow-list comes entirely from *Cntl*
    fileedit.ALLOW_PATTERNS.clear()
    # Read the CAPE control instance directly
    cntl = cli.read_cntl_cache("cape.json", solver="cfdx")
    # Check the Cntl-level allow-list: JSON file plus run-matrix CSV
    assert cntl.get_edit_allowlist() == ["cape.json", "matrix.csv"]
    # The skill's listing should pick them up, too
    result = fileedit.list_editable_files()
    assert result["success"] is True
    assert result["cntl_files"] == ["cape.json", "matrix.csv"]
    assert "cape.json" in result["files"]
    assert "matrix.csv" in result["files"]
    # The CAPE JSON file is editable with no static patterns
    result = fileedit.edit_file("cape.json", '"nProc": 4', '"nProc": 8')
    assert result["success"] is True
    assert '"nProc": 8' in open("cape.json").read()
    # So is the run-matrix file
    result = fileedit.edit_file("matrix.csv", "Mach", "mach")
    assert result["success"] is True
    assert "mach" in open("matrix.csv").read()
    # But other files in the sandbox are still off-limits
    result = fileedit.edit_file("notes.md", "x", "y")
    assert result["success"] is False
    assert "not in the edit allow-list" in result["error"]
