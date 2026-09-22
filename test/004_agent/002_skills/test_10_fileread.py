
# Third-party
import pytest

# Local imports
from cape.agent import skills
from cape.agent.skills import fileedit, fileread, skillbase, skilltools


# Save and restore fileread module state around each test
@pytest.fixture(autouse=True)
def fileread_state():
    # Save current state
    rootdir = fileread.ROOT_DIR
    yield
    # Restore
    fileread.ROOT_DIR = rootdir


# Create a sample repo with readable files
@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Create text files, including one in a subfolder and a binary
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "data.csv").write_text("Mach,Alpha\n0.2,1.1\n")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\xff\xfe")
    # Point the skills at this repo and run relative to it
    monkeypatch.setattr(fileread, "ROOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# Test the built-in skill registry
def test_01_builtin_skill():
    # Check registry and skill definition
    assert "file-reader" in skills.BUILTIN_SKILLS
    skill = skills.BUILTIN_SKILLS["file-reader"]
    assert skill.tools == ["read_file"]
    assert skill.description
    assert skill.content
    # Read-only skill is available to medium-capability models, too
    assert "file-reader" in skills.SKILL_SETS["full"]
    assert "file-reader" in skills.SKILL_SETS["medium"]
    assert "file-reader" not in skills.SKILL_SETS["low"]
    assert "file-reader" not in skills.SKILL_SETS["none"]


# Test tool registration
def test_02_tools_registered():
    # The file-editor skill reuses the same read function
    assert fileread.TOOLS["read_file"] is fileread.read_file
    assert fileedit.TOOLS["read_file"] is fileread.read_file
    # Schemas present
    names = {s["function"]["name"] for s in fileread.TOOL_SCHEMAS}
    assert names == {"read_file"}


# Test use_skill with the built-in "file-reader" skill
def test_03_use_skill():
    # Seed registry with the built-in skills
    skillbase.ACTIVE_SKILLS.clear()
    skillbase.ACTIVE_SKILLS.update(skills.BUILTIN_SKILLS)
    # Load the skill
    result = skilltools.use_skill("file-reader")
    # Check results
    assert result["success"] is True
    assert result["name"] == "file-reader"
    assert "read_file" in result["content"]


# Test reading files with no regard for the edit allow-list
def test_04_read_any_repo_file(repo):
    # Read it
    result = fileread.read_file("notes.txt")
    # Check results
    assert result["success"] is True
    assert result["file"] == "notes.txt"
    assert result["n_lines"] == 3
    assert result["truncated"] is False
    assert result["content"] == "1: alpha\n2: beta\n3: gamma"
    # A file that would never match an edit pattern is still readable
    result = fileread.read_file("sub/data.csv")
    assert result["success"] is True
    assert result["content"] == "1: Mach,Alpha\n2: 0.2,1.1"


# Test rejections
def test_05_rejections(repo):
    # Absolute path outside the root folder
    result = fileread.read_file("/etc/hostname")
    assert result["success"] is False
    assert "outside the repo root folder" in result["error"]
    # Missing file
    result = fileread.read_file("no-such-file.txt")
    assert result["success"] is False
    assert "No such file" in result["error"]
    # Bad argument type
    result = fileread.read_file(None)
    assert result["success"] is False
    # Binary file
    result = fileread.read_file("bin.dat")
    assert result["success"] is False
    assert "not a valid text file" in result["error"]


# Test the file-size limit
def test_06_size_limit(repo):
    # A file at the size limit is readable
    nbytes = fileread.MAX_READ_BYTES
    (repo / "at_limit.txt").write_bytes(b"a" * nbytes)
    result = fileread.read_file("at_limit.txt")
    assert result["success"] is True
    # One byte over the limit is rejected
    (repo / "over_limit.txt").write_bytes(b"a" * (nbytes + 1))
    result = fileread.read_file("over_limit.txt")
    assert result["success"] is False
    assert "too large" in result["error"]
