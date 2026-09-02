
# Standard library
import os

# Third-party
import pytest

# Local imports
import cape.agent.options as agentopts
from cape.agent.skills import usertools


# Script with a __main__ guard and an RST docstring
SCRIPT_GOOD = '''\
#!/usr/bin/env python3
r"""
:mod:`goodtool`: Sample user tool for testing
=============================================

Body of docstring.
"""
import os
import sys

if __name__ == "__main__":
    print("GOODTOOL_OUTPUT")
    print("ARGS:" + ",".join(sys.argv[1:]))
    print("CWD:" + os.getcwd())
'''

# Script with a guard but no docstring
SCRIPT_NODOC = '''\
import sys

if __name__ == '__main__':
    print("NODOC_OUTPUT")
    sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
'''

# Script without a guard; should not be discovered
SCRIPT_NOGUARD = '''\
print("IMPORTED")
def func():
    pass
'''

# Python 2 script that fails ast.parse but has a guard
SCRIPT_LEGACY = '''\
if __name__ == '__main__':
    print 'legacy'
'''

# Script that sleeps, for timeout testing
SCRIPT_SLEEP = '''\
import time

if __name__ == "__main__":
    time.sleep(5)
'''


# Create a sample repo with a tools/ folder; reset module state
@pytest.fixture
def toolrepo(tmp_path, monkeypatch):
    # Create tools folder and sample scripts
    fdir = tmp_path / "tools"
    fdir.mkdir()
    (fdir / "goodtool.py").write_text(SCRIPT_GOOD)
    (fdir / "nodoc.py").write_text(SCRIPT_NODOC)
    (fdir / "noguard.py").write_text(SCRIPT_NOGUARD)
    (fdir / "legacy.py").write_text(SCRIPT_LEGACY)
    (fdir / "_private.py").write_text(SCRIPT_GOOD)
    (fdir / "__init__.py").write_text(SCRIPT_GOOD)
    (fdir / "notes.txt").write_text(SCRIPT_GOOD)
    # Point the skill at this repo
    monkeypatch.setattr(usertools, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(usertools, "TOOL_DIR_NAME", "tools")
    usertools.TOOL_REGISTRY.clear()
    yield tmp_path
    usertools.TOOL_REGISTRY.clear()


# Test discovery filtering and description extraction
def test_01_discover(toolrepo):
    result = usertools.discover_user_tools()
    # Overall success
    assert result["success"] is True
    assert result["cached"] is False
    # Which scripts were discovered
    names = [tool["name"] for tool in result["tools"]]
    assert names == ["goodtool", "legacy", "nodoc"]
    # Description extraction with RST cleanup
    tools = {tool["name"]: tool for tool in result["tools"]}
    assert tools["goodtool"]["description"] == \
        "goodtool: Sample user tool for testing"
    assert tools["nodoc"]["description"] == ""
    assert tools["legacy"]["description"] == ""


# Test that results are cached and refreshable
def test_02_cache(toolrepo):
    # First discovery
    usertools.discover_user_tools()
    # Add another script
    (toolrepo / "tools" / "newtool.py").write_text(SCRIPT_NODOC)
    # Second discovery uses the cache
    result = usertools.discover_user_tools()
    assert result["cached"] is True
    assert "newtool" not in usertools.TOOL_REGISTRY
    # Refresh picks up the new script
    result = usertools.discover_user_tools(refresh=True)
    assert result["cached"] is False
    assert "newtool" in usertools.TOOL_REGISTRY


# Test discovery with no tools folder
def test_03_no_tooldir(tmp_path, monkeypatch):
    monkeypatch.setattr(usertools, "ROOT_DIR", str(tmp_path))
    usertools.TOOL_REGISTRY.clear()
    result = usertools.discover_user_tools()
    assert result["success"] is True
    assert result["tools"] == []


# Test running a script with args and cwd
def test_04_run(toolrepo):
    result = usertools.run_user_tool(
        "goodtool", argv=["-I", "1:5", "--force"])
    assert result["success"] is True
    assert result["returncode"] == 0
    assert result["argv"] == ["-I", "1:5", "--force"]
    # Script output captured
    assert "GOODTOOL_OUTPUT" in result["stdout"]
    assert "ARGS:-I,1:5,--force" in result["stdout"]
    # Script ran in the repo root
    cwd_line = [
        line for line in result["stdout"].splitlines()
        if line.startswith("CWD:")
    ][0]
    assert os.path.realpath(cwd_line[4:]) == os.path.realpath(str(toolrepo))


# Test running with the ".py" extension and a nonzero exit code
def test_05_run_exitcode(toolrepo):
    result = usertools.run_user_tool("nodoc.py", argv=["3"])
    assert result["success"] is False
    assert result["returncode"] == 3
    assert "NODOC_OUTPUT" in result["stdout"]


# Test unknown and invalid tool names
def test_06_run_bad_names(toolrepo):
    usertools.discover_user_tools()
    # Unknown tool
    result = usertools.run_user_tool("nope")
    assert result["success"] is False
    assert "available_tools" in result
    # Path components rejected
    result = usertools.run_user_tool("../goodtool")
    assert result["success"] is False
    assert "available_tools" not in result
    # Empty name
    result = usertools.run_user_tool("")
    assert result["success"] is False


# Test that the guard is re-validated before running
def test_07_guard_revalidation(toolrepo):
    usertools.discover_user_tools()
    # Remove the guard after discovery
    (toolrepo / "tools" / "goodtool.py").write_text(SCRIPT_NOGUARD)
    result = usertools.run_user_tool("goodtool")
    assert result["success"] is False
    assert "guard" in result["error"]
    # The script is pruned from the registry
    assert "goodtool" not in usertools.TOOL_REGISTRY


# Test the timeout option
def test_08_timeout(toolrepo):
    (toolrepo / "tools" / "sleeper.py").write_text(SCRIPT_SLEEP)
    usertools.discover_user_tools(refresh=True)
    result = usertools.run_user_tool("sleeper", timeout=1)
    assert result["success"] is False
    assert "timed out" in result["error"]


# Test the ToolDir option default
def test_09_tooldir_option():
    opts = agentopts.AgentOpts()
    assert opts.get_opt("ToolDir") == "tools"
