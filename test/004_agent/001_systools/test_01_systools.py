
# Standard library
import os

# Third-party
import testutils

# Local imports
import cape.agent.tools as agent_tools
from cape.agent.tools import systools


# Test registration of tools and schemas
def test_01_tools_registered():
    """Test that systools are registered in TOOLS and TOOL_SCHEMAS"""
    # Tools wired to systools functions
    assert agent_tools.TOOLS["chdir"] is systools.chdir
    assert agent_tools.TOOLS["getcwd"] is systools.getcwd
    # Schemas include system tools alongside CFD tools
    names = {s["function"]["name"] for s in agent_tools.TOOL_SCHEMAS}
    assert "chdir" in names
    assert "getcwd" in names
    assert "cape_c" in names
    # One schema per tool
    assert len(agent_tools.TOOL_SCHEMAS) == len(agent_tools.TOOLS)


# Test getcwd tool
def test_02_getcwd():
    """Test getcwd returns current folder"""
    result = systools.getcwd()
    assert result["success"] is True
    assert result["result"] == os.getcwd()


# Test chdir with non-string input
def test_03_chdir_invalid_input():
    """Test chdir rejects non-string input"""
    result = systools.chdir(123)
    assert result["success"] is False
    assert "TypeError" in result["error"]


# Test chdir with missing folder
def test_04_chdir_no_folder():
    """Test chdir rejects nonexistent folder"""
    result = systools.chdir("no-such-folder-here")
    assert result["success"] is False


# Test successful chdir
@testutils.run_sandbox(__file__)
def test_05_chdir():
    """Test chdir actually changes folder"""
    # Remember sandbox location
    sandbox = os.getcwd()
    # Create a subfolder to change into
    os.mkdir("work")
    # Change into it
    result = systools.chdir("work")
    assert result["success"] is True
    assert os.getcwd() == os.path.join(sandbox, "work")
    # Change back
    result = systools.chdir(sandbox)
    assert result["success"] is True
    assert os.getcwd() == sandbox
