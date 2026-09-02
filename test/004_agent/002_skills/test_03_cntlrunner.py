
# Standard library

# Third-party
import testutils

# Local imports
from cape.agent import skills
from cape.agent.skills import cntlrunner


# Test the built-in skill registry
def test_01_builtin_skills():
    # Check registry and skill definition
    assert "cntl-runner" in skills.BUILTIN_SKILLS
    skill = skills.BUILTIN_SKILLS["cntl-runner"]
    assert skill.tools == ["run_cntl_methods"]
    assert skill.description
    assert skill.content
    # Skill appears in "medium" and "full" skill sets
    assert "cntl-runner" in skills.SKILL_SETS["medium"]
    assert "cntl-runner" in skills.SKILL_SETS["full"]
    assert "cntl-runner" not in skills.SKILL_SETS["none"]
    assert "cntl-runner" not in skills.SKILL_SETS["low"]


# Test tool registration
def test_02_tool_registered():
    # Tool wired to module function
    assert cntlrunner.TOOLS["run_cntl_methods"] is cntlrunner.run_cntl_methods
    # Schema present
    names = {s["function"]["name"] for s in cntlrunner.TOOL_SCHEMAS}
    assert "run_cntl_methods" in names


# Test validation of *calls* input
def test_03_bad_calls():
    # Not a list
    result = cntlrunner.run_cntl_methods(calls="nope")
    assert result["success"] is False
    # Empty list
    result = cntlrunner.run_cntl_methods(calls=[])
    assert result["success"] is False
    # Entry not a dict
    result = cntlrunner.run_cntl_methods(calls=["nope"])
    assert result["success"] is False
    assert result["calls_completed"] == 0


# Test whitelist rejection before reading any JSON file
def test_04_whitelist_reject():
    # Unknown method, and a file that does not exist; whitelist check
    # must come first
    result = cntlrunner.run_cntl_methods(
        f="no-such-file.json",
        calls=[{"method": "rm_cases"}])
    assert result["success"] is False
    assert result["calls_completed"] == 0
    assert "rm_cases" in result["error"]
    # Allowed methods listed for the model
    assert "GetIndices" in result["allowed_methods"]
    assert len(result["allowed_methods"]) == len(cntlrunner.METHOD_WHITELIST)


# Test running two sequential methods on a real JSON file
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_05_run_cntl_methods():
    # Run two sequential calls
    result = cntlrunner.run_cntl_methods(
        f="cape.json",
        solver="cfdx",
        calls=[
            {"method": "get_runmatrix_keys"},
            {"method": "getval", "args": ["Mach", 0]},
        ])
    # Check overall results
    assert result["success"] is True
    assert result["calls_completed"] == 2
    # Check first call (run matrix key names)
    r0 = result["results"][0]
    assert r0["method"] == "get_runmatrix_keys"
    assert r0["success"] is True
    assert "Mach" in r0["result"]
    # Check second call (first case's Mach from matrix.csv)
    r1 = result["results"][1]
    assert r1["success"] is True
    assert r1["result"] == 0.5


# Test that NumPy results are JSON-normalized
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_06_numpy_result():
    # GetIndices returns a NumPy array
    result = cntlrunner.run_cntl_methods(
        f="cape.json",
        solver="cfdx",
        calls=[{"method": "GetIndices"}])
    # Check results
    assert result["success"] is True
    r0 = result["results"][0]
    assert r0["success"] is True
    # Converted to a list of ints
    assert isinstance(r0["result"], list)
    assert r0["result"] == list(range(len(r0["result"])))


# Test per-call failure does not abort remaining calls
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_07_continue_on_error():
    # Second call will fail (bad key name); third should still run
    result = cntlrunner.run_cntl_methods(
        f="cape.json",
        solver="cfdx",
        calls=[
            {"method": "getval", "args": ["Mach", 0]},
            {"method": "getval", "args": ["no-such-key", 0]},
            {"method": "getval", "args": ["Mach", 1]},
        ])
    # Overall result failed but all calls ran
    assert result["success"] is False
    assert result["calls_completed"] == 3
    # First call ok
    assert result["results"][0]["success"] is True
    # Second call failed with an error message
    r1 = result["results"][1]
    assert r1["success"] is False
    assert "error" in r1
    # Third call still ran
    r2 = result["results"][2]
    assert r2["success"] is True
    assert r2["result"] == 0.5
