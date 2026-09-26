"""Tests for semantic tools backed by a CAPE control instance."""

# Third-party
import testutils

# Local imports
from cape.agent.tools import cntltools


# Test registration and replacement of the old names-only tool
def test_01_tool_registered():
    assert (
        cntltools.TOOLS["describe_run_matrix_keys"]
        is cntltools.describe_run_matrix_keys)
    for names in cntltools.TOOL_SETS.values():
        assert "get_keys" not in names
    assert "describe_run_matrix_keys" in cntltools.TOOL_SETS["full"]


# Test effective definitions and bounded value summaries
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_02_describe_run_matrix_keys():
    result = cntltools.describe_run_matrix_keys(
        f="cape.json", key="M*", detail="full")
    assert result["success"] is True
    assert result["case_count"] > 0
    assert result["key_count"] == 1
    key = result["keys"][0]
    assert key["name"] == "Mach"
    assert key["type"] == "mach"
    assert key["value_type"] == "float"
    assert key["definition"]["Type"] == "mach"
    assert key["values"]["count"] == result["case_count"]
    assert key["values"]["min"] == 0.5
    assert key["values"]["max"] == 2.5


# Test compact mode, value suppression, and input validation
@testutils.run_sandbox(__file__, copyfiles=["cape.json", "matrix.csv"])
def test_03_description_options():
    result = cntltools.describe_run_matrix_keys(
        f="cape.json", key="alpha", include_values=False)
    assert result["success"] is True
    assert result["key_count"] == 1
    assert "definition" not in result["keys"][0]
    assert "values" not in result["keys"][0]
    result = cntltools.describe_run_matrix_keys(
        f="no-such-file.json", detail="verbose")
    assert result["success"] is False
    assert "detail" in result["error"]


# Test that value lists remain bounded for high-cardinality columns
def test_04_value_summary_bound():
    result = cntltools._summarize_values(range(20))
    assert result["count"] == 20
    assert result["unique_count"] == 20
    assert result["examples"] == list(range(8))
    assert result["examples_truncated"] is True
    assert result["min"] == 0
    assert result["max"] == 19
