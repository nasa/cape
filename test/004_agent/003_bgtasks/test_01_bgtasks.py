
# Standard library
import json
import os
import sys
import time

# Third-party
import testutils

# Local imports
from cape.agent import bgtasks
from cape.agent.agentcntl import _strip_background
from cape.agent.tools import cfdxtools


# Time (sec) to wait for a background task
TASK_TIMEOUT = 60.0


# Wait for one task and return newly finished tasks
def _wait_one(task):
    t0 = time.time()
    while time.time() - t0 < TASK_TIMEOUT:
        finished = bgtasks.poll_finished([task])
        if finished:
            return finished
        time.sleep(0.05)
    raise TimeoutError("background task did not finish")


# Use a local task folder inside the sandbox
class _taskdir:
    def __enter__(self):
        os.mkdir("tasks")
        bgtasks.TASK_DIR = os.path.join(os.getcwd(), "tasks")
        return bgtasks.TASK_DIR

    def __exit__(self, *exc):
        bgtasks.TASK_DIR = None
        return False


# Test stripping of trailing "&" from typed commands
def test_01_strip_background():
    # No marker
    cmdlist, bg = _strip_background(["cape", "c", "-I", "3"])
    assert bg is False
    assert cmdlist == ["cape", "c", "-I", "3"]
    # Marker as its own token
    cmdlist, bg = _strip_background(["cape", "report", "-I", "5:8", "&"])
    assert bg is True
    assert cmdlist == ["cape", "report", "-I", "5:8"]
    # Marker attached to final token
    cmdlist, bg = _strip_background(["cape", "report", "-I", "5:8&"])
    assert bg is True
    assert cmdlist == ["cape", "report", "-I", "5:8"]
    # Marker only
    cmdlist, bg = _strip_background(["&"])
    assert bg is True
    assert cmdlist == []


# Test launching and reaping a CLI background task
@testutils.run_sandbox(__file__)
def test_02_cli_task():
    with _taskdir():
        # Launch a trivial command
        task = bgtasks.launch_cli_task(
            1, [sys.executable, "-c", "print('marker')"])
        # Task description includes the command
        assert "python" in task.describe()
        # Wait for completion
        finished = _wait_one(task)
        assert len(finished) == 1
        task_out, result = finished[0]
        assert task_out is task
        assert task.reaped is True
        # Synthesized result from exit code and log file
        assert result["success"] is True
        assert result["returncode"] == 0
        assert "marker" in result["stdout"]
        # Second poll returns nothing new
        assert bgtasks.poll_finished([task]) == []
        # Log file contains the output
        with open(task.logfile) as fp:
            assert "marker" in fp.read()
        # Terminal and history notes
        note = bgtasks.format_completion_note(task, result)
        assert "task 1" in note
        assert "exit 0" in note
        record = bgtasks.format_history_record(task, result)
        assert "Background task 1 completed" in record
        assert "stdout" not in record


# Test a failing CLI background task
@testutils.run_sandbox(__file__)
def test_03_cli_task_failure():
    with _taskdir():
        # Launch a command that fails
        task = bgtasks.launch_cli_task(
            2, [sys.executable, "-c", "import sys; sys.exit(3)"])
        finished = _wait_one(task)
        _, result = finished[0]
        assert result["success"] is False
        assert result["returncode"] == 3
        # Completion note flags the failure
        note = bgtasks.format_completion_note(task, result)
        assert "exit 3" in note


# Test a tool task run by a real child process
@testutils.run_sandbox(__file__)
def test_04_tool_task():
    with _taskdir():
        # Launch the getcwd system tool in a child process
        task = bgtasks.launch_tool_task(5, "getcwd", {})
        assert task.tool_name == "getcwd"
        assert task.describe() == "getcwd(background)"
        # Wait for completion
        finished = _wait_one(task)
        _, result = finished[0]
        # Result read back from child's JSON file
        assert result["success"] is True
        assert result["result"] == os.getcwd()
        assert os.path.isfile(task.resultfile)
        assert os.path.isfile(task.specfile)


# Test the child-side main() function directly
@testutils.run_sandbox(__file__)
def test_05_child_main():
    with _taskdir() as taskdir:
        # Spec for a working tool
        resultfile = os.path.join(taskdir, "ok.result.json")
        spec = {"tool": "getcwd", "kwargs": {}, "resultfile": resultfile}
        specfile = os.path.join(taskdir, "ok.spec.json")
        with open(specfile, "w") as fp:
            json.dump(spec, fp)
        # Run in-process
        ierr = bgtasks.main([specfile])
        assert ierr == 0
        with open(resultfile) as fp:
            result = json.load(fp)
        assert result["success"] is True
        assert result["result"] == os.getcwd()
        # Spec for an unknown tool
        badfile = os.path.join(taskdir, "bad.result.json")
        spec = {"tool": "no_such_tool", "kwargs": {}, "resultfile": badfile}
        specfile = os.path.join(taskdir, "bad.spec.json")
        with open(specfile, "w") as fp:
            json.dump(spec, fp)
        ierr = bgtasks.main([specfile])
        assert ierr != 0
        with open(badfile) as fp:
            result = json.load(fp)
        assert result["success"] is False
        assert "unknown tool" in result["error"]


# Test the background option on the cape_report tool schema
def test_06_schema():
    # Only cape_report is backgroundable
    assert "cape_report" in cfdxtools.BACKGROUNDABLE_TOOLS
    # Find its schema
    for schema in cfdxtools.TOOL_SCHEMAS:
        if schema["function"]["name"] == "cape_report":
            props = schema["function"]["parameters"]["properties"]
            assert "background" in props
            break
    else:
        raise AssertionError("no cape_report schema found")
