r"""
:mod:`cape.agent.bgtasks`: Background tasks for the CAPE agent
==============================================================

This module provides background task support for the CAPE agentic
interface (see :mod:`cape.agent.agentcntl`). A *background task* is a
CAPE command, such as ``cape report -I 100:150``, that is run in a
child process so that the interactive agent session can continue with
other work. The main session checks for completed tasks each time it
displays a new prompt.

Two kinds of background task are supported:

*   **CLI tasks** are full command lines, for example when a user
    types ``cape report -I 100:150 &`` at the agent prompt. The
    trailing ``&`` requests backgrounding, following the usual shell
    convention.

*   **Tool tasks** are agent tool calls, e.g. ``cape_report()`` with
    its *background* option set. These are run by a child copy of
    this module (``python -m cape.agent.bgtasks``) so that the full
    tool result is once again available when the task finishes.

Each task redirects STDOUT and STDERR to a log file in the task folder
(``<CacheDir>/agent-tasks/`` by default, where *CacheDir* is from the
CAPE user configuration), has no STDIN, and is launched in its own
session so that Ctrl-C in the main session does not interrupt it.
Note that tasks share the file system with the main session; users
should avoid running a background task and a foreground command on the
same cases at the same time.
"""

# Standard library
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Optional

# Local imports
from . import agentutils
from .tools import cfdxtools, systools
from .tools.toolutils import _truncate_stdout
from .. import capeconfig
from ..ui.promptutils import sprintf_color


# Name of task folder inside the CAPE cache folder
TASK_DIR_NAME = "agent-tasks"
# Override for task folder (used mostly by tests)
TASK_DIR = None
# Prefix for colored terminal messages
BGTASK_PROMPT = sprintf_color("[background] ", ["italic", "purple"])


# Record and process handle for one background task
class BackgroundTask:
    r"""Record and process handle of one background task

    :Call:
        >>> task = BackgroundTask(task_id, cmdlist, tool_name=None)
    :Inputs:
        *task_id*: :class:`int`
            Task ID for this session, starting at ``1``
        *cmdlist*: :class:`list`\ [:class:`str`]
            Command being run by the child process
        *tool_name*: {``None``} | :class:`str`
            Name of agent tool being run; ``None`` for CLI tasks
    :Outputs:
        *task*: :class:`BackgroundTask`
            Record of one background task
    """
    # Attributes
    __slots__ = (
        "cmdlist",
        "logfile",
        "proc",
        "reaped",
        "resultfile",
        "specfile",
        "t0",
        "task_id",
        "tool_name",
    )

    # Initialize
    def __init__(
            self,
            task_id: int,
            cmdlist: list,
            tool_name: Optional[str] = None):
        #: :class:`int`
        #: Task ID for this session, starting at ``1``
        self.task_id = task_id
        #: :class:`list`\ [:class:`str`]
        #: Command being run by the child process
        self.cmdlist = cmdlist
        #: :class:`str` | ``None``
        #: Name of agent tool being run; ``None`` for CLI tasks
        self.tool_name = tool_name
        #: :class:`subprocess.Popen`
        #: Child process running this task
        self.proc = None
        #: :class:`bool`
        #: Whether this task has exited and been processed
        self.reaped = False
        #: :class:`str`
        #: Log file receiving the task's STDOUT and STDERR
        self.logfile = None
        #: :class:`str` | ``None``
        #: JSON file defining a tool task; ``None`` for CLI tasks
        self.specfile = None
        #: :class:`str` | ``None``
        #: JSON file of tool results; ``None`` for CLI tasks
        self.resultfile = None
        #: :class:`float`
        #: Time at which this task was launched
        self.t0 = time.time()

    # Short description of the command
    def describe(self) -> str:
        r"""Create a short text description of this task"""
        if self.tool_name is None:
            return shlex.join(self.cmdlist)
        else:
            return f"{self.tool_name}(background)"

    # Check exit status of child process
    def poll(self) -> Optional[int]:
        r"""Check exit status, ``None`` if still running"""
        return self.proc.poll()


# Get (and create) the task folder
def get_task_dir() -> str:
    r"""Get the folder for task logs, creating it if necessary

    By default this is ``agent-tasks/`` inside the CAPE *CacheDir*,
    but it can be overridden by setting the module-level *TASK_DIR*.

    :Call:
        >>> taskdir = get_task_dir()
    :Outputs:
        *taskdir*: :class:`str`
            Name of folder for background task files
    """
    # Check for override
    if TASK_DIR is not None:
        taskdir = TASK_DIR
    else:
        taskdir = os.path.join(
            capeconfig.get_cape_opt("CacheDir"), TASK_DIR_NAME)
    # Create if necessary
    os.makedirs(taskdir, exist_ok=True)
    # Output
    return taskdir


# Base file name for a task's files
def _task_fname(task_id: int, ext: str) -> str:
    return os.path.join(get_task_dir(), f"task-{task_id:04d}{ext}")


# Launch a background task for a shell command
def launch_cli_task(task_id: int, cmdlist: list) -> BackgroundTask:
    r"""Launch a command line as a background task

    :Call:
        >>> task = launch_cli_task(task_id, cmdlist)
    :Inputs:
        *task_id*: :class:`int`
            Task ID for this session, starting at ``1``
        *cmdlist*: :class:`list`\ [:class:`str`]
            Command to run in the child process
    :Outputs:
        *task*: :class:`BackgroundTask`
            Record of the newly launched task
    """
    # Initialize record
    task = BackgroundTask(task_id, cmdlist)
    # Log file for STDOUT and STDERR
    task.logfile = _task_fname(task_id, ".log")
    # Launch child process
    _popen(task)
    # Output
    return task


# Launch a background task for an agent tool
def launch_tool_task(
        task_id: int,
        tool_name: str,
        kwargs: dict) -> BackgroundTask:
    r"""Launch an agent tool call as a background task

    The tool is run by a child copy of this module (see :func:`main`)
    so that the full tool result is written to a JSON file and
    recovered by the main session once the task completes.

    :Call:
        >>> task = launch_tool_task(task_id, tool_name, kwargs)
    :Inputs:
        *task_id*: :class:`int`
            Task ID for this session, starting at ``1``
        *tool_name*: :class:`str`
            Name of agent tool, e.g. ``"cape_report"``
        *kwargs*: :class:`dict`
            Keyword arguments to the tool
    :Outputs:
        *task*: :class:`BackgroundTask`
            Record of the newly launched task
    """
    # JSON files for inputs and results
    specfile = _task_fname(task_id, ".spec.json")
    resultfile = _task_fname(task_id, ".result.json")
    # Write the tool's specification
    spec = {
        "tool": tool_name,
        "kwargs": kwargs,
        "resultfile": resultfile,
    }
    with open(specfile, "w") as fp:
        json.dump(spec, fp, cls=agentutils._NPEncoder)
    # Command to run child copy of this module
    cmdlist = [sys.executable, "-m", "cape.agent.bgtasks", specfile]
    # Initialize record
    task = BackgroundTask(task_id, cmdlist, tool_name=tool_name)
    # Save file names
    task.logfile = _task_fname(task_id, ".log")
    task.specfile = specfile
    task.resultfile = resultfile
    # Launch child process
    _popen(task)
    # Output
    return task


# Launch the child process for one task
def _popen(task: BackgroundTask):
    # Open the log file; child receives a copy of its descriptor
    with open(task.logfile, "w") as fout:
        # Launch child in its own session so that terminal Ctrl-C
        # does not reach it and with no access to parent STDIN
        task.proc = subprocess.Popen(
            task.cmdlist,
            stdin=subprocess.DEVNULL,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True)


# Check for newly completed tasks
def poll_finished(tasks: list) -> list:
    r"""Check for newly completed background tasks

    :Call:
        >>> finished = poll_finished(tasks)
    :Inputs:
        *tasks*: :class:`list`\ [:class:`BackgroundTask`]
            List of tasks launched during this session
    :Outputs:
        *finished*: :class:`list`\ [:class:`tuple`]
            Newly completed ``(task, result)`` pairs; *result* is a
            :class:`dict` in the style of a normal tool result
    """
    # List of (task, result) pairs completed since last check
    finished = []
    # Loop through tasks
    for task in tasks:
        # Skip already processed tasks
        if task.reaped:
            continue
        # Check for completion (also reaps the child process)
        returncode = task.poll()
        if returncode is None:
            continue
        # Save state and assemble result
        task.reaped = True
        finished.append((task, read_task_result(task, returncode)))
    # Output
    return finished


# Assemble a result dict for a completed task
def read_task_result(task: BackgroundTask, returncode: int) -> dict:
    r"""Assemble a tool-style result dict for a completed task

    For tool tasks the result written by the child process is read
    back; for other tasks (or a tool task whose child died early) a
    result is synthesized from the exit code and a tail of the log
    file.

    :Call:
        >>> result = read_task_result(task, returncode)
    :Inputs:
        *task*: :class:`BackgroundTask`
            Record of the completed task
        *returncode*: :class:`int`
            Exit status of the child process
    :Outputs:
        *result*: :class:`dict`
            Tool-style result, including *success* and *returncode*
    """
    # Tool tasks write their result to a JSON file
    if task.resultfile is not None:
        try:
            with open(task.resultfile) as fp:
                result = json.load(fp)
            # Make sure it looks like a tool result
            if isinstance(result, dict) and "success" in result:
                return result
        except (OSError, json.JSONDecodeError):
            pass
    # Otherwise synthesize a result from exit code and log file
    result = {
        "success": returncode == 0,
        "returncode": returncode,
    }
    # Read a tail of the task's log file
    try:
        with open(task.logfile) as fp:
            result["stdout"] = _truncate_stdout(fp.read())
    except OSError:
        pass
    # Output
    return result


# Format a note saying a task was launched
def format_launch_note(task: BackgroundTask) -> str:
    r"""Format a terminal note for a newly launched task

    :Call:
        >>> note = format_launch_note(task)
    :Inputs:
        *task*: :class:`BackgroundTask`
            Record of the new task
    :Outputs:
        *note*: :class:`str`
            Note to display to the user
    """
    return (
        f"{BGTASK_PROMPT}task {task.task_id} launched: "
        f"{task.describe()}\n  log: {task.logfile}")


# Format a note saying a task completed
def format_completion_note(task: BackgroundTask, result: dict) -> str:
    r"""Format a terminal note for a completed task

    :Call:
        >>> note = format_completion_note(task, result)
    :Inputs:
        *task*: :class:`BackgroundTask`
            Record of the completed task
        *result*: :class:`dict`
            Tool-style result of the task
    :Outputs:
        *note*: :class:`str`
            Note to display to the user
    """
    # Elapsed time
    dt = time.time() - task.t0
    # Get the exit status
    returncode = result.get("returncode")
    if returncode is None:
        returncode = 0 if result.get("success") else 1
    # Colored exit text
    if returncode == 0:
        statxt = sprintf_color(f"exit {returncode}", ["green"])
    else:
        statxt = sprintf_color(f"exit {returncode}", ["red", "bold"])
    # Format note
    return (
        f"{BGTASK_PROMPT}task {task.task_id} finished ({statxt}, "
        f"{dt:.1f} sec): {task.describe()}\n  log: {task.logfile}")


# Format a message for the LLM history when a task completes
def format_history_record(task: BackgroundTask, result: dict) -> str:
    r"""Format a history record for a completed task

    :Call:
        >>> record = format_history_record(task, result)
    :Inputs:
        *task*: :class:`BackgroundTask`
            Record of the completed task
        *result*: :class:`dict`
            Tool-style result of the task
    :Outputs:
        *record*: :class:`str`
            Message to append to the conversation history
    """
    # Compact copy of the result without huge STDOUT text
    compact = {k: v for k, v in result.items() if k != "stdout"}
    # Convert to JSON text
    txt = json.dumps(compact, cls=agentutils._NPEncoder, default=str)
    # Assemble the record
    parts = [
        f"Background task {task.task_id} completed.",
        f"Command: {task.describe()}",
        f"Result: {txt}",
        f"Full log file: {task.logfile}",
    ]
    return "\n".join(parts)


# Run one agent tool from a spec file (child-side entry point)
def main(argv: Optional[list] = None) -> int:
    r"""Run one agent tool from a JSON spec file

    This is the entry point used by child processes created by
    :func:`launch_tool_task`; it runs the tool and writes its result
    to the JSON file named in the spec.

    :Call:
        >>> ierr = main(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            Command-line args, the last of which is the spec file;
            defaults to ``sys.argv[1:]``
    :Outputs:
        *ierr*: :class:`int`
            Return code of the tool call
    """
    # Parse args
    argv = sys.argv[1:] if argv is None else argv
    specfile = argv[-1]
    # Read the spec
    with open(specfile) as fp:
        spec = json.load(fp)
    name = spec.get("tool")
    kwargs = spec.get("kwargs") or {}
    resultfile = spec["resultfile"]
    # Look up the tool, first in CAPE CLI tools, then system tools
    tool_fn = cfdxtools.TOOLS.get(name)
    if tool_fn is None:
        tool_fn = systools.TOOLS.get(name)
    # Run the tool
    if tool_fn is None:
        result = {
            "success": False,
            "error": f"unknown tool: {name}",
        }
    else:
        try:
            result = tool_fn(**kwargs)
        except Exception as e:
            result = {
                "success": False,
                "error": f"{e.__class__.__name__}: {e}",
            }
    # Make sure result is a dict with return code
    if not isinstance(result, dict):
        result = {"result": result, "success": True}
    result.setdefault("returncode", 0 if result.get("success") else 1)
    # Write the result for the parent process
    try:
        with open(resultfile, "w") as fp:
            json.dump(result, fp, cls=agentutils._NPEncoder, indent=2)
    except Exception:
        pass
    # Use tool's return code as child's exit status
    return int(result.get("returncode") or 0)


# Call main() when run as ``python -m cape.agent.bgtasks``
if __name__ == "__main__":
    sys.exit(main())
