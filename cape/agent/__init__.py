r"""
:mod:`cape.agent`: Agentic interface to CAPE
==============================================

The main interface for running the ``cape --agentic`` loop, processing
user reponses, passing them to an external LLM, and processing the
results.
"""


from __future__ import annotations

# Standard library
import json
import os
import pprint
import re
import readline
import shlex
import shutil
import sys
from subprocess import Popen
from typing import Optional

# Third-party imports
import numpy as np
from cape.cfdx import cli
from cape.util import pyrangestr
from openai import OpenAI, InternalServerError

# Local imports
from . import agentutils
from .. import capeconfig
from .tools import TOOL_SCHEMAS, TOOLS
from .tools.toolutils import normalize_kwargs
from ..argread.clitext import compile_rst, wrapline
from ..ui.promptutils import CfdxCompleter, sprintf_color, sprintf_color_rl


# Model selection
BASE_URL = "http://localhost:8000/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

# Constants
CAPE_HISTORY_LENGTH = 1000
EXIT_CMDS = (
    "exit",
    "quit",
    "exit()",
    "quit()",
)

# LLM parameters
MAX_TOOL_CALL_LOOPS = 3
SYSTEM_PROMPT = r"""

You are a helpful assistant for CAPE (Computational Aerosciences Productivity &
Execution), a NASA CFD run-matrix management tool. You have access to several
CAPE tools such as `cape_c`, which checks the status of one or more cases in
the run matrix. Each case will report one of the following status, which have
specific meanings:

* `---` means the case has not been started or set up yet.
* `INCOMP` means the case is set up but has not completed the minimum
  required iterations and is not running.
* `QUEUE` is an `INCOMP` case that has a PBS/Slurm job currently in the queue.
* `RUNNING` means the case is currently running (in progress).
* `ZOMBIE` means the case appears to be running but has not had any recent
  updates; the job likely failed for some reason.
* `FAIL`: The case encountered a failure while attempting to run CFD.
* `ERROR`: The user has marked this case a failure, and the status is final.
* `DONE` means the case has completed all required iterations and phases
  and is awaiting disposition by the user or agent.
* `PASS`: The case is `DONE` and marked as final by the user.
* `PASS*`: The case is marked as `PASS` by the user but does not meet the
  requirements for `DONE`.

Do not call the same tool again with the same or very similar arguments.

In most cases, do not create a table of results for each case; the user will
have already seen that from STDOUT during the tool call.

For most run-matrix related tool calls, including `cape_c`, it's often best to
call `cape_find` first, which finds the appropriate subset of cases and returns
the appropiate `I` parameter to use.
"""

# Special case: use CAPE directly
_solvrs = "(fun|cart|over|kes|lava|lch|us)"
REGEX_CAPE_CLI = re.compile(rf"\$?\s*(cape|py{_solvrs})( -?-?[a-z][a-z-]*)?")


# Agent prompt
AGENT_PROMPT = sprintf_color("→ Agent: ", ["bold"])
CAPE_PROMPT = sprintf_color("CAPE Input/Ouput", ["italic", "green"])
TOOL_CALL_PROMPT = sprintf_color("[tool call] ", ["italic", "purple"])
CLI_CALL_PROMPT = sprintf_color("[CLI]\n$", ["italic", "purple"])
TOOL_RESPONSE_PROMPT = sprintf_color("[tool response] ", ["italic", "purple"])
RAW_CAPE_MESSAGE = sprintf_color(
    "Detected raw CAPE command:", ["italic", "purple"])
RAW_TOOL_MESSAGE = sprintf_color(
    "Detected raw system command:", ["italic", "purple"])
# Other text
HLINE = "-" * min(int(0.9*shutil.get_terminal_size().columns), 79)
HLINE_BOLD = sprintf_color(HLINE, ["purple", "bold"])
HLINE = sprintf_color(HLINE, ["purple"])


def run_agent(
        user_message: str,
        client: OpenAI,
        history: Optional[list[dict]] = None) -> tuple[list[dict], dict]:
    r"""Run one pass of model with multi-round tool calling

    Run one user turn with up to *MAX_TOOL_CALL_LOOPS* rounds of tool
    calls. This allows the agent to chain tool calls, e.g., calling
    :func:`cape_find` followed by :func:`cape_c` with the results from
    the first call.
    """
    # Start some counters
    result = {
        "n_tool_calls": 0,
        "n_tool_fails": 0,
    }
    # Use message history or start with system prompt
    messages = history if history is not None else [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]
    # Check for apparent CLI call
    if REGEX_CAPE_CLI.match(user_message):
        # Turn into command
        cmdlist = shlex.split(user_message.lstrip("$").strip())
        # Status update
        print(HLINE)
        print(RAW_CAPE_MESSAGE)
        print(f"{CLI_CALL_PROMPT} {shlex.join(cmdlist)}")
        # Run it
        cli.main(argv=cmdlist)
        print(HLINE)
        return messages, result
    elif user_message.startswith("$"):
        # Run into command
        cmdlist = shlex.split(user_message.lstrip("$").strip())
        # Status update
        print(HLINE)
        print(RAW_TOOL_MESSAGE)
        print(f"{CLI_CALL_PROMPT} {shlex.join(cmdlist)}")
        # Run it
        proc = Popen(cmdlist)
        proc.communicate()
        print(HLINE)
        return messages, result
    # Append the user input
    messages.append({"role": "user", "content": user_message})
    print(HLINE_BOLD)
    # Main tool-calling loop (allow multiple rounds of tool calls)
    for loop_iter in range(MAX_TOOL_CALL_LOOPS):
        # Interact with LLM and get a response
        with agentutils.ThinkingSpinner("Thinking ..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        # Select the highest-ranked response
        msg = response.choices[0].message
        # Append model's response to history
        messages.append(msg.model_dump(exclude_none=True))
        # Check for special case with no tool calls
        if not msg.tool_calls:
            final_msg = msg
            break
        # Loop through tool calls
        for call in msg.tool_calls:
            # Get function name from tool call
            name = call.function.name
            # Parse function arguments from tool call
            try:
                kwargs = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                kwargs = {}
            # Format tool call
            tool_call_txt = format_tool_call(name, kwargs)
            tool_call_cli = format_cli_call(name, kwargs)
            # Print result
            print(HLINE)
            print(f"{TOOL_CALL_PROMPT}{tool_call_txt}")
            print(HLINE)
            if tool_call_cli:
                print(f"{CLI_CALL_PROMPT} {tool_call_cli}")
            # Get the actual tool
            tool_fn = TOOLS.get(name)
            # Increase tool-call count
            result["n_tool_calls"] += 1
            # Call tool if possible
            if tool_fn is None:
                # No actual tool call
                tool_result = {"ok": False, "error": f"unknown tool: {name}"}
                result["n_tool_fails"] += 1
            else:
                # Tool call: add prompt
                tool_result = tool_fn(**kwargs)
                print(HLINE)
            # Display output
            show_tool_result(tool_result)
            print(HLINE_BOLD)
            # Append message to history
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": dumps(tool_result),
                }
            )
    else:
        # If we've hit the max loops, force a final plain-text answer
        # Deliberately NOT passing `tools` here to force a text response
        with agentutils.ThinkingSpinner("Processing results ..."):
            followup = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
        # Select answer
        final_msg = followup.choices[0].message
        # Save it to history
        messages.append(final_msg.model_dump(exclude_none=True))
    # Show the response
    show_formatted_response(final_msg.content)
    # Return the messages so to be used as history for the next prompt
    return messages, result


# Format the tool's response
def show_formatted_response(msg: Optional[str]):
    if msg is None:
        return
    sys.stdout.write(f"\n{AGENT_PROMPT}")
    print(compile_rst(wrapline(msg)))
    print(HLINE_BOLD)
    print("")


# Turn a tool call into formatted function
def format_tool_call(name: str, kwargs: dict) -> str:
    # Normalize kwargs
    kw = normalize_kwargs(kwargs)
    # Parse into Python syntax
    argtxts = []
    # Loop through kwargs
    for k, v in kw.items():
        argtxts.append(f"{k}={repr(v)}")
    # Combine
    argtxt = ', '.join(argtxts)
    # Print result
    return f"{name}({argtxt})"


# Turn a tool call into CLI
def format_cli_call(name: str, kwargs: dict) -> str:
    # Check if command can be found
    cmdname = cli.CMD_FUNCS.get(name)
    # Exit if not found
    if cmdname is None:
        return ''
    # Safety
    try:
        # Get parser class
        parsercls = cli.CfdxFrontDesk._cmdparsers[cmdname]
        # Normalize kwargs
        kw = normalize_kwargs(kwargs)
        # Parse the kwargs
        parser = parsercls(**kw)
        # Reconstruct the command
        cmdlist = parser.reconstruct()
        cmdlist[0] = cmdname
        cmdlist.insert(0, "cape")
        # Output
        return shlex.join(cmdlist)
    except Exception:
        return ''


# Display the tool result
def show_tool_result(tool_result: dict):
    # Drop the STDOUT (which was already shown live)
    tool_stdout = _normalize_result(tool_result)
    # Display prompt
    print(TOOL_RESPONSE_PROMPT)
    # Convert to YAML format
    print(dumps(tool_stdout, sort_keys=False, indent=2))


# Normlaize output
def _normalize_result(result: dict):
    # Initialize normalized dict
    output = {}
    # Loop through keys
    for k, v in result.items():
        # Skip
        if k == "stdout":
            continue
        # Recurse?
        if isinstance(v, dict):
            output[k] = _normalize_result(v)
            continue
        # Check for range strings
        if isinstance(v, (list, np.ndarray)):
            try:
                vj = pyrangestr(v)
                output[k] = vj
            except TypeError:
                output[k] = v
        else:
            # Save as-is
            output[k] = v
    # Output
    return output


def main(cls: Optional[type] = None) -> None:
    # Initialize a results dictionary
    result = {
        "n_user_msgs": 0,
        "n_tool_calls": 0,
        "n_tool_fails": 0,
        "n_fails": 0,
    }
    # Get history file
    histfile = capeconfig.get_cape_opt("AgentHistoryFile")
    # If relative path, join with CacheDir
    if not os.path.isabs(histfile):
        cachedir = capeconfig.get_cape_opt("CacheDir")
        histfile = os.path.join(cachedir, histfile)
    # Read CAPE history from previous sessions
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(CAPE_HISTORY_LENGTH)
    except FileNotFoundError:
        pass
    # Enable tab completion (optional)
    readline.parse_and_bind("tab: complete")
    # Default completions class
    if cls is None:
        from ..cfdx.cli import CfdxFrontDesk
        cls = CfdxFrontDesk
    # Create and used CAPE-based autocompleter
    completer = CfdxCompleter(cls)
    readline.set_completer(completer)
    # Open the OpenAI interface to the LLM client
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    # Initialize history
    history: Optional[list[dict]] = None
    # Special formatting for initial prompt
    url = sprintf_color(BASE_URL, ["underline", "blue"])
    ctrlc = sprintf_color("Ctrl-C", "bold")
    # Initial prompt
    print(f"CAPE agent ready, using:\n   {url}")
    print(f"\nPress {ctrlc} to quit.\n")
    # Prompt message (use readline-specific version for proper wrapping)
    user_prompt = sprintf_color_rl("You: ", ["bold", "italic", "green"])
    # Loop until user requests exit
    while True:
        try:
            user_message = input(user_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # Recycle if empty prompt given
        if not user_message:
            continue
        # Check for manual exit
        if user_message.strip() in EXIT_CMDS:
            print()
            break
        # Update number of messages
        result["n_user_msgs"] += 1
        # Interact with LLM
        try:
            # Pass message and wait
            history, agent_result = run_agent(user_message, client, history)
            # Add to totals
            for k, n in agent_result.items():
                result[k] += n
        except InternalServerError as e:
            # Count failures
            result["n_fails"] += 1
            # Parse error message
            parts = e.args[0].split(' - ', 1)
            details = None if len(parts) < 2 else parts[1]
            # Show details first
            if details is not None:
                pprint.pprint(details)
            print(f"{type(e).__name__}: {parts[0]}")
            break
    # Save readline history on exit
    try:
        readline.write_history_file(histfile)
    except Exception:
        pass
    # Output
    return 0, result


# Convert to string
def dumps(v, **kw) -> str:
    return json.dumps(v, cls=agentutils._NPEncoder, **kw)

