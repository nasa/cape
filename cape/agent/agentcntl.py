r"""
:mod:`cape.agent.agentcntl`: Controller class for CAPE agent
=============================================================

This module provdes the class :class:`AgentCntl` which serves as an
object-oriented interface to the CAPE agentic capability and interface.
Most of the actual agent loop, including passing user responses on to
an external LLM and processing the results, is implemented by methods of
:class:`AgentCntl`.

The module-level list :data:`EDIT_FILE_ALLOW_LIST` is a registry of
POSIX-style file names, relative to the folder in which the agent is
launched, that skills may append to at run time (e.g. the ``fix-json``
skill registers the file the user asked to repair). The
:mod:`cape.agent.skills.fileedit` module merges these names into its
edit allow-list in addition to its own static patterns and the files
provided by :func:`cape.cfdx.cntl.Cntl.get_edit_allowlist`.
"""

# Standard library
import contextlib
import json
import os
import pprint
import re
import readline
import shlex
import shutil
from collections import namedtuple
from subprocess import PIPE, STDOUT, Popen

# Third-party imports
import numpy as np
from openai import OpenAI, InternalServerError

# Local imports
from . import agentutils
from . import bgtasks
from . import skills as agentskills
from .options import AgentOpts
from .skills import skilltools
from .skills.skillbase import discover_user_skills
from .tools import cfdxtools, cntltools, systools
from .tools.toolutils import normalize_kwargs
from ..argread.clitext import compile_rst, wrapline
from ..cfdx import cli
from ..errors import assert_isinstance
from ..ui.promptutils import CfdxCompleter, sprintf_color, sprintf_color_rl
from ..util import pyrangestr


# Model selection
BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

# Additional files (POSIX-style names, rel. to repo root) that skills
# may register for editing; merged into the file-editor allow-list
EDIT_FILE_ALLOW_LIST: list = []

# Constants
CAPE_HISTORY_LENGTH = 1000
EXIT_CMDS = (
    "exit",
    "quit",
    "exit()",
    "quit()",
)

# LLM parameters
SYSTEM_PROMPT = r"""

You are a helpful assistant for CAPE, a NASA CFD run-matrix management tool.
You have access to several CAPE tools such as `cape_c`, which checks the
status of one or more cases in the run matrix. Each case will report one of the
following status, which have specific meanings:

* `---` means the case has not been started or set up yet.
* `INCOMP` means the case is set up but has not completed the minimum
  required iterations and is not running.
* `QUEUE` is an `INCOMP` case that has a PBS/Slurm job currently in the queue.
* `RUNNING` means the case is currently running (in progress).
* `ZOMBIE` means the case appears to be running but has not had any recent
  updates.
* `FAIL`: The case encountered a failure while attempting to run CFD.
* `ERROR`: The user has marked this case a failure, and the status is final.
* `DONE` means the case has completed all required iterations and phases
   and is awaiting disposition by the user or agent.
* `PASS`: The case is `DONE` and marked as final by the user.
* `PASS*`: The case is marked `PASS` by the user but does not meet the
  requirements for `DONE`.

Slow commands such as report generation can be run as background
tasks using the tool's `background` option; their full results are
delivered in a follow-up message once the task completes.

Some tools such as `view_subfig` return images for you to look at;
those images arrive in the message immediately after the tool result,
so wait for that message before describing or analyzing the image.

Do not call the same tool again with the same or very similar arguments.

In most cases, do not create a table of results for each case; the user will
have already seen that from STDOUT during the tool call.

For most run-matrix related tool calls, including `cape_c`, it's often best to
call `cape_find` first, which finds the appropriate subset of cases and returns
the appropriate `I` parameter to use. Indexing for this `I` is ALWAYS 0-based
Python-like, so the first case is `0` and `600:602` means `600,601`.
"""

# Special case: use CAPE directly
_solvrs = "(fun|cart|over|kes|lava|lch|us)"
REGEX_CAPE_CLI = re.compile(rf"\$?\s*(cape|py{_solvrs}) -")


# Agent prompt
AGENT_PROMPT = sprintf_color("→ Agent: ", ["bold"])
CAPE_PROMPT = sprintf_color("CAPE Input/Ouput", ["italic", "green"])
TOOL_CALL_PROMPT = sprintf_color("[tool call] ", ["italic", "purple"])
CLI_CALL_PROMPT = sprintf_color("[CLI]\n$", ["italic", "purple"])
TOOL_RESPONSE_PROMPT = sprintf_color("[tool response] ", ["italic", "purple"])
REASONING_PROMPT = sprintf_color("[reasoning]", ["italic", "purple"])
RAW_CAPE_MESSAGE = sprintf_color(
    "Detected raw CAPE command:", ["italic", "purple"])
RAW_TOOL_MESSAGE = sprintf_color(
    "Detected raw system command:", ["italic", "purple"])
# Other text
HLINE = "─" * min(int(0.9*shutil.get_terminal_size().columns), 79)
HLINE_BOLD = sprintf_color(HLINE, ["purple", "bold"])
HLINE_ORANGE = sprintf_color(HLINE, ["orange"])
HLINE_GREEN = sprintf_color(HLINE, ["green"])
HLINE = sprintf_color(HLINE, ["purple"])


# Output class for main()
AgentResult = namedtuple("AgentResult", ("returncode", "result"))


# Control class
class AgentCntl:
    r"""Controller class for the CAPE agentic interface

    This class implements the agent loop behind ``cape --agentic``:
    reading user input, passing messages to an external LLM server,
    and processing any tool calls in its response. The tools and
    skills exposed to the LLM are filtered based on the options for
    the model in use (see :mod:`cape.agent.options`); user skills are
    discovered from the folder in which the agent is launched (see
    :mod:`cape.agent.skills`).

    :Call:
        >>> cntl = AgentCntl(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of CAPE-agentic JSON file (defaults to
            ``"cape-agent.json"``)
    :Outputs:
        *cntl*: :class:`AgentCntl`
            Controller for the CAPE agent loop
    """
    # Attributes
    __slots__ = (
        "RootDir",
        "base_url",
        "client",
        "fdir",
        "fname",
        "history",
        "loaded_skills",
        "model",
        "opts",
        "skills",
        "system_prompt",
        "tasks",
        "tool_schemas",
        "tools",
    )

    #: Name of default JSON file
    _fjson_default = "cape-agent.json"

    # Initialize
    def __init__(self, fname: str | None = None):
        # Default file name
        fname = self._fjson_default if fname is None else fname
        # Make sure it's a string
        assert_isinstance(fname, str, "Name of CAPE-agentic JSON file")
        #: :class:`str`
        #: Root folder for this controller
        self.RootDir = os.getcwd()
        # Get actual name of root file (follows links if necessary)
        fjson = os.path.realpath(fname)
        # Absolutize
        if os.path.isabs(fjson):
            # Already absolute
            fjson_rel = os.path.relpath(fjson, self.RootDir)
        else:
            # Already relative
            fjson_rel = fjson
        #: :class:`str`
        #: JSON file name (follows links if necessary) rel. to root dir
        self.fname = os.path.basename(fjson_rel)
        #: :class:`str`
        #: Folder in which JSON file is located, relative to root dir
        self.fdir = os.path.dirname(fjson_rel)
        # Read options
        self.read_opts(fname)
        #: :class:`str`
        #: Base URL of LLM server's OpenAI-compatible API
        self.base_url = self.opts.get_opt("URL", vdef=BASE_URL)
        #: :class:`openai.OpenAI`
        #: Client interface to LLM server
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        #: :class:`str`
        #: Name of LLM model currently in use
        self.model = self.get_model()
        #: :class:`list`\ [:class:`dict`] | ``None``
        #: Message history for current conversation
        self.history = None
        #: :class:`list`\ [:class:`bgtasks.BackgroundTask`]
        #: Background tasks launched during this session
        self.tasks = []
        # Filter tools to those appropriate for this model
        self.assemble_tools()
        # Assemble skills available for this model
        self.assemble_skills()

    # Read options
    def read_opts(self, fname: str):
        # Check if file exists
        if os.path.isfile(fname):
            # Read it
            self.opts = AgentOpts(fname)
        else:
            # Default options if file name does not exist
            print(f"No agents file '{fname}' found; using defaults")
            self.opts = AgentOpts()

    # Get the name of the model to use
    def get_model(self) -> str:
        # Get user setting, if any
        model = self.opts.get_opt("Model")
        # Query server for list of available models
        try:
            model_list = self.client.models.list()
        except Exception:
            model_list = None
        # Check for empty or failed query
        if (model_list is None) or not model_list.data:
            # Fall back to user setting or system default
            return model if model else MODEL
        # Get names of models available from server
        names = [m.id for m in model_list.data]
        # Check for user setting
        if model:
            # Check if user's model is served
            if model in names:
                return model
            # Warn that user's model is not available
            print(
                f"Warning: model '{model}' not in v1/models; "
                f"using '{names[0]}'")
        # Default to first model from ``v1/models``
        return names[0]

    # Filter tools to those for this model's *ToolSet*
    def assemble_tools(self):
        # Get descriptive name of how many tools to expose
        toolset = self.opts.get_ModelOpt(self.model, "ToolSet", vdef="full")
        # Get list of CAPE CLI tools for this set; default to all
        names_cfdx = cfdxtools.TOOL_SETS.get(toolset)
        if names_cfdx is None:
            names_cfdx = list(cfdxtools.TOOL_DICT)
        # Get list of CNTL tools for this set; default to all
        names_cntl = cntltools.TOOL_SETS.get(toolset)
        if names_cntl is None:
            names_cntl = list(cntltools.TOOL_DICT)
        # Combine tool names from both modules
        names = names_cfdx + names_cntl
        # Convert to a set for faster checks
        nameset = set(names)
        #: :class:`dict`\ [:class:`str`]
        #: Map of tool names to functions for current model
        self.tools = {name: cfdxtools.TOOLS[name] for name in names_cfdx}
        self.tools.update({name: cntltools.TOOLS[name] for name in names_cntl})
        #: :class:`list`\ [:class:`dict`]
        #: JSON schemas for tools available to current model
        self.tool_schemas = [
            schema for schema in cfdxtools.TOOL_SCHEMAS
            if schema["function"]["name"] in nameset
        ]
        self.tool_schemas += [
            schema for schema in cntltools.TOOL_SCHEMAS
            if schema["function"]["name"] in nameset
        ]
        # Always include all system tools
        self.tools.update(systools.TOOLS)
        self.tool_schemas += systools.TOOL_SCHEMAS

    # Assemble skills available for this model's *SkillSet*
    def assemble_skills(self):
        # Get descriptive name of how many skills to expose
        skillset = self.opts.get_ModelOpt(self.model, "SkillSet", vdef="full")
        # Get list of built-in skill names for this set; default to all
        names = agentskills.SKILL_SETS.get(skillset)
        if names is None:
            names = list(agentskills.BUILTIN_SKILLS)
        #: :class:`dict`\ [:class:`str`]
        #: Map of skill names to :class:`Skill` definitions
        self.skills = {
            name: agentskills.BUILTIN_SKILLS[name] for name in names
        }
        #: :class:`set`\ [:class:`str`]
        #: Skills whose tool schemas have been loaded this session
        self.loaded_skills = set()
        # Add user skills from launch dir unless skills are turned off
        if skillset != "none":
            # Discover from <RootDir>/.agents/skills/<NAME>/SKILL.md
            user_skills = discover_user_skills(self.RootDir)
            # User skills override built-ins of the same name
            self.skills.update(user_skills)
            # Report user skills found
            if user_skills:
                n = len(user_skills)
                print(f"Loaded {n} user skill(s) from .agents/skills")
        # Make skills available to the ``use_skill`` tool
        agentskills.skillbase.ACTIVE_SKILLS.clear()
        agentskills.skillbase.ACTIVE_SKILLS.update(self.skills)
        # Configure the folder in which the user-tools skill looks
        agentskills.usertools.ROOT_DIR = self.RootDir
        agentskills.usertools.TOOL_DIR_NAME = self.opts.get_opt(
            "ToolDir", vdef="tools")
        agentskills.usertools.TOOL_REGISTRY.clear()
        # Configure the allow-list for the file-editor skill
        agentskills.fileedit.ROOT_DIR = self.RootDir
        agentskills.fileedit.ALLOW_PATTERNS.clear()
        agentskills.fileedit.ALLOW_PATTERNS.extend(
            self.opts.get_opt("EditAllowList", vdef=[]))
        # Reset files registered for editing by other skills
        EDIT_FILE_ALLOW_LIST.clear()
        #: :class:`str`
        #: System prompt including listing of available skills
        self.system_prompt = genr8_system_prompt(self.skills)
        # Include skill management only if skills are available
        if self.skills:
            self.tools.update(skilltools.TOOLS)
            # Use controller wrapper to activate tools after loading skill
            self.tools["use_skill"] = self.use_skill
            self.tool_schemas += skilltools.TOOL_SCHEMAS

    # Load a skill's instructions and activate its tool schemas
    def use_skill(self, name: str) -> dict:
        r"""Load an available skill and activate the tools it provides"""
        # Load the skill instructions
        result = skilltools.use_skill(name)
        # Stop if *name* is not an available skill
        if not result["success"]:
            return result
        # Activate any tools provided by the skill
        added, active = self.activate_skill_tools(name)
        result["tools_added"] = added
        result["tools_active"] = active
        # Output
        return result

    # Add tool functions and schemas for one skill
    def activate_skill_tools(self, name: str) -> tuple[list, list]:
        r"""Activate tools belonging to one available skill"""
        # Get skill definition and its tool module
        skill = self.skills[name]
        mod = agentskills.SKILL_TOOL_MODULES.get(name)
        # Skills without built-in tools need no schema activation
        if mod is None or not skill.tools:
            self.loaded_skills.add(name)
            return [], []
        # Map schema names to schemas
        schemas = {
            schema["function"]["name"]: schema
            for schema in mod.TOOL_SCHEMAS
        }
        # Validate all tools before changing controller state
        for tool_name in skill.tools:
            if tool_name not in mod.TOOLS:
                raise KeyError(
                    f"Skill '{name}' has no tool function '{tool_name}'")
            if tool_name not in schemas:
                raise KeyError(
                    f"Skill '{name}' has no tool schema '{tool_name}'")
            old_tool = self.tools.get(tool_name)
            if old_tool is not None and old_tool is not mod.TOOLS[tool_name]:
                raise ValueError(
                    f"Skill tool '{tool_name}' conflicts with an active tool")
        # Add only tools explicitly declared by the skill
        added = []
        if name not in self.loaded_skills:
            active_schemas = {
                schema["function"]["name"]
                for schema in self.tool_schemas
            }
            for tool_name in skill.tools:
                if tool_name not in active_schemas:
                    self.tools[tool_name] = mod.TOOLS[tool_name]
                    self.tool_schemas.append(schemas[tool_name])
                    added.append(tool_name)
            self.loaded_skills.add(name)
        # Output both newly added and currently active tools
        return added, list(skill.tools)

    # Run one user prompt with multi-round tool calling
    def run_agent(
            self,
            user_message: str,
            spinner: bool = True,
            capture_subprocess: bool = False,
            section_handler=None) -> dict:
        r"""Run one pass of model with multi-round tool calling

        Run one user turn with up to *MaxToolCallLoops* rounds of tool
        calls. This allows the agent to chain tool calls, e.g., calling
        :func:`cape_find` followed by :func:`cape_c` with the results from
        the first call.

        :Inputs:
            *user_message*: :class:`str`
                User prompt or direct command
            *spinner*: {``True``} | ``False``
                Show the readline thinking spinner; Textual supplies its own
            *capture_subprocess*: {``False``} | ``True``
                Pipe direct system-command output through :data:`sys.stdout`
            *section_handler*: {``None``} | callable
                Optional presentation hook called at the start and end of
                reasoning and tool sections
        :Outputs:
            *result*: :class:`dict`
                Counts of tool calls and failures for this turn
        """
        # Start some counters
        result = {
            "n_tool_calls": 0,
            "n_tool_fails": 0,
        }
        # Get max tool call loops for this model
        max_loops = self.opts.get_ModelOpt(self.model, "MaxToolCallLoops")
        # Initialize message history with system prompt
        if self.history is None:
            self.history = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]
        # Use message history
        messages = self.history
        # Check for apparent CLI call
        if REGEX_CAPE_CLI.match(user_message):
            # Turn into command
            cmdlist = shlex.split(user_message.lstrip("$").strip())
            # Check for shell-style background request (trailing "&")
            cmdlist, bg = _strip_background(cmdlist)
            # Check for background request
            if bg:
                # Launch as background task
                self.launch_cli_task(messages, cmdlist)
                return result
            # Keep later turns aware of direct execution without an LLM call
            messages.append({
                "role": "user",
                "content": (
                    "I issued this command directly through the CLI; "
                    "this is a record, not a request to execute it again. "
                    "No LLM response is needed. Command:\n" +
                    shlex.join(cmdlist)),
            })
            # Status update
            print(HLINE)
            print(RAW_CAPE_MESSAGE)
            print(f"{CLI_CALL_PROMPT} {shlex.join(cmdlist)}")
            # Run it
            cli.main(argv=cmdlist)
            print(HLINE)
            return result
        elif user_message.startswith("$"):
            # Run into command
            cmdlist = shlex.split(user_message.lstrip("$").strip())
            # Check for shell-style background request (trailing "&")
            cmdlist, bg = _strip_background(cmdlist)
            if not cmdlist:
                return result
            # Check for background request
            if bg:
                # Launch as background task
                self.launch_cli_task(messages, cmdlist)
                return result
            # Keep later turns aware of direct execution without an LLM call
            messages.append({
                "role": "user",
                "content": (
                    "I issued this command directly through the CLI; "
                    "this is a record, not a request to execute it again. "
                    "No LLM response is needed. Command:\n" +
                    shlex.join(cmdlist)),
            })
            # Status update
            print(HLINE)
            print(RAW_TOOL_MESSAGE)
            print(f"{CLI_CALL_PROMPT} {shlex.join(cmdlist)}")
            # Run it
            try:
                if capture_subprocess:
                    proc = Popen(
                        cmdlist,
                        stdout=PIPE,
                        stderr=STDOUT,
                        text=True,
                        errors="replace")
                    for line in proc.stdout or ():
                        print(line, end="")
                    proc.wait()
                else:
                    proc = Popen(cmdlist)
                    proc.communicate()
            except Exception:
                print("System command failed")
            print(HLINE)
            return result
        # Append the user input
        messages.append({"role": "user", "content": user_message})
        # Main tool-calling loop (allow multiple rounds of tool calls)
        for loop_iter in range(max_loops):
            # Interact with LLM and get a response
            thinking = agentutils.ThinkingSpinner("Thinking ...") \
                if spinner else contextlib.nullcontext()
            with thinking:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                )
            # Select the highest-ranked response
            msg = response.choices[0].message
            # Show server-provided reasoning after the response completes
            show_reasoning_opt = self.opts.get_opt("ShowReasoning")
            if section_handler is not None and get_reasoning_text(msg):
                section_handler(
                    "start", "reasoning", "[reasoning]",
                    bool(show_reasoning_opt))
                show_reasoning(msg, framed=False)
                section_handler("end", "reasoning", "", True)
            elif show_reasoning_opt:
                show_reasoning(msg)
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
                if section_handler is not None:
                    section_handler(
                        "start", "tool", f"[tool call] {tool_call_txt}",
                        bool(self.opts.get_opt("ShowToolResult")))
                else:
                    start_section("tool_call", tool_call_txt)
                # Display CLI equivalent if appropriate
                if tool_call_cli:
                    start_section("cli", tool_call_cli)
                # Get the actual tool
                tool_fn = self.tools.get(name)
                # Increase tool-call count
                result["n_tool_calls"] += 1
                # Set flag for user interrupt
                user_interrupt = False
                # Call tool if possible
                if tool_fn is None:
                    # No actual tool call
                    tool_result = {
                        "ok": False, "error": f"unknown tool: {name}"}
                    result["n_tool_fails"] += 1
                elif (name in cfdxtools.BACKGROUNDABLE_TOOLS and
                        kwargs.pop("background", False)):
                    # Launch as background task instead of blocking
                    tool_result = self.launch_tool_task(
                        name, kwargs, result)
                else:
                    # Tool call: add prompt
                    try:
                        tool_result = tool_fn(**kwargs)
                    except Exception as e:
                        # Get error class
                        ecls = e.__class__.__name__
                        print("Tool evaluation failed:")
                        print(f"   {ecls}: {e.args[0]}")
                        tool_result = {
                            "success": False,
                            "reason": ecls,
                        }
                        result["n_tool_fails"] += 1
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt")
                        user_interrupt = True
                        tool_result = {
                            "success": False,
                            "reason": "User interrupted tool call",
                        }
                    if section_handler is None:
                        print(HLINE)
                # Pull out any images for multimodal delivery
                images = None
                if isinstance(tool_result, dict):
                    images = tool_result.pop("images", None)
                # Display output if turned on
                if section_handler is not None or \
                        self.opts.get_opt("ShowToolResult"):
                    show_tool_result(tool_result)
                    if section_handler is None:
                        print(HLINE_BOLD)
                if section_handler is not None:
                    section_handler("end", "tool", "", True)
                # Append message to history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": dumps(tool_result),
                    }
                )
                # Attach images so the model can look at them
                if images:
                    messages.append(genr8_image_message(images))
                # Exit if interrupted
                if user_interrupt:
                    break
        else:
            # If we've hit the max loops, force a final plain-text answer
            # Deliberately NOT passing `tools` here to force a text response
            thinking = agentutils.ThinkingSpinner("Processing results ...") \
                if spinner else contextlib.nullcontext()
            with thinking:
                followup = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            # Select answer
            final_msg = followup.choices[0].message
            # Show server-provided reasoning after the response completes
            show_reasoning_opt = self.opts.get_opt("ShowReasoning")
            if section_handler is not None and get_reasoning_text(final_msg):
                section_handler(
                    "start", "reasoning", "[reasoning]",
                    bool(show_reasoning_opt))
                show_reasoning(final_msg, framed=False)
                section_handler("end", "reasoning", "", True)
            elif show_reasoning_opt:
                show_reasoning(final_msg)
            # Save it to history
            messages.append(final_msg.model_dump(exclude_none=True))
        # Start the response section
        if section_handler is not None:
            # Start a section
            section_handler(
                "start", "response", "Agent:", True)
            # Display the response
            show_formatted_response(final_msg.content.lstrip())
            # End the section
            section_handler("end", "response", "", True)
        else:
            # Start the "section"
            start_response()
            # Show the response
            show_formatted_response(final_msg.content.lstrip())
            # Extra divider
            print(HLINE_GREEN)
        # Return counters for this pass
        return result

    # Launch a CLI command as a background task
    def launch_cli_task(self, messages: list, cmdlist: list):
        r"""Launch a typed command as a background task

        Runs *cmdlist* as a background task and records the launch in
        the conversation history.
        """
        # Keep later turns aware of direct execution without an LLM call
        messages.append({
            "role": "user",
            "content": (
                "I issued this command through the CLI as a background "
                "task; this is a record, not a request to execute it "
                "again. No LLM response is needed. Its results will "
                "arrive in a follow-up message. Command:\n" +
                shlex.join(cmdlist)),
        })
        # Status update
        print(HLINE)
        # Try to launch the task
        try:
            task = bgtasks.launch_cli_task(len(self.tasks) + 1, cmdlist)
        except OSError as e:
            print(f"Could not launch background task: {e}")
            print(HLINE)
            return
        # Save task and report
        self.tasks.append(task)
        print(bgtasks.format_launch_note(task))
        print(HLINE)

    # Run an allow-listed tool call as a background task
    def launch_tool_task(self, name: str, kwargs: dict, result: dict) -> dict:
        r"""Launch a tool call as a background task

        Returns the tool's immediate result; its full result is
        delivered as a follow-up message when the task completes.
        """
        # Try to launch the task
        try:
            task = bgtasks.launch_tool_task(len(self.tasks) + 1, name, kwargs)
        except Exception as e:
            result["n_tool_fails"] += 1
            return {
                "success": False,
                "error": f"could not launch task: {e.__class__.__name__}: {e}",
            }
        # Save task and report
        self.tasks.append(task)
        print(bgtasks.format_launch_note(task))
        print(HLINE)
        # Immediate tool response; full result delivered on completion
        return {
            "success": True,
            "background": True,
            "task_id": task.task_id,
            "logfile": task.logfile,
            "message": (
                "This command was launched as a background task; its "
                "full result will arrive in a follow-up message after "
                "it completes."),
        }

    # Check for completed background tasks
    def reap_tasks(self, section_handler=None):
        r"""Notify user and history of completed background tasks"""
        # Loop through newly finished tasks
        for task, tool_result in bgtasks.poll_finished(self.tasks):
            if section_handler is not None and task.tool_name:
                section_handler(
                    "start", "tool",
                    f"[tool result] background task {task.task_id}",
                    bool(self.opts.get_opt("ShowToolResult")))
            # Terminal notification
            if section_handler is None:
                print(HLINE)
            print(bgtasks.format_completion_note(task, tool_result))
            if section_handler is None:
                print(HLINE)
            # Display tool-style result if turned on
            if task.tool_name and (
                    section_handler is not None or
                    self.opts.get_opt("ShowToolResult")):
                show_tool_result(tool_result)
            if section_handler is not None and task.tool_name:
                section_handler("end", "tool", "", True)
            # Inform the conversation, if there is one
            if self.history is not None:
                self.history.append({
                    "role": "user",
                    "content": bgtasks.format_history_record(
                        task, tool_result),
                })

    # Run main loop
    def main(self, cls: type | None = None) -> AgentResult:
        # Initialize a results dictionary
        result = {
            "n_user_msgs": 0,
            "n_tool_calls": 0,
            "n_tool_fails": 0,
            "n_fails": 0,
        }
        # Get history file
        histfile = agentutils.get_agent_histfile()
        # Read CAPE history from previous sessions
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(CAPE_HISTORY_LENGTH)
        except FileNotFoundError:
            pass
        # Enable tab completion (optional)
        readline.parse_and_bind("tab: complete")
        # Search history
        readline.parse_and_bind(r'"\e[A": history-search-backward')
        readline.parse_and_bind(r'"\e[B": history-search-forward')
        # Default completions class
        if cls is None:
            from ..cfdx.cli import CfdxFrontDesk
            cls = CfdxFrontDesk
        # Create and used CAPE-based autocompleter
        completer = CfdxCompleter(cls)
        readline.set_completer(completer)
        # Special formatting for initial prompt
        url = sprintf_color(self.base_url, ["underline", "blue"])
        model = sprintf_color(self.model, ["underline", "blue"])
        ctrlc = sprintf_color("Ctrl-C", "bold")
        # Initial prompt
        print(f"CAPE agent ready, using:\n   {url}\n   {model}")
        print(f"\nPress {ctrlc} to quit.\n")
        # Prompt message (use readline-specific version for proper wrapping)
        user_prompt = sprintf_color_rl("You: ", ["bold", "italic", "green"])
        # Loop until user requests exit
        while True:
            # Notify user of completed background tasks
            self.reap_tasks()
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
                agent_result = self.run_agent(user_message)
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
        # Warn about background tasks still running; being in their
        # own sessions, they are not interrupted by this exit
        running = [
            task for task in self.tasks
            if not task.reaped and task.poll() is None
        ]
        if running:
            print(f"Note: {len(running)} background task(s) still running:")
            for task in running:
                print(f"  {task.task_id}: {task.describe()}")
                print(f"    log: {task.logfile}")
        # Save readline history on exit
        try:
            readline.write_history_file(histfile)
        except Exception:
            pass
        # Output
        return 0, result


# Build system prompt, appending a listing of available skills
def genr8_system_prompt(skills: dict) -> str:
    r"""Build the system prompt, listing available agent skills

    :Call:
        >>> prompt = genr8_system_prompt(skills)
    :Inputs:
        *skills*: :class:`dict`\ [:class:`.skills.skillbase.Skill`]
            Map of skill names to skill definitions
    :Outputs:
        *prompt*: :class:`str`
            System prompt for the LLM
    """
    # Base prompt if no skills
    if not skills:
        return SYSTEM_PROMPT
    # Assemble skill listing
    lines = [
        SYSTEM_PROMPT.strip(),
        "",
        "## Agent skills",
        "",
        "You have access to *agent skills*: documented workflows that"
        " describe how and when to use certain tools and how to chain"
        " tool calls together. Before starting a task that matches a"
        " skill's description, call the `use_skill` tool with the skill"
        " name to read its full instructions and activate its tools.",
        "",
        "Available skills:",
    ]
    # Add one line per skill
    for name in sorted(skills):
        lines.append(f"* `{name}`: {skills[name].description}")
    # Combine
    return "\n".join(lines)


# Remove a trailing "&" background marker from a command
def _strip_background(cmdlist: list) -> tuple[list, bool]:
    r"""Remove a trailing ``&`` for backgrounded shell-style commands

    :Call:
        >>> cmdlist, bg = _strip_background(cmdlist)
    :Inputs:
        *cmdlist*: :class:`list`\ [:class:`str`]
            Command split into tokens
    :Outputs:
        *cmdlist*: :class:`list`\ [:class:`str`]
            Command without any trailing ``&`` token
        *bg*: :class:`bool`
            Whether a trailing ``&`` was found
    """
    # Check for trailing "&", either its own token or tacked onto the
    # final token ("... &" or "...&")
    if cmdlist and cmdlist[-1].endswith("&"):
        # Remove the "&" character
        last = cmdlist.pop()[:-1]
        # Put the token back unless the "&" was its own token
        if last:
            cmdlist.append(last)
        # Background requested
        return cmdlist, True
    # No background request
    return cmdlist, False


# Package tool images as a multimodal user message
def genr8_image_message(images: list) -> dict:
    r"""Build a user message carrying image content for the model

    :Call:
        >>> msg = genr8_image_message(images)
    :Inputs:
        *images*: :class:`list`
            Image entries from a tool result; each item is either a
            :class:`dict` with *url* (and optional *case*/*file*) or a
            bare data-URL :class:`str`
    :Outputs:
        *msg*: :class:`dict`
            Multimodal user message for the chat API
    """
    # Introductory text so the model knows what it's seeing
    content = [
        {
            "type": "text",
            "text": (
                "The previous tool call returned the CAPE report "
                "subfigure image(s) below for you to look at."
            ),
        }
    ]
    # Loop through images
    for img in images:
        # Parse the entry
        if isinstance(img, dict):
            url = img.get("url")
            case = img.get("case")
            fimg = img.get("file")
        else:
            url = img
            case = None
            fimg = None
        # Skip malformed entries
        if not url:
            continue
        # Label the image when possible
        if case is not None or fimg is not None:
            content.append({
                "type": "text",
                "text": f"Case {case}: {fimg}",
            })
        # Attach the image
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    # Output
    return {"role": "user", "content": content}


# Format the model's response
def show_formatted_response(msg: str | None):
    if msg is None:
        return
    print(compile_rst(wrapline(msg)))
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
        # Remove parameters that are only meaningful to the agent
        kw.pop("background", None)
        # Parse the kwargs
        parser = parsercls(**kw)
        # Reconstruct the command
        cmdlist = parser.reconstruct()
        cmdlist[0] = cmdname
        cmdlist.insert(0, "cape")
        # Output
        return "$ " + shlex.join(cmdlist)
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


# Non-TUI final response start
def start_response(title: str = "Agent: ", txt: str | None = None):
    r"""Produce the header at the start of agent's actual response

    :Call:
        >>> start_response(title, txt)
    :Inputs:
        *title*: :class:`str`
            Section title, diplayed purple as ``f"[{title}]"``
        *txt*: {``None``} | :class:`str`
            Optional text after section title
    """
    # Start the section
    print(HLINE_ORANGE)
    # Create title
    msg = sprintf_color(title, ["orange", "bold", "italic"])
    # Add the optional text
    if txt:
        msg += f" {txt}"
    # Display the line
    print(msg)


# Non-TUI section start
def start_section(title: str, txt: str | None = None):
    r"""Produce the header at the start of a section, non-TUI

    :Call:
        >>> start_section(title, txt)
    :Inputs:
        *title*: :class:`str`
            Section title, diplayed purple as ``f"[{title}]"``
        *txt*: {``None``} | :class:`str`
            Optional text after section title
    """
    # Start the section
    print(HLINE)
    # Create title
    msg = sprintf_color(f"[{title}]", ["purple", "italic"])
    # Add the optional text
    if txt:
        msg += f" {txt}"
    # Display the line
    print(msg)


# Display reasoning exposed by an OpenAI-compatible model server
def get_reasoning_text(message) -> str:
    r"""Get normalized reasoning text from a completion message

    :Inputs:
        *message*: :class:`object`
            Chat-completion response message
    :Outputs:
        *text*: :class:`str`
            Reasoning text, or an empty string when none was exposed
    """
    reasoning = getattr(message, "reasoning", None)
    if not reasoning:
        reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        return ""
    if isinstance(reasoning, str):
        return reasoning.strip()
    return dumps(reasoning, sort_keys=False, indent=2).strip()


# Display reasonint content in non-TUI interface
def show_reasoning(message, framed: bool = True) -> bool:
    r"""Display post-response reasoning content when available

    This uses the nonstandard ``reasoning`` field exposed by current vLLM
    servers, with ``reasoning_content`` as a fallback for older compatible
    servers. The OpenAI Python client keeps unknown response fields as model
    extras, so :func:`getattr` works even when the installed SDK does not
    declare either field.

    :Call:
        >>> shown = show_reasoning(message)
    :Inputs:
        *message*: :class:`object`
            Chat-completion response message
        *framed*: {``True``} | ``False``
            Print the classic prompt and horizontal rules
    :Outputs:
        *shown*: :class:`bool`
            Whether nonempty reasoning content was displayed
    """
    text = get_reasoning_text(message)
    if not text:
        return False
    if framed:
        start_section("reasoning")
    # Use the same paragraph-aware wrapping as the final response. Structured
    # reasoning extensions are still valid plain text after serialization.
    print(compile_rst(wrapline(text)))
    return True


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


# Convert to string
def dumps(v, **kw) -> str:
    return json.dumps(v, cls=agentutils._NPEncoder, **kw)
