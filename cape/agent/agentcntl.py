r"""
:mod:`cape.agent.agentcntl`: Controller class for CAPE agent
=============================================================

This module provdes the class :class:`AgentCntl` which serves as an
object-oriented interface to the CAPE agentic capability and interface.
Most of the actual agent loop, including passing user responses on to
an external LLM and processing the results, is implemented by methods of
:class:`AgentCntl`.
"""

# Standard library
import json
import os
import pprint
import re
import readline
import shlex
import shutil
import sys
from collections import namedtuple
from subprocess import Popen

# Third-party imports
import numpy as np
from openai import OpenAI, InternalServerError

# Local imports
from . import agentutils
from . import skills as agentskills
from .options import AgentOpts
from .skills import skilltools
from .skills.skillbase import discover_user_skills
from .tools import cfdxtools, cntltools, systools
from .tools.toolutils import normalize_kwargs
from .. import capeconfig
from ..argread.clitext import compile_rst, wrapline
from ..cfdx import cli
from ..errors import assert_isinstance
from ..ui.promptutils import CfdxCompleter, sprintf_color, sprintf_color_rl
from ..util import pyrangestr


# Model selection
BASE_URL = "http://localhost:8000/v1"
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
REGEX_CAPE_CLI = re.compile(rf"\$?\s*(cape|py{_solvrs}) -")


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
        "model",
        "opts",
        "skills",
        "system_prompt",
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
        #: :class:`str`
        #: System prompt including listing of available skills
        self.system_prompt = genr8_system_prompt(self.skills)
        # Always include the skill-management tools
        self.tools.update(skilltools.TOOLS)
        self.tool_schemas += skilltools.TOOL_SCHEMAS
        # Merge tools provided by active skills
        for name, mod in agentskills.SKILL_TOOL_MODULES.items():
            if name in self.skills:
                self.tools.update(mod.TOOLS)
                self.tool_schemas += mod.TOOL_SCHEMAS

    # Run one user prompt with multi-round tool calling
    def run_agent(self, user_message: str) -> dict:
        r"""Run one pass of model with multi-round tool calling

        Run one user turn with up to *MaxToolCallLoops* rounds of tool
        calls. This allows the agent to chain tool calls, e.g., calling
        :func:`cape_find` followed by :func:`cape_c` with the results from
        the first call.
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
            # Status update
            print(HLINE)
            print(RAW_TOOL_MESSAGE)
            print(f"{CLI_CALL_PROMPT} {shlex.join(cmdlist)}")
            # Run it
            try:
                proc = Popen(cmdlist)
                proc.communicate()
            except Exception:
                print("System command failed")
            print(HLINE)
            return result
        # Append the user input
        messages.append({"role": "user", "content": user_message})
        print(HLINE_BOLD)
        # Main tool-calling loop (allow multiple rounds of tool calls)
        for loop_iter in range(max_loops):
            # Interact with LLM and get a response
            with agentutils.ThinkingSpinner("Thinking ..."):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
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
                tool_fn = self.tools.get(name)
                # Increase tool-call count
                result["n_tool_calls"] += 1
                # Call tool if possible
                if tool_fn is None:
                    # No actual tool call
                    tool_result = {
                        "ok": False, "error": f"unknown tool: {name}"}
                    result["n_tool_fails"] += 1
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
                        tool_result = {
                            "success": False,
                            "reason": "User interrupted tool call",
                        }
                    print(HLINE)
                # Display output if turned on
                if self.opts.get_opt("ShowToolResult"):
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
                followup = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            # Select answer
            final_msg = followup.choices[0].message
            # Save it to history
            messages.append(final_msg.model_dump(exclude_none=True))
        # Show the response
        show_formatted_response(final_msg.content)
        # Return counters for this pass
        return result

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
        " name to read its full instructions.",
        "",
        "Available skills:",
    ]
    # Add one line per skill
    for name in sorted(skills):
        lines.append(f"* `{name}`: {skills[name].description}")
    # Combine
    return "\n".join(lines)


# Format the model's response
def show_formatted_response(msg: str | None):
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


# Convert to string
def dumps(v, **kw) -> str:
    return json.dumps(v, cls=agentutils._NPEncoder, **kw)
