r"""
:mod:`cape.agent.skills.usertools`: Discover and run repo scripts
==================================================================

This module defines the built-in agent skill ``"user-tools"``, which
teaches the CAPE agent how to use customized scripts the user has placed
in the ``tools/`` folder of the repo in which the agent is launched.

Many CAPE-based repos accumulate local scripts that extend CAPE's
built-in capabilities, e.g. scripts to restart failed adaptations or
gather surface data. Such scripts are ordinary Python files with an
``if __name__ == "__main__":`` block, usually with a help message
available from ``-h`` or ``--help``.

The skill provides two tools:

* :func:`discover_user_tools`: scan the tools folder for executable
  scripts (those with a ``__main__`` guard) and extract a one-line
  description from each script's docstring
* :func:`run_user_tool`: run one of the discovered scripts with
  user-provided command-line arguments

Scripts are run using the same Python executable as the agent itself,
so they import the same CAPE installation, and with the repo root as
the working folder so relative file names (e.g. ``"pyFun.json"``)
resolve as they would if the user ran the script from the repo root.
"""

# Standard library
import ast
import glob
import os
import re
import subprocess
import sys

# Local imports
from ..tools import toolutils


# Regex fallback for detecting a __main__ guard
REGEX_MAIN_GUARD = re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]")

# Folder in which agent was launched; defaults to cwd
ROOT_DIR: str | None = None

# Name of folder (relative to *ROOT_DIR*) containing user tools
TOOL_DIR_NAME = "tools"

# Cache of discovered tools, keyed by script name without extension
TOOL_REGISTRY: dict = {}


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "name": {
        "description": (
            "Name of a user script from the repo's tools folder, with "
            "or without the '.py' extension, as listed by "
            "discover_user_tools."
        ),
        "type": "string",
    },
    "argv": {
        "description": (
            "Command-line arguments to the script, e.g. "
            "['-I', '1:5', '--force']. Use ['--help'] to show the "
            "script's own help message. Omit for no arguments."
        ),
        "type": ["array", "null"],
        "items": {"type": "string"},
    },
    "refresh": {
        "description": (
            "Rescan the tools folder even if tools have already been "
            "discovered in this session."
        ),
        "type": ["boolean", "null"],
    },
    "timeout": {
        "description": (
            "Maximum number of seconds to wait for the script to "
            "finish. Omit for no limit."
        ),
        "type": ["integer", "null"],
        "minimum": 1,
    },
}


# Get absolute path of tools folder
def _get_tooldir() -> str:
    # Substitute cwd if *ROOT_DIR* not set
    rootdir = os.getcwd() if ROOT_DIR is None else ROOT_DIR
    return os.path.join(rootdir, TOOL_DIR_NAME)


# Check source text for a top-level __main__ guard
def _has_main_guard(src: str) -> bool:
    r"""Check if Python source has an ``if __name__ == "__main__"`` guard

    Uses :mod:`ast` to check top-level ``if`` statements; falls back to
    a regular expression if the file cannot be parsed.

    :Call:
        >>> flag = _has_main_guard(src)
    :Inputs:
        *src*: :class:`str`
            Source text of a Python script
    :Outputs:
        *flag*: :class:`bool`
            Whether a ``__main__`` guard is present (or likely is, if
            the source does not parse)
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fallback to regex for unparseable (e.g. Python 2) scripts
        return bool(REGEX_MAIN_GUARD.search(src))
    # Check top-level "if" statements for the __main__ comparison
    for node in tree.body:
        # Check for "if" with a simple comparison
        if not isinstance(node, ast.If):
            continue
        # Get the comparison
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        # Check LHS: __name__
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        # Check for exactly one "== __main__" comparison
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        # Check RHS: "__main__"
        rhs = test.comparators[0]
        if isinstance(rhs, ast.Constant) and rhs.value == "__main__":
            return True
    return False


# Extract one-line description from a script's module docstring
def _extract_description(tree: ast.Module) -> str:
    r"""Extract a one-line description from a module docstring

    Takes the first nonempty line of the docstring and performs light
    cleanup of common RST markup (``:mod:`` roles and backticks).

    :Call:
        >>> desc = _extract_description(tree)
    :Inputs:
        *tree*: :class:`ast.Module`
            Parsed Python script
    :Outputs:
        *desc*: :class:`str`
            One-line summary, or empty string if no docstring
    """
    # Get raw docstring
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    # Get first nonempty line
    for line in doc.splitlines():
        # Strip whitespace and common RST clutter
        line = line.strip()
        if not line:
            continue
        # Remove leading ":mod:" role and all backticks
        line = re.sub(r"^:mod:", "", line)
        line = line.replace("`", "").strip()
        return line
    return ""


# Discover executable scripts from the tools folder
def discover_user_tools(refresh: bool | None = False) -> dict:
    r"""Discover executable scripts in the repo's tools folder

    Scans ``<ROOT_DIR>/<TOOL_DIR_NAME>/`` for Python scripts (top-level
    only; subfolders are not scanned) containing an
    ``if __name__ == "__main__":`` block. Files whose name starts with
    an underscore (including ``__init__.py``) are skipped. Results are
    cached for the rest of the agent session unless *refresh* is true.

    :Call:
        >>> result = discover_user_tools(refresh=False)
    :Inputs:
        *refresh*: ``True`` | {``False``} | ``None``
            Rescan even if tools were already discovered
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *tooldir*, and *tools*: a list of
            dicts with *name*, *file*, and *description* for each
            discovered script
    :Versions:
        * 2026-09-02 ``@ddalle``: v1.0
    """
    # Use cached registry if present
    if TOOL_REGISTRY and not refresh:
        return {
            "success": True,
            "cached": True,
            "tooldir": _get_tooldir(),
            "tools": [TOOL_REGISTRY[name] for name in sorted(TOOL_REGISTRY)],
        }
    # Absolute path of tools folder
    tooldir = _get_tooldir()
    # Clear the registry (also covers refresh=True)
    TOOL_REGISTRY.clear()
    # Check for folder
    if not os.path.isdir(tooldir):
        return {
            "success": True,
            "tooldir": tooldir,
            "tools": [],
            "message": "No tools folder found in this repo",
        }
    # Loop through Python files, top-level only
    for fpath in sorted(glob.glob(os.path.join(tooldir, "*.py"))):
        # Get the file name
        fname = os.path.basename(fpath)
        # Skip private files and __init__.py
        if fname.startswith("_"):
            continue
        # Read the file
        try:
            with open(fpath) as fp:
                src = fp.read()
        except (OSError, UnicodeDecodeError):
            continue
        # Check for a __main__ guard
        if not _has_main_guard(src):
            continue
        # Parse for the docstring (ok if it fails; description is blank)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        desc = "" if tree is None else _extract_description(tree)
        # Register the tool
        name = fname[:-len(".py")]
        TOOL_REGISTRY[name] = {
            "name": name,
            "file": fname,
            "description": desc,
        }
    # Output
    return {
        "success": True,
        "cached": False,
        "tooldir": tooldir,
        "tools": [TOOL_REGISTRY[name] for name in sorted(TOOL_REGISTRY)],
    }


# Run one of the discovered user scripts
def run_user_tool(
        name: str,
        argv: list | None = None,
        timeout: int | None = None) -> dict:
    r"""Run a script from the repo's tools folder

    The script must have been discovered by :func:`discover_user_tools`
    (discovery runs automatically if it has not been). The script's
    ``__main__`` guard is re-validated before running. The script is
    executed with the same Python executable as the CAPE agent and with
    the repo root folder as the working directory.

    :Call:
        >>> result = run_user_tool(name, argv=None, timeout=None)
    :Inputs:
        *name*: :class:`str`
            Name of a discovered script, with or without ``".py"``
        *argv*: {``None``} | :class:`list`
            Command-line arguments to the script, e.g.
            ``["-I", "1:5", "--force"]``; use ``["--help"]`` to show
            the script's own help message
        *timeout*: {``None``} | :class:`int`
            Maximum seconds to wait for the script
    :Outputs:
        *result*: :class:`dict`
            Keys include *success* (``True`` only if return code is 0),
            *returncode*, *stdout*, and *argv*
    :Versions:
        * 2026-09-02 ``@ddalle``: v1.0
    """
    # Check type of *name*
    if not isinstance(name, str) or not name:
        return {
            "success": False,
            "error": "'name' must be a nonempty string",
        }
    # Run discovery if the registry is empty
    if not TOOL_REGISTRY:
        discover_user_tools()
    # Normalize the name: strip the extension if provided
    name_py = name[:-len(".py")] if name.endswith(".py") else name
    # Reject any path components
    if os.sep in name_py or (os.altsep and os.altsep in name_py):
        return {
            "success": False,
            "error": f"Invalid tool name: '{name}'",
        }
    # Look up the script
    toolinfo = TOOL_REGISTRY.get(name_py)
    if toolinfo is None:
        return {
            "success": False,
            "error": f"Unknown user tool: '{name}'",
            "available_tools": sorted(TOOL_REGISTRY),
        }
    # Absolute path of the script
    fpath = os.path.join(_get_tooldir(), toolinfo["file"])
    # Resolve any links and check it's still in the tools folder
    freal = os.path.realpath(fpath)
    if os.path.dirname(freal) != os.path.realpath(_get_tooldir()):
        return {
            "success": False,
            "error": f"Script '{name}' resolves outside the tools folder",
        }
    # Re-validate the __main__ guard before running
    try:
        with open(freal) as fp:
            src = fp.read()
    except OSError as e:
        return {"success": False, "error": f"Could not read script: {e}"}
    if not _has_main_guard(src):
        TOOL_REGISTRY.pop(name_py, None)
        return {
            "success": False,
            "error": f"Script '{name}' has no __main__ guard",
        }
    # Normalize argv to a list of strings
    if argv is None:
        argv_tool = []
    elif isinstance(argv, list):
        argv_tool = [str(v) for v in argv]
    else:
        argv_tool = [str(argv)]
    # Folder to run in: repo root
    rootdir = os.getcwd() if ROOT_DIR is None else ROOT_DIR
    # Assemble the command
    cmd = [sys.executable, freal] + argv_tool
    # Run the script
    try:
        proc = subprocess.run(
            cmd,
            cwd=rootdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # Convert partial output to text
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
        if out:
            sys.stdout.write(out)
        return {
            "success": False,
            "error": f"Script timed out after {timeout} seconds",
            "timeout": timeout,
            "name": name_py,
            "argv": argv_tool,
            "stdout": toolutils._truncate_stdout(out or ""),
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"Could not run script: {e}",
            "name": name_py,
            "argv": argv_tool,
        }
    # Echo the script's output so the user sees it live
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    # Output
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "name": name_py,
        "argv": argv_tool,
        "stdout": toolutils._truncate_stdout(proc.stdout or ""),
    }


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# user-tools: running the repo's custom scripts

Use this skill to run customized scripts that the user has placed in
the `tools/` folder of the repo. These scripts extend CAPE's built-in
capabilities for this particular project but are not part of CAPE
itself; they use CAPE's Python infrastructure and usually accept
CAPE-style command-line arguments such as `-I 1:5` or `--force`.

## When to use

* The user asks what custom tools, scripts, or utilities this repo has.
* The user refers to one of these scripts by name or by *file name*.
* A task matches the description of a discovered script and the fixed
  CAPE tools do not provide the capability.

## Workflow

1. Call `discover_user_tools` to list the available scripts. Each
   entry has a `name`, `file`, and a one-line `description`. Skipped
   files (folders, `_`-prefixed files, and scripts without a
   `__main__` guard) do not appear. Use `refresh=true` if the user says
   they have just added or edited a script.
2. If you are unsure how to use a script, call `run_user_tool` with
   `argv=["--help"]` to show the script's own help message.
3. Call `run_user_tool` with the desired `argv`. Arguments are each a
   list item, e.g. `["-I", "1:5", "--force"]`. Scripts run with the
   repo root as the working folder (so file names like `pyFun.json`
   work as expected) and use the same Python environment as this agent.

## Safety: confirm before destructive runs

Discovery and `--help` calls are always safe to run directly. However,
most of these scripts *change* something: resubmitting jobs, editing
the run matrix, writing data book files, or deleting files. Before
running a script whose description or help text indicates it modifies
state, show the user the exact command you intend to run and ask for
confirmation -- unless the user's current request already explicitly
asked for that action.

## Results

The result includes `success` (true only when the script's exit code
was 0), `returncode`, and `stdout` (truncated if very long). The user
sees the full output live during the call; do not repeat it back. On
failure, inspect `stdout` for the script's own error message before
reporting.
"""

# Simplified skill definition
SKILL_DICT = {
    "user-tools": {
        "description": (
            "Discover and run the custom Python scripts in this "
            "repo's tools/ folder. Use when the user wants to list or "
            "run repo-specific utilities beyond CAPE's built-in tools."
        ),
        "content": SKILL_CONTENT,
        "tools": ["discover_user_tools", "run_user_tool"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "discover_user_tools": {
        "description": (
            "List executable Python scripts (those with a __main__ "
            "guard) in the repo's tools folder, with a one-line "
            "description of each. Call use_skill('user-tools') for "
            "full instructions first."
        ),
        "parameters": ["refresh"],
        "required": [],
    },
    "run_user_tool": {
        "description": (
            "Run a script from the repo's tools folder with "
            "command-line arguments; the script runs with the repo "
            "root as working folder using the agent's Python "
            "environment. Call use_skill('user-tools') for full "
            "instructions first."
        ),
        "parameters": ["name", "argv", "timeout"],
        "required": ["name"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)
