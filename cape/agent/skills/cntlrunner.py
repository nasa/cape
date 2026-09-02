r"""
:mod:`cape.agent.skills.cntlrunner`: Run whitelisted ``Cntl`` methods
======================================================================

This module defines the built-in agent skill ``"cntl-runner"``, which
teaches the CAPE agent how to call methods of a CAPE run matrix control
instance (:class:`cape.cfdx.cntl.Cntl` and subclasses) that are not
covered by the fixed CLI tools.

The skill provides the :func:`run_cntl_methods` tool, which reads a CAPE
JSON file into a *Cntl* instance using
:func:`cape.cfdx.cli.read_cntl_cache` and then runs one or more methods
of that instance from a whitelist of read-only methods.
"""

# Standard library
import contextlib
import io
import json
import sys

# Local imports
from ..agentutils import _NPEncoder
from ..tools import toolutils
from ...cfdx import cli


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "f": {
        "description": (
            "Name of CAPE JSON file to read. If empty, CAPE will find "
            "the most appropriate file. If the user has specified a "
            "file, continue to use that file until they request a "
            "different one."
        ),
        "type": ["string", "null"],
    },
    "solver": {
        "description": (
            "Name of CAPE solver module, e.g. 'pycart' or 'pyfun'. "
            "Determined automatically from the JSON file if empty."
        ),
        "type": ["string", "null"],
    },
    "calls": {
        "description": (
            "Ordered list of method calls to run on the Cntl instance. "
            "Each entry has 'method' (required), 'args' (optional list "
            "of positional args), and 'kwargs' (optional dict of "
            "keyword args). Methods must be in the whitelist; unknown "
            "names are rejected before any call is run."
        ),
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "args": {"type": "array", "items": {}},
                "kwargs": {"type": "object"},
            },
            "required": ["method"],
            "additionalProperties": False,
        },
    },
}


# Whitelist of Cntl methods this skill may call, with summaries
METHOD_WHITELIST = {
    "CountQueuedCases": (
        "Count cases with a job currently in the PBS/Slurm queue. "
        "Optional filters: I, cons, re, filter, u."),
    "GetCurrentIter": (
        "Current iteration number for case index i. "
        "Args: i (int). Optional: force (bool)."),
    "GetIndices": (
        "Indices of cases matching subset constraints, returned as a "
        "list. Kwargs: cons, re, filter, glob, I, status, etc."),
    "GetLastIter": (
        "Final iteration number required for case index i. "
        "Args: i (int)."),
    "check_case_status": (
        "Status ('---', 'INCOMP', 'QUEUE', 'RUNNING', 'ZOMBIE', "
        "'FAIL', 'ERROR', 'DONE', 'PASS', 'PASS*') of case index i. "
        "Args: i (int). Optional: force (bool)."),
    "get_report_comps": (
        "List of (component, coefficient) pairs tracked in report "
        "rep (optional)."),
    "get_report_figs": (
        "List of figure names in report rep (optional)."),
    "get_report_subfigs": (
        "List of subfigure names in report rep (optional)."),
    "get_runmatrix_keys": (
        "Dict of run matrix keys with their types and dtypes. "
        "Optional: keyname (filename pattern to filter keys)."),
    "get_subfigs": (
        "List of all subfigures, or subfigures of one report if "
        "report is given."),
    "getval": (
        "Value of run matrix key opt for case index i, or a special "
        "value like 'progress' or 'iter'. Args: opt (str), i (int)."),
    "inspect_json": (
        "Show the JSON options at jq path jq (default '.'), "
        "optionally truncated to maxdepth levels."),
}


# Run one or more whitelisted Cntl methods
def run_cntl_methods(
        f: str | None = None,
        solver: str | None = None,
        calls: list | None = None) -> dict:
    r"""Read a CAPE JSON file and run whitelisted ``Cntl`` methods on it

    The JSON file is read into a *Cntl* instance using
    :func:`cape.cfdx.cli.read_cntl_cache`. Each entry of *calls* is a
    dict with a required ``"method"`` key and optional ``"args"`` and
    ``"kwargs"``, which must name a method in *METHOD_WHITELIST*.

    A failing call flags its own entry but does not prevent remaining
    calls from running; mutating methods are not in the whitelist.

    :Call:
        >>> result = run_cntl_methods(f=None, solver=None, calls=calls)
    :Inputs:
        *f*: :class:`str` | ``None``
            Name of CAPE JSON file (or use most recent)
        *solver*: :class:`str` | ``None``
            Solver module (or determine based on *f*)
        *calls*: :class:`list`\ [:class:`dict`]
            Ordered list of method calls; each has ``"method"`` and
            optional ``"args"`` (list) and ``"kwargs"`` (dict)
    :Outputs:
        *result*: :class:`dict`
            Keys include *success*, *results* (one entry per call), and
            *stdout*
    :Versions:
        * 2026-09-01 ``@ddalle``: v1.0
    """
    # Check overall type of *calls*
    if not isinstance(calls, list) or len(calls) == 0:
        return {
            "success": False,
            "error": (
                "'calls' must be a nonempty list of dicts with a "
                "'method' key"),
            "allowed_methods": sorted(METHOD_WHITELIST),
        }
    # Validate all calls *before* reading the JSON file
    for j, call in enumerate(calls):
        # Check type
        if not isinstance(call, dict):
            return {
                "success": False,
                "calls_completed": 0,
                "error": f"Call {j} is {type(call).__name__}, not dict",
                "allowed_methods": sorted(METHOD_WHITELIST),
            }
        # Get method name
        method = call.get("method")
        # Check that it's a whitelisted string
        if not isinstance(method, str) or method not in METHOD_WHITELIST:
            return {
                "success": False,
                "calls_completed": 0,
                "error": (
                    f"Call {j}: method {method!r} is not in the "
                    f"whitelist"),
                "allowed_methods": sorted(METHOD_WHITELIST),
            }
    # Capture STDOUT while also displaying it live
    buf = io.StringIO()
    results = []
    success = True
    with contextlib.redirect_stdout(toolutils.Tee(sys.stdout, buf)):
        # Read *cntl* (errors are handled by the calling agent loop)
        cntl = cli.read_cntl_cache(f, solver)
        # Run each call in order
        for j, call in enumerate(calls):
            # Get the method
            method = call["method"]
            func = getattr(cntl, method)
            # Normalize the args
            args = [
                toolutils.normalize_tool_arg(v)
                for v in call.get("args") or []
            ]
            kwargs = toolutils.normalize_kwargs(call.get("kwargs") or {})
            # Run it, flagging failures without aborting the other calls
            try:
                v = func(*args, **kwargs)
            except Exception as e:
                # NOTE: a fail-fast abort (for future mutating skills)
                # would go here: ``break`` instead of only flagging
                success = False
                results.append({
                    "method": method,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                })
            else:
                results.append({
                    "method": method,
                    "success": True,
                    "result": _jsonify(v),
                })
    # Output
    return {
        "success": success,
        "calls_completed": len(results),
        "f": f,
        "solver": solver,
        "results": results,
        "stdout": toolutils._truncate_stdout(buf.getvalue()),
    }


# Convert a result to JSON-compatible values
def _jsonify(v):
    # Try strict JSON round-trip (handles NumPy types)
    try:
        return json.loads(json.dumps(v, cls=_NPEncoder))
    except (TypeError, ValueError):
        # Fallback to repr for anything not serializable
        return repr(v)


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# cntl-runner: running Cntl methods

Use this skill when a task requires the CAPE Python API of the run
matrix control instance (`Cntl`) that the fixed CLI tools do not
expose. The skill's `run_cntl_methods` tool reads a CAPE JSON file into
a `Cntl` instance (using `cape.cfdx.cli.read_cntl_cache`, so repeated
reads of the same file are cached) and runs one or more methods from a
whitelist of read-only methods. All whitelisted methods are read-only;
do not attempt to call a method that is not in the whitelist.

## When to use

* The user asks a question about the run matrix, JSON options, report
  setup, or case status that the fixed tools cannot answer.
* You need a specific value, e.g. one run matrix key for one case, or
  one item from the JSON options.

## How to call

```json
{
  "f": "cape.json",
  "solver": null,
  "calls": [
    {"method": "get_runmatrix_keys"},
    {"method": "getval", "args": ["mach", 0]},
    {"method": "check_case_status", "args": [0]}
  ]
}
```

* `f`: JSON file name. Omit (null) to use the most recently modified
  CAPE JSON file in the repo. Once the user names a file, keep using it.
* `solver`: optional; usually determined from the JSON file.
* `calls`: an ordered list. Each call has a required `method` and
  optional `args` (list) and `kwargs` (dict).

The result has one entry per call (`method`, `success`, and `result` or
`error`). A failing call does not stop the other calls, but the overall
`success` will be false; inspect the entry's `error` before reporting
results.

## Whitelisted methods

* `GetIndices(cons=None, re=None, filter=None, I=None, status=None,
  ...)`: find case indices matching run matrix constraints. Prefer the
  fixed `cape_find` tool when it suffices; use this to chain the
  indices into further `Cntl` calls in one round.
* `get_runmatrix_keys(keyname=None)`: describe the run matrix keys.
* `getval(opt, i)`: one run matrix value (or special key like
  `"progress"`, `"iter"`) for case `i`.
* `check_case_status(i, force=False)`: status of one case.
* `GetCurrentIter(i)`: current iteration of case `i`.
* `GetLastIter(i)`: final required iteration of case `i`.
* `CountQueuedCases(I=None, cons=None, re=None, ...)`: count cases
  with a job currently in the queue.
* `get_subfigs(report=None)`, `get_report_comps(rep=None)`,
  `get_report_subfigs(rep=None)`, `get_report_figs(rep=None)`: report
  layout queries.
* `inspect_json(jq=".", maxdepth=None)`: read a subset of the JSON
  options, e.g. `jq=".RunControl"` shows the *RunControl* section.

## Chaining example

To report the status of all cases with `mach>1.2`:

1. Call with
   `calls=[{"method": "GetIndices", "kwargs": {"cons": "mach>1.2"}}]`.
2. Read the returned index list, then call again with a `calls` list
   containing one `check_case_status` entry per index.
"""

# Simplified skill definition (mirrors TOOL_DICT pattern for tools)
SKILL_DICT = {
    "cntl-runner": {
        "description": (
            "Read a CAPE JSON file into a Cntl instance and run "
            "whitelisted read-only Python API methods on it. Use for "
            "questions the fixed CLI tools cannot answer."
        ),
        "content": SKILL_CONTENT,
        "tools": ["run_cntl_methods"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "run_cntl_methods": {
        "description": (
            "Read a CAPE JSON file into a Cntl instance and run an "
            "ordered list of whitelisted read-only methods on it. Only "
            "use methods from the cntl-runner skill's whitelist; call "
            "use_skill('cntl-runner') for full instructions first."
        ),
        "parameters": ["f", "solver", "calls"],
        "required": ["calls"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)
