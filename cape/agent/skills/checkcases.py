r"""
:mod:`cape.agent.skills.checkcases`: Efficient case-status queries
===================================================================

This module defines the built-in agent skill ``"check-cases"``, which
teaches the CAPE agent how to use the full capability of the ``cape
check`` command (:func:`cape.cfdx.cli.cape_c`) efficiently.

The fixed ``cape_c`` tool exposes only *f*, *I*, and *add_cols*, so
every call computes the default columns (including ``status``,
``progress``, ``queue``, and ``cpu-hours``), which are the most
expensive ones. The skill's :func:`cape_check` tool exposes explicit
column and counter selection and the full set of case-subset filters
so the agent computes only what the user's question needs.
"""

# Local imports
from ..tools import toolutils
from ...cfdx import cli


# Explicit-column split (used only for documentation in this module)
CHEAP_COLS = (
    "i",
    "frun",
    "group",
    "case",
    "maxiter",
)
COSTLY_COLS = (
    "status",
    "progress",
    "iter",
    "queue",
    "job",
    "job-id",
    "phase",
    "cpu-hours",
    "cpu-abbrev",
    "dirsize",
    "files",
)


# Parameter definitions for the tool schema
SKILL_PARAMS = {
    "f": {
        "description": (
            "Name of CAPE JSON file to use. If empty, CAPE finds the "
            "most appropriate file. If the user specified a file, keep "
            "using it until they name a different one."
        ),
        "type": ["string", "null"],
    },
    "I": {
        "description": (
            "Case indices using Python slice syntax: '8', '5:11', or "
            "'14,17:20'. Prefer the cons, re, user, status, marked, "
            "or unmarked filters when the user describes cases by "
            "property instead of index."
        ),
        "type": ["string", "null"],
    },
    "cols": {
        "description": (
            "Explicit comma-separated list of columns to show, "
            "REPLACING the slow default columns. Cheap columns: 'i', "
            "'frun', 'group', 'case', 'maxiter', and any run matrix "
            "key (e.g. 'mach', 'user'). Expensive columns, computed "
            "per case: 'status', 'progress', 'iter', 'queue', 'job', "
            "'phase', 'cpu-hours', 'dirsize', 'files'. Example for a "
            "fast query: cols='i,frun,user', counters=''."
        ),
        "type": ["string", "null"],
    },
    "counters": {
        "description": (
            "Comma-separated list of columns to summarize with counts "
            "after the table. Default 'status', which forces a status "
            "check of every case. Pass an empty string '' to skip all "
            "counters for fast queries that do not need status counts."
        ),
        "type": ["string", "null"],
    },
    "hide_cols": {
        "description": (
            "Comma-separated list of default columns to hide, e.g. "
            "'progress,cpu-hours' to keep default status output but "
            "skip those two slower columns."
        ),
        "type": ["string", "null"],
    },
    "status": {
        "description": (
            "Only show cases with this status: '---', 'INCOMP', "
            "'QUEUE', 'FAIL', 'ERROR', 'DONE', 'PASS', 'PASS*', or "
            "'ZOMBIE'. Requires computing the status of every "
            "selected case, so it is not a cheap filter."
        ),
        "type": ["string", "null"],
    },
    "cons": {
        "description": (
            "Constraint on run matrix keys, e.g. 'mach>1.0'. Get "
            "explicit run matrix keys unless user is explicit. Extra"
            "keys 'Mach', 'alpha', and 'beta' are likely available "
            "even if not in the .RunMatrix.Keys list.  Protect string "
            "values with quotes. Cheap."
        ),
        "type": ["string", "null"],
    },
    "re": {
        "description": (
            "Only cases whose full folder name contains a match for "
            "this regular expression. Cheap."
        ),
        "type": ["string", "null"],
    },
    "user": {
        "description": "Only cases owned by this user. Cheap.",
        "type": ["string", "null"],
    },
    "me": {
        "description": (
            "Only cases owned by the current user, equivalent to "
            "user='$USER'. Cheap."
        ),
        "type": ["boolean", "null"],
    },
    "marked": {
        "description": "Only cases with PASS/ERROR markings. Cheap.",
        "type": ["boolean", "null"],
    },
    "unmarked": {
        "description": "Only cases without PASS/ERROR markings. Cheap.",
        "type": ["boolean", "null"],
    },
    "j": {
        "description": "Also show the PBS/Slurm job ID column.",
        "type": ["boolean", "null"],
    },
    "nproc": {
        "description": (
            "Number of parallel workers for expensive columns; "
            "default 8. Raise to 16-32 when checking expensive "
            "columns (status, progress) for hundreds of cases."
        ),
        "type": ["integer", "null"],
    },
}


# Run ``cape check`` with the full option surface
def cape_check(*a, **kw) -> dict:
    r"""Check case status with explicit column and counter selection

    This wraps :func:`cape.cfdx.cli.cape_c` after translating a few
    agent-friendly inputs: comma-separated strings for *cols*,
    *counters*, and *hide_cols* are split into lists, and an empty
    string for *counters* becomes an empty list (disabling all
    counter output).

    :Call:
        >>> result = cape_check(*a, **kw)
    :Inputs:
        *a*: :class:`tuple`
            Positional arguments passed to the CAPE CLI function
        *kw*: :class:`dict`
            Keyword arguments passed to the CAPE CLI function
    :Outputs:
        *result*: :class:`dict`
            Wrapped CLI result
    """
    # Split comma-separated list options into lists
    for opt in ("cols", "counters", "hide_cols"):
        v = kw.get(opt)
        if isinstance(v, str):
            kw[opt] = [vj.strip() for vj in v.split(",") if vj.strip()]
    # Allow full stdout length for status tables
    kw["__long_stdout"] = True
    return toolutils.wrap_cli(cli.cape_c, *a, **kw)


# Allow CLI-equivalent display ("$ cape check ...") in the agent UI
cli.CMD_FUNCS.setdefault("cape_check", "check")


# Full Markdown instructions provided to the agent via ``use_skill``
SKILL_CONTENT = r"""
# check-cases: status checks that only compute what you need

Use `cape_check` for any question about the state, location, owner, or
run matrix values of cases. It wraps `cape check`, which by default
computes SIX columns per case (`i`, `frun`, `status`, `progress`,
`queue`, `cpu-hours`), four of which require per-case filesystem,
queue, or solver-output reads -- and it counts statuses afterward,
forcing a status pass even if status is never shown.

**Always decide which columns the question actually needs, then pass
`cols` and usually `counters=''`.** Cheap columns are `i`, `frun`,
`group`, `case`, `maxiter`, and every run matrix key (e.g. `mach`,
`alpha`, `user`, if defined in the run matrix). Expensive columns,
computed per case: `status`, `progress`, `iter`, `queue`, `job`,
`job-id`, `phase`, `cpu-hours`, `cpu-abbrev`, `dirsize`, `files`.

## Recipes

* "Who owns cases 100:500?" ->
  `cape_check(I='100:500', cols='i,frun,user', counters='')`.
  Do NOT use the defaults; no status check happens at all.
* "List all the cases" -> `cols='i,frun'`, `counters=''`.
* "Which cases are zombies?" -> `status='ZOMBIE'` with default
  columns. Computing status is required to filter by it, so this is
  intrinsically not a cheap query.
* Status sweep of a large matrix -> use the default columns (omit
  `cols` and `counters`), but consider `nproc=16` or `nproc=32` if
  there are hundreds of cases.
* Default-ish view minus the slowest column ->
  `hide_cols='progress,cpu-hours'`.
* "Show my cases that are already in the queue" ->
  `me=true, status='QUEUE'`.
* "Which cases have Mach above 1.2?" ->
  `cons='mach>1.2', cols='i,frun,mach', counters=''`.

## Filters

Filter cases cheaply by property instead of hand-building `I` when
the user describes cases by property: `cons` (run matrix
constraints), `re` (folder name contains a match for this regex),
`user` / `me` (owner), `marked` / `unmarked` (PASS/ERROR markings).
The `I` selector and these filters can be combined.

## Results

* The table is streamed to the user live; do not repeat it back.
  Summarize only what answers the question.
* When the `status` column is shown, the tool result's `status` key
  maps each status name to the list of case indices with that status;
  `n` counts the cases processed.
* Case indices are 0-based Python slices: cases 5-10 are `I='5:11'`.
"""

# Simplified skill definition
SKILL_DICT = {
    "check-cases": {
        "description": (
            "Check case status, owner, location, or run matrix values "
            "using explicit column selection so only the needed data "
            "is computed."
        ),
        "content": SKILL_CONTENT,
        "tools": ["cape_check"],
    },
}

# Simplified tool definitions not in OpenAPI format
TOOL_DICT = {
    "cape_check": {
        "description": (
            "Check status, progress, owner, or run matrix values of "
            "one or more cases, computing only the columns you "
            "request. Call use_skill('check-cases') for guidance on "
            "efficient column selection first."
        ),
        "parameters": [
            "f",
            "I",
            "cols",
            "counters",
            "hide_cols",
            "status",
            "cons",
            "re",
            "user",
            "me",
            "marked",
            "unmarked",
            "j",
            "nproc",
        ],
        "required": [],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register tools
toolutils.register_module_tools(SKILL_PARAMS)
