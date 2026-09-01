r"""
:mod:`cape.agent.tools.cfdxtools`: CAPE CLI tools for :mod:`cape.agent`
=======================================================================

This module provides definitions and tool schemas for tool calls to most
of the low-level CLI functions defined in :mod:`cape.cfdx.cli`.
"""

# Standard library
from typing import Callable

# Local imports
from .toolutils import register_module_tools, wrap_cli
from ...cfdx import cli


# List of parameters common to **all** run-matrix commands
CAPE_PARAMS = {
    "f": {
        "description": (
            "Name of JSON file to use. CAPE will find the most appropriate "
            "file if left empty. If users specify a file, continue to use "
            "that file until user specifically requests a different one."
            "Synonyms: file, json."
        ),
        "type": ["string", "null"],
    },
    "I": {
        "description": (
            "Indices of cases to consider. Indexing follows Python syntax. "
            "This can be a single case "
            "like '14', a comma-separated list like '14,19,20', "
            "a range such as '14:20', or a combination like 14,17:20. "
            "Examples:\n* Case 8 -> '8'\n* Cases 5-10 -> '5:11'"
        ),
        "type": ["string", "null"],
    },
    "re": {
        "description": (
            "Consider cases containing matches for this regular expression."
            "Synonym: regex."
        ),
        "type": ["string", "null"]
    },
    "cons": {
        "description": (
            "Add constraints on the run matrix keys. The user specifies the "
            "key and constraint. These can be specified as logical operators."
            "Example: Run cases where run matrix key called 'mach' is "
            "greater than 1.0 -> 'mach>1.0'. Protect the right-hand side with "
            "quotes if searching for a string value."
        ),
        "type": ["string", "null"]
    },
    "filter": {
        "description": (
            "Limit command to cases containing a string of text specified "
            "by the user. "
            "Example: Only show cases containing 'm3': filter='m3'"
        ),
        "type": ["string", "null"]
    },
    "user": {
        "description": (
            "Limit to cases owned by this specific user"
        ),
        "type": ["string", "null"]
    },
    "me": {
        "description": (
            "Limit to cases owned by the current user, equivalent to "
            "user='$USER'"
        ),
        "type": ["boolean", "null"]
    },
    "unmarked": {
        "description": (
            "Only consider cases with no PASS/ERROR markings. This should be "
            "used if a user asks to only see unmarked cases or cases that are"
            "not passed or is looking for cases with a specific status other "
            "than PASS/PASS*/ERROR."
        ),
        "type": ["boolean", "null"]
    },
    "marked": {
        "description": (
            "Only consider cases with PASS/ERROR markings. This should be "
            "used if a user asks to only see marked cases or cases that are "
            "passed."
        ),
        "type": ["boolean", "null"]
    },
    "add_cols": {
        "description": (
            "Additional columns to show in case status output. Multiple "
            "additional columns can be specified as comma-separated list."
        ),
        "type": ["string", "null"]
    },
    "h": {
        "description": (
            "Display help message and exit. The help message describes all "
            "commands a user has access to and can call through CAPE. "
            "Synonym: help."
        ),
        "type": ["boolean", "null"]
    },
    "n": {
        "description": (
            "Submit at most n cases. The user specifies the number of cases "
            "to run. This should run all avaiable cases available up to the "
            "number specified. Cases cannot be in status PASS or DONE."
            "Synonym: N."
        ),
        "type": ["integer", "null"]
    },
    "j": {
        "description": (
            "List the PBS/Slurm job ID. This is used when the user checks "
            "the status of one or more cases."
        ),
        "type": ["boolean", "null"]
    },
    "jq": {
        "description": (
            "Path to item or subset of options using jq syntax, e.g. "
            "'.RunControl.nProc' or '.Config.Components[0]'. Default: "
            "'.', which displays the entire JSON options."
        ),
        "type": ["string", "null"]
    },
    "maxdepth": {
        "description": (
            "Maximum depth of dicts to show when inspecting JSON "
            "options; deeper dicts are replaced by {}."
        ),
        "type": ["integer", "null"]
    },
    "batch": {
        "description": (
            "Submit PBS/Slurm job and run this command."
        ),
        "type": ["boolean", "null"]
    },
    "e": {
        "description": (
            "Execute the command EXEC."
            "Synonym: exec."
        ),
        "type": ["string", "null"]
    },
    "extend": {
        "description": (
            "Number of times to extend case. Default: 1"
        ),
        "type": ["integer"]
    },
    "no-restart": {
        "description": (
            "Only submit new cases when submitting jobs."
        ),
        "type": ["boolean", "null"]
    },
    "no-start": {
        "description": (
            "Only set up cases. Do not start or submit cases to run."
        ),
        "type": ["boolean", "null"]
    },
    "q": {
        "description": (
            "Submit to a specific PBS/Slurm queue. The target queue is "
            "specified by the user. This command overrides the queue value "
            "in the input JSON file."
            "Synonyms: queue."
        ),
        "type": ["string", "null"]
    },
    "qsub": {
        "description": (
            "After extending or modifying a case, also submit it. "
            "Default: true"
        ),
        "type": ["boolean", "null"]
    },
    "report": {
        "description": "Name of specific report to generate. Optional",
        "type": ["string", "null"]
    },
    "subfig": {
        "description": (
            "Name of the report subfigure to create and open its image."
        ),
        "type": ["string", "null"]
    },
    "start": {
        "description": (
            "Set this option to 'false' in order to set a case up but not "
            "start or submit it."
        ),
        "type": ["boolean", "null"]
    },
    "u": {
        "description": (
            "Pretend to be the user UID. The original user is able to act "
            "as the user of the UID they specify."
        ),
        "type": ["string", "null"]
    },
    "x": {
        "description": (
            "Execute a Python script after reading the JSON file. The script "
            "is specified by the user and can only run after the JSON."
        ),
        "type": ["string", "null"]
    },
}


def cape_report(*a, **kw) -> dict:
    # Get notional result
    result = wrap_cli(cli.cape_report, *a, **kw)
    # Add follow-up if appropriate
    fpdf = result.get("reportfile")
    if fpdf:
        result["follow-up"] = "open_pdf"
    # Output
    return result


def cape_open_pdf(*a, **kw) -> dict:
    fpdf = kw.get("fpdf")
    return wrap_cli(cli.cape_open_pdf, fpdf, wait=True)


def cape_c(*a, **kw) -> dict:
    kw["__long_stdout"] = True
    return wrap_cli(cli.cape_c, *a, **kw)


def cape_inspect_json(*a, **kw) -> dict:
    # Function to keep dict outputs intact under wrap_cli()
    def shim(*a, **kw):
        ierr, v = cli.cape_inspect_json(*a, **kw)
        return ierr, {"result": v}
    # Wrap the shim so the inspected value is always in "result"
    return wrap_cli(shim, *a, **kw)


# Simplified definitions not in OpenAPI format
TOOL_DICT = {
    "cape_find": {
        "description": "find the indices of requested cases",
        "parameters": [
            "I",
            "cons",
            "f",
            "filter",
            "marked",
            "me",
            "re",
            "unmarked",
            "user",
        ],
    },
    "cape_c": {
        "description": (
            "Check the status of one or more cases. A status of '---' "
            "means the case has not been started or set up yet. You "
            "can use *add_cols* to show the values of more run matrix "
            "values."
        ),
        "parameters": [
            "f",
            "I",
            "add_cols",
        ],
    },
    "cape_inspect_json": {
        "description": (
            "Show an item or subset of the JSON options, e.g. "
            "jq='.RunControl.nProc'. Use maxdepth to limit the size "
            "of the output."
        ),
        "parameters": ["f", "jq", "maxdepth"],
    },
    "cape_apply": {
        "description": "Re-apply settings to one or more cases",
        "properties": ["f", "I", "qsub"],
        "required": ["I"],
    },
    "cape_approve": {
        "description": "PASS/approve one or more cases",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_defail": {
        "description": "Clean up failure files, or 'defail' cases",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_extend": {
        "description": "Extend case(s) by running more iterations",
        "properties": ["f", "I", "extend", "qsub"],
        "required": ["I", "extend"],
    },
    "cape_report": {
        "description": "Generate a PDF report for one or more cases",
        "properties": ["f", "I", "report"],
        "required": ["I"],
    },
    "cape_fail": {
        "description": "FAIL/mark cases as errors",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_unmark": {
        "description": "Remove PASS/ERROR markings from cases",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_dezombie": {
        "description": "Clean up ZOMBIE cases (jobs stalled)",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_clean": {
        "description": "Remove extra files not necessary for running a case",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_archive": {
        "description": (
            "Archive cases to long-term storage; then delete files not "
            "needed for post-processing"),
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_open_pdf": {
        "description": "Open a PDF for user to view",
        "properteis": ["fpdf"],
        "required": ["fpdf"],
    },
    "cape_open_subfig": {
        "description": (
            "Create one report subfigure for the selected cases and "
            "open (display) its image"
        ),
        "parameters": ["f", "I", "subfig"],
        "required": ["subfig"],
    },
    "cape_unarchive": {
        "description": "Expand files from archive",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_skeleton": {
        "description": (
            "Clean up case folder after ALL processing is finished; "
            "leave only key files"
        ),
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_rm": {
        "description": "Delete entire case folders",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_run": {
        "description": "Run CFD solver in the current (case) folder",
        "properties": []
    },
    "cape_start": {
        "description": "Set up and/or start/submit cases",
        "properties": ["f", "I", "start"],
        "required": ["I"],
    },
    "cape_qdel": {
        "description": "Delete PBS/Slurm job of case(s)",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_check_db": {
        "description": "Check completion of all databook components",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_check_fm": {
        "description": "Check completion of all force & moment components",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_check_ll": {
        "description": "Check completion of all line load components",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_check_triqfm": {
        "description": "Check completion of TriqFM components",
        "properties": ["f", "I"],
        "required": ["I"],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Tool sets per capability
TOOL_SETS = {
    "none": [],
    "low": [
        "cape_find",
        "cape_c",
        "cape_inspect_json",
        "cape_clean",
        "cape_open_pdf",
        "cape_open_subfig",
        "cape_report",
        "cape_start",
    ],
    "medium": [
        "cape_find",
        "cape_c",
        "cape_approve",
        "cape_check_db",
        "cape_clean",
        "cape_fail",
        "cape_open_pdf",
        "cape_open_subfig",
        "cape_qdel",
        "cape_report",
        "cape_rm",
        "cape_run",
        "cape_start",
        "cape_unmark",
    ],
    "full": list(TOOL_DICT.keys())
}


# Function generator
def genr8_func(funcname: str) -> Callable:
    # Create a function
    def fn(*a, **kw):
        return wrap_cli(getattr(cli, funcname), *a, **kw)
    # Return it
    return fn


# Register tools
register_module_tools(CAPE_PARAMS)
