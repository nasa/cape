r"""
:mod:`cape.agent.tools.cfdxtools`: CAPE CLI tools for :mod:`cape.agent`
=======================================================================

This module provides definitions and tool schemas for tool calls to most
of the low-level CLI functions defined in :mod:`cape.cfdx.cli`.
"""

# Local imports
from .toolutils import wrap_cli
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
    "report": {
        "description": "Name of specific report to generate. Optional",
        "type": ["string", "null"]
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
SUBSET_PARAMS = {
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
}

# Specifications for other common parameters
OTHER_CAPE_PARAMS = {
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


def cape_c(*a, **kw) -> dict:
    return wrap_cli(cli.cape_c, *a, **kw)


def cape_find(*a, **kw) -> dict:
    return wrap_cli(cli.cape_find, *a, **kw)


def cape_report(*a, **kw) -> dict:
    # Get notional result
    result = wrap_cli(cli.cape_report, *a, **kw)
    # Add follow-up if appropriate
    fpdf = result.get("reportfile")
    if fpdf:
        result["follow-up"] = "open_pdf"
    # Output
    return result


def cape_approve(*a, **kw) -> dict:
    return wrap_cli(cli.cape_approve, *a, **kw)


def cape_fail(*a, **kw) -> dict:
    return wrap_cli(cli.cape_fail, *a, **kw)


def cape_unmark(*a, **kw) -> dict:
    return wrap_cli(cli.cape_unmark, *a, **kw)


def cape_defail(*a, **kw) -> dict:
    return wrap_cli(cli.cape_dezombie, *a, **kw)


def cape_dezombie(*a, **kw) -> dict:
    return wrap_cli(cli.cape_dezombie, *a, **kw)


def cape_open_pdf(*a, **kw) -> dict:
    fpdf = kw.get("fpdf")
    return wrap_cli(cli.cape_open_pdf, fpdf, wait=True)


def cape_extend(*a, **kw) -> dict:
    return wrap_cli(cli.cape_extend, *a, **kw)


def cape_clean(*a, **kw) -> dict:
    return wrap_cli(cli.cape_clean, *a, **kw)


# Simplified definitions not in OpenAPI format
TOOL_DICT = {
    "cape_find": {
        "function": cape_find,
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
        "function": cape_c,
        "description": "check the status of one or more cases",
        "parameters": [
            "f",
            "I",
            "add_cols",
        ],
    },
    "cape_approve": {
        "function": cape_approve,
        "description": "PASS/approve one or more cases",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_defail": {
        "function": cape_defail,
        "description": "Clean up failure files, or 'defail' cases",
        "properties": ["f", "I"],
        "required": ["I"],
    },
    "cape_extend": {
        "function": cape_extend,
        "description": "Extend case(s) by running more iterations",
        "properties": ["f", "I", "extend"],
        "required": ["I", "extend"],
    },
    "cape_report": {
        "function": cape_report,
        "description": "Generate a PDF report for one or more cases",
        "properties": ["f", "I", "report"],
        "required": ["I"],
    },
    "open_pdf": {
        "function": cape_open_pdf,
        "description": "open a PDF for the user to view",
        "properties": ["fpdf"],
        "required": ["fpdf"],
    },
}


# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Fill out definitions for one tool
def _add_schema(name: str):
    # Get short options
    opts = TOOL_DICT[name]
    # Initialize parameter properties
    tool_arg_props = {
        opt: CAPE_PARAMS[opt] for opt in opts.get("parameters", [])
    }
    tool_args = {
        "type": "object",
        "properties": tool_arg_props,
        "required": opts.get("required", []),
    }
    # Initialize function schema
    tool_func = {
        "name": name,
        "description": opts["description"],
        "parameters": tool_args,
    }
    # Initialize tool schema
    tool_schema = {
        "type": "function",
        "function": tool_func,
    }
    # Append to overall schema
    TOOL_SCHEMAS.append(tool_schema)
    # Append to list
    TOOLS[name] = opts["function"]


# Add the definitions
for name in TOOL_DICT:
    try:
        _add_schema(name)
    except Exception:
        breakpoint()
