r"""
:mod:`cape.agent.tools.cntl`: CAPE API tools for :mod:`cape.agent`
=======================================================================

This module provides definitions and tool schemas for tool calls to most
of the low-level CLI functions defined in :mod:`cape.cfdx.cli`.
"""

# Standard library
from typing import Callable

# Local imports
from .toolutils import register_module_tools
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
    "report": {
        "description": "Name of specific report to generate. Optional",
        "type": ["string", "null"]
    },
}


def get_subfigs(f: str | None = None, report: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the subfigures
    return {
        "report": report,
        "subfigures": cntl.get_subfigs(report),
    }


def get_reports(f: str | None = None) -> dict:
    # Read *cntl*
    cntl = cli.read_cntl_q(f)
    # List the reports
    return {
        "reports": cntl.opts.get_ReportList(),
    }


# Simplified definitions not in OpenAPI format
TOOL_DICT = {
    "get_subfigs": {
        "description": (
            "List the subfigures, either of all reports if 'report' is not "
            "given or of a specific named report."
        ),
        "parameters": [
            "f",
            "report",
        ],
    },
    "get_reports": {
        "description": "Get list of reports available",
        "parameters": ["f"],
    }
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Tool sets per capability
TOOL_SETS = {
    "none": [],
    "low": [
        "get_subfigs",
        "get_reports",
    ],
    "medium": [
        "get_subfigs",
        "get_reports",
    ],
    "full": list(TOOL_DICT.keys())
}


# Function generator
def genr8_func(funcname: str) -> Callable:
    # Create a function
    def fn(*a, **kw):
        return {}
    # Return it
    return fn


# Register tools
register_module_tools(CAPE_PARAMS)
