r"""
:mod:`cape.agent.tools.systools`: System tools for :mod:`cape.agent`
=====================================================================

This module provides definitions and tool schemas for basic system tool
calls, such as querying or changing the working directory.
"""

# Standard library
import os

# Local imports
from .toolutils import register_module_tools


# List of parameters common to multiple system tools
SYS_PARAMS = {
    "dirname": {
        "description": (
            "Name of folder to change to, can be either an "
            "absolute path or a relative path."
        ),
        "type": "string",
    },
}


# Change folder
def chdir(dirname: str) -> dict:
    # Check type
    if not isinstance(dirname, str):
        return {
            "success": False,
            "result": "Input was not a string",
            "error": f"TypeError: got {type(dirname).__name__}",
        }
    # Check for folder
    if not os.path.isdir(dirname):
        return {
            "success": False,
            "result": f"No such: folder {dirname}",
        }
    # Attempt to change directory
    try:
        os.chdir(dirname)
    except Exception:
        return {
            "success": False,
            "result": f"Could not change to {dirname}",
        }
    return {
        "success": True,
        "result": f"Folder changed to {os.path.basename(dirname)}",
    }


# Show folder
def getcwd() -> dict:
    # Get the current location
    dirname = os.getcwd()
    # Return it
    return {
        "success": True,
        "result": dirname,
    }


# Simplified definitions not in OpenAPI format
TOOL_DICT = {
    "chdir": {
        "description": "Change the working directory",
        "parameters": ["dirname"],
    },
    "getcwd": {
        "description": (
            "Show name of current working directory; also answer 'Where "
            "am I?' questions."
        ),
        "parameters": [],
    },
}

# JSON-schema tool definitions, OpenAI-compatible
TOOL_SCHEMAS = []
TOOLS = {}


# Register all the tools
register_module_tools(SYS_PARAMS)
