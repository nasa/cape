"""
Tools exposed to the CAPE agent.

For now there's just one: run_cape_c, which wraps
cape.cfdx.cli.cape_c() -- the same entry point the `cape` console
script uses.
"""

from __future__ import annotations

# Standard library
import os

# Third-party

# Local imports
from . import cfdxtools


# Change folder
def chdir(dirname: str):
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
        breakpoint()
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


# Fork JSON-chema defns from cfdxtools
TOOL_SCHEMAS = list(cfdxtools.TOOL_SCHEMAS)


# JSON-schema tool definitions, OpenAI-compatible
# (works with llama.cpp's /v1/chat/completions "tools" param
#  when the server is started with --jinja
#  and the model's chat template supports tool calling).
TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "chdir",
            "description": "Change the working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "dirname": {
                        "type": "string",
                        "description": (
                            "Name of folder to change to, can be either an "
                            "absolute path or a relative path."
                        ),
                    },
                },
                "required": [],
            }
        },
    })
TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "getcwd",
            "description": (
                "Show name of current working directory; also answer 'Where "
                "am I?' questions."
            ),
            "parameters": {},
        }
    })

TOOLS = dict(cfdxtools.TOOLS)
TOOLS["chdir"] = chdir
TOOLS["getcwd"] = getcwd
