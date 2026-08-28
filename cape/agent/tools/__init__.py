"""
Tools exposed to the CAPE agent.

For now there's just one: run_cape_c, which wraps
cape.cfdx.cli.cape_c() -- the same entry point the `cape` console
script uses.
"""

from __future__ import annotations

# Standard library
import contextlib
import io
import os
import sys
import shutil
from typing import Callable

# Third-party
from cape.cfdx import cli

# Local imports
from .cfdxtools import SUBSET_PARAMS, OTHER_CAPE_PARAMS


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


def normalize_kwargs(kw: dict) -> dict:
    # Filter out empty strings from kwargs
    kw_tool = {
        k: normalize_tool_arg(v) for k, v in kw.items()
        if v not in (None, "", "None", "null")
    }
    # Output
    return kw_tool


def normalize_tool_arg(v):
    # Check for special cases
    if v in ("null", "None"):
        return None
    elif v in ("True", "true"):
        return True
    elif v in ("False", "false"):
        return False
    else:
        return v


def wrap_cli(func: Callable, *a, **kw) -> dict:
    # Normalize the args
    a_tool = (normalize_tool_arg(aj) for aj in a)
    # Normalize the kwargs
    kw_tool = normalize_kwargs(kw)
    # Redirect STDOUT while able
    buf = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        # Run the tool
        try:
            # Call the nominal tool
            ierr, result = func(*a_tool, **kw_tool)
            # Output
            if isinstance(result, dict):
                # Add information
                result["success"] = True
                result["returncode"] = ierr
            else:
                # Combine information
                result = {
                    "result": result,
                    "success": True,
                    "returncode": ierr,
                }
        except Exception as e:
            result = {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
            }
    # Save the captured STDOUT
    result["stdout"] = buf.getvalue()
    # Output
    return result


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


# Delete folder
def rm(dirname: str):
    if not isinstance(dirname, str):
        return {
            "success": False,
            "result": "Input was not a string",
            "error": f"TypeError: got {type(dirname).__name__}",
        }

    if not os.path.isdir(dirname):
        return {
            "success": False,
            "result": f"No such: folder {dirname}",
        }

    try:
        shutil.rmtree(dirname)
    except Exception:
        return {
            "success": False,
            "result": f"Could not delete folder {dirname}",
        }
    return {
        "success": True,
        "result": f"Deleted folder {dirname}",
    }


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


# Class to capture STDOUT and report it
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


# JSON-schema tool definitions, OpenAI-compatible
# (works with llama.cpp's /v1/chat/completions "tools" param
#  when the server is started with --jinja
#  and the model's chat template supports tool calling).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "cape_find",
            "description": (
                "Find the indices of requested cases."
            ),
            "parameters": {
                "type": "object",
                "properties": SUBSET_PARAMS,
                "required": [],
            }
        },
    },
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
    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "open_pdf",
            "description": (
                "Open a PDF for the user to view and wait for user to close."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fpdf": {
                        "type": "string",
                        "description": "Name of the file to open"
                    }
                },
                "required": ["fpdf"],
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cape_c",
            "description": (
                "Check the status of one or more cases. A status "
                "of '---' means the case has not been started yet. A status "
                "of 'PASS' means completed and approved. 'PASS*' means the "
                "case is approved but does not appear to be DONE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "f": SUBSET_PARAMS["f"],
                    "I": SUBSET_PARAMS["I"],
                    "add_cols": OTHER_CAPE_PARAMS["add_cols"],
                },
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cape_approve",
            "description": "PASS/approve one or more cases",
            "parameters": {
                "properties": {
                    "f": SUBSET_PARAMS["f"],
                    "I": SUBSET_PARAMS["I"],
                }
            },
            "required": ["I"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cape_defail",
            "description": "Clean up failure files, or 'defail' cases",
            "parameters": {
                "properties": {
                    "f": SUBSET_PARAMS["f"],
                    "I": SUBSET_PARAMS["I"],
                }
            },
            "required": ["I"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cape_extend",
            "description": "Extend case(s) by running more iterations",
            "parameters": {
                "properties": {
                    "f": SUBSET_PARAMS["f"],
                    "I": SUBSET_PARAMS["I"],
                    "extend": {
                        "description": (
                            "Number of times to extend case. Default: 1"
                        ),
                        "type": ["integer"]
                    }
                }
            },
            "required": ["I", "extend"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cape_report",
            "description": (
                "Generate a PDF report for one or more cases."
            ),
            "parameters": {
                "properties": {
                    "f": SUBSET_PARAMS["f"],
                    "I": SUBSET_PARAMS["I"],
                    "report": {
                        "description": (
                            "Name of specific report to generate. Optional"
                        ),
                        "type": ["string", "null"]
                    }
                },
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rm",
            "description": (
                "Delete the provided directory and all of the content inside."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dirname": {
                        "type": "string",
                        "description": (
                            "Name of folder to delete, can be either an "
                            "absolute path or a relative path."
                        ),
                    },
                },
                "required": [],
            }
        },
    },
]

TOOLS = {
    "chdir": chdir,
    "cape_approve": cape_approve,
    "cape_c": cape_c,
    "cape_defail": cape_defail,
    "cape_extend": cape_extend,
    "cape_find": cape_find,
    "cape_report": cape_report,
    "open_pdf": cape_open_pdf,
    "getcwd": getcwd,
    "rm": rm,
}
