r"""
:mod:`cape.agent.tools.toolutils`: Common functions for CAPE tool usage
=======================================================================

This module provides a function :func:`wrap_cli` that places several
preprocessing, protection, and postprocessing steps around a tool call
eminated by an LLM.

This includes normalizing the arguments, e.g. replacing the string
``"true"`` with the Python value ``True``, removing any keyword
arguments set to ``None``, capturing STDOUT, and more.
"""

# Standard library
import contextlib
import io
import sys
from typing import Callable


def normalize_kwargs(kw: dict) -> dict:
    r"""Filter keyword args to tool, removing ``''`` and ``None``

    :Call:
        >>> kw_tool = normalize_kwargs(kw)
    :Inputs:
        *kw*: :class:`dict`
            Keyword args created by LLM tool caller
    :Outputs:
        *kw_tool*: :class:`dict`
            Modified *kw* as such:

            * String conversion: ``"true"`` -> ``True``
            * Remove keys set to ``None`` after normalization
    """
    # Filter out empty strings from kwargs
    kw_tool = {
        k: normalize_tool_arg(v) for k, v in kw.items()
        if v not in (None, "", "None", "null")
    }
    # Output
    return kw_tool


def normalize_tool_arg(v):
    r"""Normalize strings of special values to Pythonic values

    E.g. ``"null"`` -> ``None``
    """
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
